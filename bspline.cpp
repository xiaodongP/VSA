#include "bspline.h"

#include <algorithm>
#include <cmath>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>
#include <fstream>
#include <limits>
#include <stdexcept>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::Vector3d;
using std::string;
using std::vector;

namespace {

static bool finite_double(double x) {
    return std::isfinite(x);
}

static void validate_degree(int degree) {
    if (degree < 0) {
        throw std::invalid_argument("B-spline degree must be non-negative");
    }
}

static void validate_knots(const vector<double>& knots, int degree, int control_count) {
    validate_degree(degree);
    if (control_count <= 0) {
        throw std::invalid_argument("B-spline needs at least one control point");
    }
    if (control_count < degree + 1) {
        throw std::invalid_argument("control_count must be at least degree + 1");
    }
    if ((int)knots.size() != control_count + degree + 1) {
        throw std::invalid_argument("knot count must equal control_count + degree + 1");
    }
    for (double k : knots) {
        if (!finite_double(k)) {
            throw std::invalid_argument("knots must be finite");
        }
    }
    for (int i = 1; i < (int)knots.size(); i++) {
        if (knots[i] < knots[i - 1]) {
            throw std::invalid_argument("knots must be nondecreasing");
        }
    }
    if (knots[degree] >= knots[knots.size() - degree - 1]) {
        throw std::invalid_argument("B-spline parameter domain is empty");
    }
}

static double domain_start(const vector<double>& knots, int degree) {
    return knots[degree];
}

static double domain_end(const vector<double>& knots, int degree) {
    return knots[knots.size() - degree - 1];
}

static double checked_parameter(
    double t,
    const vector<double>& knots,
    int degree,
    bool for_derivative) {
    if (!finite_double(t)) {
        throw std::invalid_argument("B-spline parameter must be finite");
    }

    const double a = domain_start(knots, degree);
    const double b = domain_end(knots, degree);
    const double scale = std::max(1.0, std::abs(b - a));
    const double eps = 1e-12 * scale;

    if (t < a - eps || t > b + eps) {
        throw std::out_of_range("B-spline parameter outside knot domain");
    }
    t = std::max(a, std::min(b, t));

    if (for_derivative) {
        if (t == a) return std::nextafter(a, b);
        if (t == b) return std::nextafter(b, a);
    }
    return t;
}

static double basis_recursive(
    int i,
    int degree,
    double t,
    const vector<double>& knots) {
    if (i < 0 || i + degree + 1 >= (int)knots.size()) return 0.0;

    if (degree == 0) {
        const int last_basis = (int)knots.size() - 2;
        const double end = knots.back();
        if (knots[i] <= t && t < knots[i + 1]) return 1.0;
        if (t == end && i == last_basis) return 1.0;
        return 0.0;
    }

    double value = 0.0;
    const double left_den = knots[i + degree] - knots[i];
    if (left_den > 0.0) {
        value += (t - knots[i]) / left_den *
                 basis_recursive(i, degree - 1, t, knots);
    }

    const double right_den = knots[i + degree + 1] - knots[i + 1];
    if (right_den > 0.0) {
        value += (knots[i + degree + 1] - t) / right_den *
                 basis_recursive(i + 1, degree - 1, t, knots);
    }

    return value;
}

static double basis_derivative_recursive(
    int i,
    int degree,
    double t,
    int derivative_order,
    const vector<double>& knots) {
    if (derivative_order < 0) {
        throw std::invalid_argument("derivative order must be non-negative");
    }
    if (derivative_order == 0) {
        return basis_recursive(i, degree, t, knots);
    }
    if (degree == 0 || derivative_order > degree) {
        return 0.0;
    }
    if (i < 0 || i + degree + 1 >= (int)knots.size()) return 0.0;

    double value = 0.0;
    const double left_den = knots[i + degree] - knots[i];
    if (left_den > 0.0) {
        value += degree / left_den *
                 basis_derivative_recursive(
                     i, degree - 1, t, derivative_order - 1, knots);
    }

    const double right_den = knots[i + degree + 1] - knots[i + 1];
    if (right_den > 0.0) {
        value -= degree / right_den *
                 basis_derivative_recursive(
                     i + 1, degree - 1, t, derivative_order - 1, knots);
    }

    return value;
}

static void validate_curve(const BSplineCurve3D& curve) {
    validate_knots(curve.knots, curve.degree, (int)curve.control_points.size());
}

static void validate_surface(const BSplineSurface3D& surface) {
    if (surface.control_grid.empty() || surface.control_grid[0].empty()) {
        throw std::invalid_argument("surface control grid must be non-empty");
    }
    const int nu = (int)surface.control_grid.size();
    const int nv = (int)surface.control_grid[0].size();
    for (const vector<Vector3d>& row : surface.control_grid) {
        if ((int)row.size() != nv) {
            throw std::invalid_argument("surface control grid must be rectangular");
        }
    }
    validate_knots(surface.knots_u, surface.degree_u, nu);
    validate_knots(surface.knots_v, surface.degree_v, nv);
}

static void write_obj_vertex(std::ofstream& fout, const Vector3d& p) {
    fout << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
}

static void add_fit_coefficient(
    MatrixXd& dense_A,
    vector<Triplet<double>>& sparse_triplets,
    bool use_sparse,
    int row,
    int col,
    double value) {
    if (std::abs(value) <= 0.0) return;
    if (use_sparse) {
        sparse_triplets.push_back(Triplet<double>(row, col, value));
    } else {
        dense_A(row, col) += value;
    }
}

static bool solve_fit_system(
    const MatrixXd& dense_A,
    const vector<Triplet<double>>& sparse_triplets,
    bool use_sparse,
    int row_count,
    int unknown_count,
    const MatrixXd& B,
    MatrixXd& X) {
    if (unknown_count == 0) {
        X.resize(0, 3);
        return true;
    }

    if (use_sparse) {
        SparseMatrix<double> A(row_count, unknown_count);
        A.setFromTriplets(sparse_triplets.begin(), sparse_triplets.end());
        Eigen::SparseQR<SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success) return false;
        X = solver.solve(B);
        return solver.info() == Eigen::Success;
    }

    X = dense_A.colPivHouseholderQr().solve(B);
    return X.rows() == unknown_count && X.cols() == 3;
}

} // namespace

vector<double> make_open_uniform_knot_vector(int control_count, int degree) {
    validate_degree(degree);
    if (control_count < degree + 1) {
        throw std::invalid_argument("control_count must be at least degree + 1");
    }

    vector<double> knots(control_count + degree + 1, 0.0);
    const int interior_count = control_count - degree - 1;
    for (int i = 0; i <= degree; i++) {
        knots[i] = 0.0;
        knots[control_count + i] = 1.0;
    }
    for (int j = 1; j <= interior_count; j++) {
        knots[degree + j] = (double)j / (double)(interior_count + 1);
    }
    return knots;
}

double bspline_basis(
    int i,
    int degree,
    double t,
    const vector<double>& knots) {
    validate_degree(degree);
    const int basis_count = (int)knots.size() - degree - 1;
    if (basis_count <= 0) {
        throw std::invalid_argument("invalid knot vector for basis evaluation");
    }
    validate_knots(knots, degree, basis_count);
    t = checked_parameter(t, knots, degree, false);
    if (t == domain_end(knots, degree) && i == basis_count - 1) {
        return 1.0;
    }
    if (t == domain_end(knots, degree)) {
        return 0.0;
    }
    return basis_recursive(i, degree, t, knots);
}

double bspline_basis_derivative(
    int i,
    int degree,
    double t,
    int derivative_order,
    const vector<double>& knots) {
    validate_degree(degree);
    if (derivative_order < 0) {
        throw std::invalid_argument("derivative order must be non-negative");
    }
    const int basis_count = (int)knots.size() - degree - 1;
    if (basis_count <= 0) {
        throw std::invalid_argument("invalid knot vector for basis derivative");
    }
    validate_knots(knots, degree, basis_count);
    t = checked_parameter(t, knots, degree, derivative_order > 0);
    return basis_derivative_recursive(i, degree, t, derivative_order, knots);
}

BSplineCurve3D::BSplineCurve3D()
    : degree(3) {}

BSplineCurve3D::BSplineCurve3D(
    int degree_,
    const vector<double>& knots_,
    const vector<Vector3d>& control_points_)
    : degree(degree_),
      knots(knots_),
      control_points(control_points_) {
    validate_curve(*this);
}

Vector3d BSplineCurve3D::evaluate(double t) const {
    validate_curve(*this);
    t = checked_parameter(t, knots, degree, false);
    if (t == domain_start(knots, degree)) return control_points.front();
    if (t == domain_end(knots, degree)) return control_points.back();

    Vector3d value = Vector3d::Zero();
    for (int i = 0; i < (int)control_points.size(); i++) {
        value += bspline_basis(i, degree, t, knots) * control_points[i];
    }
    return value;
}

Vector3d BSplineCurve3D::derivative(double t, int order) const {
    validate_curve(*this);
    if (order < 0) {
        throw std::invalid_argument("derivative order must be non-negative");
    }
    if (order == 0) return evaluate(t);
    t = checked_parameter(t, knots, degree, true);

    Vector3d value = Vector3d::Zero();
    for (int i = 0; i < (int)control_points.size(); i++) {
        value += bspline_basis_derivative(i, degree, t, order, knots) *
                 control_points[i];
    }
    return value;
}

vector<Vector3d> BSplineCurve3D::sample(int count) const {
    validate_curve(*this);
    if (count <= 0) {
        throw std::invalid_argument("sample count must be positive");
    }

    vector<Vector3d> points;
    points.reserve(count);
    const double a = domain_start(knots, degree);
    const double b = domain_end(knots, degree);
    if (count == 1) {
        points.push_back(evaluate(a));
        return points;
    }

    for (int i = 0; i < count; i++) {
        double t = a + (b - a) * (double)i / (double)(count - 1);
        points.push_back(evaluate(t));
    }
    return points;
}

BSplineCurveFitResult::BSplineCurveFitResult()
    : mean_error(0.0),
      max_error(0.0),
      success(false) {}

BSplineSurface3D::BSplineSurface3D()
    : degree_u(3),
      degree_v(3) {}

BSplineSurface3D::BSplineSurface3D(
    int degree_u_,
    int degree_v_,
    const vector<double>& knots_u_,
    const vector<double>& knots_v_,
    const vector<vector<Vector3d>>& control_grid_)
    : degree_u(degree_u_),
      degree_v(degree_v_),
      knots_u(knots_u_),
      knots_v(knots_v_),
      control_grid(control_grid_) {
    validate_surface(*this);
}

Vector3d BSplineSurface3D::evaluate(double u, double v) const {
    validate_surface(*this);
    u = checked_parameter(u, knots_u, degree_u, false);
    v = checked_parameter(v, knots_v, degree_v, false);

    Vector3d value = Vector3d::Zero();
    const int nu = (int)control_grid.size();
    const int nv = (int)control_grid[0].size();
    for (int i = 0; i < nu; i++) {
        double bu = bspline_basis(i, degree_u, u, knots_u);
        if (bu == 0.0) continue;
        for (int j = 0; j < nv; j++) {
            double bv = bspline_basis(j, degree_v, v, knots_v);
            if (bv == 0.0) continue;
            value += bu * bv * control_grid[i][j];
        }
    }
    return value;
}

Vector3d BSplineSurface3D::derivative(
    double u,
    double v,
    int order_u,
    int order_v) const {
    validate_surface(*this);
    if (order_u < 0 || order_v < 0) {
        throw std::invalid_argument("derivative order must be non-negative");
    }
    if (order_u == 0 && order_v == 0) return evaluate(u, v);

    u = checked_parameter(u, knots_u, degree_u, order_u > 0);
    v = checked_parameter(v, knots_v, degree_v, order_v > 0);

    Vector3d value = Vector3d::Zero();
    const int nu = (int)control_grid.size();
    const int nv = (int)control_grid[0].size();
    for (int i = 0; i < nu; i++) {
        double bu = bspline_basis_derivative(
            i, degree_u, u, order_u, knots_u);
        if (bu == 0.0) continue;
        for (int j = 0; j < nv; j++) {
            double bv = bspline_basis_derivative(
                j, degree_v, v, order_v, knots_v);
            if (bv == 0.0) continue;
            value += bu * bv * control_grid[i][j];
        }
    }
    return value;
}

vector<double> chord_length_parameters(
    const vector<Vector3d>& polyline) {
    if (polyline.empty()) {
        throw std::invalid_argument("polyline must be non-empty");
    }

    vector<double> params(polyline.size(), 0.0);
    double total = 0.0;
    for (int i = 1; i < (int)polyline.size(); i++) {
        double d = (polyline[i] - polyline[i - 1]).norm();
        if (!finite_double(d)) {
            throw std::invalid_argument("polyline points must be finite");
        }
        total += d;
        params[i] = total;
    }

    if (total <= 1e-14) {
        if (polyline.size() == 1) return params;
        for (int i = 0; i < (int)polyline.size(); i++) {
            params[i] = (double)i / (double)(polyline.size() - 1);
        }
        return params;
    }

    for (double& t : params) {
        t /= total;
    }
    params.front() = 0.0;
    params.back() = 1.0;
    return params;
}

BSplineCurveFitResult fit_cubic_bspline_curve_least_squares(
    const vector<Vector3d>& polyline,
    int num_control_points,
    double fairness_weight) {
    BSplineCurveFitResult result;

    try {
        if ((int)polyline.size() < 2) {
            throw std::invalid_argument("polyline needs at least two points");
        }
        if (num_control_points < 4) {
            throw std::invalid_argument("cubic curve needs at least four control points");
        }
        if (!finite_double(fairness_weight) || fairness_weight < 0.0) {
            throw std::invalid_argument("fairness weight must be finite and non-negative");
        }

        const int degree = 3;
        const int unknown_count = num_control_points - 2;
        vector<double> knots = make_open_uniform_knot_vector(num_control_points, degree);
        vector<double> params = chord_length_parameters(polyline);

        const int data_rows = (int)polyline.size();
        const int fair_rows = unknown_count;
        const int row_count = data_rows + fair_rows;
        const bool use_sparse =
            (row_count * std::max(1, unknown_count) > 200000) ||
            unknown_count > 256 ||
            row_count > 4096;

        MatrixXd dense_A;
        if (!use_sparse) {
            dense_A = MatrixXd::Zero(row_count, unknown_count);
        }
        vector<Triplet<double>> sparse_triplets;
        if (use_sparse) {
            sparse_triplets.reserve((size_t)data_rows * (size_t)std::min(num_control_points, degree + 1) +
                                    (size_t)fair_rows * 3);
        }

        MatrixXd B = MatrixXd::Zero(row_count, 3);
        const Vector3d first = polyline.front();
        const Vector3d last = polyline.back();

        for (int r = 0; r < data_rows; r++) {
            const double t = params[r];
            B.row(r) = polyline[r].transpose();

            double b0 = bspline_basis(0, degree, t, knots);
            double bn = bspline_basis(num_control_points - 1, degree, t, knots);
            B.row(r) -= (b0 * first + bn * last).transpose();

            for (int ci = 1; ci <= num_control_points - 2; ci++) {
                double value = bspline_basis(ci, degree, t, knots);
                add_fit_coefficient(
                    dense_A, sparse_triplets, use_sparse,
                    r, ci - 1, value);
            }
        }

        const double fair_scale = std::sqrt(fairness_weight);
        if (fair_scale > 0.0) {
            for (int ci = 1; ci <= num_control_points - 2; ci++) {
                int r = data_rows + (ci - 1);
                int stencil[3] = {ci - 1, ci, ci + 1};
                double coeffs[3] = {1.0, -2.0, 1.0};

                for (int k = 0; k < 3; k++) {
                    int idx = stencil[k];
                    double coeff = fair_scale * coeffs[k];
                    if (idx == 0) {
                        B.row(r) -= (coeff * first).transpose();
                    } else if (idx == num_control_points - 1) {
                        B.row(r) -= (coeff * last).transpose();
                    } else {
                        add_fit_coefficient(
                            dense_A, sparse_triplets, use_sparse,
                            r, idx - 1, coeff);
                    }
                }
            }
        }

        MatrixXd X;
        if (!solve_fit_system(
                dense_A, sparse_triplets, use_sparse,
                row_count, unknown_count, B, X)) {
            throw std::runtime_error("least-squares solve failed");
        }

        vector<Vector3d> control_points(num_control_points);
        control_points.front() = first;
        control_points.back() = last;
        for (int i = 0; i < unknown_count; i++) {
            control_points[i + 1] = X.row(i).transpose();
        }

        result.curve = BSplineCurve3D(degree, knots, control_points);
        double sum_error = 0.0;
        double max_error = 0.0;
        for (int i = 0; i < (int)polyline.size(); i++) {
            double err = (result.curve.evaluate(params[i]) - polyline[i]).norm();
            sum_error += err;
            max_error = std::max(max_error, err);
        }
        result.mean_error = sum_error / (double)polyline.size();
        result.max_error = max_error;
        result.success = true;
        result.message = use_sparse ? "sparse least-squares solve" : "dense least-squares solve";
    } catch (const std::exception& e) {
        result.success = false;
        result.message = e.what();
    }

    return result;
}

void sample_bspline_surface(
    const BSplineSurface3D& surface,
    int sample_u,
    int sample_v,
    MatrixXd& V,
    MatrixXi& F) {
    validate_surface(surface);
    if (sample_u < 2 || sample_v < 2) {
        throw std::invalid_argument("surface sampling needs at least 2 samples per direction");
    }

    V.resize(sample_u * sample_v, 3);
    F.resize((sample_u - 1) * (sample_v - 1) * 2, 3);

    const double ua = domain_start(surface.knots_u, surface.degree_u);
    const double ub = domain_end(surface.knots_u, surface.degree_u);
    const double va = domain_start(surface.knots_v, surface.degree_v);
    const double vb = domain_end(surface.knots_v, surface.degree_v);

    auto id = [sample_v](int i, int j) {
        return i * sample_v + j;
    };

    for (int i = 0; i < sample_u; i++) {
        double u = ua + (ub - ua) * (double)i / (double)(sample_u - 1);
        for (int j = 0; j < sample_v; j++) {
            double v = va + (vb - va) * (double)j / (double)(sample_v - 1);
            V.row(id(i, j)) = surface.evaluate(u, v).transpose();
        }
    }

    int f = 0;
    for (int i = 0; i < sample_u - 1; i++) {
        for (int j = 0; j < sample_v - 1; j++) {
            int a = id(i, j);
            int b = id(i + 1, j);
            int c = id(i, j + 1);
            int d = id(i + 1, j + 1);
            F.row(f++) << a, b, c;
            F.row(f++) << b, d, c;
        }
    }
}

bool export_bspline_curve_polyline_obj(
    const string& filename,
    const BSplineCurve3D& curve,
    int sample_count) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;

    vector<Vector3d> points = curve.sample(sample_count);
    fout << "# B-spline curve polyline\n";
    for (const Vector3d& p : points) {
        write_obj_vertex(fout, p);
    }
    for (int i = 0; i + 1 < (int)points.size(); i++) {
        fout << "l " << (i + 1) << " " << (i + 2) << "\n";
    }
    return true;
}

bool export_bspline_surface_mesh_obj(
    const string& filename,
    const BSplineSurface3D& surface,
    int sample_u,
    int sample_v) {
    MatrixXd V;
    MatrixXi F;
    sample_bspline_surface(surface, sample_u, sample_v, V, F);

    std::ofstream fout(filename);
    if (!fout.is_open()) return false;

    fout << "# B-spline surface sampled mesh\n";
    for (int i = 0; i < V.rows(); i++) {
        fout << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
    }
    for (int i = 0; i < F.rows(); i++) {
        fout << "f " << (F(i, 0) + 1)
             << " " << (F(i, 1) + 1)
             << " " << (F(i, 2) + 1) << "\n";
    }
    return true;
}

bool export_bspline_surface_control_net_obj(
    const string& filename,
    const BSplineSurface3D& surface) {
    validate_surface(surface);
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;

    const int nu = (int)surface.control_grid.size();
    const int nv = (int)surface.control_grid[0].size();
    fout << "# B-spline surface control net\n";
    for (int i = 0; i < nu; i++) {
        for (int j = 0; j < nv; j++) {
            write_obj_vertex(fout, surface.control_grid[i][j]);
        }
    }

    auto id = [nv](int i, int j) {
        return i * nv + j + 1;
    };
    for (int i = 0; i < nu; i++) {
        for (int j = 0; j + 1 < nv; j++) {
            fout << "l " << id(i, j) << " " << id(i, j + 1) << "\n";
        }
    }
    for (int j = 0; j < nv; j++) {
        for (int i = 0; i + 1 < nu; i++) {
            fout << "l " << id(i, j) << " " << id(i + 1, j) << "\n";
        }
    }
    return true;
}

bool export_control_points_obj(
    const string& filename,
    const vector<Vector3d>& points) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;

    fout << "# B-spline control points\n";
    for (const Vector3d& p : points) {
        write_obj_vertex(fout, p);
    }
    return true;
}

bool export_polyline_obj(
    const string& filename,
    const vector<Vector3d>& points) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;

    fout << "# Polyline\n";
    for (const Vector3d& p : points) {
        write_obj_vertex(fout, p);
    }
    for (int i = 0; i + 1 < (int)points.size(); i++) {
        fout << "l " << (i + 1) << " " << (i + 2) << "\n";
    }
    return true;
}

bool export_bspline_curve_control_polygon_obj(
    const string& filename,
    const BSplineCurve3D& curve) {
    validate_curve(curve);
    return export_polyline_obj(filename, curve.control_points);
}
