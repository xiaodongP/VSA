#include "polyharmonic_3d_extension.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::Matrix;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::map;
using std::set;
using std::string;
using std::vector;

namespace {

struct WeightedVectorRow {
    vector<std::pair<int, double>> coeffs;
    Vector3d target = Vector3d::Zero();
    double weight = 1.0;
};

static double cross2(const Vector2d& a, const Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

static double tri_area2d(const Vector2d& a, const Vector2d& b, const Vector2d& c) {
    return 0.5 * std::abs(cross2(b - a, c - a));
}

static double cotangent(const Vector2d& a, const Vector2d& b, const Vector2d& c) {
    Vector2d u = b - a;
    Vector2d v = c - a;
    double area2 = cross2(u, v);
    if (std::abs(area2) < 1e-16) return 0.0;
    return u.dot(v) / area2;
}

static bool angle_obtuse(double a2, double b2, double c2) {
    return a2 + b2 < c2;
}

static void add_cotan_weight(map<std::pair<int, int>, double>& weights, int i, int j, double w) {
    if (i == j) return;
    if (i > j) std::swap(i, j);
    weights[std::make_pair(i, j)] += w;
}

static SparseMatrix<double> build_cotan_laplacian(
    const MatrixXd& UV,
    const MatrixXi& F) {
    map<std::pair<int, int>, double> weights;
    for (int fi = 0; fi < F.rows(); fi++) {
        int i = F(fi, 0), j = F(fi, 1), k = F(fi, 2);
        Vector2d pi = UV.row(i), pj = UV.row(j), pk = UV.row(k);
        double coti = cotangent(pi, pj, pk);
        double cotj = cotangent(pj, pk, pi);
        double cotk = cotangent(pk, pi, pj);
        add_cotan_weight(weights, j, k, 0.5 * coti);
        add_cotan_weight(weights, k, i, 0.5 * cotj);
        add_cotan_weight(weights, i, j, 0.5 * cotk);
    }
    vector<Triplet<double>> trips;
    vector<double> diag(UV.rows(), 0.0);
    for (const auto& kv : weights) {
        int i = kv.first.first;
        int j = kv.first.second;
        double w = kv.second;
        diag[i] += w;
        diag[j] += w;
        trips.emplace_back(i, j, -w);
        trips.emplace_back(j, i, -w);
    }
    for (int i = 0; i < (int)diag.size(); i++) trips.emplace_back(i, i, diag[i]);
    SparseMatrix<double> L(UV.rows(), UV.rows());
    L.setFromTriplets(trips.begin(), trips.end());
    return L;
}

static vector<double> build_lumped_voronoi_mass(
    const MatrixXd& UV,
    const MatrixXi& F,
    double min_mass) {
    vector<double> mass(UV.rows(), 0.0);
    for (int fi = 0; fi < F.rows(); fi++) {
        int ids[3] = {F(fi, 0), F(fi, 1), F(fi, 2)};
        Vector2d p[3] = {UV.row(ids[0]), UV.row(ids[1]), UV.row(ids[2])};
        double area = tri_area2d(p[0], p[1], p[2]);
        if (area <= 1e-18) continue;
        double l2[3] = {
            (p[1] - p[2]).squaredNorm(),
            (p[2] - p[0]).squaredNorm(),
            (p[0] - p[1]).squaredNorm()};
        bool obtuse0 = angle_obtuse(l2[1], l2[2], l2[0]);
        bool obtuse1 = angle_obtuse(l2[2], l2[0], l2[1]);
        bool obtuse2 = angle_obtuse(l2[0], l2[1], l2[2]);
        if (obtuse0 || obtuse1 || obtuse2) {
            for (int i = 0; i < 3; i++) {
                mass[ids[i]] += (i == 0 && obtuse0) || (i == 1 && obtuse1) || (i == 2 && obtuse2)
                                    ? 0.5 * area
                                    : 0.25 * area;
            }
            continue;
        }
        double cot0 = cotangent(p[0], p[1], p[2]);
        double cot1 = cotangent(p[1], p[2], p[0]);
        double cot2 = cotangent(p[2], p[0], p[1]);
        mass[ids[0]] += (l2[2] * cot1 + l2[1] * cot2) / 8.0;
        mass[ids[1]] += (l2[0] * cot2 + l2[2] * cot0) / 8.0;
        mass[ids[2]] += (l2[1] * cot0 + l2[0] * cot1) / 8.0;
    }
    for (double& m : mass) m = std::max(m, min_mass);
    return mass;
}

static SparseMatrix<double> left_scale_rows(
    const SparseMatrix<double>& A,
    const vector<double>& scale) {
    vector<Triplet<double>> trips;
    trips.reserve(A.nonZeros());
    for (int k = 0; k < A.outerSize(); k++) {
        for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            trips.emplace_back(it.row(), it.col(), scale[it.row()] * it.value());
        }
    }
    SparseMatrix<double> S(A.rows(), A.cols());
    S.setFromTriplets(trips.begin(), trips.end());
    return S;
}

static SparseMatrix<double> build_polyharmonic_q(
    const SparseMatrix<double>& L,
    const vector<double>& mass,
    PolyharmonicContinuityMode mode,
    double regularization) {
    vector<double> inv_mass(mass.size());
    for (int i = 0; i < (int)mass.size(); i++) inv_mass[i] = 1.0 / std::max(mass[i], 1e-18);
    SparseMatrix<double> Q;
    if (mode == PolyharmonicContinuityMode::G1) {
        SparseMatrix<double> MinvL = left_scale_rows(L, inv_mass);
        Q = L.transpose() * MinvL;
    } else {
        SparseMatrix<double> MinvL = left_scale_rows(L, inv_mass);
        SparseMatrix<double> A = L * MinvL;
        Q = A.transpose() * left_scale_rows(A, inv_mass);
    }
    if (regularization > 0.0) {
        vector<Triplet<double>> reg;
        reg.reserve(Q.rows());
        for (int i = 0; i < Q.rows(); i++) reg.emplace_back(i, i, regularization);
        SparseMatrix<double> R(Q.rows(), Q.cols());
        R.setFromTriplets(reg.begin(), reg.end());
        Q = Q + R;
    }
    return Q;
}

static double clamp01(double x) {
    return std::max(0.0, std::min(1.0, x));
}

static void add_unique_value(vector<double>& values, double x, double tol) {
    for (double v : values) {
        if (std::abs(v - x) <= tol) return;
    }
    values.push_back(x);
}

static vector<int> collect_isocurve_vertices(
    const MatrixXd& UV,
    int fixed_coord,
    double value,
    double tol) {
    int other = 1 - fixed_coord;
    vector<int> ids;
    for (int i = 0; i < UV.rows(); i++) {
        if (std::abs(UV(i, fixed_coord) - value) <= tol) ids.push_back(i);
    }
    std::sort(ids.begin(), ids.end(), [&](int a, int b) {
        if (std::abs(UV(a, other) - UV(b, other)) > 1e-12) {
            return UV(a, other) < UV(b, other);
        }
        return a < b;
    });
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

static vector<vector<int>> collect_configured_isocurves(
    const RectangularDomainExtensionResult& domain,
    const Polyharmonic3DExtensionConfig& config) {
    double diag = (domain.rectangle_max - domain.rectangle_min).norm();
    double tol = std::max(config.isocurve_tolerance, std::max(1.0, diag) * 1e-9);
    vector<double> u_values;
    vector<double> v_values;
    if (config.include_rectangle_boundary_isocurves) {
        add_unique_value(u_values, domain.rectangle_min.x(), tol);
        add_unique_value(u_values, domain.rectangle_max.x(), tol);
        add_unique_value(v_values, domain.rectangle_min.y(), tol);
        add_unique_value(v_values, domain.rectangle_max.y(), tol);
    }
    for (double u : config.labeled_isocurve_u_values) add_unique_value(u_values, u, tol);
    for (double v : config.labeled_isocurve_v_values) add_unique_value(v_values, v, tol);

    vector<vector<int>> curves;
    for (double u : u_values) {
        vector<int> ids = collect_isocurve_vertices(domain.full_uv_vertices, 0, u, tol);
        if (ids.size() >= 3) curves.push_back(ids);
    }
    for (double v : v_values) {
        vector<int> ids = collect_isocurve_vertices(domain.full_uv_vertices, 1, v, tol);
        if (ids.size() >= 3) curves.push_back(ids);
    }
    return curves;
}

static VectorXd finite_difference_coefficients(
    const vector<double>& s,
    int begin,
    int count,
    int order) {
    VectorXd rhs = VectorXd::Zero(count);
    double factorial = 1.0;
    for (int i = 2; i <= order; i++) factorial *= (double)i;
    rhs(order) = factorial;
    double center = s[begin + std::min(1, count - 1)];
    Eigen::MatrixXd A(count, count);
    for (int p = 0; p < count; p++) {
        for (int j = 0; j < count; j++) {
            A(p, j) = std::pow(s[begin + j] - center, p);
        }
    }
    return A.colPivHouseholderQr().solve(rhs);
}

static void append_isocurve_fairness_rows(
    const RectangularDomainExtensionResult& domain,
    const Polyharmonic3DExtensionConfig& config,
    vector<WeightedVectorRow>& rows) {
    int order = config.mode == PolyharmonicContinuityMode::G1 ? 2 : 3;
    int stencil = order + 1;
    double tol = std::max(config.isocurve_tolerance, 1e-10);
    vector<vector<int>> curves = collect_configured_isocurves(domain, config);
    for (const vector<int>& ids : curves) {
        if ((int)ids.size() < stencil) continue;
        vector<double> s(ids.size(), 0.0);
        for (int i = 1; i < (int)ids.size(); i++) {
            Vector2d a = domain.full_uv_vertices.row(ids[i - 1]);
            Vector2d b = domain.full_uv_vertices.row(ids[i]);
            s[i] = s[i - 1] + (b - a).norm();
        }
        for (int begin = 0; begin + stencil <= (int)ids.size(); begin++) {
            double span = s[begin + stencil - 1] - s[begin];
            if (span <= tol) continue;
            VectorXd c = finite_difference_coefficients(s, begin, stencil, order);
            WeightedVectorRow row;
            row.weight = config.isocurve_weight_scale * span;
            for (int j = 0; j < stencil; j++) {
                if (std::isfinite(c(j))) row.coeffs.push_back({ids[begin + j], c(j)});
            }
            if ((int)row.coeffs.size() == stencil) rows.push_back(row);
        }
    }
}

static std::array<vector<std::pair<int, double>>, 2> triangle_jacobian_rows(
    const MatrixXd& UV,
    const MatrixXi& F,
    int fi) {
    int i0 = F(fi, 0), i1 = F(fi, 1), i2 = F(fi, 2);
    Vector2d p0 = UV.row(i0), p1 = UV.row(i1), p2 = UV.row(i2);
    Eigen::Matrix2d D;
    D.col(0) = p1 - p0;
    D.col(1) = p2 - p0;
    Eigen::Matrix2d inv = D.inverse();
    std::array<vector<std::pair<int, double>>, 2> rows;
    for (int col = 0; col < 2; col++) {
        double a = inv(0, col);
        double b = inv(1, col);
        rows[col].push_back({i0, -a - b});
        rows[col].push_back({i1, a});
        rows[col].push_back({i2, b});
    }
    return rows;
}

static Matrix<double, 3, 2> triangle_jacobian(
    const MatrixXd& V,
    const MatrixXd& UV,
    const MatrixXi& F,
    int fi) {
    auto rows = triangle_jacobian_rows(UV, F, fi);
    Matrix<double, 3, 2> J;
    for (int col = 0; col < 2; col++) {
        Vector3d value = Vector3d::Zero();
        for (const auto& kv : rows[col]) value += kv.second * V.row(kv.first).transpose();
        J.col(col) = value;
    }
    return J;
}

static Matrix<double, 3, 2> closest_reference_rotation(const Matrix<double, 3, 2>& J) {
    Eigen::JacobiSVD<Matrix<double, 3, 2>> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU().leftCols<2>() * svd.matrixV().transpose();
}

static void append_mesh_fairness_rows(
    const RectangularDomainExtensionResult& domain,
    const MatrixXd& initial_extension,
    const Polyharmonic3DExtensionConfig& config,
    vector<WeightedVectorRow>& rows) {
    int fcount = domain.full_faces.rows();
    vector<std::array<vector<std::pair<int, double>>, 2>> jac_rows(fcount);
    vector<Matrix<double, 3, 2>> rotations(fcount);
    for (int fi = 0; fi < fcount; fi++) {
        jac_rows[fi] = triangle_jacobian_rows(domain.full_uv_vertices, domain.full_faces, fi);
        rotations[fi] = closest_reference_rotation(
            triangle_jacobian(initial_extension, domain.full_uv_vertices, domain.full_faces, fi));
    }

    map<std::pair<int, int>, vector<int>> edge_faces;
    for (int fi = 0; fi < fcount; fi++) {
        for (int e = 0; e < 3; e++) {
            int a = domain.full_faces(fi, e);
            int b = domain.full_faces(fi, (e + 1) % 3);
            if (a > b) std::swap(a, b);
            edge_faces[{a, b}].push_back(fi);
        }
    }
    for (const auto& kv : edge_faces) {
        if (kv.second.size() != 2) continue;
        int fa = kv.second[0];
        int fb = kv.second[1];
        int va = kv.first.first;
        int vb = kv.first.second;
        double edge_length = (initial_extension.row(va) - initial_extension.row(vb)).norm();
        double weight = config.mesh_fairness_weight_scale * std::max(edge_length, 1e-8);
        for (int col = 0; col < 2; col++) {
            map<int, double> coeff_map;
            for (const auto& c : jac_rows[fa][col]) coeff_map[c.first] += c.second;
            for (const auto& c : jac_rows[fb][col]) coeff_map[c.first] -= c.second;
            WeightedVectorRow row;
            row.weight = weight;
            row.target = rotations[fa].col(col) - rotations[fb].col(col);
            for (const auto& c : coeff_map) {
                if (std::abs(c.second) > 1e-14) row.coeffs.push_back(c);
            }
            rows.push_back(row);
        }
    }
}

static void append_initial_position_rows(
    const RectangularDomainExtensionResult& domain,
    const MatrixXd& initial_positions,
    vector<WeightedVectorRow>& rows) {
    for (int i = 0; i < initial_positions.rows(); i++) {
        bool is_original = i < (int)domain.original_vertex_mask.size() &&
                           domain.original_vertex_mask[i];
        if (is_original) continue;
        WeightedVectorRow row;
        row.coeffs.push_back({i, 1.0});
        row.target = initial_positions.row(i).transpose();
        row.weight = 1.0;
        rows.push_back(row);
    }
}

static void add_rows_to_normal_system(
    const vector<WeightedVectorRow>& rows,
    int vertex_count,
    vector<Triplet<double>>& trips,
    MatrixXd& B,
    double scale) {
    if (scale <= 0.0) return;
    for (const WeightedVectorRow& row : rows) {
        double w = scale * row.weight;
        if (w <= 0.0) continue;
        for (const auto& a : row.coeffs) {
            B.row(a.first) += w * a.second * row.target.transpose();
            for (const auto& b : row.coeffs) {
                trips.emplace_back(a.first, b.first, w * a.second * b.second);
            }
        }
    }
    (void)vertex_count;
}

static double rows_energy(const vector<WeightedVectorRow>& rows, const MatrixXd& X) {
    double energy = 0.0;
    for (const WeightedVectorRow& row : rows) {
        Vector3d value = -row.target;
        for (const auto& c : row.coeffs) value += c.second * X.row(c.first).transpose();
        energy += row.weight * value.squaredNorm();
    }
    return energy;
}

static double quadratic_energy(const SparseMatrix<double>& Q, const MatrixXd& X) {
    MatrixXd QX = Q * X;
    return std::max(0.0, (QX.array() * X.array()).sum());
}

static double surface_area(
    const MatrixXd& V,
    const MatrixXi& F,
    const vector<bool>* face_mask) {
    double area = 0.0;
    for (int fi = 0; fi < F.rows(); fi++) {
        if (face_mask && (fi >= (int)face_mask->size() || !(*face_mask)[fi])) continue;
        Vector3d a = V.row(F(fi, 0));
        Vector3d b = V.row(F(fi, 1));
        Vector3d c = V.row(F(fi, 2));
        area += 0.5 * (b - a).cross(c - a).norm();
    }
    return area;
}

static double bbox_diagonal(const MatrixXd& V, const vector<bool>* vertex_mask) {
    bool any = false;
    Vector3d mn = Vector3d::Zero();
    Vector3d mx = Vector3d::Zero();
    for (int i = 0; i < V.rows(); i++) {
        if (vertex_mask && (i >= (int)vertex_mask->size() || !(*vertex_mask)[i])) continue;
        Vector3d p = V.row(i);
        if (!any) {
            mn = mx = p;
            any = true;
        } else {
            mn = mn.cwiseMin(p);
            mx = mx.cwiseMax(p);
        }
    }
    return any ? (mx - mn).norm() : 0.0;
}

static void compute_boundary_curvature(
    const RectangularDomainExtensionResult& domain,
    const Polyharmonic3DExtensionConfig& config,
    const MatrixXd& V,
    double& mean_curvature,
    double& max_curvature) {
    Polyharmonic3DExtensionConfig boundary_config = config;
    boundary_config.labeled_isocurve_u_values.clear();
    boundary_config.labeled_isocurve_v_values.clear();
    boundary_config.include_rectangle_boundary_isocurves = true;
    vector<vector<int>> curves = collect_configured_isocurves(domain, boundary_config);
    double sum = 0.0;
    int count = 0;
    max_curvature = 0.0;
    for (const vector<int>& ids : curves) {
        for (int i = 1; i + 1 < (int)ids.size(); i++) {
            Vector3d a = V.row(ids[i - 1]);
            Vector3d b = V.row(ids[i]);
            Vector3d c = V.row(ids[i + 1]);
            Vector3d e0 = b - a;
            Vector3d e1 = c - b;
            double l0 = e0.norm();
            double l1 = e1.norm();
            if (l0 <= 1e-12 || l1 <= 1e-12) continue;
            double cosang = clamp01((e0.dot(e1) / (l0 * l1) + 1.0) * 0.5) * 2.0 - 1.0;
            double kappa = std::acos(cosang) / (0.5 * (l0 + l1));
            sum += kappa;
            max_curvature = std::max(max_curvature, kappa);
            count++;
        }
    }
    mean_curvature = count > 0 ? sum / (double)count : 0.0;
}

static bool write_obj(const string& filename, const MatrixXd& V, const MatrixXi& F) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    for (int i = 0; i < V.rows(); i++) {
        out << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
    }
    for (int i = 0; i < F.rows(); i++) {
        out << "f " << F(i, 0) + 1 << " " << F(i, 1) + 1 << " " << F(i, 2) + 1 << "\n";
    }
    return true;
}

} // namespace

const char* to_string(PolyharmonicContinuityMode mode) {
    switch (mode) {
    case PolyharmonicContinuityMode::G1: return "G1";
    case PolyharmonicContinuityMode::G2: return "G2";
    }
    return "Unknown";
}

Polyharmonic3DExtensionResult extend_polyharmonic_3d(
    const RectangularDomainExtensionResult& domain,
    const MatrixXd& original_positions,
    const Polyharmonic3DExtensionConfig& config) {
    Polyharmonic3DExtensionResult result;
    result.mode = config.mode;
    if (!domain.valid) {
        result.reason = "rectangular domain extension is invalid: " + domain.reason;
        return result;
    }
    if (domain.full_uv_vertices.cols() != 2 || domain.full_faces.cols() != 3) {
        result.reason = "domain UV mesh dimensions are invalid";
        return result;
    }
    if (original_positions.cols() != 3 ||
        original_positions.rows() != domain.full_uv_vertices.rows()) {
        result.reason = "original_positions must have one 3D row per full UV vertex";
        return result;
    }
    int n = domain.full_uv_vertices.rows();
    vector<int> fixed, unknown;
    for (int i = 0; i < n; i++) {
        bool is_original = i < (int)domain.original_vertex_mask.size() &&
                           domain.original_vertex_mask[i];
        if (is_original) fixed.push_back(i);
        else unknown.push_back(i);
    }
    if (fixed.empty() || unknown.empty()) {
        result.reason = "polyharmonic extension requires both fixed and artificial vertices";
        return result;
    }

    double w_fair = clamp01(config.fairness_weight);
    double w_iso = clamp01(config.isocurve_fairness_weight);
    Polyharmonic3DExtensionResult initial_extension;
    bool need_mesh_fairness = w_fair > 0.0 && (1.0 - w_iso) > 0.0;
    if (need_mesh_fairness) {
        Polyharmonic3DExtensionConfig pure_config = config;
        pure_config.fairness_weight = 0.0;
        pure_config.export_debug = false;
        initial_extension = extend_polyharmonic_3d(domain, original_positions, pure_config);
        if (!initial_extension.valid) {
            result.reason = "initial pure extension failed: " + initial_extension.reason;
            return result;
        }
    }

    SparseMatrix<double> L = build_cotan_laplacian(domain.full_uv_vertices, domain.full_faces);
    vector<double> mass = build_lumped_voronoi_mass(
        domain.full_uv_vertices, domain.full_faces, std::max(config.min_mass, 1e-18));
    result.min_mass = *std::min_element(mass.begin(), mass.end());
    result.max_mass = *std::max_element(mass.begin(), mass.end());
    SparseMatrix<double> Qext = build_polyharmonic_q(L, mass, config.mode, 0.0);
    SparseMatrix<double> Q = Qext;
    MatrixXd B = MatrixXd::Zero(n, 3);

    vector<WeightedVectorRow> iso_rows;
    vector<WeightedVectorRow> mesh_rows;
    vector<WeightedVectorRow> initial_rows;
    if (w_fair > 0.0) {
        vector<Triplet<double>> fairness_trips;
        if (w_iso > 0.0) {
            append_isocurve_fairness_rows(domain, config, iso_rows);
            add_rows_to_normal_system(
                iso_rows, n, fairness_trips, B, w_fair * w_iso);
        }
        if (need_mesh_fairness) {
            append_mesh_fairness_rows(domain, initial_extension.extended_vertices, config, mesh_rows);
            add_rows_to_normal_system(
                mesh_rows, n, fairness_trips, B, w_fair * (1.0 - w_iso));
        }
        SparseMatrix<double> Qfair(n, n);
        Qfair.setFromTriplets(fairness_trips.begin(), fairness_trips.end());
        Q = (1.0 - w_fair) * Qext + Qfair;
    }
    if (config.initial_position_weight > 0.0) {
        vector<Triplet<double>> initial_trips;
        append_initial_position_rows(domain, original_positions, initial_rows);
        add_rows_to_normal_system(
            initial_rows,
            n,
            initial_trips,
            B,
            config.initial_position_weight);
        SparseMatrix<double> Qinitial(n, n);
        Qinitial.setFromTriplets(initial_trips.begin(), initial_trips.end());
        Q = Q + Qinitial;
    }

    map<int, int> unknown_to_col;
    for (int i = 0; i < (int)unknown.size(); i++) unknown_to_col[unknown[i]] = i;
    map<int, int> fixed_to_col;
    for (int i = 0; i < (int)fixed.size(); i++) fixed_to_col[fixed[i]] = i;

    vector<Triplet<double>> Quu_trips;
    MatrixXd Quf = MatrixXd::Zero((int)unknown.size(), (int)fixed.size());
    for (int k = 0; k < Q.outerSize(); k++) {
        for (SparseMatrix<double>::InnerIterator it(Q, k); it; ++it) {
            int r = it.row();
            int c = it.col();
            auto ur = unknown_to_col.find(r);
            if (ur == unknown_to_col.end()) continue;
            auto uc = unknown_to_col.find(c);
            if (uc != unknown_to_col.end()) {
                Quu_trips.emplace_back(ur->second, uc->second, it.value());
            } else {
                auto fc = fixed_to_col.find(c);
                if (fc != fixed_to_col.end()) Quf(ur->second, fc->second) += it.value();
            }
        }
    }
    SparseMatrix<double> Quu((int)unknown.size(), (int)unknown.size());
    Quu.setFromTriplets(Quu_trips.begin(), Quu_trips.end());
    Eigen::SparseLU<SparseMatrix<double>> solver;
    solver.compute(Quu);
    if (solver.info() != Eigen::Success) {
        SparseMatrix<double> R(n, n);
        vector<Triplet<double>> reg_trips;
        reg_trips.reserve(n);
        for (int i = 0; i < n; i++) reg_trips.emplace_back(i, i, std::max(0.0, config.regularization));
        R.setFromTriplets(reg_trips.begin(), reg_trips.end());
        Q = Q + R;
        Quu_trips.clear();
        Quf.setZero();
        for (int k = 0; k < Q.outerSize(); k++) {
            for (SparseMatrix<double>::InnerIterator it(Q, k); it; ++it) {
                int r = it.row();
                int c = it.col();
                auto ur = unknown_to_col.find(r);
                if (ur == unknown_to_col.end()) continue;
                auto uc = unknown_to_col.find(c);
                if (uc != unknown_to_col.end()) {
                    Quu_trips.emplace_back(ur->second, uc->second, it.value());
                } else {
                    auto fc = fixed_to_col.find(c);
                    if (fc != fixed_to_col.end()) Quf(ur->second, fc->second) += it.value();
                }
            }
        }
        Quu.setZero();
        Quu.setFromTriplets(Quu_trips.begin(), Quu_trips.end());
        solver.compute(Quu);
        if (solver.info() != Eigen::Success) {
            result.reason = "polyharmonic sparse factorization failed";
            return result;
        }
    }

    MatrixXd Xf(fixed.size(), 3);
    for (int i = 0; i < (int)fixed.size(); i++) Xf.row(i) = original_positions.row(fixed[i]);
    MatrixXd Bu(unknown.size(), 3);
    for (int i = 0; i < (int)unknown.size(); i++) Bu.row(i) = B.row(unknown[i]);
    MatrixXd rhs = Bu - Quf * Xf;
    MatrixXd Xu = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        result.reason = "polyharmonic sparse solve failed";
        return result;
    }

    result.extended_vertices = original_positions;
    for (int i = 0; i < (int)unknown.size(); i++) {
        result.extended_vertices.row(unknown[i]) = Xu.row(i);
    }
    result.faces = domain.full_faces;
    result.original_vertex_mask = domain.original_vertex_mask;
    result.original_face_mask = domain.original_face_mask;
    result.fixed_vertex_count = (int)fixed.size();
    result.unknown_vertex_count = (int)unknown.size();
    result.mean_unknown_displacement = 0.0;
    result.max_unknown_displacement = 0.0;
    for (int idx : unknown) {
        double d = (result.extended_vertices.row(idx) - original_positions.row(idx)).norm();
        result.mean_unknown_displacement += d;
        result.max_unknown_displacement = std::max(result.max_unknown_displacement, d);
    }
    result.mean_unknown_displacement /= (double)unknown.size();
    MatrixXd all = result.extended_vertices;
    result.residual_norm = (Q * all).norm();
    result.extension_energy = quadratic_energy(Qext, all);
    if (iso_rows.empty()) append_isocurve_fairness_rows(domain, config, iso_rows);
    result.isocurve_fairness_energy = rows_energy(iso_rows, all);
    result.mesh_fairness_energy = rows_energy(mesh_rows, all);
    compute_boundary_curvature(
        domain, config, result.extended_vertices,
        result.mean_boundary_curvature, result.max_boundary_curvature);
    result.original_surface_area = surface_area(
        result.extended_vertices, domain.full_faces, &domain.original_face_mask);
    result.extended_surface_area = surface_area(
        result.extended_vertices, domain.full_faces, nullptr);
    if (result.original_surface_area > 1e-12) {
        result.surface_area_growth = result.extended_surface_area / result.original_surface_area;
    }
    result.original_bbox_diagonal = bbox_diagonal(original_positions, &domain.original_vertex_mask);
    result.extended_bbox_diagonal = bbox_diagonal(result.extended_vertices, nullptr);
    if (result.original_bbox_diagonal > 1e-12) {
        result.bbox_growth = result.extended_bbox_diagonal / result.original_bbox_diagonal;
    }
    result.valid = true;
    result.reason = "ok";
    if (config.export_debug) export_polyharmonic_3d_extension_debug(config.debug_prefix, result);
    return result;
}

bool export_polyharmonic_3d_extension_debug(
    const string& prefix,
    const Polyharmonic3DExtensionResult& result) {
    bool ok = true;
    ok = write_obj(prefix + "_extended_mesh.obj", result.extended_vertices, result.faces) && ok;
    {
        std::ofstream out(prefix + "_summary.csv");
        ok = out.is_open() && ok;
        out << "valid,reason,mode,fixed_vertices,unknown_vertices,min_mass,max_mass,"
            << "mean_unknown_displacement,max_unknown_displacement,residual_norm,"
            << "extension_energy,isocurve_fairness_energy,mesh_fairness_energy,"
            << "mean_boundary_curvature,max_boundary_curvature,"
            << "original_surface_area,extended_surface_area,surface_area_growth,"
            << "original_bbox_diagonal,extended_bbox_diagonal,bbox_growth\n";
        out << (result.valid ? 1 : 0) << "," << result.reason << ","
            << to_string(result.mode) << ","
            << result.fixed_vertex_count << ","
            << result.unknown_vertex_count << ","
            << result.min_mass << ","
            << result.max_mass << ","
            << result.mean_unknown_displacement << ","
            << result.max_unknown_displacement << ","
            << result.residual_norm << ","
            << result.extension_energy << ","
            << result.isocurve_fairness_energy << ","
            << result.mesh_fairness_energy << ","
            << result.mean_boundary_curvature << ","
            << result.max_boundary_curvature << ","
            << result.original_surface_area << ","
            << result.extended_surface_area << ","
            << result.surface_area_growth << ","
            << result.original_bbox_diagonal << ","
            << result.extended_bbox_diagonal << ","
            << result.bbox_growth << "\n";
    }
    return ok;
}
