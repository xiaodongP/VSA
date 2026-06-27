#include "initial_bspline_surface.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector3d;
using std::array;
using std::string;
using std::vector;

namespace {

static bool finite_vec3(const Vector3d& p) {
    return std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z());
}

static bool valid_side_polyline(const vector<Vector3d>& side) {
    if (side.size() < 2) return false;
    for (const Vector3d& p : side) {
        if (!finite_vec3(p)) return false;
    }
    return true;
}

static bool validate_boundary(const QuadLikeBoundary& boundary, string& reason) {
    if (!boundary.valid) {
        reason = "quad-like boundary is not valid";
        return false;
    }
    for (int s = 0; s < 4; s++) {
        if (!valid_side_polyline(boundary.side_polylines[s])) {
            reason = "each side polyline must contain at least two finite points";
            return false;
        }
    }
    return true;
}

static void enforce_shared_corners(
    const QuadLikeBoundary& boundary,
    array<BSplineCurve3D, 4>& curves) {
    const Vector3d p00 = boundary.side_polylines[0].front();
    const Vector3d p10 = boundary.side_polylines[0].back();
    const Vector3d p11 = boundary.side_polylines[1].back();
    const Vector3d p01 = boundary.side_polylines[2].back();

    curves[0].control_points.front() = p00;
    curves[0].control_points.back() = p10;

    curves[1].control_points.front() = p10;
    curves[1].control_points.back() = p11;

    curves[2].control_points.front() = p11;
    curves[2].control_points.back() = p01;

    curves[3].control_points.front() = p01;
    curves[3].control_points.back() = p00;
}

static vector<vector<Vector3d>> make_boundary_control_grid(
    const array<BSplineCurve3D, 4>& curves,
    int nu,
    int nv) {
    vector<vector<Vector3d>> grid(nu, vector<Vector3d>(nv, Vector3d::Zero()));

    for (int i = 0; i < nu; i++) {
        grid[i][0] = curves[0].control_points[i];
        grid[i][nv - 1] = curves[2].control_points[nu - 1 - i];
    }
    for (int j = 0; j < nv; j++) {
        grid[nu - 1][j] = curves[1].control_points[j];
        grid[0][j] = curves[3].control_points[nv - 1 - j];
    }
    return grid;
}

static int interior_index(int i, int j, int nv) {
    return (i - 1) * (nv - 2) + (j - 1);
}

static bool solve_internal_control_points(
    const array<BSplineCurve3D, 4>& curves,
    vector<vector<Vector3d>>& grid,
    const vector<double>& knots_u,
    const vector<double>& knots_v,
    int degree_u,
    int degree_v,
    int sample_u,
    int sample_v,
    double& mean_error,
    double& max_error) {
    int nu = (int)grid.size();
    int nv = (int)grid[0].size();
    int unknown_count = (nu - 2) * (nv - 2);
    if (unknown_count <= 0) return false;

    int su = std::max(2, sample_u);
    int sv = std::max(2, sample_v);
    int row_count = su * sv;
    MatrixXd A = MatrixXd::Zero(row_count, unknown_count);
    MatrixXd B = MatrixXd::Zero(row_count, 3);

    int row = 0;
    for (int iu = 0; iu < su; iu++) {
        double u = (double)iu / (double)(su - 1);
        vector<double> bu(nu, 0.0);
        for (int i = 0; i < nu; i++) {
            bu[i] = bspline_basis(i, degree_u, u, knots_u);
        }
        for (int iv = 0; iv < sv; iv++) {
            double v = (double)iv / (double)(sv - 1);
            vector<double> bv(nv, 0.0);
            for (int j = 0; j < nv; j++) {
                bv[j] = bspline_basis(j, degree_v, v, knots_v);
            }

            Vector3d rhs = evaluate_coons_patch(curves, u, v);
            for (int i = 0; i < nu; i++) {
                for (int j = 0; j < nv; j++) {
                    double coeff = bu[i] * bv[j];
                    if (coeff == 0.0) continue;
                    if (i == 0 || i == nu - 1 || j == 0 || j == nv - 1) {
                        rhs -= coeff * grid[i][j];
                    } else {
                        A(row, interior_index(i, j, nv)) = coeff;
                    }
                }
            }
            B.row(row) = rhs.transpose();
            row++;
        }
    }

    MatrixXd X = A.colPivHouseholderQr().solve(B);
    if (X.rows() != unknown_count || X.cols() != 3) return false;

    for (int i = 1; i < nu - 1; i++) {
        for (int j = 1; j < nv - 1; j++) {
            grid[i][j] = X.row(interior_index(i, j, nv)).transpose();
        }
    }

    BSplineSurface3D fitted(degree_u, degree_v, knots_u, knots_v, grid);
    double sum = 0.0;
    max_error = 0.0;
    row = 0;
    for (int iu = 0; iu < su; iu++) {
        double u = (double)iu / (double)(su - 1);
        for (int iv = 0; iv < sv; iv++) {
            double v = (double)iv / (double)(sv - 1);
            double err = (fitted.evaluate(u, v) -
                          evaluate_coons_patch(curves, u, v)).norm();
            sum += err;
            max_error = std::max(max_error, err);
            row++;
        }
    }
    mean_error = sum / (double)std::max(1, row);
    return true;
}

static void compute_boundary_errors(
    const InitialBSplineSurfacePatch& patch,
    int sample_count,
    double& mean_error,
    double& max_error) {
    int n = std::max(2, sample_count);
    double sum = 0.0;
    int count = 0;
    max_error = 0.0;

    for (int i = 0; i < n; i++) {
        double t = (double)i / (double)(n - 1);
        array<Vector3d, 4> surface_pts = {
            patch.surface.evaluate(t, 0.0),
            patch.surface.evaluate(1.0, t),
            patch.surface.evaluate(t, 1.0),
            patch.surface.evaluate(0.0, t)
        };
        array<Vector3d, 4> curve_pts = {
            patch.boundary_curves[0].evaluate(t),
            patch.boundary_curves[1].evaluate(t),
            patch.boundary_curves[2].evaluate(1.0 - t),
            patch.boundary_curves[3].evaluate(1.0 - t)
        };
        for (int s = 0; s < 4; s++) {
            double err = (surface_pts[s] - curve_pts[s]).norm();
            sum += err;
            max_error = std::max(max_error, err);
            count++;
        }
    }
    mean_error = sum / (double)std::max(1, count);
}

static bool export_matrix_obj(
    const string& filename,
    const MatrixXd& V,
    const MatrixXi& F,
    const string& comment) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "# " << comment << "\n";
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

} // namespace

InitialBSplineSurfaceConfig::InitialBSplineSurfaceConfig()
    : control_count_u(4),
      control_count_v(4),
      boundary_fairness_weight(1e-6),
      coons_fit_samples_u(12),
      coons_fit_samples_v(12),
      boundary_check_samples(25) {}

InitialBSplineSurfacePatch::InitialBSplineSurfacePatch()
    : boundary_mean_error(0.0),
      boundary_max_error(0.0),
      coons_fit_mean_error(0.0),
      coons_fit_max_error(0.0),
      valid(false) {}

InitialBSplineSurfacePatch build_initial_bspline_surface_from_quad_boundary(
    const QuadLikeBoundary& boundary,
    const InitialBSplineSurfaceConfig& cfg) {
    InitialBSplineSurfacePatch patch;
    string reason;
    if (!validate_boundary(boundary, reason)) {
        patch.reason = reason;
        return patch;
    }

    int nu = cfg.control_count_u;
    int nv = cfg.control_count_v;
    if (nu < 4 || nv < 4) {
        patch.reason = "bicubic surface needs at least 4 control points per direction";
        return patch;
    }

    BSplineCurveFitResult bottom = fit_cubic_bspline_curve_least_squares(
        boundary.side_polylines[0], nu, cfg.boundary_fairness_weight);
    BSplineCurveFitResult right = fit_cubic_bspline_curve_least_squares(
        boundary.side_polylines[1], nv, cfg.boundary_fairness_weight);
    BSplineCurveFitResult top = fit_cubic_bspline_curve_least_squares(
        boundary.side_polylines[2], nu, cfg.boundary_fairness_weight);
    BSplineCurveFitResult left = fit_cubic_bspline_curve_least_squares(
        boundary.side_polylines[3], nv, cfg.boundary_fairness_weight);

    if (!bottom.success || !right.success || !top.success || !left.success) {
        patch.reason = "boundary curve fitting failed";
        return patch;
    }

    patch.boundary_curves = {bottom.curve, right.curve, top.curve, left.curve};
    enforce_shared_corners(boundary, patch.boundary_curves);

    vector<double> knots_u = make_open_uniform_knot_vector(nu, 3);
    vector<double> knots_v = make_open_uniform_knot_vector(nv, 3);
    for (int s = 0; s < 4; s++) {
        patch.boundary_curves[s].degree = 3;
    }
    patch.boundary_curves[0].knots = knots_u;
    patch.boundary_curves[2].knots = knots_u;
    patch.boundary_curves[1].knots = knots_v;
    patch.boundary_curves[3].knots = knots_v;

    vector<vector<Vector3d>> grid = make_boundary_control_grid(
        patch.boundary_curves, nu, nv);

    if (!solve_internal_control_points(
            patch.boundary_curves, grid, knots_u, knots_v,
            3, 3,
            cfg.coons_fit_samples_u, cfg.coons_fit_samples_v,
            patch.coons_fit_mean_error,
            patch.coons_fit_max_error)) {
        patch.reason = "internal control point solve failed";
        return patch;
    }

    patch.surface = BSplineSurface3D(3, 3, knots_u, knots_v, grid);
    compute_boundary_errors(
        patch, cfg.boundary_check_samples,
        patch.boundary_mean_error, patch.boundary_max_error);

    patch.valid = true;
    patch.reason = "ok";
    return patch;
}

Vector3d evaluate_coons_patch(
    const array<BSplineCurve3D, 4>& boundary_curves,
    double u,
    double v) {
    Vector3d bottom = boundary_curves[0].evaluate(u);
    Vector3d right = boundary_curves[1].evaluate(v);
    Vector3d top = boundary_curves[2].evaluate(1.0 - u);
    Vector3d left = boundary_curves[3].evaluate(1.0 - v);

    Vector3d p00 = boundary_curves[0].evaluate(0.0);
    Vector3d p10 = boundary_curves[0].evaluate(1.0);
    Vector3d p11 = boundary_curves[1].evaluate(1.0);
    Vector3d p01 = boundary_curves[3].evaluate(0.0);

    Vector3d ruled =
        (1.0 - v) * bottom + v * top +
        (1.0 - u) * left + u * right;
    Vector3d bilinear =
        (1.0 - u) * (1.0 - v) * p00 +
        u * (1.0 - v) * p10 +
        u * v * p11 +
        (1.0 - u) * v * p01;
    return ruled - bilinear;
}

bool export_coons_patch_obj(
    const string& filename,
    const array<BSplineCurve3D, 4>& boundary_curves,
    int sample_u,
    int sample_v) {
    int su = std::max(2, sample_u);
    int sv = std::max(2, sample_v);
    MatrixXd V(su * sv, 3);
    MatrixXi F((su - 1) * (sv - 1) * 2, 3);
    auto id = [sv](int i, int j) { return i * sv + j; };

    for (int i = 0; i < su; i++) {
        double u = (double)i / (double)(su - 1);
        for (int j = 0; j < sv; j++) {
            double v = (double)j / (double)(sv - 1);
            V.row(id(i, j)) = evaluate_coons_patch(boundary_curves, u, v).transpose();
        }
    }

    int f = 0;
    for (int i = 0; i < su - 1; i++) {
        for (int j = 0; j < sv - 1; j++) {
            int a = id(i, j);
            int b = id(i + 1, j);
            int c = id(i, j + 1);
            int d = id(i + 1, j + 1);
            F.row(f++) << a, b, c;
            F.row(f++) << b, d, c;
        }
    }

    return export_matrix_obj(filename, V, F, "Coons initialization surface");
}

bool export_boundary_consistency_report(
    const string& filename,
    const InitialBSplineSurfacePatch& patch) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;

    fout << "valid," << (patch.valid ? 1 : 0) << "\n";
    fout << "reason," << patch.reason << "\n";
    fout << "boundary_mean_error," << patch.boundary_mean_error << "\n";
    fout << "boundary_max_error," << patch.boundary_max_error << "\n";
    fout << "coons_fit_mean_error," << patch.coons_fit_mean_error << "\n";
    fout << "coons_fit_max_error," << patch.coons_fit_max_error << "\n";
    fout << "check_equations,"
         << "S(u,0)=bottom(u);"
         << "S(1,v)=right(v);"
         << "S(u,1)=top(1-u);"
         << "S(0,v)=left(1-v)\n";
    return true;
}

bool export_initial_bspline_surface_debug(
    const string& prefix,
    const InitialBSplineSurfacePatch& patch,
    int surface_sample_u,
    int surface_sample_v) {
    if (!patch.valid) return false;

    bool ok = true;
    ok = export_bspline_curve_polyline_obj(
        prefix + "_boundary_bottom.obj", patch.boundary_curves[0], 64) && ok;
    ok = export_bspline_curve_polyline_obj(
        prefix + "_boundary_right.obj", patch.boundary_curves[1], 64) && ok;
    ok = export_bspline_curve_polyline_obj(
        prefix + "_boundary_top.obj", patch.boundary_curves[2], 64) && ok;
    ok = export_bspline_curve_polyline_obj(
        prefix + "_boundary_left.obj", patch.boundary_curves[3], 64) && ok;
    ok = export_coons_patch_obj(
        prefix + "_coons_surface.obj", patch.boundary_curves,
        surface_sample_u, surface_sample_v) && ok;
    ok = export_bspline_surface_control_net_obj(
        prefix + "_control_net.obj", patch.surface) && ok;
    ok = export_bspline_surface_mesh_obj(
        prefix + "_sampled_surface.obj", patch.surface,
        surface_sample_u, surface_sample_v) && ok;
    ok = export_boundary_consistency_report(
        prefix + "_boundary_report.csv", patch) && ok;
    return ok;
}
