#include "fit_bspline_surface_interior.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::string;
using std::vector;

namespace {

static bool finite_vec3(const Vector3d& p) {
    return std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z());
}

static int interior_unknown_index(int i, int j, int nv) {
    return (i - 1) * (nv - 2) + (j - 1);
}

static bool is_boundary_cp(int i, int j, int nu, int nv) {
    return i == 0 || j == 0 || i == nu - 1 || j == nv - 1;
}

static Vector3d surface_control_contribution(
    const BSplineSurface3D& surface,
    double u,
    double v,
    bool boundary_only) {
    int nu = (int)surface.control_grid.size();
    int nv = (int)surface.control_grid[0].size();
    Vector3d value = Vector3d::Zero();
    for (int i = 0; i < nu; i++) {
        double bu = bspline_basis(i, surface.degree_u, u, surface.knots_u);
        if (bu == 0.0) continue;
        for (int j = 0; j < nv; j++) {
            if (boundary_only != is_boundary_cp(i, j, nu, nv)) continue;
            double bv = bspline_basis(j, surface.degree_v, v, surface.knots_v);
            if (bv == 0.0) continue;
            value += bu * bv * surface.control_grid[i][j];
        }
    }
    return value;
}

static SurfaceFitErrorStats compute_error_stats(
    const BSplineSurface3D& surface,
    const vector<SurfaceFitSample>& samples) {
    SurfaceFitErrorStats stats;
    if (samples.empty()) return stats;

    double sum = 0.0;
    double sum_sq = 0.0;
    double max_err = 0.0;
    double weight_sum = 0.0;
    for (const SurfaceFitSample& s : samples) {
        if (s.weight <= 0.0) continue;
        double err = (surface.evaluate(s.uv.x(), s.uv.y()) - s.position).norm();
        sum += s.weight * err;
        sum_sq += s.weight * err * err;
        max_err = std::max(max_err, err);
        weight_sum += s.weight;
    }

    if (weight_sum > 0.0) {
        stats.mean_error = sum / weight_sum;
        stats.rms_error = std::sqrt(sum_sq / weight_sum);
        stats.max_error = max_err;
    }
    return stats;
}

static void add_scalar_row(
    MatrixXd& A,
    MatrixXd& B,
    int& row,
    int unknown_index,
    double coeff,
    const Vector3d& rhs) {
    A(row, unknown_index) += coeff;
    B.row(row) = rhs.transpose();
    row++;
}

static void add_vector_row(
    MatrixXd& A,
    MatrixXd& B,
    int row,
    int unknown_count,
    const vector<double>& coeffs,
    const Vector3d& rhs) {
    for (int k = 0; k < unknown_count; k++) {
        A(row, k) = coeffs[k];
    }
    B.row(row) = rhs.transpose();
}

static bool sampled_surface_has_flipped_normals(
    const BSplineSurface3D& surface,
    int sample_u,
    int sample_v) {
    int su = std::max(2, sample_u);
    int sv = std::max(2, sample_v);
    Vector3d reference = Vector3d::Zero();
    bool has_reference = false;

    for (int i = 0; i < su - 1; i++) {
        double u0 = (double)i / (double)(su - 1);
        double u1 = (double)(i + 1) / (double)(su - 1);
        for (int j = 0; j < sv - 1; j++) {
            double v0 = (double)j / (double)(sv - 1);
            double v1 = (double)(j + 1) / (double)(sv - 1);
            Vector3d p00 = surface.evaluate(u0, v0);
            Vector3d p10 = surface.evaluate(u1, v0);
            Vector3d p01 = surface.evaluate(u0, v1);
            Vector3d n = (p10 - p00).cross(p01 - p00);
            if (n.norm() <= 1e-14 || !finite_vec3(n)) continue;
            n.normalize();
            if (!has_reference) {
                reference = n;
                has_reference = true;
            } else if (reference.dot(n) < -0.2) {
                return true;
            }
        }
    }
    return false;
}

static bool control_points_outside_bbox(
    const BSplineSurface3D& surface,
    const vector<SurfaceFitSample>& samples,
    double padding_factor) {
    if (samples.empty()) return false;
    Vector3d mn = samples.front().position;
    Vector3d mx = samples.front().position;
    for (const SurfaceFitSample& s : samples) {
        mn = mn.cwiseMin(s.position);
        mx = mx.cwiseMax(s.position);
    }
    Vector3d ext = mx - mn;
    double pad = std::max(1e-8, padding_factor * std::max(1e-8, ext.norm()));
    mn.array() -= pad;
    mx.array() += pad;

    for (const auto& row : surface.control_grid) {
        for (const Vector3d& p : row) {
            if ((p.array() < mn.array()).any() || (p.array() > mx.array()).any()) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

SurfaceFitSample::SurfaceFitSample()
    : uv(Vector2d::Zero()),
      position(Vector3d::Zero()),
      normal(Vector3d::UnitZ()),
      weight(1.0),
      has_normal(false) {}

SurfaceInteriorFitConfig::SurfaceInteriorFitConfig()
    : fairness_weight(1e-4),
      initial_weight(1e-3),
      point_to_plane_weight(0.0),
      enable_point_to_plane(false),
      control_bbox_padding_factor(2.0),
      surface_sample_u(16),
      surface_sample_v(16) {}

SurfaceFitErrorStats::SurfaceFitErrorStats()
    : mean_error(0.0),
      rms_error(0.0),
      max_error(0.0) {}

SurfaceInteriorFitResult::SurfaceInteriorFitResult()
    : valid(false),
      normal_flip_detected(false),
      control_points_out_of_bounds(false) {}

vector<SurfaceFitSample> make_region_vertex_fit_samples(
    const MatrixXd& V,
    const RegionSquareParameterizationResult& parameterization) {
    vector<SurfaceFitSample> samples;
    samples.reserve(parameterization.region_vertex_ids.size());
    for (int vid : parameterization.region_vertex_ids) {
        if (vid < 0 || vid >= V.rows()) continue;
        if (!std::isfinite(parameterization.UV(vid, 0)) ||
            !std::isfinite(parameterization.UV(vid, 1))) {
            continue;
        }
        SurfaceFitSample s;
        s.uv = parameterization.UV.row(vid).transpose();
        s.position = V.row(vid).transpose();
        s.weight = 1.0;
        samples.push_back(s);
    }
    return samples;
}

SurfaceInteriorFitResult fit_bspline_surface_interior_control_points(
    const InitialBSplineSurfacePatch& initial_patch,
    const vector<SurfaceFitSample>& samples,
    const SurfaceInteriorFitConfig& cfg) {
    SurfaceInteriorFitResult result;
    result.patch = initial_patch;

    if (!initial_patch.valid) {
        result.reason = "initial patch is invalid";
        return result;
    }
    if (samples.empty()) {
        result.reason = "no fit samples";
        return result;
    }

    const BSplineSurface3D& surface0 = initial_patch.surface;
    int nu = (int)surface0.control_grid.size();
    int nv = (int)surface0.control_grid[0].size();
    if (nu < 3 || nv < 3) {
        result.reason = "surface control grid is too small";
        return result;
    }
    int unknown_count = (nu - 2) * (nv - 2);
    if (unknown_count <= 0) {
        result.reason = "no interior control points";
        return result;
    }

    result.before = compute_error_stats(surface0, samples);

    int fit_rows = 0;
    for (const SurfaceFitSample& s : samples) {
        if (s.weight > 0.0) fit_rows++;
        if (cfg.enable_point_to_plane && cfg.point_to_plane_weight > 0.0 && s.has_normal) {
            fit_rows++;
        }
    }
    int fair_rows = 0;
    if (cfg.fairness_weight > 0.0) {
        for (int i = 1; i < nu - 1; i++) {
            for (int j = 1; j < nv - 1; j++) {
                if (i > 0 && i < nu - 1) fair_rows++;
                if (j > 0 && j < nv - 1) fair_rows++;
            }
        }
    }
    int initial_rows = cfg.initial_weight > 0.0 ? unknown_count : 0;
    int row_count = fit_rows + fair_rows + initial_rows;

    MatrixXd A = MatrixXd::Zero(row_count, unknown_count);
    MatrixXd B = MatrixXd::Zero(row_count, 3);
    int row = 0;

    for (const SurfaceFitSample& s : samples) {
        if (s.weight <= 0.0) continue;
        double scale = std::sqrt(s.weight);
        vector<double> coeffs(unknown_count, 0.0);
        for (int i = 1; i < nu - 1; i++) {
            double bu = bspline_basis(i, surface0.degree_u, s.uv.x(), surface0.knots_u);
            if (bu == 0.0) continue;
            for (int j = 1; j < nv - 1; j++) {
                double bv = bspline_basis(j, surface0.degree_v, s.uv.y(), surface0.knots_v);
                coeffs[interior_unknown_index(i, j, nv)] = scale * bu * bv;
            }
        }
        Vector3d rhs = scale * (s.position -
            surface_control_contribution(surface0, s.uv.x(), s.uv.y(), true));
        add_vector_row(A, B, row, unknown_count, coeffs, rhs);
        row++;

        if (cfg.enable_point_to_plane && cfg.point_to_plane_weight > 0.0 && s.has_normal) {
            Vector3d n = s.normal;
            if (n.norm() > 1e-12) {
                n.normalize();
                double pscale = std::sqrt(s.weight * cfg.point_to_plane_weight);
                vector<double> pcoeffs(unknown_count, 0.0);
                for (int i = 1; i < nu - 1; i++) {
                    double bu = bspline_basis(i, surface0.degree_u, s.uv.x(), surface0.knots_u);
                    for (int j = 1; j < nv - 1; j++) {
                        double bv = bspline_basis(j, surface0.degree_v, s.uv.y(), surface0.knots_v);
                        pcoeffs[interior_unknown_index(i, j, nv)] = pscale * bu * bv;
                    }
                }
                double target = n.dot(s.position -
                    surface_control_contribution(surface0, s.uv.x(), s.uv.y(), true));
                B.row(row) = (pscale * target * n).transpose();
                for (int k = 0; k < unknown_count; k++) A(row, k) = pcoeffs[k];
                row++;
            }
        }
    }

    if (cfg.fairness_weight > 0.0) {
        double scale = std::sqrt(cfg.fairness_weight);
        for (int i = 1; i < nu - 1; i++) {
            for (int j = 1; j < nv - 1; j++) {
                int center = interior_unknown_index(i, j, nv);
                if (i > 0 && i < nu - 1) {
                    Vector3d rhs = Vector3d::Zero();
                    A(row, center) += -2.0 * scale;
                    if (i - 1 == 0) rhs -= scale * surface0.control_grid[i - 1][j];
                    else A(row, interior_unknown_index(i - 1, j, nv)) += scale;
                    if (i + 1 == nu - 1) rhs -= scale * surface0.control_grid[i + 1][j];
                    else A(row, interior_unknown_index(i + 1, j, nv)) += scale;
                    B.row(row) = rhs.transpose();
                    row++;
                }
                if (j > 0 && j < nv - 1) {
                    Vector3d rhs = Vector3d::Zero();
                    A(row, center) += -2.0 * scale;
                    if (j - 1 == 0) rhs -= scale * surface0.control_grid[i][j - 1];
                    else A(row, interior_unknown_index(i, j - 1, nv)) += scale;
                    if (j + 1 == nv - 1) rhs -= scale * surface0.control_grid[i][j + 1];
                    else A(row, interior_unknown_index(i, j + 1, nv)) += scale;
                    B.row(row) = rhs.transpose();
                    row++;
                }
            }
        }
    }

    if (cfg.initial_weight > 0.0) {
        double scale = std::sqrt(cfg.initial_weight);
        for (int i = 1; i < nu - 1; i++) {
            for (int j = 1; j < nv - 1; j++) {
                int idx = interior_unknown_index(i, j, nv);
                add_scalar_row(A, B, row, idx, scale, scale * surface0.control_grid[i][j]);
            }
        }
    }

    MatrixXd X = A.colPivHouseholderQr().solve(B);
    if (X.rows() != unknown_count || X.cols() != 3) {
        result.reason = "least-squares solve failed";
        return result;
    }

    BSplineSurface3D fitted = surface0;
    for (int i = 1; i < nu - 1; i++) {
        for (int j = 1; j < nv - 1; j++) {
            fitted.control_grid[i][j] =
                X.row(interior_unknown_index(i, j, nv)).transpose();
        }
    }

    result.patch.surface = fitted;
    result.after = compute_error_stats(fitted, samples);
    result.normal_flip_detected = sampled_surface_has_flipped_normals(
        fitted, cfg.surface_sample_u, cfg.surface_sample_v);
    result.control_points_out_of_bounds = control_points_outside_bbox(
        fitted, samples, cfg.control_bbox_padding_factor);

    if (result.normal_flip_detected) {
        result.reason = "sampled surface normal flip detected";
        return result;
    }
    if (result.control_points_out_of_bounds) {
        result.reason = "control point moved outside padded region bounding box";
        return result;
    }

    result.valid = true;
    result.reason = "ok";
    return result;
}

bool export_surface_fit_error_heatmap_obj(
    const string& filename,
    const BSplineSurface3D& surface,
    const vector<SurfaceFitSample>& samples) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "# Surface fit point error heatmap as sample points\n";
    double max_err = 0.0;
    vector<double> errors(samples.size(), 0.0);
    for (int i = 0; i < (int)samples.size(); i++) {
        errors[i] = (surface.evaluate(samples[i].uv.x(), samples[i].uv.y()) -
                     samples[i].position).norm();
        max_err = std::max(max_err, errors[i]);
    }
    for (int i = 0; i < (int)samples.size(); i++) {
        double t = max_err > 1e-14 ? errors[i] / max_err : 0.0;
        const Vector3d& p = samples[i].position;
        fout << "v " << p.x() << " " << p.y() << " " << p.z()
             << " " << t << " 0 " << (1.0 - t) << "\n";
    }
    for (int i = 0; i < (int)samples.size(); i++) {
        fout << "p " << (i + 1) << "\n";
    }
    return true;
}

bool export_surface_fit_report_csv(
    const string& filename,
    const SurfaceInteriorFitResult& result) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "valid," << (result.valid ? 1 : 0) << "\n";
    fout << "reason," << result.reason << "\n";
    fout << "before_mean," << result.before.mean_error << "\n";
    fout << "before_rms," << result.before.rms_error << "\n";
    fout << "before_max," << result.before.max_error << "\n";
    fout << "after_mean," << result.after.mean_error << "\n";
    fout << "after_rms," << result.after.rms_error << "\n";
    fout << "after_max," << result.after.max_error << "\n";
    fout << "normal_flip_detected," << (result.normal_flip_detected ? 1 : 0) << "\n";
    fout << "control_points_out_of_bounds," << (result.control_points_out_of_bounds ? 1 : 0) << "\n";
    return true;
}

bool export_surface_interior_fit_debug(
    const string& prefix,
    const SurfaceInteriorFitResult& result,
    const vector<SurfaceFitSample>& samples) {
    bool ok = true;
    ok = export_bspline_surface_control_net_obj(
        prefix + "_control_net.obj", result.patch.surface) && ok;
    ok = export_bspline_surface_mesh_obj(
        prefix + "_sampled_surface.obj", result.patch.surface,
        24, 24) && ok;
    ok = export_surface_fit_error_heatmap_obj(
        prefix + "_error_heatmap.obj", result.patch.surface, samples) && ok;
    ok = export_surface_fit_report_csv(
        prefix + "_report.csv", result) && ok;
    return ok;
}
