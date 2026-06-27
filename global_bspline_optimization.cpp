#include "global_bspline_optimization.h"

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::Vector3d;
using std::map;
using std::set;
using std::string;
using std::vector;

namespace {

static bool is_boundary_cp(int i, int j, int nu, int nv) {
    return i == 0 || j == 0 || i == nu - 1 || j == nv - 1;
}

static int cp_id_at(const SharedSplinePatch& patch, int i, int j) {
    return patch.topology.control_point_ids[i][j];
}

static void sync_shared_boundary_curves_from_pool(SharedSplineAssembly& assembly) {
    for (SharedSplineBoundary& b : assembly.shared_boundaries) {
        if (b.control_point_ids.empty()) continue;
        b.curve.control_points.clear();
        b.curve.control_points.reserve(b.control_point_ids.size());
        for (int id : b.control_point_ids) {
            if (id >= 0 && id < (int)assembly.pool.values.size()) {
                b.curve.control_points.push_back(assembly.pool.values[id]);
            }
        }
    }
}

static SurfaceFitErrorStats compute_patch_error(
    const SharedSplinePatch& patch,
    const GlobalControlPointPool& pool,
    const vector<SurfaceFitSample>& samples) {
    SurfaceFitErrorStats stats;
    if (samples.empty()) return stats;
    BSplineSurface3D surface = materialize_patch_surface(patch, pool);
    double sum = 0.0;
    double sum_sq = 0.0;
    double max_err = 0.0;
    double wsum = 0.0;
    for (const SurfaceFitSample& s : samples) {
        if (s.weight <= 0.0) continue;
        double err = (surface.evaluate(s.uv.x(), s.uv.y()) - s.position).norm();
        sum += s.weight * err;
        sum_sq += s.weight * err * err;
        max_err = std::max(max_err, err);
        wsum += s.weight;
    }
    if (wsum > 0.0) {
        stats.mean_error = sum / wsum;
        stats.rms_error = std::sqrt(sum_sq / wsum);
        stats.max_error = max_err;
    }
    return stats;
}

static SurfaceFitErrorStats combine_error_stats(
    const SharedSplineAssembly& assembly,
    const vector<GlobalBSplineRegionInput>& inputs) {
    SurfaceFitErrorStats stats;
    double sum = 0.0;
    double sum_sq = 0.0;
    double max_err = 0.0;
    double wsum = 0.0;
    for (const GlobalBSplineRegionInput& input : inputs) {
        if (input.patch_index < 0 || input.patch_index >= (int)assembly.patches.size()) continue;
        BSplineSurface3D surface = materialize_patch_surface(
            assembly.patches[input.patch_index], assembly.pool);
        for (const SurfaceFitSample& s : input.samples) {
            if (s.weight <= 0.0) continue;
            double err = (surface.evaluate(s.uv.x(), s.uv.y()) - s.position).norm();
            sum += s.weight * err;
            sum_sq += s.weight * err * err;
            max_err = std::max(max_err, err);
            wsum += s.weight;
        }
    }
    if (wsum > 0.0) {
        stats.mean_error = sum / wsum;
        stats.rms_error = std::sqrt(sum_sq / wsum);
        stats.max_error = max_err;
    }
    return stats;
}

static SurfaceFitErrorStats compute_boundary_drift(
    const SharedSplineAssembly& before,
    const SharedSplineAssembly& after,
    int sample_count) {
    SurfaceFitErrorStats stats;
    int count = std::max(2, sample_count);
    double sum = 0.0;
    double sum_sq = 0.0;
    double max_err = 0.0;
    double wsum = 0.0;
    int boundary_count = std::min(
        (int)before.shared_boundaries.size(),
        (int)after.shared_boundaries.size());
    for (int bi = 0; bi < boundary_count; bi++) {
        const SharedSplineBoundary& a = before.shared_boundaries[bi];
        const SharedSplineBoundary& b = after.shared_boundaries[bi];
        for (int k = 0; k < count; k++) {
            double t = (double)k / (double)(count - 1);
            double err = (a.curve.evaluate(t) - b.curve.evaluate(t)).norm();
            sum += err;
            sum_sq += err * err;
            max_err = std::max(max_err, err);
            wsum += 1.0;
        }
    }
    if (wsum > 0.0) {
        stats.mean_error = sum / wsum;
        stats.rms_error = std::sqrt(sum_sq / wsum);
        stats.max_error = max_err;
    }
    return stats;
}

static void add_coeff(
    vector<Triplet<double>>& triplets,
    int row,
    int col,
    double value) {
    if (std::abs(value) > 0.0) triplets.push_back(Triplet<double>(row, col, value));
}

static double estimate_condition_from_normal_matrix(const MatrixXd& dense_normal) {
    if (dense_normal.rows() == 0) return 0.0;
    Eigen::SelfAdjointEigenSolver<MatrixXd> eig(dense_normal);
    if (eig.info() != Eigen::Success) return std::numeric_limits<double>::infinity();
    double min_pos = std::numeric_limits<double>::infinity();
    double max_val = 0.0;
    for (int i = 0; i < eig.eigenvalues().size(); i++) {
        double v = eig.eigenvalues()(i);
        max_val = std::max(max_val, std::abs(v));
        if (v > 1e-14) min_pos = std::min(min_pos, v);
    }
    if (!std::isfinite(min_pos) || min_pos <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return max_val / min_pos;
}

} // namespace

GlobalBSplineOptimizationConfig::GlobalBSplineOptimizationConfig()
    : fairness_weight(1e-4),
      initial_weight(1e-3),
      optimize_shared_boundary_control_points(false),
      boundary_fit_weight(1.0),
      boundary_fairness_weight(1e-4),
      boundary_fit_sample_count(16),
      boundary_drift_sample_count(64),
      max_boundary_drift(std::numeric_limits<double>::infinity()),
      rollback_on_excessive_boundary_drift(true),
      compare_with_fixed_boundary(true),
      sample_u(16),
      sample_v(16) {}

GlobalBSplineRegionReport::GlobalBSplineRegionReport()
    : patch_index(-1),
      region_id(-1),
      optimized(false) {}

GlobalBSplineOptimizationResult::GlobalBSplineOptimizationResult()
    : total_control_points(0),
      variable_control_points(0),
      spline_patch_count(0),
      source_vertex_count(0),
      source_triangle_count(0),
      matrix_rows(0),
      matrix_cols(0),
      matrix_nonzeros(0),
      normal_matrix_condition_estimate(0.0),
      movable_boundary_enabled(false),
      has_fixed_boundary_comparison(false),
      boundary_drift_exceeded(false),
      used_fixed_boundary_fallback(false),
      fixed_boundary_variable_control_points(0),
      valid(false) {}

GlobalBSplineOptimizationResult optimize_global_bspline_control_points(
    const SharedSplineAssembly& assembly,
    const vector<GlobalBSplineRegionInput>& region_inputs,
    const GlobalBSplineOptimizationConfig& cfg) {
    GlobalBSplineOptimizationResult result;
    result.optimized_assembly = assembly;
    sync_shared_boundary_curves_from_pool(result.optimized_assembly);
    result.total_control_points = (int)assembly.pool.values.size();
    result.spline_patch_count = (int)assembly.patches.size();
    result.movable_boundary_enabled = cfg.optimize_shared_boundary_control_points;

    if (!assembly.valid) {
        result.reason = "input assembly is invalid";
        return result;
    }

    GlobalBSplineOptimizationResult fixed_result;
    if (cfg.optimize_shared_boundary_control_points &&
        cfg.compare_with_fixed_boundary) {
        GlobalBSplineOptimizationConfig fixed_cfg = cfg;
        fixed_cfg.optimize_shared_boundary_control_points = false;
        fixed_cfg.compare_with_fixed_boundary = false;
        fixed_result = optimize_global_bspline_control_points(
            assembly, region_inputs, fixed_cfg);
        if (fixed_result.valid) {
            result.has_fixed_boundary_comparison = true;
            result.fixed_boundary_assembly = fixed_result.optimized_assembly;
            result.fixed_boundary_sampled_mesh = fixed_result.sampled_mesh;
            result.fixed_boundary_adjacency_report = fixed_result.adjacency_report;
            result.fixed_boundary_global_after = fixed_result.global_after;
            result.fixed_boundary_variable_control_points =
                fixed_result.variable_control_points;
        }
    }

    map<int, const GlobalBSplineRegionInput*> input_by_patch;
    for (const GlobalBSplineRegionInput& input : region_inputs) {
        if (input.patch_index >= 0 && input.patch_index < (int)assembly.patches.size()) {
            input_by_patch[input.patch_index] = &input;
        } else {
            result.skipped_regions.push_back(
                "patch " + std::to_string(input.patch_index) +
                ": unsupported patch index for region " +
                std::to_string(input.region_id));
        }
        result.source_vertex_count += (int)input.samples.size();
    }

    set<int> variable_cp_ids;
    vector<GlobalBSplineRegionReport> reports;
    for (int pi = 0; pi < (int)assembly.patches.size(); pi++) {
        const SharedSplinePatch& patch = assembly.patches[pi];
        GlobalBSplineRegionReport report;
        report.patch_index = pi;
        report.region_id = patch.region_id;

        auto input_it = input_by_patch.find(pi);
        if (input_it == input_by_patch.end() || input_it->second->samples.empty()) {
            report.reason = "missing or empty region samples";
            result.skipped_regions.push_back("patch " + std::to_string(pi) + ": " + report.reason);
            reports.push_back(report);
            continue;
        }

        report.before = compute_patch_error(patch, assembly.pool, input_it->second->samples);
        int nu = (int)patch.topology.control_point_ids.size();
        int nv = nu > 0 ? (int)patch.topology.control_point_ids[0].size() : 0;
        if (nu < 3 || nv < 3) {
            report.reason = "patch topology has no interior control points";
            result.skipped_regions.push_back("patch " + std::to_string(pi) + ": " + report.reason);
            reports.push_back(report);
            continue;
        }
        for (int i = 1; i < nu - 1; i++) {
            for (int j = 1; j < nv - 1; j++) {
                int id = cp_id_at(patch, i, j);
                if (id >= 0 && id < (int)assembly.pool.values.size()) {
                    variable_cp_ids.insert(id);
                }
            }
        }
        report.optimized = true;
        report.reason = "queued";
        reports.push_back(report);
    }

    if (cfg.optimize_shared_boundary_control_points) {
        for (const SharedSplineBoundary& boundary : assembly.shared_boundaries) {
            for (int k = 1; k + 1 < (int)boundary.control_point_ids.size(); k++) {
                int id = boundary.control_point_ids[k];
                if (id >= 0 && id < (int)assembly.pool.values.size()) {
                    variable_cp_ids.insert(id);
                }
            }
        }
    }

    if (variable_cp_ids.empty()) {
        result.region_reports = reports;
        result.reason = "no supported variable control points";
        return result;
    }

    map<int, int> cp_to_unknown;
    for (int id : variable_cp_ids) {
        int idx = (int)cp_to_unknown.size();
        cp_to_unknown[id] = idx;
    }
    result.variable_control_points = (int)cp_to_unknown.size();
    result.matrix_cols = result.variable_control_points;

    int row_count = 0;
    for (int ri = 0; ri < (int)reports.size(); ri++) {
        if (!reports[ri].optimized) continue;
        const GlobalBSplineRegionInput& input = *input_by_patch[reports[ri].patch_index];
        row_count += (int)input.samples.size();
        const SharedSplinePatch& patch = assembly.patches[reports[ri].patch_index];
        int nu = (int)patch.topology.control_point_ids.size();
        int nv = (int)patch.topology.control_point_ids[0].size();
        if (cfg.fairness_weight > 0.0) row_count += 2 * (nu - 2) * (nv - 2);
        if (cfg.initial_weight > 0.0) row_count += (nu - 2) * (nv - 2);
    }
    if (cfg.optimize_shared_boundary_control_points) {
        for (const SharedSplineBoundary& boundary : assembly.shared_boundaries) {
            int n = (int)boundary.control_point_ids.size();
            if (cfg.boundary_fit_weight > 0.0) {
                row_count += std::max(2, cfg.boundary_fit_sample_count);
            }
            if (cfg.boundary_fairness_weight > 0.0) {
                row_count += std::max(0, n - 2);
            }
            if (cfg.initial_weight > 0.0) {
                row_count += std::max(0, n - 2);
            }
        }
    }
    result.matrix_rows = row_count;

    vector<Triplet<double>> triplets;
    MatrixXd B = MatrixXd::Zero(row_count, 3);
    int row = 0;

    for (GlobalBSplineRegionReport& report : reports) {
        if (!report.optimized) continue;
        const SharedSplinePatch& patch = assembly.patches[report.patch_index];
        const GlobalBSplineRegionInput& input = *input_by_patch[report.patch_index];
        BSplineSurface3D surface = materialize_patch_surface(patch, assembly.pool);
        int nu = (int)patch.topology.control_point_ids.size();
        int nv = (int)patch.topology.control_point_ids[0].size();

        for (const SurfaceFitSample& s : input.samples) {
            double scale = std::sqrt(std::max(0.0, s.weight));
            Vector3d fixed = Vector3d::Zero();
            for (int i = 0; i < nu; i++) {
                double bu = bspline_basis(i, patch.degree_u, s.uv.x(), patch.knots_u);
                if (bu == 0.0) continue;
                for (int j = 0; j < nv; j++) {
                    double coeff = bu * bspline_basis(j, patch.degree_v, s.uv.y(), patch.knots_v);
                    if (coeff == 0.0) continue;
                    int cp_id = cp_id_at(patch, i, j);
                    auto uit = cp_to_unknown.find(cp_id);
                    if (uit == cp_to_unknown.end()) {
                        fixed += coeff * assembly.pool.values[cp_id];
                    } else {
                        add_coeff(triplets, row, uit->second, scale * coeff);
                    }
                }
            }
            B.row(row) = (scale * (s.position - fixed)).transpose();
            row++;
        }

        if (cfg.fairness_weight > 0.0) {
            double scale = std::sqrt(cfg.fairness_weight);
            for (int i = 1; i < nu - 1; i++) {
                for (int j = 1; j < nv - 1; j++) {
                    int center_id = cp_id_at(patch, i, j);
                    auto center_unknown = cp_to_unknown.find(center_id);
                    if (center_unknown == cp_to_unknown.end()) continue;

                    auto add_fair_term = [&](int ia, int ja, int ib, int jb) {
                        Vector3d fixed = Vector3d::Zero();
                        int ids[3] = {cp_id_at(patch, ia, ja), center_id, cp_id_at(patch, ib, jb)};
                        double coeffs[3] = {1.0, -2.0, 1.0};
                        for (int k = 0; k < 3; k++) {
                            auto uit = cp_to_unknown.find(ids[k]);
                            if (uit == cp_to_unknown.end()) fixed += scale * coeffs[k] * assembly.pool.values[ids[k]];
                            else add_coeff(triplets, row, uit->second, scale * coeffs[k]);
                        }
                        B.row(row) = (-fixed).transpose();
                        row++;
                    };

                    add_fair_term(i - 1, j, i + 1, j);
                    add_fair_term(i, j - 1, i, j + 1);
                }
            }
        }

        if (cfg.initial_weight > 0.0) {
            double scale = std::sqrt(cfg.initial_weight);
            for (int i = 1; i < nu - 1; i++) {
                for (int j = 1; j < nv - 1; j++) {
                    int cp_id = cp_id_at(patch, i, j);
                    auto uit = cp_to_unknown.find(cp_id);
                    if (uit == cp_to_unknown.end()) continue;
                    add_coeff(triplets, row, uit->second, scale);
                    B.row(row) = (scale * assembly.pool.values[cp_id]).transpose();
                    row++;
                }
            }
        }
    }

    if (cfg.optimize_shared_boundary_control_points) {
        for (const SharedSplineBoundary& boundary : assembly.shared_boundaries) {
            int n = (int)boundary.control_point_ids.size();
            if (n < 2) continue;

            if (cfg.boundary_fit_weight > 0.0) {
                int sample_count = std::max(2, cfg.boundary_fit_sample_count);
                double scale = std::sqrt(cfg.boundary_fit_weight);
                for (int k = 0; k < sample_count; k++) {
                    double t = (double)k / (double)(sample_count - 1);
                    Vector3d target = boundary.curve.evaluate(t);
                    Vector3d fixed = Vector3d::Zero();
                    for (int a = 0; a < n; a++) {
                        double coeff = bspline_basis(
                            a, boundary.curve.degree, t, boundary.curve.knots);
                        if (coeff == 0.0) continue;
                        int cp_id = boundary.control_point_ids[a];
                        auto uit = cp_to_unknown.find(cp_id);
                        if (uit == cp_to_unknown.end()) {
                            fixed += coeff * assembly.pool.values[cp_id];
                        } else {
                            add_coeff(triplets, row, uit->second, scale * coeff);
                        }
                    }
                    B.row(row) = (scale * (target - fixed)).transpose();
                    row++;
                }
            }

            if (cfg.boundary_fairness_weight > 0.0) {
                double scale = std::sqrt(cfg.boundary_fairness_weight);
                for (int a = 1; a + 1 < n; a++) {
                    Vector3d fixed = Vector3d::Zero();
                    int ids[3] = {
                        boundary.control_point_ids[a - 1],
                        boundary.control_point_ids[a],
                        boundary.control_point_ids[a + 1]};
                    double coeffs[3] = {1.0, -2.0, 1.0};
                    for (int k = 0; k < 3; k++) {
                        auto uit = cp_to_unknown.find(ids[k]);
                        if (uit == cp_to_unknown.end()) {
                            fixed += scale * coeffs[k] * assembly.pool.values[ids[k]];
                        } else {
                            add_coeff(triplets, row, uit->second, scale * coeffs[k]);
                        }
                    }
                    B.row(row) = (-fixed).transpose();
                    row++;
                }
            }

            if (cfg.initial_weight > 0.0) {
                double scale = std::sqrt(cfg.initial_weight);
                for (int a = 1; a + 1 < n; a++) {
                    int cp_id = boundary.control_point_ids[a];
                    auto uit = cp_to_unknown.find(cp_id);
                    if (uit == cp_to_unknown.end()) continue;
                    add_coeff(triplets, row, uit->second, scale);
                    B.row(row) = (scale * assembly.pool.values[cp_id]).transpose();
                    row++;
                }
            }
        }
    }

    result.matrix_rows = row;
    SparseMatrix<double> A(row, result.matrix_cols);
    A.setFromTriplets(triplets.begin(), triplets.end());
    result.matrix_nonzeros = (int)A.nonZeros();
    if (A.rows() == 0 || A.cols() == 0 || A.nonZeros() == 0) {
        result.reason = "assembled sparse system is empty";
        result.region_reports = reports;
        return result;
    }

    SparseMatrix<double> AtA = A.transpose() * A;
    MatrixXd AtB = A.transpose() * B.topRows(row);
    Eigen::SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(AtA);
    if (solver.info() != Eigen::Success) {
        result.reason = "sparse normal-equation factorization failed";
        result.region_reports = reports;
        return result;
    }
    MatrixXd X = solver.solve(AtB);
    if (solver.info() != Eigen::Success) {
        result.reason = "sparse normal-equation solve failed";
        result.region_reports = reports;
        return result;
    }

    if (result.matrix_cols <= 128) {
        result.normal_matrix_condition_estimate =
            estimate_condition_from_normal_matrix(MatrixXd(AtA));
    } else {
        result.normal_matrix_condition_estimate = -1.0;
    }

    for (const auto& kv : cp_to_unknown) {
        result.optimized_assembly.pool.values[kv.first] = X.row(kv.second).transpose();
    }
    sync_shared_boundary_curves_from_pool(result.optimized_assembly);

    result.global_before = combine_error_stats(assembly, region_inputs);
    result.global_after = combine_error_stats(result.optimized_assembly, region_inputs);
    result.boundary_drift = compute_boundary_drift(
        assembly,
        result.optimized_assembly,
        cfg.boundary_drift_sample_count);
    result.boundary_drift_exceeded =
        cfg.optimize_shared_boundary_control_points &&
        result.boundary_drift.max_error > cfg.max_boundary_drift;

    for (GlobalBSplineRegionReport& report : reports) {
        if (!report.optimized) continue;
        const GlobalBSplineRegionInput& input = *input_by_patch[report.patch_index];
        report.after = compute_patch_error(
            result.optimized_assembly.patches[report.patch_index],
            result.optimized_assembly.pool,
            input.samples);
        report.reason = "ok";
    }
    result.region_reports = reports;

    if (result.boundary_drift_exceeded &&
        cfg.rollback_on_excessive_boundary_drift &&
        result.has_fixed_boundary_comparison) {
        result.used_fixed_boundary_fallback = true;
        result.optimized_assembly = result.fixed_boundary_assembly;
        result.sampled_mesh = result.fixed_boundary_sampled_mesh;
        result.adjacency_report = result.fixed_boundary_adjacency_report;
        result.global_after = result.fixed_boundary_global_after;
        result.boundary_drift = compute_boundary_drift(
            assembly,
            result.optimized_assembly,
            cfg.boundary_drift_sample_count);
        for (GlobalBSplineRegionReport& report : result.region_reports) {
            if (!report.optimized) continue;
            const GlobalBSplineRegionInput& input = *input_by_patch[report.patch_index];
            report.after = compute_patch_error(
                result.optimized_assembly.patches[report.patch_index],
                result.optimized_assembly.pool,
                input.samples);
            report.reason = "ok_fixed_boundary_fallback";
        }
        result.valid = true;
        result.reason = "ok_fixed_boundary_fallback";
        return result;
    }

    sample_shared_spline_assembly(
        result.optimized_assembly,
        cfg.sample_u,
        cfg.sample_v,
        result.sampled_mesh,
        result.adjacency_report);

    result.valid = true;
    result.reason = "ok";
    return result;
}

bool export_global_bspline_region_report_csv(
    const string& filename,
    const GlobalBSplineOptimizationResult& result) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "patch_index,region_id,optimized,reason,before_mean,before_rms,before_max,after_mean,after_rms,after_max\n";
    for (const GlobalBSplineRegionReport& r : result.region_reports) {
        fout << r.patch_index << ","
             << r.region_id << ","
             << (r.optimized ? 1 : 0) << ","
             << r.reason << ","
             << r.before.mean_error << ","
             << r.before.rms_error << ","
             << r.before.max_error << ","
             << r.after.mean_error << ","
             << r.after.rms_error << ","
             << r.after.max_error << "\n";
    }
    return true;
}

bool export_global_bspline_summary_csv(
    const string& filename,
    const GlobalBSplineOptimizationResult& result) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "valid," << (result.valid ? 1 : 0) << "\n";
    fout << "reason," << result.reason << "\n";
    fout << "total_control_points," << result.total_control_points << "\n";
    fout << "variable_control_points," << result.variable_control_points << "\n";
    fout << "spline_patch_count," << result.spline_patch_count << "\n";
    fout << "source_vertex_count," << result.source_vertex_count << "\n";
    fout << "source_triangle_count," << result.source_triangle_count << "\n";
    fout << "matrix_rows," << result.matrix_rows << "\n";
    fout << "matrix_cols," << result.matrix_cols << "\n";
    fout << "matrix_nonzeros," << result.matrix_nonzeros << "\n";
    fout << "normal_matrix_condition_estimate," << result.normal_matrix_condition_estimate << "\n";
    fout << "movable_boundary_enabled," << (result.movable_boundary_enabled ? 1 : 0) << "\n";
    fout << "has_fixed_boundary_comparison," << (result.has_fixed_boundary_comparison ? 1 : 0) << "\n";
    fout << "fixed_boundary_variable_control_points," << result.fixed_boundary_variable_control_points << "\n";
    fout << "boundary_drift_mean," << result.boundary_drift.mean_error << "\n";
    fout << "boundary_drift_rms," << result.boundary_drift.rms_error << "\n";
    fout << "boundary_drift_max," << result.boundary_drift.max_error << "\n";
    fout << "boundary_drift_exceeded," << (result.boundary_drift_exceeded ? 1 : 0) << "\n";
    fout << "used_fixed_boundary_fallback," << (result.used_fixed_boundary_fallback ? 1 : 0) << "\n";
    fout << "fixed_boundary_after_mean," << result.fixed_boundary_global_after.mean_error << "\n";
    fout << "fixed_boundary_after_rms," << result.fixed_boundary_global_after.rms_error << "\n";
    fout << "fixed_boundary_after_max," << result.fixed_boundary_global_after.max_error << "\n";
    fout << "global_before_mean," << result.global_before.mean_error << "\n";
    fout << "global_before_rms," << result.global_before.rms_error << "\n";
    fout << "global_before_max," << result.global_before.max_error << "\n";
    fout << "global_after_mean," << result.global_after.mean_error << "\n";
    fout << "global_after_rms," << result.global_after.rms_error << "\n";
    fout << "global_after_max," << result.global_after.max_error << "\n";
    for (const string& skipped : result.skipped_regions) {
        fout << "skipped," << skipped << "\n";
    }
    return true;
}

bool export_global_bspline_optimization_debug(
    const string& prefix,
    const GlobalBSplineOptimizationResult& result) {
    bool ok = true;
    ok = export_shared_spline_debug(
        prefix,
        result.optimized_assembly,
        result.sampled_mesh,
        result.adjacency_report) && ok;
    ok = export_global_bspline_region_report_csv(
        prefix + "_region_report.csv", result) && ok;
    ok = export_global_bspline_summary_csv(
        prefix + "_summary.csv", result) && ok;
    if (result.movable_boundary_enabled) {
        ok = export_shared_spline_debug(
            prefix + "_movable_boundary",
            result.optimized_assembly,
            result.sampled_mesh,
            result.adjacency_report) && ok;
        if (result.has_fixed_boundary_comparison) {
            ok = export_shared_spline_debug(
                prefix + "_fixed_boundary",
                result.fixed_boundary_assembly,
                result.fixed_boundary_sampled_mesh,
                result.fixed_boundary_adjacency_report) && ok;
        }
    }
    return ok;
}
