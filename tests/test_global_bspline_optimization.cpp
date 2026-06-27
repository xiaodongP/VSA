#include "global_bspline_optimization.h"

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Eigen::Vector3d;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            cerr << "  FAIL: " << msg << endl; \
            g_fail++; \
        } else { \
            cout << "  OK:   " << msg << endl; \
            g_pass++; \
        } \
    } while (0)

static bool file_nonempty(const string& filename) {
    std::ifstream fin(filename, std::ios::binary | std::ios::ate);
    return fin.is_open() && fin.tellg() > 0;
}

static BSplineSurface3D make_flat_patch(double x0, double x1) {
    const int n = 5;
    vector<vector<Vector3d>> grid(n, vector<Vector3d>(n));
    for (int i = 0; i < n; i++) {
        double u = (double)i / (double)(n - 1);
        for (int j = 0; j < n; j++) {
            double v = (double)j / (double)(n - 1);
            grid[i][j] = Vector3d((1.0 - u) * x0 + u * x1, v, 0.0);
        }
    }
    return BSplineSurface3D(
        3, 3,
        make_open_uniform_knot_vector(n, 3),
        make_open_uniform_knot_vector(n, 3),
        grid);
}

static vector<SurfaceFitSample> make_patch_samples(int patch_index) {
    vector<SurfaceFitSample> samples;
    for (int i = 0; i <= 8; i++) {
        for (int j = 0; j <= 8; j++) {
            double u = (double)i / 8.0;
            double v = (double)j / 8.0;
            double x = patch_index == 0 ? u : 1.0 + u;
            double y = patch_index == 0 ? v : 1.0 - v;
            double bump = 0.22 * std::sin(3.141592653589793 * u) *
                          std::sin(3.141592653589793 * v);
            SurfaceFitSample s;
            s.uv = Eigen::Vector2d(u, v);
            s.position = Vector3d(x, y, bump);
            s.weight = (i == 0 || j == 0 || i == 8 || j == 8) ? 0.25 : 1.0;
            samples.push_back(s);
        }
    }
    return samples;
}

static vector<SurfaceFitSample> make_patch_samples_with_shared_boundary_ridge(int patch_index) {
    vector<SurfaceFitSample> samples;
    for (int i = 0; i <= 10; i++) {
        for (int j = 0; j <= 10; j++) {
            double u = (double)i / 10.0;
            double v = (double)j / 10.0;
            double x = patch_index == 0 ? u : 1.0 + u;
            double y = patch_index == 0 ? v : 1.0 - v;
            double distance_to_shared = patch_index == 0 ? (1.0 - u) : u;
            double ridge = 0.18 * std::sin(3.141592653589793 * y) *
                           std::exp(-3.0 * distance_to_shared);
            SurfaceFitSample s;
            s.uv = Eigen::Vector2d(u, v);
            s.position = Vector3d(x, y, ridge);
            s.weight = 1.0;
            samples.push_back(s);
        }
    }
    return samples;
}

static SharedSplineAssembly make_two_patch_flat_assembly() {
    BSplineSurface3D a = make_flat_patch(0.0, 1.0);
    BSplineSurface3D b = make_flat_patch(1.0, 2.0);
    for (int j = 0; j < 5; j++) {
        b.control_grid[0][j] = a.control_grid[4][4 - j];
    }
    return build_two_patch_shared_boundary_assembly(a, 100, 1, b, 200, 3, true);
}

static vector<GlobalBSplineRegionInput> make_two_patch_inputs(
    const vector<SurfaceFitSample>& samples0,
    const vector<SurfaceFitSample>& samples1) {
    vector<GlobalBSplineRegionInput> inputs;
    GlobalBSplineRegionInput in0;
    in0.patch_index = 0;
    in0.region_id = 100;
    in0.samples = samples0;
    inputs.push_back(in0);

    GlobalBSplineRegionInput in1;
    in1.patch_index = 1;
    in1.region_id = 200;
    in1.samples = samples1;
    inputs.push_back(in1);
    return inputs;
}

static void test_global_two_patch_optimization() {
    cout << "\n=== Global two-patch optimization ===" << endl;
    SharedSplineAssembly assembly = make_two_patch_flat_assembly();
    CHECK(assembly.valid, "shared assembly builds");

    vector<GlobalBSplineRegionInput> inputs =
        make_two_patch_inputs(make_patch_samples(0), make_patch_samples(1));

    GlobalBSplineRegionInput skipped;
    skipped.patch_index = 999;
    skipped.region_id = 999;
    inputs.push_back(skipped);

    GlobalBSplineOptimizationConfig cfg;
    cfg.fairness_weight = 1e-6;
    cfg.initial_weight = 1e-8;
    cfg.sample_u = 9;
    cfg.sample_v = 9;

    GlobalBSplineOptimizationResult result =
        optimize_global_bspline_control_points(assembly, inputs, cfg);

    cout << "  reason=" << result.reason << endl;
    cout << "  matrix rows/cols/nnz="
         << result.matrix_rows << " / "
         << result.matrix_cols << " / "
         << result.matrix_nonzeros << endl;
    cout << "  condition estimate="
         << result.normal_matrix_condition_estimate << endl;
    cout << "  global before mean/rms/max="
         << result.global_before.mean_error << " / "
         << result.global_before.rms_error << " / "
         << result.global_before.max_error << endl;
    cout << "  global after mean/rms/max="
         << result.global_after.mean_error << " / "
         << result.global_after.rms_error << " / "
         << result.global_after.max_error << endl;

    CHECK(result.valid, "global optimization succeeds");
    CHECK(result.total_control_points == 2 * 5 * 5 - 5,
          "shared boundary control points are single global ids");
    CHECK(result.variable_control_points == 2 * 3 * 3,
          "only interior control points are variables");
    CHECK(result.spline_patch_count == 2, "two spline patches are optimized");
    CHECK(result.matrix_rows > result.matrix_cols, "least-squares system is overdetermined");
    CHECK(result.matrix_nonzeros > 0, "sparse matrix has nonzeros");
    CHECK(std::isfinite(result.normal_matrix_condition_estimate) &&
              result.normal_matrix_condition_estimate > 0.0,
          "condition estimate is finite and positive");
    CHECK(result.global_after.rms_error < result.global_before.rms_error,
          "global RMS error decreases");
    CHECK(result.region_reports.size() == 2, "reports emitted for supported patches");
    CHECK(result.region_reports[0].after.rms_error < result.region_reports[0].before.rms_error,
          "region 0 RMS error decreases");
    CHECK(result.region_reports[1].after.rms_error < result.region_reports[1].before.rms_error,
          "region 1 RMS error decreases");
    CHECK(result.adjacency_report.watertight_across_shared_boundary,
          "sampled optimized mesh remains watertight across shared boundary");
    CHECK(result.adjacency_report.max_boundary_sample_difference < 1e-12,
          "shared boundary did not move inconsistently");

    CHECK(export_global_bspline_optimization_debug(
              "global_bspline_optimization", result),
          "debug exports succeed");
    CHECK(file_nonempty("global_bspline_optimization_summary.csv") &&
              file_nonempty("global_bspline_optimization_region_report.csv") &&
              file_nonempty("global_bspline_optimization_watertight_mesh.obj") &&
              file_nonempty("global_bspline_optimization_colored_patches.ply") &&
              file_nonempty("global_bspline_optimization_adjacency_report.csv"),
          "debug export files are non-empty");
}

static void test_movable_shared_boundary_optimization() {
    cout << "\n=== Movable shared boundary optimization ===" << endl;
    SharedSplineAssembly assembly = make_two_patch_flat_assembly();
    CHECK(assembly.valid, "shared assembly builds");

    vector<GlobalBSplineRegionInput> inputs =
        make_two_patch_inputs(
            make_patch_samples_with_shared_boundary_ridge(0),
            make_patch_samples_with_shared_boundary_ridge(1));

    GlobalBSplineOptimizationConfig fixed_cfg;
    fixed_cfg.fairness_weight = 1e-6;
    fixed_cfg.initial_weight = 1e-8;
    fixed_cfg.sample_u = 11;
    fixed_cfg.sample_v = 11;
    GlobalBSplineOptimizationResult fixed =
        optimize_global_bspline_control_points(assembly, inputs, fixed_cfg);
    CHECK(fixed.valid, "fixed-boundary baseline succeeds");

    GlobalBSplineOptimizationConfig movable_cfg = fixed_cfg;
    movable_cfg.optimize_shared_boundary_control_points = true;
    movable_cfg.boundary_fit_weight = 0.05;
    movable_cfg.boundary_fairness_weight = 1e-5;
    movable_cfg.boundary_fit_sample_count = 21;
    movable_cfg.boundary_drift_sample_count = 65;
    movable_cfg.max_boundary_drift = 0.5;
    movable_cfg.rollback_on_excessive_boundary_drift = true;

    GlobalBSplineOptimizationResult movable =
        optimize_global_bspline_control_points(assembly, inputs, movable_cfg);

    cout << "  fixed after mean/rms/max="
         << movable.fixed_boundary_global_after.mean_error << " / "
         << movable.fixed_boundary_global_after.rms_error << " / "
         << movable.fixed_boundary_global_after.max_error << endl;
    cout << "  movable after mean/rms/max="
         << movable.global_after.mean_error << " / "
         << movable.global_after.rms_error << " / "
         << movable.global_after.max_error << endl;
    cout << "  boundary drift mean/rms/max="
         << movable.boundary_drift.mean_error << " / "
         << movable.boundary_drift.rms_error << " / "
         << movable.boundary_drift.max_error << endl;
    cout << "  fixed/movable variables="
         << movable.fixed_boundary_variable_control_points << " / "
         << movable.variable_control_points << endl;

    CHECK(movable.valid, "movable-boundary optimization succeeds");
    CHECK(movable.movable_boundary_enabled, "movable boundary mode is recorded");
    CHECK(movable.has_fixed_boundary_comparison, "fixed-boundary comparison is recorded");
    CHECK(movable.variable_control_points ==
              movable.fixed_boundary_variable_control_points + 3,
          "three internal shared-boundary control points become variables");
    CHECK(movable.global_after.rms_error < fixed.global_after.rms_error,
          "movable boundary improves RMS error over fixed boundary");
    CHECK(movable.boundary_drift.max_error > 1e-5,
          "shared boundary drifts from its original position");
    CHECK(movable.boundary_drift.max_error < movable_cfg.max_boundary_drift,
          "boundary drift stays below configured maximum");
    CHECK(!movable.used_fixed_boundary_fallback,
          "fallback is not used when drift is acceptable");
    CHECK(movable.adjacency_report.watertight_across_shared_boundary,
          "movable shared-boundary result remains watertight");
    CHECK(movable.adjacency_report.max_boundary_sample_difference < 1e-12,
          "two adjacent patches still share identical boundary samples");

    CHECK(export_global_bspline_optimization_debug(
              "global_bspline_optimization_movable", movable),
          "movable debug exports succeed");
    CHECK(file_nonempty("global_bspline_optimization_movable_summary.csv") &&
              file_nonempty("global_bspline_optimization_movable_fixed_boundary_watertight_mesh.obj") &&
              file_nonempty("global_bspline_optimization_movable_movable_boundary_watertight_mesh.obj"),
          "movable comparison export files are non-empty");
}

static void test_movable_shared_boundary_rollback() {
    cout << "\n=== Movable shared boundary drift rollback ===" << endl;
    SharedSplineAssembly assembly = make_two_patch_flat_assembly();
    vector<GlobalBSplineRegionInput> inputs =
        make_two_patch_inputs(
            make_patch_samples_with_shared_boundary_ridge(0),
            make_patch_samples_with_shared_boundary_ridge(1));

    GlobalBSplineOptimizationConfig cfg;
    cfg.fairness_weight = 1e-6;
    cfg.initial_weight = 1e-8;
    cfg.optimize_shared_boundary_control_points = true;
    cfg.boundary_fit_weight = 0.05;
    cfg.boundary_fairness_weight = 1e-5;
    cfg.max_boundary_drift = 1e-8;
    cfg.rollback_on_excessive_boundary_drift = true;
    cfg.sample_u = 11;
    cfg.sample_v = 11;

    GlobalBSplineOptimizationResult result =
        optimize_global_bspline_control_points(assembly, inputs, cfg);

    cout << "  reason=" << result.reason << endl;
    cout << "  boundary drift after fallback mean/rms/max="
         << result.boundary_drift.mean_error << " / "
         << result.boundary_drift.rms_error << " / "
         << result.boundary_drift.max_error << endl;

    CHECK(result.valid, "rollback result is valid");
    CHECK(result.boundary_drift_exceeded, "excessive drift is detected");
    CHECK(result.used_fixed_boundary_fallback, "fixed-boundary fallback is used");
    CHECK(result.boundary_drift.max_error < 1e-12,
          "fallback restores fixed shared boundary");
    CHECK(result.adjacency_report.watertight_across_shared_boundary,
          "fallback mesh remains watertight");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Global B-spline Optimization Tests" << endl;
    cout << "========================================" << endl;

    test_global_two_patch_optimization();
    test_movable_shared_boundary_optimization();
    test_movable_shared_boundary_rollback();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
