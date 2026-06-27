#include "tensor_product_bspline_fitter.h"

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
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

static Eigen::Vector3d smooth_surface(double u, double v, double z_scale) {
    const double pi = 3.14159265358979323846;
    return Eigen::Vector3d(
        u,
        v,
        z_scale * std::sin(pi * u) * std::cos(pi * v) + 0.08 * u * u - 0.04 * v);
}

static TensorProductBSplineFitInput make_grid_input(
    int n,
    TensorProductFitBaseline baseline) {
    TensorProductBSplineFitInput input;
    input.baseline = baseline;
    input.label = to_string(baseline);
    input.UV.resize(n * n, 2);
    input.positions.resize(n * n, 3);
    input.original_vertex_mask.assign(n * n, false);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            int id = j * n + i;
            double u = (double)i / (double)(n - 1);
            double v = (double)j / (double)(n - 1);
            input.UV.row(id) << u, v;
            bool original = (u <= 0.62 || v <= 0.62);
            input.original_vertex_mask[id] = original;
            double scale = baseline == TensorProductFitBaseline::ArapOnly ? 0.11 : 0.16;
            Eigen::Vector3d p = smooth_surface(u, v, scale);
            if (baseline == TensorProductFitBaseline::ArapOnly && !original) {
                p.z() = 0.02 * (u + v);
            }
            if (baseline == TensorProductFitBaseline::LabelingExtension && !original) {
                const double pi = 3.14159265358979323846;
                p.z() += 0.015 * std::sin(2.0 * pi * u) * std::sin(2.0 * pi * v);
            }
            input.positions.row(id) = p.transpose();
        }
    }

    input.F.resize(2 * (n - 1) * (n - 1), 3);
    input.original_face_mask.clear();
    int f = 0;
    for (int j = 0; j + 1 < n; j++) {
        for (int i = 0; i + 1 < n; i++) {
            int a = j * n + i;
            int b = j * n + i + 1;
            int c = (j + 1) * n + i + 1;
            int d = (j + 1) * n + i;
            input.F.row(f++) << a, b, c;
            input.original_face_mask.push_back(
                input.original_vertex_mask[a] && input.original_vertex_mask[b] && input.original_vertex_mask[c]);
            input.F.row(f++) << a, c, d;
            input.original_face_mask.push_back(
                input.original_vertex_mask[a] && input.original_vertex_mask[c] && input.original_vertex_mask[d]);
        }
    }
    return input;
}

static TensorProductBSplineFitConfig cfg(
    int controls,
    const string& prefix,
    bool trimmed_only) {
    TensorProductBSplineFitConfig config;
    config.control_count_u = controls;
    config.control_count_v = controls;
    config.control_net_fairness_weight = 1e-7;
    config.normal_equation_regularization = 1e-10;
    config.weak_support_threshold = 5e-3;
    config.fit_original_vertices_only = trimmed_only;
    config.export_debug = true;
    config.debug_prefix = prefix;
    return config;
}

static void test_control_grid_sweep() {
    cout << "\n=== Control grid sweep ===" << endl;
    double previous_rms = 1e100;
    for (int controls : {4, 6, 8, 10}) {
        TensorProductBSplineFitInput input =
            make_grid_input(17, TensorProductFitBaseline::ExtensionOnly);
        TensorProductBSplineFitResult result =
            fit_tensor_product_cubic_bspline_surface(
                input, cfg(controls, "tensor_fit_grid_" + std::to_string(controls), false));
        cout << "  grid=" << controls << "x" << controls
             << " rms=" << result.original_region_error.rms_error
             << " weak=" << result.weak_support.weak_control_point_count
             << " cond=" << result.condition_estimate << endl;
        CHECK(result.valid, "fit succeeds for configured control grid");
        CHECK(std::isfinite(result.original_region_error.rms_error), "original-region RMS is finite");
        CHECK(std::isfinite(result.extension_region_smoothness), "extension smoothness is finite");
        CHECK(result.condition_estimate > 0.0, "condition estimate is reported");
        if (controls == 10) {
            CHECK(result.original_region_error.rms_error <= previous_rms + 2e-3,
                  "larger control grid does not significantly worsen RMS");
        }
        previous_rms = result.original_region_error.rms_error;
    }
}

static void test_baselines_and_weak_support() {
    cout << "\n=== Baseline comparison ===" << endl;
    TensorProductBSplineFitInput trimmed =
        make_grid_input(17, TensorProductFitBaseline::TrimmedOnly);
    TensorProductBSplineFitInput arap =
        make_grid_input(17, TensorProductFitBaseline::ArapOnly);
    TensorProductBSplineFitInput extension =
        make_grid_input(17, TensorProductFitBaseline::ExtensionOnly);
    TensorProductBSplineFitInput labeling =
        make_grid_input(17, TensorProductFitBaseline::LabelingExtension);

    TensorProductBSplineFitResult a = fit_tensor_product_cubic_bspline_surface(
        trimmed, cfg(8, "tensor_fit_baseline_A_trimmed", true));
    TensorProductBSplineFitResult b = fit_tensor_product_cubic_bspline_surface(
        arap, cfg(8, "tensor_fit_baseline_B_arap", false));
    TensorProductBSplineFitResult c = fit_tensor_product_cubic_bspline_surface(
        extension, cfg(8, "tensor_fit_baseline_C_extension", false));
    TensorProductBSplineFitResult d = fit_tensor_product_cubic_bspline_surface(
        labeling, cfg(8, "tensor_fit_baseline_D_labeling_extension", false));

    CHECK(a.valid, "baseline A trimmed-only fitting succeeds");
    CHECK(b.valid, "baseline B ordinary ARAP plus fitting succeeds");
    CHECK(c.valid, "baseline C extension plus fitting succeeds");
    CHECK(d.valid, "baseline D labeling plus extension plus fitting succeeds");
    CHECK(a.weak_support.weak_control_point_count >= c.weak_support.weak_control_point_count,
          "extension reduces weak control point count");
    CHECK(c.original_region_error.rms_error < 0.03, "extension baseline fits original region");
    CHECK(d.extension_region_smoothness >= 0.0, "labeling extension smoothness is reported");
    CHECK(file_nonempty("tensor_fit_baseline_C_extension_sampled_base_mesh.obj"),
          "sampled base mesh is exported");
    CHECK(file_nonempty("tensor_fit_baseline_C_extension_control_points.csv"),
          "control points are exported");
    CHECK(file_nonempty("tensor_fit_baseline_C_extension_knots.csv"),
          "knots are exported");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Tensor-Product B-spline Fitter Tests" << endl;
    cout << "========================================" << endl;

    test_control_grid_sweep();
    test_baselines_and_weak_support();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;
    return g_fail;
}
