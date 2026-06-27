#include "trimmed_bspline_pipeline.h"

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
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

static bool file_contains(const string& filename, const string& needle) {
    std::ifstream fin(filename);
    if (!fin.is_open()) return false;
    string content(
        (std::istreambuf_iterator<char>(fin)),
        std::istreambuf_iterator<char>());
    return content.find(needle) != string::npos;
}

static int vid(int nx, int x, int y) {
    return y * (nx + 1) + x;
}

static void build_quad_region_with_neighbors(
    int nx,
    int ny,
    MatrixXd& V,
    MatrixXi& F,
    vector<int>& labels) {
    vector<Vector3d> vertices;
    for (int y = 0; y <= ny; y++) {
        for (int x = 0; x <= nx; x++) {
            double z = 0.04 * std::sin(0.6 * x) * std::cos(0.5 * y);
            vertices.push_back(Vector3d((double)x, (double)y, z));
        }
    }

    vector<Eigen::Vector3i> faces;
    labels.clear();
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int v00 = vid(nx, x, y);
            int v10 = vid(nx, x + 1, y);
            int v01 = vid(nx, x, y + 1);
            int v11 = vid(nx, x + 1, y + 1);
            faces.push_back(Eigen::Vector3i(v00, v10, v11));
            labels.push_back(1);
            faces.push_back(Eigen::Vector3i(v00, v11, v01));
            labels.push_back(1);
        }
    }
    auto add_out = [&](int a, int b, const Vector3d& offset, int label) {
        int o = (int)vertices.size();
        vertices.push_back(0.5 * (vertices[a] + vertices[b]) + offset);
        faces.push_back(Eigen::Vector3i(b, a, o));
        labels.push_back(label);
    };
    for (int x = 0; x < nx; x++) add_out(vid(nx, x, 0), vid(nx, x + 1, 0), Vector3d(0, -0.35, 0), 10);
    for (int y = 0; y < ny; y++) add_out(vid(nx, nx, y), vid(nx, nx, y + 1), Vector3d(0.35, 0, 0), 11);
    for (int x = nx - 1; x >= 0; x--) add_out(vid(nx, x + 1, ny), vid(nx, x, ny), Vector3d(0, 0.35, 0), 12);
    for (int y = ny - 1; y >= 0; y--) add_out(vid(nx, 0, y + 1), vid(nx, 0, y), Vector3d(-0.35, 0, 0), 13);

    V.resize((int)vertices.size(), 3);
    for (int i = 0; i < (int)vertices.size(); i++) V.row(i) = vertices[i].transpose();
    F.resize((int)faces.size(), 3);
    for (int i = 0; i < (int)faces.size(); i++) F.row(i) = faces[i].transpose();
}

static void test_single_region_pipeline() {
    cout << "\n=== Single region trimmed B-spline pipeline ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_quad_region_with_neighbors(4, 3, V, F, labels);

    TrimmedBSplinePipelineConfig cfg;
    cfg.output_dir = "trimmed_bspline_output";
    cfg.control_count_u = 6;
    cfg.control_count_v = 6;
    cfg.surface_sample_u = 28;
    cfg.surface_sample_v = 28;
    cfg.extension_sample_u = 12;
    cfg.extension_sample_v = 12;

    TrimmedBSplinePipelineResult result =
        run_single_region_trimmed_bspline_pipeline(V, F, labels, 1, cfg);

    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "pipeline succeeds");
    CHECK(result.baseline_results.size() == 4, "four ablation baselines are produced");
    CHECK(result.metrics.valid, "metrics are valid");
    CHECK(result.metrics.original_region_rms_error >= 0.0, "original fitting RMS is reported");
    CHECK(result.metrics.boundary_max_error >= 0.0, "boundary max error is reported");
    CHECK(result.metrics.linear_system_condition_estimate > 0.0, "linear condition estimate is reported");
    CHECK(file_nonempty("trimmed_bspline_output/labels.json"), "labels.json is exported");
    CHECK(file_nonempty("trimmed_bspline_output/guiding_frame.obj"), "guiding_frame.obj is exported");
    CHECK(file_nonempty("trimmed_bspline_output/parameterized_region.obj"), "parameterized_region.obj is exported");
    CHECK(file_nonempty("trimmed_bspline_output/uv_trim_loops.obj"), "uv_trim_loops.obj is exported");
    CHECK(file_nonempty("trimmed_bspline_output/extended_uv_mesh.obj"), "extended_uv_mesh.obj is exported");
    CHECK(file_nonempty("trimmed_bspline_output/extended_3d_mesh.obj"), "extended_3d_mesh.obj is exported");
    CHECK(file_nonempty("trimmed_bspline_output/bspline_control_net.obj"), "bspline_control_net.obj is exported");
    CHECK(file_nonempty("trimmed_bspline_output/bspline_full_surface.obj"), "bspline_full_surface.obj is exported");
    CHECK(file_nonempty("trimmed_bspline_output/bspline_trimmed_surface.obj"), "bspline_trimmed_surface.obj is exported");
    CHECK(file_nonempty("trimmed_bspline_output/trimmed_bspline_asset.json"),
          "reusable trimmed B-spline asset JSON is exported");
    CHECK(file_contains("trimmed_bspline_output/trimmed_bspline_asset.json",
                        "\"boundary_ribbons\""),
          "reusable asset JSON contains boundary ribbons");
    CHECK(file_nonempty("trimmed_bspline_output/bspline_trimmed_surface_asset.obj"),
          "asset-derived trimmed surface mesh is exported");
    CHECK(file_nonempty("trimmed_bspline_output/trimmed_bspline_asset_mesh_validation.json"),
          "asset-derived mesh validation is exported");
    CHECK(file_nonempty("trimmed_bspline_output/region_topology_output_report.json"),
          "region topology output report is exported");
    CHECK(file_nonempty("trimmed_bspline_output/bspline_trimmed_surface_abc_preview.obj"),
          "ABC boundary-controlled preview mesh is exported");
    CHECK(file_nonempty("trimmed_bspline_output/trimmed_bspline_abc_preview_validation.json"),
          "ABC preview validation is exported");
    CHECK(file_nonempty("trimmed_bspline_output/abc_ribbon_preview_report.json"),
          "ABC ribbon-driven preview report is exported");
    CHECK(file_nonempty("trimmed_bspline_output/abc_preview_local_debug.json"),
          "ABC preview local debug report is exported");
    CHECK(file_nonempty("trimmed_bspline_output/abc_preview_near_trim_corner_faces.obj"),
          "ABC preview trim-corner face debug mesh is exported");
    CHECK(file_nonempty("trimmed_bspline_output/abc_preview_segment_transition_faces.obj"),
          "ABC preview segment-transition face debug mesh is exported");
    CHECK(file_nonempty("trimmed_bspline_output/abc_preview_trim_corner_markers_uv.obj"),
          "ABC preview trim corner UV markers are exported");
    CHECK(file_nonempty("trimmed_bspline_output/abc_boundary_ribbon_strips.obj"),
          "ABC boundary ribbon strips are exported");
    CHECK(file_nonempty("trimmed_bspline_output/abc_boundary_ribbon_report.json"),
          "ABC boundary ribbon report is exported");
    CHECK(file_nonempty("trimmed_bspline_output/abc_boundary_ribbon_surfaces.obj"),
          "ABC boundary ribbon B-spline surfaces are exported");
    CHECK(file_nonempty("trimmed_bspline_output/abc_boundary_ribbon_surfaces_report.json"),
          "ABC boundary ribbon surface report is exported");
    CHECK(file_nonempty("trimmed_bspline_output/metrics.json"), "metrics.json is exported");
    CHECK(file_nonempty("trimmed_bspline_output/pipeline.log"), "pipeline.log is exported");
}

static void test_quadlike_fallback_pipeline() {
    cout << "\n=== Quad-like fallback pipeline ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_quad_region_with_neighbors(4, 3, V, F, labels);
    for (int& label : labels) {
        if (label != 1) label = 10;
    }

    TrimmedBSplinePipelineConfig cfg;
    cfg.output_dir = "trimmed_bspline_output_fallback";
    cfg.control_count_u = 6;
    cfg.control_count_v = 6;
    cfg.surface_sample_u = 24;
    cfg.surface_sample_v = 24;
    cfg.extension_sample_u = 10;
    cfg.extension_sample_v = 10;

    TrimmedBSplinePipelineResult result =
        run_single_region_trimmed_bspline_pipeline(V, F, labels, 1, cfg);

    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "pipeline succeeds with quad-like fallback");
    CHECK(result.metrics.ambiguous, "fallback marks labeling as ambiguous");
    CHECK(file_nonempty("trimmed_bspline_output_fallback/labeling_quadlike_fallback_pca.obj"),
          "fallback PCA debug is exported");
    CHECK(file_nonempty("trimmed_bspline_output_fallback/bspline_trimmed_surface.obj"),
          "fallback trimmed surface is exported");
    CHECK(file_nonempty("trimmed_bspline_output_fallback/trimmed_bspline_asset.json"),
          "fallback reusable asset is exported");
}

static void test_fast_preview_pipeline() {
    cout << "\n=== Fast preview pipeline ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_quad_region_with_neighbors(4, 3, V, F, labels);

    TrimmedBSplinePipelineConfig cfg;
    cfg.output_dir = "trimmed_bspline_output_fast";
    cfg.control_count_u = 5;
    cfg.control_count_v = 5;
    cfg.surface_sample_u = 20;
    cfg.surface_sample_v = 20;
    cfg.extension_sample_u = 8;
    cfg.extension_sample_v = 8;
    cfg.export_debug_artifacts = false;
    cfg.run_ablation_baselines = false;
    cfg.estimate_condition_number = false;
    cfg.enable_smoothed_arap = false;
    cfg.enable_extension_fairness = false;

    TrimmedBSplinePipelineResult result =
        run_single_region_trimmed_bspline_pipeline(V, F, labels, 1, cfg);

    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "fast preview pipeline succeeds");
    CHECK(result.baseline_results.size() == 1, "fast preview skips ablation baselines");
    CHECK(result.metrics.linear_system_condition_estimate == 0.0,
          "fast preview skips condition estimate");
    CHECK(file_nonempty("trimmed_bspline_output_fast/bspline_trimmed_surface.obj"),
          "fast preview trimmed surface is exported");
    CHECK(file_nonempty("trimmed_bspline_output_fast/trimmed_bspline_asset.json"),
          "fast preview reusable asset is exported");
    CHECK(file_nonempty("trimmed_bspline_output_fast/metrics.json"),
          "fast preview metrics are exported");
}

static void test_reproduction_doc() {
    cout << "\n=== Reproduction document ===" << endl;
    CHECK(write_trimmed_bspline_reproduction_doc("TRIMMED_BSPLINE_REPRODUCTION.md"),
          "reproduction document is written");
    CHECK(file_nonempty("TRIMMED_BSPLINE_REPRODUCTION.md"),
          "reproduction document is non-empty");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Trimmed B-spline Pipeline Tests" << endl;
    cout << "========================================" << endl;

    test_single_region_pipeline();
    test_quadlike_fallback_pipeline();
    test_fast_preview_pipeline();
    test_reproduction_doc();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;
    return g_fail;
}
