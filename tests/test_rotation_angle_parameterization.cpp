#include "rotation_angle_parameterization.h"
#include "bezier_guiding_frame.h"
#include "trimmed_labeling.h"
#include "trimmed_region_input.h"

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
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

static int vid(int nx, int x, int y) {
    return y * (nx + 1) + x;
}

static void build_rect_grid_with_neighbors(
    int nx,
    int ny,
    bool curved,
    MatrixXd& V,
    MatrixXi& F,
    vector<int>& labels) {
    vector<Vector3d> vertices;
    for (int y = 0; y <= ny; y++) {
        for (int x = 0; x <= nx; x++) {
            double z = curved ? 0.06 * std::sin(0.8 * x) * std::cos(0.4 * y) : 0.0;
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

static void build_hole_grid(MatrixXd& V, MatrixXi& F, vector<int>& labels) {
    int nx = 3, ny = 3;
    V.resize((nx + 1) * (ny + 1), 3);
    for (int y = 0; y <= ny; y++) {
        for (int x = 0; x <= nx; x++) V.row(vid(nx, x, y)) << x, y, 0.0;
    }
    F.resize(nx * ny * 2, 3);
    labels.assign(F.rows(), 1);
    int f = 0;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int v00 = vid(nx, x, y), v10 = vid(nx, x + 1, y);
            int v01 = vid(nx, x, y + 1), v11 = vid(nx, x + 1, y + 1);
            F.row(f++) << v00, v10, v11;
            F.row(f++) << v00, v11, v01;
        }
    }
    int center = 2 * (1 * nx + 1);
    labels[center] = 0;
    labels[center + 1] = 0;
}

static AutomaticLabelingResult manual_labeling() {
    AutomaticLabelingResult labeling;
    labeling.valid = true;
    labeling.reason = "manual";
    labeling.final_label_count = 4;
    labeling.final_corner_count = 4;
    for (int i = 0; i < 4; i++) {
        AbstractSide side;
        side.side_index = i;
        side.label_group_ids.push_back(i);
        side.segment_ids.push_back(i);
        labeling.abstract_sides.push_back(side);
    }
    return labeling;
}

static BezierGuidingFrameResult make_frame(
    const MatrixXd& V,
    const MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling) {
    BezierGuidingFrameConfig cfg;
    cfg.export_debug = false;
    cfg.use_virtual_corners = false;
    cfg.max_degree = 3;
    return build_bezier_guiding_frame(V, F, input, labeling, cfg);
}

static void test_disk_rotation_and_stitching(bool curved) {
    cout << "\n=== Disk " << (curved ? "curved" : "flat") << " rotation + KKT ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_rect_grid_with_neighbors(4, 3, curved, V, F, labels);
    BoundarySegmentationResult input = build_trimmed_region_input(V, F, labels, 1);
    AutomaticLabelingResult labeling = manual_labeling();
    BezierGuidingFrameResult frame = make_frame(V, F, input, labeling);

    RotationAngleInitializationConfig rcfg;
    rcfg.debug_prefix = curved ? "rotation_angle_curved" : "rotation_angle_flat";
    RotationAngleInitializationResult rot =
        initialize_rotation_angles_section421(V, F, input, labeling, rcfg);

    KktGlobalStitchingConfig kcfg;
    kcfg.debug_prefix = curved ? "kkt_stitching_curved" : "kkt_stitching_flat";
    kcfg.enable_smoothed_arap = curved;
    kcfg.lambda_smooth = curved ? 1e-4 : 0.0;
    KktGlobalStitchingResult uv =
        parameterize_kkt_global_stitching(V, F, input, labeling, frame, rot, kcfg);

    cout << "  rot reason=" << rot.reason << endl;
    cout << "  kkt reason=" << uv.reason << endl;
    cout << "  label_err=" << rot.max_label_orientation_error
         << " flips=" << uv.flipped_triangle_count
         << " mean_arap=" << uv.mean_arap_residual << endl;
    CHECK(input.valid, "input segmentation is valid");
    CHECK(frame.valid, "guiding frame is valid");
    CHECK(rot.valid, "rotation-angle initialization succeeds");
    CHECK(rot.max_fan_closure_error < 1e-3, "fan closure error is small");
    CHECK(rot.max_noncontractible_loop_error == 0.0, "disk has no non-contractible loop error");
    CHECK(rot.max_label_orientation_error < 5e-3, "label edge orientation error is small");
    CHECK(uv.valid, "KKT stitching succeeds");
    CHECK(uv.max_label_coordinate_error < 1e-7, "label coordinate constraints are satisfied");
    CHECK(uv.max_length_ratio_error < 1e-7, "length-ratio constraint residual is small");
    CHECK(uv.flipped_triangle_count == 0, "stitched UV has no flips");
    CHECK(uv.mean_arap_residual >= 0.0, "mean ARAP residual is reported");
    CHECK(file_nonempty(kcfg.debug_prefix + "_after_distortion.csv"),
          "after distortion heatmap is exported");
    CHECK(file_nonempty(kcfg.debug_prefix + "_before_distortion.csv"),
          "before distortion heatmap is exported");
    CHECK(file_nonempty(kcfg.debug_prefix + "_kkt_diagnostics.csv"),
          "KKT diagnostics are exported");
}

static void test_hole_unsupported() {
    cout << "\n=== Hole unsupported ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_hole_grid(V, F, labels);
    BoundarySegmentationResult input = build_trimmed_region_input(V, F, labels, 1);
    AutomaticLabelingResult labeling = manual_labeling();
    RotationAngleInitializationResult rot =
        initialize_rotation_angles_section421(V, F, input, labeling);

    cout << "  reason=" << rot.reason << endl;
    CHECK(input.valid, "annulus input is valid at boundary-input stage");
    CHECK(!rot.valid, "multiply-connected rotation init is rejected");
    CHECK(rot.unsupported_multiply_connected, "unsupported flag is set");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Rotation-Angle + KKT Stitching Tests" << endl;
    cout << "========================================" << endl;

    test_disk_rotation_and_stitching(false);
    test_disk_rotation_and_stitching(true);
    test_hole_unsupported();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;
    return g_fail;
}
