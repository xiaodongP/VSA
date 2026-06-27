#include "constrained_arap_parameterization.h"
#include "bezier_guiding_frame.h"
#include "trimmed_labeling.h"
#include "trimmed_region_input.h"

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::cerr;
using std::cout;
using std::endl;
using std::map;
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

static int grid_vid(int nx, int x, int y) {
    return y * (nx + 1) + x;
}

static void build_rect_grid_with_boundary_neighbors(
    int nx,
    int ny,
    bool curved,
    bool split_south,
    MatrixXd& V,
    MatrixXi& F,
    vector<int>& labels) {
    vector<Vector3d> vertices;
    vertices.reserve((nx + 1) * (ny + 1) + 2 * nx + 2 * ny);
    for (int y = 0; y <= ny; y++) {
        for (int x = 0; x <= nx; x++) {
            double z = curved ? 0.08 * std::sin(0.7 * x) * std::cos(0.6 * y) : 0.0;
            vertices.push_back(Vector3d((double)x, (double)y, z));
        }
    }

    vector<Eigen::Vector3i> faces;
    labels.clear();
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int v00 = grid_vid(nx, x, y);
            int v10 = grid_vid(nx, x + 1, y);
            int v01 = grid_vid(nx, x, y + 1);
            int v11 = grid_vid(nx, x + 1, y + 1);
            faces.push_back(Eigen::Vector3i(v00, v10, v11));
            labels.push_back(1);
            faces.push_back(Eigen::Vector3i(v00, v11, v01));
            labels.push_back(1);
        }
    }

    auto add_outside = [&](int a, int b, const Vector3d& p, int label) {
        int o = (int)vertices.size();
        vertices.push_back(p);
        faces.push_back(Eigen::Vector3i(b, a, o));
        labels.push_back(label);
    };

    for (int x = 0; x < nx; x++) {
        int a = grid_vid(nx, x, 0);
        int b = grid_vid(nx, x + 1, 0);
        int label = split_south && x == 0 ? 14 : 10;
        add_outside(a, b, 0.5 * (vertices[a] + vertices[b]) + Vector3d(0, -0.35, 0), label);
    }
    for (int y = 0; y < ny; y++) {
        int a = grid_vid(nx, nx, y);
        int b = grid_vid(nx, nx, y + 1);
        add_outside(a, b, 0.5 * (vertices[a] + vertices[b]) + Vector3d(0.35, 0, 0), 11);
    }
    for (int x = nx - 1; x >= 0; x--) {
        int a = grid_vid(nx, x + 1, ny);
        int b = grid_vid(nx, x, ny);
        add_outside(a, b, 0.5 * (vertices[a] + vertices[b]) + Vector3d(0, 0.35, 0), 12);
    }
    for (int y = ny - 1; y >= 0; y--) {
        int a = grid_vid(nx, 0, y + 1);
        int b = grid_vid(nx, 0, y);
        add_outside(a, b, 0.5 * (vertices[a] + vertices[b]) + Vector3d(-0.35, 0, 0), 13);
    }

    V.resize((int)vertices.size(), 3);
    for (int i = 0; i < (int)vertices.size(); i++) V.row(i) = vertices[i].transpose();
    F.resize((int)faces.size(), 3);
    for (int i = 0; i < (int)faces.size(); i++) F.row(i) = faces[i].transpose();
}

static AutomaticLabelingResult make_manual_labeling(
    const vector<vector<int>>& side_segments) {
    AutomaticLabelingResult labeling;
    labeling.valid = true;
    labeling.reason = "manual";
    labeling.final_label_count = (int)side_segments.size();
    labeling.final_corner_count = (int)side_segments.size();
    for (int i = 0; i < (int)side_segments.size(); i++) {
        AbstractSide side;
        side.side_index = i;
        side.label_group_ids.push_back(i);
        side.segment_ids = side_segments[i];
        labeling.abstract_sides.push_back(side);
    }
    return labeling;
}

static BezierGuidingFrameResult build_frame(
    const MatrixXd& V,
    const MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling,
    const string& dir) {
    BezierGuidingFrameConfig cfg;
    cfg.export_debug = false;
    cfg.debug_output_dir = dir;
    cfg.use_virtual_corners = false;
    cfg.max_degree = 3;
    return build_bezier_guiding_frame(V, F, input, labeling, cfg);
}

static ConstrainedArapConfig arap_config(const string& prefix) {
    ConstrainedArapConfig cfg;
    cfg.debug_output_prefix = prefix;
    cfg.export_debug = true;
    cfg.max_iterations = 8;
    cfg.fail_on_flips = true;
    cfg.use_labeling_orientation = false;
    return cfg;
}

static bool uv_close(const MatrixXd& UV, int vid, double u, double v, double eps = 1e-6) {
    return std::abs(UV(vid, 0) - u) < eps && std::abs(UV(vid, 1) - v) < eps;
}

static void test_flat_patch_constraints() {
    cout << "\n=== Flat patch constraints ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_rect_grid_with_boundary_neighbors(4, 2, false, false, V, F, labels);
    BoundarySegmentationResult input =
        build_trimmed_region_input(V, F, labels, 1);
    AutomaticLabelingResult labeling = make_manual_labeling({{0}, {1}, {2}, {3}});
    BezierGuidingFrameResult frame = build_frame(V, F, input, labeling, "unused");

    ConstrainedArapResult result =
        parameterize_constrained_arap_mvp(
            V, F, input, labeling, frame, arap_config("constrained_arap_flat"));

    cout << "  reason=" << result.reason << endl;
    cout << "  width=" << result.width << " height=" << result.height << endl;
    CHECK(input.valid, "boundary input is valid");
    CHECK(frame.valid, "guiding frame is valid");
    CHECK(result.valid, "constrained ARAP succeeds");
    CHECK(std::abs(result.width - 4.0) < 1e-5, "free aspect width follows frame length");
    CHECK(std::abs(result.height - 2.0) < 1e-5, "free aspect height follows frame length");
    CHECK(uv_close(result.UV, grid_vid(4, 0, 0), 0.0, 0.0), "south-west corner fixed");
    CHECK(uv_close(result.UV, grid_vid(4, 4, 0), 4.0, 0.0), "south-east corner fixed");
    CHECK(uv_close(result.UV, grid_vid(4, 4, 2), 4.0, 2.0), "north-east corner fixed");
    CHECK(uv_close(result.UV, grid_vid(4, 0, 2), 0.0, 2.0), "north-west corner fixed");
    CHECK(result.flipped_triangle_count == 0, "no flipped triangles");
    CHECK(result.max_conformal_distortion < 1.05, "flat rectangle is nearly conformal");
    CHECK(file_nonempty("constrained_arap_flat_triangle_stats.csv"),
          "triangle stats CSV is exported");
}

static void test_curved_patch_metrics() {
    cout << "\n=== Curved patch metrics ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_rect_grid_with_boundary_neighbors(4, 3, true, false, V, F, labels);
    BoundarySegmentationResult input =
        build_trimmed_region_input(V, F, labels, 1);
    AutomaticLabelingResult labeling = make_manual_labeling({{0}, {1}, {2}, {3}});
    BezierGuidingFrameResult frame = build_frame(V, F, input, labeling, "unused");

    ConstrainedArapResult result =
        parameterize_constrained_arap_mvp(
            V, F, input, labeling, frame, arap_config("constrained_arap_curved"));

    cout << "  reason=" << result.reason << endl;
    cout << "  mean_arap_residual=" << result.mean_arap_residual
         << " max_conformal=" << result.max_conformal_distortion << endl;
    CHECK(result.valid, "curved patch parameterization succeeds");
    CHECK(result.triangle_stats.size() == (size_t)input.region.face_ids.size(),
          "one stat per region face");
    CHECK(result.flipped_triangle_count == 0, "curved patch has no flips");
    CHECK(result.mean_arap_residual >= 0.0, "ARAP residual is finite");
    CHECK(result.max_area_ratio > 0.0, "area ratio is reported");
}

static void test_multisegment_side_ratio() {
    cout << "\n=== Multi-segment side ratio ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_rect_grid_with_boundary_neighbors(4, 2, false, true, V, F, labels);
    BoundarySegmentationResult input =
        build_trimmed_region_input(V, F, labels, 1);
    AutomaticLabelingResult labeling = make_manual_labeling({{0, 1}, {2}, {3}, {4}});
    BezierGuidingFrameResult frame = build_frame(V, F, input, labeling, "unused");

    ConstrainedArapResult result =
        parameterize_constrained_arap_mvp(
            V, F, input, labeling, frame, arap_config("constrained_arap_multisegment"));

    cout << "  reason=" << result.reason << endl;
    CHECK(input.perimeter_segments.size() == 5, "south side is split into two segments");
    CHECK(result.valid, "multi-segment side parameterization succeeds");
    CHECK(std::abs(result.UV(grid_vid(4, 1, 0), 0) - 1.0) < 1e-6,
          "first south segment keeps 1:3 length proportion");
    CHECK(std::abs(result.UV(grid_vid(4, 1, 0), 1)) < 1e-6,
          "south split vertex remains on v=0");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Constrained ARAP MVP Tests" << endl;
    cout << "========================================" << endl;

    test_flat_patch_constraints();
    test_curved_patch_metrics();
    test_multisegment_side_ratio();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
