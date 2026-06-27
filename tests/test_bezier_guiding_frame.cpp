#include "bezier_guiding_frame.h"
#include "trimmed_labeling.h"
#include "trimmed_region_input.h"

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector2d;
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

static void build_polygon_fan_region(
    const vector<Vector2d>& polygon,
    const vector<int>& adjacent_region_per_edge,
    int region_id,
    MatrixXd& V,
    MatrixXi& F,
    vector<int>& labels) {
    const int n = (int)polygon.size();
    int outside_count = 0;
    for (int adj : adjacent_region_per_edge) {
        if (adj >= 0) outside_count++;
    }

    V.resize(n + 1 + outside_count, 3);
    Vector2d center = Vector2d::Zero();
    for (const Vector2d& p : polygon) center += p;
    center /= (double)n;

    for (int i = 0; i < n; i++) {
        V.row(i) << polygon[i].x(), polygon[i].y(), 0.0;
    }
    V.row(n) << center.x(), center.y(), 0.0;

    F.resize(n + outside_count, 3);
    labels.clear();
    labels.reserve(F.rows());
    for (int i = 0; i < n; i++) {
        F.row(i) << n, i, (i + 1) % n;
        labels.push_back(region_id);
    }

    int outside_vertex = n + 1;
    int outside_face = n;
    for (int i = 0; i < n; i++) {
        int adj = adjacent_region_per_edge[i];
        if (adj < 0) continue;
        Vector2d a = polygon[i];
        Vector2d b = polygon[(i + 1) % n];
        Vector2d e = b - a;
        Vector2d outward(e.y(), -e.x());
        if (outward.norm() > 1e-14) outward.normalize();
        Vector2d p = 0.5 * (a + b) + 0.25 * outward;
        V.row(outside_vertex) << p.x(), p.y(), 0.0;
        F.row(outside_face) << (i + 1) % n, i, outside_vertex;
        labels.push_back(adj);
        outside_vertex++;
        outside_face++;
    }
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

static BezierGuidingFrameConfig frame_config(const string& dir) {
    BezierGuidingFrameConfig config;
    config.debug_output_dir = dir;
    config.export_debug = true;
    config.initial_degree = 1;
    config.max_degree = 4;
    config.fitting_tolerance = 1e-5;
    config.lambda_frame = 1e-5;
    return config;
}

static void check_shared_endpoints(const BezierGuidingFrameResult& result) {
    for (int i = 0; i < (int)result.curves.size(); i++) {
        const BezierCurve3D& c = result.curves[i];
        int next = (i + 1) % result.curves.size();
        CHECK((c.control_points.front() - result.shared_corners[i]).norm() < 1e-10,
              "curve start shares corner control point");
        CHECK((c.control_points.back() - result.shared_corners[next]).norm() < 1e-10,
              "curve end shares next corner control point");
    }
}

static void test_complete_rectangle() {
    cout << "\n=== Complete rectangle ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_polygon_fan_region(
        {Vector2d(0, 0), Vector2d(3, 0), Vector2d(3, 2), Vector2d(0, 2)},
        {10, 11, 12, 13},
        1, V, F, labels);
    BoundarySegmentationResult input =
        build_trimmed_region_input(V, F, labels, 1);
    AutomaticLabelingConfig labeling_config;
    labeling_config.export_debug = false;
    labeling_config.label_energy_threshold = 10.0;
    AutomaticLabelingResult labeling =
        run_automatic_labeling(V, F, input, labeling_config);

    BezierGuidingFrameResult frame =
        build_bezier_guiding_frame(
            V, F, input, labeling, frame_config("bezier_frame_debug_rectangle"));

    cout << "  reason=" << frame.reason << endl;
    CHECK(frame.valid, "rectangle frame is valid");
    CHECK(frame.curves.size() == 4, "rectangle has four frame curves");
    CHECK(frame.max_error < 1e-6, "straight rectangle fits to near-zero error");
    check_shared_endpoints(frame);
    CHECK(file_nonempty("bezier_frame_debug_rectangle/iteration_report.csv"),
          "iteration report is exported");
}

static void test_missing_corner_without_virtuals() {
    cout << "\n=== Missing virtual corner ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_polygon_fan_region(
        {Vector2d(0, 0), Vector2d(2, 0), Vector2d(2, 1), Vector2d(0, 1)},
        {20, 21, 22, 23},
        2, V, F, labels);
    BoundarySegmentationResult input =
        build_trimmed_region_input(V, F, labels, 2);
    AutomaticLabelingResult labeling =
        make_manual_labeling({{0}, {1}, {2}, {3}});
    BezierGuidingFrameConfig config = frame_config("bezier_frame_debug_missing_corner");
    config.use_virtual_corners = true;

    BezierGuidingFrameResult frame =
        build_bezier_guiding_frame(V, F, input, labeling, config);

    cout << "  reason=" << frame.reason << endl;
    CHECK(frame.valid, "frame works without optional virtual corners");
    CHECK(frame.shared_corners.size() == 4, "four shared corners are inferred");
    check_shared_endpoints(frame);
}

static void test_two_parallel_labels() {
    cout << "\n=== Two near-parallel labels ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_polygon_fan_region(
        {Vector2d(0, 0), Vector2d(4, 0), Vector2d(4, 1), Vector2d(0, 1)},
        {30, 31, 32, 33},
        3, V, F, labels);
    BoundarySegmentationResult input =
        build_trimmed_region_input(V, F, labels, 3);
    AutomaticLabelingResult labeling = make_manual_labeling({{0}, {2}});

    BezierGuidingFrameResult frame =
        build_bezier_guiding_frame(
            V, F, input, labeling, frame_config("bezier_frame_debug_parallel"));

    cout << "  reason=" << frame.reason << endl;
    CHECK(frame.valid, "two-label frame is valid");
    CHECK(frame.curves.size() == 2, "two labels produce two frame curves");
    check_shared_endpoints(frame);
}

static void test_multisegment_label() {
    cout << "\n=== Multi-segment label ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_polygon_fan_region(
        {Vector2d(0, 0), Vector2d(1, 0), Vector2d(2, 0),
         Vector2d(2, 1), Vector2d(0, 1)},
        {40, 41, 42, 43, 44},
        4, V, F, labels);
    BoundarySegmentationResult input =
        build_trimmed_region_input(V, F, labels, 4);
    AutomaticLabelingResult labeling =
        make_manual_labeling({{0, 1}, {2}, {3}, {4}});

    BezierGuidingFrameResult frame =
        build_bezier_guiding_frame(
            V, F, input, labeling, frame_config("bezier_frame_debug_multisegment"));

    cout << "  reason=" << frame.reason << endl;
    CHECK(frame.valid, "multi-segment label frame is valid");
    CHECK(frame.labeled_samples[0].size() >= 3,
          "first label contains samples from multiple segments");
    CHECK(frame.curves.size() == 4, "multi-segment case keeps four curves");
}

static void test_noisy_boundary() {
    cout << "\n=== Noisy boundary ===" << endl;
    vector<Vector2d> polygon;
    vector<int> adj;
    for (int i = 0; i <= 4; i++) {
        polygon.push_back(Vector2d((double)i, 0.02 * std::sin((double)i)));
        adj.push_back(50);
    }
    for (int i = 1; i <= 3; i++) {
        polygon.push_back(Vector2d(4.0 + 0.02 * std::sin((double)i), (double)i));
        adj.push_back(51);
    }
    for (int i = 3; i >= 0; i--) {
        polygon.push_back(Vector2d((double)i, 3.0 + 0.02 * std::cos((double)i)));
        adj.push_back(52);
    }
    for (int i = 2; i >= 1; i--) {
        polygon.push_back(Vector2d(0.02 * std::cos((double)i), (double)i));
        adj.push_back(53);
    }
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_polygon_fan_region(polygon, adj, 5, V, F, labels);
    BoundarySegmentationResult input =
        build_trimmed_region_input(V, F, labels, 5);
    AutomaticLabelingResult labeling =
        make_manual_labeling({{0}, {1}, {2}, {3}});
    BezierGuidingFrameConfig config = frame_config("bezier_frame_debug_noisy");
    config.fitting_tolerance = 1e-3;
    config.max_degree = 5;

    BezierGuidingFrameResult frame =
        build_bezier_guiding_frame(V, F, input, labeling, config);

    cout << "  reason=" << frame.reason << endl;
    CHECK(frame.valid, "noisy boundary frame is valid");
    CHECK(frame.final_degree >= 2, "noisy boundary elevates degree");
    CHECK(frame.max_error < 0.08, "noisy boundary fit error is bounded");
}

static void test_missing_segment_hermite_initialization() {
    cout << "\n=== Missing segment Hermite initialization ===" << endl;
    vector<vector<Vector3d>> samples(3);
    samples[0] = {Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(2, 0, 0)};
    samples[1] = {};
    samples[2] = {Vector3d(2, 1, 0), Vector3d(1, 1.1, 0), Vector3d(0, 1, 0)};
    vector<Vector3d> corners = {
        Vector3d(0, 0, 0),
        Vector3d(2, 0, 0),
        Vector3d(2, 1, 0)};

    BezierGuidingFrameConfig config = frame_config("bezier_frame_debug_missing_segment");
    config.max_degree = 3;
    config.fitting_tolerance = 1e-6;

    BezierGuidingFrameResult frame =
        build_bezier_guiding_frame_from_samples(samples, corners, config);

    cout << "  reason=" << frame.reason << endl;
    CHECK(frame.valid, "direct sample frame with a missing segment is valid");
    CHECK(frame.curves.size() == 3, "three labels produce three frame curves");
    CHECK(frame.final_degree >= 3, "missing segment forces cubic handling");
    CHECK(frame.curves[1].degree == frame.final_degree,
          "missing segment has the final elevated degree");
    check_shared_endpoints(frame);
}

int main() {
    cout << "========================================" << endl;
    cout << "  Bezier Guiding Frame Tests" << endl;
    cout << "========================================" << endl;

    test_complete_rectangle();
    test_missing_corner_without_virtuals();
    test_two_parallel_labels();
    test_multisegment_label();
    test_noisy_boundary();
    test_missing_segment_hermite_initialization();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
