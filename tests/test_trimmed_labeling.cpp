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
        Vector2d p = 0.5 * (a + b) + 0.35 * outward;
        V.row(outside_vertex) << p.x(), p.y(), 0.0;
        F.row(outside_face) << (i + 1) % n, i, outside_vertex;
        labels.push_back(adj);
        outside_vertex++;
        outside_face++;
    }
}

static AutomaticLabelingConfig test_config(const string& dir) {
    AutomaticLabelingConfig config;
    config.export_debug = true;
    config.debug_output_dir = dir;
    config.label_energy_threshold = 10.0;
    return config;
}

static void test_rectangle_labeling() {
    cout << "\n=== Rectangle automatic labeling ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_polygon_fan_region(
        {Vector2d(0, 0), Vector2d(2, 0), Vector2d(2, 1), Vector2d(0, 1)},
        {10, 11, 12, 13},
        1, V, F, labels);

    BoundarySegmentationResult input =
        build_trimmed_region_input(V, F, labels, 1);
    AutomaticLabelingConfig config = test_config("labeling_debug_rectangle");
    config.enable_vsa_orientation = true;
    config.use_supplied_orientation_axes = true;
    config.orientation_axis_u = Eigen::Vector3d::UnitX();
    config.orientation_axis_v = Eigen::Vector3d::UnitY();
    AutomaticLabelingResult result =
        run_automatic_labeling(V, F, input, config);

    cout << "  reason=" << result.reason << endl;
    CHECK(input.valid, "segmentation input is valid");
    CHECK(result.valid, "rectangle labeling is admissible");
    CHECK(result.final_label_count == 4, "rectangle keeps four labels");
    CHECK(result.final_corner_count == 4, "rectangle keeps four strong corners");
    CHECK(result.abstract_sides.size() == 4, "four abstract sides are produced");
    CHECK(result.orientation.valid, "orientation resolver succeeds");
    CHECK(result.orientation.confidence > 0.9, "orientation confidence is high");
    CHECK(file_nonempty("labeling_debug_rectangle/segments.obj"),
          "segments debug OBJ exists");
    CHECK(file_nonempty("labeling_debug_rectangle/candidate_metrics.json"),
          "candidate metrics JSON exists");
    CHECK(file_nonempty("labeling_debug_rectangle/final_labels.json"),
          "final labels JSON exists");
}

static void test_pentagon_reduction() {
    cout << "\n=== Pentagon reduction ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    vector<Vector2d> polygon;
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < 5; i++) {
        double a = 2.0 * pi * (double)i / 5.0;
        polygon.push_back(Vector2d(std::cos(a), std::sin(a)));
    }
    build_polygon_fan_region(polygon, {20, 21, 22, 23, 24}, 2, V, F, labels);

    BoundarySegmentationResult input =
        build_trimmed_region_input(V, F, labels, 2);
    AutomaticLabelingResult result =
        run_automatic_labeling(V, F, input, test_config("labeling_debug_pentagon"));

    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "pentagon reduces to an admissible case");
    CHECK(result.final_label_count <= 4, "pentagon removes at least one label");
    CHECK(!result.operation_log.empty(), "reduction operation is logged");
}

static void test_parallel_concatenation() {
    cout << "\n=== Parallel concatenation ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_polygon_fan_region(
        {Vector2d(0, 0), Vector2d(1, 0), Vector2d(2, 0),
         Vector2d(2, 1), Vector2d(0, 1)},
        {30, 31, 32, 33, 34},
        3, V, F, labels);

    BoundarySegmentationResult input =
        build_trimmed_region_input(V, F, labels, 3);
    AutomaticLabelingResult result =
        run_automatic_labeling(V, F, input, test_config("labeling_debug_parallel"));

    cout << "  reason=" << result.reason << endl;
    CHECK(input.perimeter_segments.size() == 5, "straight split starts as five labels");
    CHECK(result.valid, "parallel split labeling is admissible");
    CHECK(result.final_label_count == 4, "parallel split is merged to four labels");
    bool saw_parallel_merge = false;
    for (const string& line : result.operation_log) {
        if (line.find("Parallel") != string::npos) saw_parallel_merge = true;
    }
    CHECK(saw_parallel_merge, "parallel merge is logged");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Trimmed Automatic Labeling Tests" << endl;
    cout << "========================================" << endl;

    test_rectangle_labeling();
    test_pentagon_reduction();
    test_parallel_concatenation();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
