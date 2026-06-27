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

static void build_grid_mesh(int nx, int ny, MatrixXd& V, MatrixXi& F) {
    V.resize((nx + 1) * (ny + 1), 3);
    auto vid = [nx](int x, int y) {
        return y * (nx + 1) + x;
    };
    for (int y = 0; y <= ny; y++) {
        for (int x = 0; x <= nx; x++) {
            V.row(vid(x, y)) << (double)x, (double)y, 0.0;
        }
    }

    F.resize(nx * ny * 2, 3);
    int f = 0;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int v00 = vid(x, y);
            int v10 = vid(x + 1, y);
            int v01 = vid(x, y + 1);
            int v11 = vid(x + 1, y + 1);
            F.row(f++) << v00, v10, v11;
            F.row(f++) << v00, v11, v01;
        }
    }
}

static int cell_first_face(int nx, int x, int y) {
    return 2 * (y * nx + x);
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
        double norm = outward.norm();
        if (norm > 1e-14) outward /= norm;
        Vector2d p = 0.5 * (a + b) + 0.35 * outward;
        V.row(outside_vertex) << p.x(), p.y(), 0.0;
        F.row(outside_face) << (i + 1) % n, i, outside_vertex;
        labels.push_back(adj);
        outside_vertex++;
        outside_face++;
    }
}

static void test_rectangle_perimeter() {
    cout << "\n=== Rectangle perimeter ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_polygon_fan_region(
        {Vector2d(0, 0), Vector2d(2, 0), Vector2d(2, 1), Vector2d(0, 1)},
        {-1, -1, -1, -1},
        7, V, F, labels);

    BoundarySegmentationResult result =
        build_trimmed_region_input(V, F, labels, 7);

    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "rectangle input is valid");
    CHECK(result.loops.size() == 1, "one boundary loop");
    CHECK(result.perimeter_loop_index == 0, "loop is perimeter");
    CHECK(result.loops[0].vertex_ids.size() == 4, "perimeter has four vertices");
    CHECK(result.perimeter_segments.size() == 1, "single mesh-boundary segment");
    CHECK(result.perimeter_segments[0].touches_mesh_boundary, "segment records mesh boundary");
    CHECK(export_trimmed_region_input_debug_obj(
              "trimmed_region_input_rectangle.obj", V, F, result) &&
              file_nonempty("trimmed_region_input_rectangle.obj"),
          "debug OBJ export succeeds");
}

static void test_pentagon_shared_perimeter() {
    cout << "\n=== Pentagon shared perimeter ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    vector<Vector2d> polygon;
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < 5; i++) {
        double a = 2.0 * pi * (double)i / 5.0;
        polygon.push_back(Vector2d(std::cos(a), std::sin(a)));
    }
    build_polygon_fan_region(polygon, {9, 9, 9, 9, 9}, 3, V, F, labels);

    BoundarySegmentationResult result =
        build_trimmed_region_input(V, F, labels, 3);

    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "pentagon input is valid");
    CHECK(result.loops[0].vertex_ids.size() == 5, "perimeter has five vertices");
    CHECK(result.perimeter_segments.size() == 1, "same adjacent region forms one chain");
    CHECK(result.perimeter_segments[0].adjacent_region_id == 9,
          "segment stores adjacent region id");
}

static void test_concave_region() {
    cout << "\n=== Concave L-shaped region ===" << endl;
    MatrixXd V;
    MatrixXi F;
    build_grid_mesh(2, 2, V, F);
    vector<int> labels(F.rows(), 0);
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            if (x == 1 && y == 1) continue;
            int f0 = cell_first_face(2, x, y);
            labels[f0] = 5;
            labels[f0 + 1] = 5;
        }
    }

    BoundarySegmentationResult result =
        build_trimmed_region_input(V, F, labels, 5);

    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "concave region input is valid");
    CHECK(result.loops.size() == 1, "concave region still has one loop");
    CHECK(result.loops[result.perimeter_loop_index].vertex_ids.size() == 8,
          "L-shaped perimeter preserves eight mesh boundary vertices");
    CHECK(!result.perimeter_segments.empty(), "concave perimeter is segmented");
}

static void test_region_with_hole() {
    cout << "\n=== Region with hole ===" << endl;
    MatrixXd V;
    MatrixXi F;
    build_grid_mesh(3, 3, V, F);
    vector<int> labels(F.rows(), 8);
    int center = cell_first_face(3, 1, 1);
    labels[center] = 0;
    labels[center + 1] = 0;

    BoundarySegmentationResult result =
        build_trimmed_region_input(V, F, labels, 8);

    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "region with one hole is accepted as paper input");
    CHECK(result.loops.size() == 2, "two loops are extracted");
    CHECK(result.perimeter_loop_index >= 0, "perimeter loop is identified");
    int inner_count = 0;
    for (const AuthoritativeBoundaryLoop& loop : result.loops) {
        if (!loop.is_perimeter) inner_count++;
    }
    CHECK(inner_count == 1, "one inner loop is distinguished");
}

static void test_feature_barrier_split() {
    cout << "\n=== Feature barrier split ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    build_polygon_fan_region(
        {Vector2d(0, 0), Vector2d(2, 0), Vector2d(2, 1), Vector2d(0, 1)},
        {-1, -1, -1, -1},
        4, V, F, labels);

    BoundarySegmentationConfig config;
    config.feature_edges.insert(EdgeKey(0, 1));

    BoundarySegmentationResult result =
        build_trimmed_region_input(V, F, labels, 4, config);

    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "feature-marked input is valid");
    CHECK(result.perimeter_segments.size() == 2,
          "feature edge cuts perimeter into two chains");
    int feature_segments = 0;
    for (const BoundarySegment& segment : result.perimeter_segments) {
        if (segment.touches_feature_barrier) feature_segments++;
    }
    CHECK(feature_segments == 1, "one segment records feature barrier");
}

static void test_discontinuous_shared_chains() {
    cout << "\n=== Discontinuous shared chains ===" << endl;
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    vector<Vector2d> polygon = {
        Vector2d(0, 0), Vector2d(1, 0), Vector2d(2, 0.8),
        Vector2d(1.5, 1.8), Vector2d(0.5, 1.8), Vector2d(-0.3, 0.8)};
    build_polygon_fan_region(polygon, {2, 2, 3, 2, 2, 4}, 1, V, F, labels);

    BoundarySegmentationResult result =
        build_trimmed_region_input(V, F, labels, 1);

    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "shared-chain input is valid");
    CHECK(result.perimeter_segments.size() == 4,
          "adjacent-region changes create four continuous chains");
    int adjacent_two_segments = 0;
    for (const BoundarySegment& segment : result.perimeter_segments) {
        if (segment.adjacent_region_id == 2) adjacent_two_segments++;
    }
    CHECK(adjacent_two_segments == 2,
          "two disjoint chains with the same adjacent region are not merged");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Trimmed Region Input Tests" << endl;
    cout << "========================================" << endl;

    test_rectangle_perimeter();
    test_pentagon_shared_perimeter();
    test_concave_region();
    test_region_with_hole();
    test_feature_barrier_split();
    test_discontinuous_shared_chains();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
