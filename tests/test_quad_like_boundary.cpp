#include "quad_like_boundary.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Eigen::Vector3d;
using std::array;
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

static RegionBoundaryLoop make_quad_loop(bool clockwise) {
    vector<Vector3d> pts;
    auto push = [&](double x, double y) {
        pts.push_back(Vector3d(x, y, 0.08 * x + 0.03 * y));
    };

    for (int i = 0; i < 8; i++) push(4.0 * i / 8.0, 0.05 * std::sin((double)i));
    for (int i = 0; i < 6; i++) push(4.0 + 0.04 * std::sin((double)i), 2.0 * i / 6.0);
    for (int i = 0; i < 8; i++) push(4.0 - 4.0 * i / 8.0, 2.0 + 0.04 * std::cos((double)i));
    for (int i = 0; i < 6; i++) push(-0.03 * std::cos((double)i), 2.0 - 2.0 * i / 6.0);

    if (clockwise) std::reverse(pts.begin(), pts.end());

    RegionBoundaryLoop loop;
    loop.closed = true;
    loop.positions = pts;
    loop.vertex_ids.resize(pts.size());
    for (int i = 0; i < (int)pts.size(); i++) loop.vertex_ids[i] = i;
    return loop;
}

static Vector3d side_delta(const vector<Vector3d>& side) {
    if (side.size() < 2) return Vector3d::Zero();
    return side.back() - side.front();
}

static void test_auto_quad_split() {
    cout << "\n=== Auto quad-like split ===" << endl;
    RegionBoundaryLoop loop = make_quad_loop(false);

    QuadLikeBoundaryConfig cfg;
    cfg.max_corner_candidates = 18;
    cfg.min_quality_score = 0.20;
    QuadLikeBoundaryResult result = split_quad_like_boundary(loop, cfg);

    cout << "  reason=" << result.reason
         << " quality=" << result.boundary.quality_score << endl;

    CHECK(result.success, "automatic quad-like split succeeds");
    CHECK(result.boundary.valid, "boundary is valid");
    CHECK(!result.used_manual_fallback, "automatic split did not use manual fallback");
    CHECK(result.boundary.quality_score > 0.20, "quality score is above threshold");

    for (int s = 0; s < 4; s++) {
        CHECK(result.boundary.corner_loop_indices[s] >= 0 &&
                  result.boundary.corner_loop_indices[s] < (int)loop.positions.size(),
              "corner index is in range");
        CHECK(result.boundary.side_polylines[s].size() >= 2,
              "side polyline has at least two points");
    }

    Vector3d bottom = side_delta(result.boundary.side_polylines[0]);
    Vector3d right = side_delta(result.boundary.side_polylines[1]);
    Vector3d top = side_delta(result.boundary.side_polylines[2]);
    Vector3d left = side_delta(result.boundary.side_polylines[3]);

    CHECK(bottom.x() > 0.0, "bottom side is oriented left-to-right");
    CHECK(right.y() > 0.0, "right side is oriented bottom-to-top");
    CHECK(top.x() < 0.0, "top side is oriented right-to-left");
    CHECK(left.y() < 0.0, "left side is oriented top-to-bottom");

    CHECK(export_quad_like_boundary_pca_debug_obj(
              "quad_like_boundary_auto_pca.obj", result.debug) &&
              file_nonempty("quad_like_boundary_auto_pca.obj"),
          "PCA debug OBJ export succeeds");
}

static void test_clockwise_loop_still_orients_sides() {
    cout << "\n=== Clockwise input orientation ===" << endl;
    RegionBoundaryLoop loop = make_quad_loop(true);
    QuadLikeBoundaryConfig cfg;
    cfg.max_corner_candidates = 18;
    cfg.min_quality_score = 0.20;

    QuadLikeBoundaryResult result = split_quad_like_boundary(loop, cfg);
    cout << "  reason=" << result.reason
         << " quality=" << result.boundary.quality_score << endl;

    CHECK(result.success, "clockwise input also splits successfully");
    CHECK(result.boundary.valid, "clockwise result is valid");
    CHECK(result.boundary.side_polylines[0].size() >= 2, "bottom side exists");
}

static void test_manual_fallback_and_config() {
    cout << "\n=== Manual fallback ===" << endl;
    RegionBoundaryLoop loop = make_quad_loop(false);
    std::ofstream fout("quad_like_manual_corners.txt");
    fout << "corner_loop_indices 0 8 14 22\n";
    fout.close();

    array<int, 4> manual = {{-1, -1, -1, -1}};
    string reason;
    CHECK(load_quad_like_boundary_manual_config(
              "quad_like_manual_corners.txt", manual, reason),
          "manual config loads");

    QuadLikeBoundaryConfig cfg;
    cfg.use_manual_corners = true;
    cfg.manual_corner_loop_indices = manual;
    cfg.min_quality_score = 1.1;

    QuadLikeBoundaryResult result = split_quad_like_boundary(loop, cfg);
    cout << "  reason=" << result.reason
         << " quality=" << result.boundary.quality_score << endl;

    CHECK(result.success, "manual fallback succeeds when automatic threshold fails");
    CHECK(result.used_manual_fallback, "manual fallback flag is set");
    CHECK(result.boundary.valid, "manual fallback boundary is valid");
    CHECK(result.boundary.corner_loop_indices[0] == 0,
          "manual corner participates in oriented side convention");

    CHECK(export_quad_like_boundary_pca_debug_obj(
              "quad_like_boundary_manual_pca.obj", result.debug) &&
              file_nonempty("quad_like_boundary_manual_pca.obj"),
          "manual PCA debug OBJ export succeeds");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Quad-like Boundary Tests" << endl;
    cout << "========================================" << endl;

    test_auto_quad_split();
    test_clockwise_loop_still_orients_sides();
    test_manual_fallback_and_config();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
