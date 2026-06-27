#include "region_boundary.h"

#include <Eigen/Dense>

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

static void test_single_loop_success() {
    cout << "\n=== Single loop success ===" << endl;
    MatrixXd V;
    MatrixXi F;
    build_grid_mesh(2, 2, V, F);
    vector<int> labels(F.rows(), 0);
    int f0 = cell_first_face(2, 0, 0);
    labels[f0] = 7;
    labels[f0 + 1] = 7;

    RegionBoundaryExtractionResult result =
        extract_region_boundary_loop(V, F, labels, 7);

    cout << "  reason=" << result.reason << endl;
    CHECK(result.success, "single-cell region extracts successfully");
    CHECK(result.loop.closed, "loop is closed");
    CHECK(result.loop.vertex_ids.size() == 4, "loop has four vertices");
    CHECK(result.boundary_edges.size() == 4, "loop has four boundary edges");
    CHECK(result.boundary_loop_count == 1, "one boundary loop detected");
    CHECK(result.region_connected, "target region is connected");
    CHECK(export_region_boundary_debug_obj(
              "region_boundary_single_loop.obj", V, F, labels, 7, result) &&
              file_nonempty("region_boundary_single_loop.obj"),
          "debug OBJ export succeeds");
}

static void test_multiple_loops_failure() {
    cout << "\n=== Multiple loops failure ===" << endl;
    MatrixXd V;
    MatrixXi F;
    build_grid_mesh(3, 3, V, F);
    vector<int> labels(F.rows(), 5);

    int center = cell_first_face(3, 1, 1);
    labels[center] = 0;
    labels[center + 1] = 0;

    RegionBoundaryExtractionResult result =
        extract_region_boundary_loop(V, F, labels, 5);

    cout << "  reason=" << result.reason << endl;
    CHECK(!result.success, "annulus-like region is rejected");
    CHECK(result.boundary_loop_count == 2, "two boundary loops detected");
    CHECK(result.reason.find("multiple boundary loops") != string::npos,
          "failure reason mentions multiple loops");
    CHECK(export_region_boundary_debug_obj(
              "region_boundary_multiple_loops.obj", V, F, labels, 5, result) &&
              file_nonempty("region_boundary_multiple_loops.obj"),
          "failure debug OBJ export succeeds");
}

static void test_nonmanifold_boundary_failure() {
    cout << "\n=== Non-manifold boundary failure ===" << endl;
    MatrixXd V(5, 3);
    V << 0, 0, 0,
         1, 0, 0,
         0, 1, 0,
         0, -1, 0,
         1, 1, 0;
    MatrixXi F(3, 3);
    F << 0, 1, 2,
         1, 0, 3,
         0, 1, 4;
    vector<int> labels = {2, 0, 0};

    RegionBoundaryExtractionResult result =
        extract_region_boundary_loop(V, F, labels, 2);

    cout << "  reason=" << result.reason << endl;
    CHECK(!result.success, "non-manifold target edge is rejected");
    CHECK(result.nonmanifold_boundary, "non-manifold flag is set");
    CHECK(result.reason.find("non-manifold") != string::npos,
          "failure reason mentions non-manifold");
}

static void test_disconnected_region_failure() {
    cout << "\n=== Disconnected region failure ===" << endl;
    MatrixXd V(8, 3);
    V << 0, 0, 0,
         1, 0, 0,
         1, 1, 0,
         0, 1, 0,
         3, 0, 0,
         4, 0, 0,
         4, 1, 0,
         3, 1, 0;
    MatrixXi F(4, 3);
    F << 0, 1, 2,
         0, 2, 3,
         4, 5, 6,
         4, 6, 7;
    vector<int> labels(F.rows(), 3);

    RegionBoundaryExtractionResult result =
        extract_region_boundary_loop(V, F, labels, 3);

    cout << "  reason=" << result.reason << endl;
    CHECK(!result.success, "disconnected target region is rejected");
    CHECK(!result.region_connected, "region_connected flag is false");
    CHECK(result.reason.find("not edge-connected") != string::npos,
          "failure reason mentions connectivity");
}

static void test_duplicate_edge_failure() {
    cout << "\n=== Duplicate edge failure ===" << endl;
    MatrixXd V(3, 3);
    V << 0, 0, 0,
         1, 0, 0,
         0, 1, 0;
    MatrixXi F(1, 3);
    F << 0, 1, 1;
    vector<int> labels = {4};

    RegionBoundaryExtractionResult result =
        extract_region_boundary_loop(V, F, labels, 4);

    cout << "  reason=" << result.reason << endl;
    CHECK(!result.success, "degenerate duplicate-edge face is rejected");
    CHECK(result.duplicate_edge, "duplicate edge flag is set");
    CHECK(result.reason.find("duplicate") != string::npos ||
              result.reason.find("degenerate") != string::npos,
          "failure reason mentions duplicate or degenerate edge");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Region Boundary Loop Tests" << endl;
    cout << "========================================" << endl;

    test_single_loop_success();
    test_multiple_loops_failure();
    test_nonmanifold_boundary_failure();
    test_disconnected_region_failure();
    test_duplicate_edge_failure();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
