#include "shared_spline_boundary.h"

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

static BSplineSurface3D make_patch_a() {
    const int n = 5;
    vector<vector<Vector3d>> grid(n, vector<Vector3d>(n));
    for (int i = 0; i < n; i++) {
        double u = (double)i / (double)(n - 1);
        for (int j = 0; j < n; j++) {
            double v = (double)j / (double)(n - 1);
            double x = u;
            double y = v;
            double z = 0.18 * std::sin(3.141592653589793 * u) *
                       std::sin(3.141592653589793 * v);
            grid[i][j] = Vector3d(x, y, z);
        }
    }
    return BSplineSurface3D(
        3, 3,
        make_open_uniform_knot_vector(n, 3),
        make_open_uniform_knot_vector(n, 3),
        grid);
}

static BSplineSurface3D make_patch_b_opposite_shared_orientation() {
    const int n = 5;
    vector<vector<Vector3d>> grid(n, vector<Vector3d>(n));
    for (int i = 0; i < n; i++) {
        double u = (double)i / (double)(n - 1);
        for (int j = 0; j < n; j++) {
            double v = (double)j / (double)(n - 1);
            double x = 1.0 + u;
            double y = 1.0 - v;
            double z = 0.12 * std::sin(3.141592653589793 * u) *
                       std::sin(3.141592653589793 * v);
            grid[i][j] = Vector3d(x, y, z);
        }
    }
    return BSplineSurface3D(
        3, 3,
        make_open_uniform_knot_vector(n, 3),
        make_open_uniform_knot_vector(n, 3),
        grid);
}

static vector<int> right_side_ids(const SharedSplinePatch& patch) {
    vector<int> ids;
    int nu = (int)patch.topology.control_point_ids.size();
    int nv = (int)patch.topology.control_point_ids[0].size();
    for (int j = 0; j < nv; j++) ids.push_back(patch.topology.control_point_ids[nu - 1][j]);
    return ids;
}

static vector<int> left_side_ids(const SharedSplinePatch& patch) {
    vector<int> ids;
    int nv = (int)patch.topology.control_point_ids[0].size();
    for (int j = 0; j < nv; j++) ids.push_back(patch.topology.control_point_ids[0][j]);
    return ids;
}

static void test_opposite_orientation_shared_boundary() {
    cout << "\n=== Opposite-orientation shared boundary ===" << endl;
    BSplineSurface3D a = make_patch_a();
    BSplineSurface3D b = make_patch_b_opposite_shared_orientation();

    SharedSplineAssembly assembly =
        build_two_patch_shared_boundary_assembly(
            a, 10, 1,
            b, 20, 3,
            true);

    cout << "  build reason=" << assembly.reason << endl;
    CHECK(assembly.valid, "shared assembly builds");
    CHECK(assembly.shared_boundaries.size() == 1, "one shared boundary is stored");
    CHECK(assembly.patches.size() == 2, "two patch topologies are stored");

    vector<int> a_ids = right_side_ids(assembly.patches[0]);
    vector<int> b_ids = left_side_ids(assembly.patches[1]);
    bool reversed_ids_match = a_ids.size() == b_ids.size();
    for (int k = 0; k < (int)a_ids.size() && reversed_ids_match; k++) {
        reversed_ids_match = a_ids[k] == b_ids[(int)b_ids.size() - 1 - k];
    }
    CHECK(reversed_ids_match, "opposite side uses reversed shared control point ids");
    CHECK((int)assembly.pool.values.size() ==
              2 * 5 * 5 - (int)a_ids.size(),
          "shared control points are stored once in global pool");

    SharedSplineSampledMesh mesh;
    SharedSplineAdjacencyReport report;
    bool sampled = sample_shared_spline_assembly(assembly, 9, 9, mesh, report);
    cout << "  sample/report reason=" << report.reason << endl;
    cout << "  max_boundary_sample_difference="
         << report.max_boundary_sample_difference << endl;
    cout << "  shared edges two faces="
         << report.shared_boundary_edges_with_two_faces << " / "
         << report.shared_boundary_edge_count << endl;

    CHECK(sampled, "sampled mesh passes adjacency report");
    CHECK(mesh.V.rows() == 2 * 9 * 9 - 9,
          "shared boundary sample vertices are generated once");
    CHECK(mesh.F.rows() == 2 * (9 - 1) * (9 - 1) * 2,
          "both patches contribute expected triangles");
    CHECK(report.max_boundary_sample_difference < 1e-12,
          "shared boundary samples are geometrically identical");
    CHECK(report.shared_boundary_edges_with_two_faces ==
              report.shared_boundary_edge_count,
          "all shared boundary edges have two incident faces");
    CHECK(report.nonmanifold_edge_count == 0,
          "sampled mesh has no non-manifold edges");
    CHECK(report.watertight_across_shared_boundary,
          "mesh is watertight across shared boundary");

    CHECK(export_shared_spline_debug(
              "shared_spline_two_patch", assembly, mesh, report),
          "debug exports succeed");
    CHECK(file_nonempty("shared_spline_two_patch_colored_patches.ply") &&
              file_nonempty("shared_spline_two_patch_shared_boundary.obj") &&
              file_nonempty("shared_spline_two_patch_control_point_ids.csv") &&
              file_nonempty("shared_spline_two_patch_watertight_mesh.obj") &&
              file_nonempty("shared_spline_two_patch_adjacency_report.csv"),
          "debug export files are non-empty");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Shared Spline Boundary Tests" << endl;
    cout << "========================================" << endl;

    test_opposite_orientation_shared_boundary();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
