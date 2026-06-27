#include "rectangular_domain_extension.h"

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
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

static RectangularDomainExtensionConfig test_config(const string& prefix) {
    RectangularDomainExtensionConfig cfg;
    cfg.debug_prefix = prefix;
    cfg.sample_count_u = 12;
    cfg.sample_count_v = 12;
    cfg.margin = 0.05;
    cfg.export_debug = true;
    return cfg;
}

static RectangularDomainExtensionInput make_l_shape_input() {
    RectangularDomainExtensionInput input;
    input.UV.resize(6, 2);
    input.UV << 0, 0,
                2, 0,
                2, 1,
                1, 1,
                1, 2,
                0, 2;
    input.region_vertex_ids = {0, 1, 2, 3, 4, 5};
    input.local_faces.resize(4, 3);
    input.local_faces << 0, 1, 3,
                         1, 2, 3,
                         0, 3, 5,
                         3, 4, 5;
    input.local_face_to_global_face = {0, 1, 2, 3};
    TrimLoop2D loop;
    loop.is_perimeter = true;
    loop.vertex_ids = input.region_vertex_ids;
    for (int vid : loop.vertex_ids) loop.uv_polyline.push_back(input.UV.row(vid).transpose());
    input.trim_loops.push_back(loop);
    return input;
}

static RectangularDomainExtensionInput make_hole_input() {
    RectangularDomainExtensionInput input;
    input.UV.resize(8, 2);
    input.UV << 0, 0,
                3, 0,
                3, 3,
                0, 3,
                1, 1,
                2, 1,
                2, 2,
                1, 2;
    input.region_vertex_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    input.local_faces.resize(8, 3);
    input.local_faces << 0, 1, 5,
                         0, 5, 4,
                         1, 2, 6,
                         1, 6, 5,
                         2, 3, 7,
                         2, 7, 6,
                         3, 0, 4,
                         3, 4, 7;
    input.local_face_to_global_face = {0, 1, 2, 3, 4, 5, 6, 7};
    TrimLoop2D outer;
    outer.is_perimeter = true;
    outer.vertex_ids = {0, 1, 2, 3};
    for (int vid : outer.vertex_ids) outer.uv_polyline.push_back(input.UV.row(vid).transpose());
    TrimLoop2D inner;
    inner.is_perimeter = false;
    inner.vertex_ids = {4, 5, 6, 7};
    for (int vid : inner.vertex_ids) inner.uv_polyline.push_back(input.UV.row(vid).transpose());
    input.trim_loops = {outer, inner};
    return input;
}

static bool has_artificial_centroid_near(
    const RectangularDomainExtensionResult& result,
    const Vector2d& target,
    double radius) {
    for (int i = 0; i < result.full_faces.rows(); i++) {
        if (result.original_face_mask[i]) continue;
        Eigen::Vector3i f = result.full_faces.row(i).transpose();
        Vector2d c = (result.full_uv_vertices.row(f.x()).transpose() +
                      result.full_uv_vertices.row(f.y()).transpose() +
                      result.full_uv_vertices.row(f.z()).transpose()) / 3.0;
        if ((c - target).norm() <= radius) return true;
    }
    return false;
}

static bool has_nonmanifold_edges(const RectangularDomainExtensionResult& result) {
    std::map<std::pair<int, int>, int> edge_count;
    for (int fi = 0; fi < result.full_faces.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            int a = result.full_faces(fi, k);
            int b = result.full_faces(fi, (k + 1) % 3);
            if (a > b) std::swap(a, b);
            edge_count[{a, b}]++;
        }
    }
    for (const auto& kv : edge_count) {
        if (kv.second > 2) return true;
    }
    return false;
}

static void test_concave_gap_extension() {
    cout << "\n=== Concave gap extension ===" << endl;
    RectangularDomainExtensionInput input = make_l_shape_input();
    RectangularDomainExtensionResult result =
        build_rectangular_domain_extension(input, test_config("rect_extension_lshape"));
    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "L-shape extension succeeds");
    CHECK(result.original_face_count > 0, "original region is covered by CDT face mask");
    CHECK(result.artificial_face_count > 0, "artificial faces are generated");
    CHECK(!has_nonmanifold_edges(result), "full rectangular extension mesh is manifold");
    CHECK(has_artificial_centroid_near(result, Vector2d(1.5, 1.5), 0.45),
          "concave missing corner is filled by artificial triangles");
    CHECK(result.authoritative_trim_loops.size() == 1, "perimeter trim loop is preserved");
    CHECK(file_nonempty("rect_extension_lshape_full_uv.obj") &&
              file_nonempty("rect_extension_lshape_trim_loops.obj") &&
              file_nonempty("rect_extension_lshape_face_mask.csv"),
          "debug files are exported");
}

static void test_inner_hole_extension() {
    cout << "\n=== Inner hole extension ===" << endl;
    RectangularDomainExtensionInput input = make_hole_input();
    RectangularDomainExtensionResult result =
        build_rectangular_domain_extension(input, test_config("rect_extension_hole"));
    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "hole extension succeeds");
    CHECK(result.authoritative_trim_loops.size() == 2, "outer and inner trim loops are preserved");
    CHECK(result.original_face_count > 0, "annulus region is covered by CDT face mask");
    CHECK(!has_nonmanifold_edges(result), "hole extension mesh is manifold");
    CHECK(has_artificial_centroid_near(result, Vector2d(1.5, 1.5), 0.45),
          "inner hole is filled by artificial triangles");
    CHECK(result.original_vertex_count == (int)input.region_vertex_ids.size(),
          "original vertex mask count is correct");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Rectangular Domain Extension Tests" << endl;
    cout << "========================================" << endl;

    test_concave_gap_extension();
    test_inner_hole_extension();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;
    return g_fail;
}
