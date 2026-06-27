#include "polyharmonic_3d_extension.h"
#include "rectangular_domain_extension.h"

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

static RectangularDomainExtensionConfig rect_cfg(const string& prefix) {
    RectangularDomainExtensionConfig cfg;
    cfg.debug_prefix = prefix;
    cfg.sample_count_u = 10;
    cfg.sample_count_v = 10;
    cfg.margin = 0.05;
    cfg.export_debug = false;
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

static MatrixXd positions_from_uv(
    const RectangularDomainExtensionResult& domain,
    const string& mode) {
    MatrixXd P(domain.full_uv_vertices.rows(), 3);
    for (int i = 0; i < domain.full_uv_vertices.rows(); i++) {
        double u = domain.full_uv_vertices(i, 0);
        double v = domain.full_uv_vertices(i, 1);
        if (mode == "plane") {
            P.row(i) << u, v, 0.4 * u - 0.2 * v + 1.0;
        } else if (mode == "cylinder") {
            double theta = 0.35 * u;
            P.row(i) << std::cos(theta), v, std::sin(theta);
        } else {
            P.row(i) << u, v, 0.15 * u * v;
        }
        if (i >= (int)domain.original_vertex_mask.size() ||
            !domain.original_vertex_mask[i]) {
            P.row(i).setZero();
        }
    }
    return P;
}

static Polyharmonic3DExtensionConfig ext_cfg(
    const string& prefix,
    PolyharmonicContinuityMode mode) {
    Polyharmonic3DExtensionConfig cfg;
    cfg.debug_prefix = prefix;
    cfg.mode = mode;
    cfg.export_debug = true;
    cfg.regularization = 1e-8;
    return cfg;
}

static Polyharmonic3DExtensionConfig fair_cfg(
    const string& prefix,
    double fairness_weight,
    double isocurve_weight) {
    Polyharmonic3DExtensionConfig cfg = ext_cfg(prefix, PolyharmonicContinuityMode::G2);
    cfg.fairness_weight = fairness_weight;
    cfg.isocurve_fairness_weight = isocurve_weight;
    cfg.labeled_isocurve_u_values = {1.0};
    cfg.labeled_isocurve_v_values = {1.0};
    return cfg;
}

static void test_plane_stays_plane() {
    cout << "\n=== Plane extension ===" << endl;
    RectangularDomainExtensionResult domain =
        build_rectangular_domain_extension(make_l_shape_input(), rect_cfg("unused"));
    MatrixXd P = positions_from_uv(domain, "plane");
    Polyharmonic3DExtensionResult result =
        extend_polyharmonic_3d(domain, P, ext_cfg("polyharmonic_plane_g2", PolyharmonicContinuityMode::G2));
    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "G2 plane extension succeeds");
    double max_plane_error = 0.0;
    const double normal_scale = std::sqrt(1.0 + 0.4 * 0.4 + 0.2 * 0.2);
    for (int i = 0; i < result.extended_vertices.rows(); i++) {
        double x = result.extended_vertices(i, 0);
        double y = result.extended_vertices(i, 1);
        double z = result.extended_vertices(i, 2);
        max_plane_error = std::max(
            max_plane_error,
            std::abs(z - (0.4 * x - 0.2 * y + 1.0)) / normal_scale);
    }
    CHECK(max_plane_error < 1e-4, "extended artificial vertices remain on the plane");
    CHECK(file_nonempty("polyharmonic_plane_g2_extended_mesh.obj"), "debug mesh exported");
    CHECK(result.surface_area_growth > 0.0, "surface area growth is reported");
    CHECK(result.bbox_growth > 0.0, "bounding-box growth is reported");
}

static void test_cylinder_trend() {
    cout << "\n=== Cylinder trend ===" << endl;
    RectangularDomainExtensionResult domain =
        build_rectangular_domain_extension(make_l_shape_input(), rect_cfg("unused"));
    MatrixXd P = positions_from_uv(domain, "cylinder");
    Polyharmonic3DExtensionResult result =
        extend_polyharmonic_3d(domain, P, ext_cfg("polyharmonic_cylinder_g2", PolyharmonicContinuityMode::G2));
    cout << "  reason=" << result.reason << endl;
    CHECK(result.valid, "G2 cylinder extension succeeds");
    double max_radius_error = 0.0;
    for (int i = 0; i < result.extended_vertices.rows(); i++) {
        if (i < (int)domain.original_vertex_mask.size() && domain.original_vertex_mask[i]) continue;
        double x = result.extended_vertices(i, 0);
        double z = result.extended_vertices(i, 2);
        max_radius_error = std::max(max_radius_error, std::abs(std::sqrt(x * x + z * z) - 1.0));
    }
    CHECK(max_radius_error < 0.35, "cylinder extension keeps the radius trend bounded");
}

static void test_g1_g2_compare() {
    cout << "\n=== G1/G2 comparison ===" << endl;
    RectangularDomainExtensionResult domain =
        build_rectangular_domain_extension(make_l_shape_input(), rect_cfg("unused"));
    MatrixXd P = positions_from_uv(domain, "quadratic");
    Polyharmonic3DExtensionResult g1 =
        extend_polyharmonic_3d(domain, P, ext_cfg("polyharmonic_compare_g1", PolyharmonicContinuityMode::G1));
    Polyharmonic3DExtensionResult g2 =
        extend_polyharmonic_3d(domain, P, ext_cfg("polyharmonic_compare_g2", PolyharmonicContinuityMode::G2));
    CHECK(g1.valid, "G1 extension succeeds");
    CHECK(g2.valid, "G2 extension succeeds");
    CHECK((g1.extended_vertices - g2.extended_vertices).norm() > 1e-8,
          "G1 and G2 produce distinguishable extensions");
}

static void test_hole_and_skinny() {
    cout << "\n=== Hole and skinny triangles ===" << endl;
    RectangularDomainExtensionInput input = make_hole_input();
    input.UV(5, 0) = 2.001; // create a skinny original UV triangle near the hole.
    RectangularDomainExtensionResult domain =
        build_rectangular_domain_extension(input, rect_cfg("unused"));
    MatrixXd P = positions_from_uv(domain, "plane");
    Polyharmonic3DExtensionResult result =
        extend_polyharmonic_3d(domain, P, ext_cfg("polyharmonic_hole_skinny_g2", PolyharmonicContinuityMode::G2));
    cout << "  reason=" << result.reason << endl;
    CHECK(domain.valid, "rectangular extension with hole succeeds");
    CHECK(result.valid, "polyharmonic extension handles hole and skinny triangles");
    CHECK(result.unknown_vertex_count > 0, "hole/artificial vertices are solved");
    CHECK(result.min_mass > 0.0, "lumped mass is clamped positive");
}

static void test_3d_fairness_modes() {
    cout << "\n=== 3D fairness modes ===" << endl;
    RectangularDomainExtensionResult domain =
        build_rectangular_domain_extension(make_l_shape_input(), rect_cfg("unused"));
    MatrixXd P = positions_from_uv(domain, "quadratic");

    Polyharmonic3DExtensionResult pure =
        extend_polyharmonic_3d(domain, P, ext_cfg("polyharmonic_fairness_pure_g2", PolyharmonicContinuityMode::G2));
    Polyharmonic3DExtensionResult iso =
        extend_polyharmonic_3d(domain, P, fair_cfg("polyharmonic_fairness_iso_g2", 0.45, 1.0));
    Polyharmonic3DExtensionResult full =
        extend_polyharmonic_3d(domain, P, fair_cfg("polyharmonic_fairness_full_g2", 0.45, 0.5));

    CHECK(pure.valid, "pure G2 extension succeeds");
    CHECK(iso.valid, "extension plus isocurve fairness succeeds");
    CHECK(full.valid, "full fairness extension succeeds");
    CHECK(iso.isocurve_fairness_energy <= pure.isocurve_fairness_energy + 1e-6,
          "isocurve fairness does not increase isocurve energy");
    CHECK(std::isfinite(full.mesh_fairness_energy), "mesh fairness energy is reported");
    CHECK(std::isfinite(full.mean_boundary_curvature), "boundary curvature is reported");
    CHECK(full.surface_area_growth > 0.0, "full fairness surface area growth is positive");
    CHECK(full.bbox_growth > 0.0, "full fairness bounding-box growth is positive");
    CHECK(file_nonempty("polyharmonic_fairness_full_g2_summary.csv"), "full fairness summary is exported");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Polyharmonic 3D Extension Tests" << endl;
    cout << "========================================" << endl;

    test_plane_stays_plane();
    test_cylinder_trend();
    test_g1_g2_compare();
    test_hole_and_skinny();
    test_3d_fairness_modes();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;
    return g_fail;
}
