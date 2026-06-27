#include "bspline.h"

#include <Eigen/Dense>

#include <cmath>
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

static bool near(double a, double b, double tol) {
    return std::abs(a - b) <= tol;
}

static bool near_vec(const Vector3d& a, const Vector3d& b, double tol) {
    return (a - b).norm() <= tol;
}

static bool file_nonempty(const string& filename) {
    std::ifstream fin(filename, std::ios::binary | std::ios::ate);
    return fin.is_open() && fin.tellg() > 0;
}

static BSplineCurve3D make_test_curve() {
    vector<Vector3d> cp;
    cp.push_back(Vector3d(0.0, 0.0, 0.0));
    cp.push_back(Vector3d(1.0, 2.0, 0.5));
    cp.push_back(Vector3d(2.0, -1.0, 1.0));
    cp.push_back(Vector3d(3.0, 1.0, 0.0));
    cp.push_back(Vector3d(4.0, 0.0, -0.5));
    cp.push_back(Vector3d(5.0, 2.0, 0.0));
    return BSplineCurve3D(3, make_open_uniform_knot_vector((int)cp.size(), 3), cp);
}

static BSplineSurface3D make_plane_surface() {
    vector<vector<Vector3d>> grid(4, vector<Vector3d>(4));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            grid[i][j] = Vector3d((double)i, (double)j, 2.0);
        }
    }
    return BSplineSurface3D(
        3, 3,
        make_open_uniform_knot_vector(4, 3),
        make_open_uniform_knot_vector(4, 3),
        grid);
}

static BSplineSurface3D make_test_surface() {
    vector<vector<Vector3d>> grid(5, vector<Vector3d>(4));
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 4; j++) {
            double x = (double)i;
            double y = (double)j;
            double z = 0.2 * std::sin(0.7 * x) + 0.15 * std::cos(1.1 * y);
            grid[i][j] = Vector3d(x, y, z);
        }
    }
    return BSplineSurface3D(
        3, 3,
        make_open_uniform_knot_vector(5, 3),
        make_open_uniform_knot_vector(4, 3),
        grid);
}

static void test_basis_properties() {
    cout << "\n=== Basis properties ===" << endl;
    const int degree = 3;
    const int control_count = 7;
    vector<double> knots = make_open_uniform_knot_vector(control_count, degree);

    bool nonnegative = true;
    bool sums_to_one = true;
    for (int s = 0; s <= 40; s++) {
        double t = (double)s / 40.0;
        double sum = 0.0;
        for (int i = 0; i < control_count; i++) {
            double b = bspline_basis(i, degree, t, knots);
            if (b < -1e-12) nonnegative = false;
            sum += b;
        }
        if (!near(sum, 1.0, 1e-10)) sums_to_one = false;
    }

    CHECK(nonnegative, "basis functions are non-negative on [0,1]");
    CHECK(sums_to_one, "basis functions sum to one on [0,1]");
}

static void test_curve_endpoints() {
    cout << "\n=== Curve endpoints ===" << endl;
    BSplineCurve3D curve = make_test_curve();
    CHECK(near_vec(curve.evaluate(0.0), curve.control_points.front(), 1e-12),
          "open knot curve passes through first control point");
    CHECK(near_vec(curve.evaluate(1.0), curve.control_points.back(), 1e-12),
          "open knot curve passes through last control point");
}

static void test_plane_surface() {
    cout << "\n=== Plane surface ===" << endl;
    BSplineSurface3D surface = make_plane_surface();

    bool planar = true;
    for (int i = 0; i <= 6; i++) {
        for (int j = 0; j <= 6; j++) {
            Vector3d p = surface.evaluate((double)i / 6.0, (double)j / 6.0);
            if (!near(p.z(), 2.0, 1e-12)) planar = false;
        }
    }
    CHECK(planar, "planar control net evaluates to planar surface");

    MatrixXd V;
    MatrixXi F;
    sample_bspline_surface(surface, 5, 6, V, F);
    CHECK(V.rows() == 30 && F.rows() == 40,
          "surface sampling creates expected vertex and face counts");
}

static void test_surface_edges() {
    cout << "\n=== Surface edges ===" << endl;
    BSplineSurface3D surface = make_test_surface();
    const int nu = (int)surface.control_grid.size();
    const int nv = (int)surface.control_grid[0].size();

    vector<Vector3d> u0_cp, u1_cp, v0_cp, v1_cp;
    for (int i = 0; i < nu; i++) {
        v0_cp.push_back(surface.control_grid[i][0]);
        v1_cp.push_back(surface.control_grid[i][nv - 1]);
    }
    for (int j = 0; j < nv; j++) {
        u0_cp.push_back(surface.control_grid[0][j]);
        u1_cp.push_back(surface.control_grid[nu - 1][j]);
    }

    BSplineCurve3D v0_curve(3, surface.knots_u, v0_cp);
    BSplineCurve3D v1_curve(3, surface.knots_u, v1_cp);
    BSplineCurve3D u0_curve(3, surface.knots_v, u0_cp);
    BSplineCurve3D u1_curve(3, surface.knots_v, u1_cp);

    bool edges_match = true;
    for (int s = 0; s <= 20; s++) {
        double t = (double)s / 20.0;
        edges_match = edges_match &&
            near_vec(surface.evaluate(t, 0.0), v0_curve.evaluate(t), 1e-11) &&
            near_vec(surface.evaluate(t, 1.0), v1_curve.evaluate(t), 1e-11) &&
            near_vec(surface.evaluate(0.0, t), u0_curve.evaluate(t), 1e-11) &&
            near_vec(surface.evaluate(1.0, t), u1_curve.evaluate(t), 1e-11);
    }
    CHECK(edges_match, "surface boundary curves match control-grid boundary curves");
}

static void test_derivatives() {
    cout << "\n=== Derivatives ===" << endl;
    BSplineCurve3D curve = make_test_curve();
    const double t = 0.37;
    const double h = 1e-5;

    Vector3d d1_fd = (curve.evaluate(t + h) - curve.evaluate(t - h)) / (2.0 * h);
    Vector3d d1 = curve.derivative(t, 1);
    CHECK(near_vec(d1, d1_fd, 1e-5), "curve first derivative matches finite difference");

    Vector3d d2_fd = (curve.evaluate(t + h) -
                      2.0 * curve.evaluate(t) +
                      curve.evaluate(t - h)) / (h * h);
    Vector3d d2 = curve.derivative(t, 2);
    CHECK(near_vec(d2, d2_fd, 5e-4), "curve second derivative matches finite difference");

    BSplineSurface3D surface = make_test_surface();
    const double u = 0.42;
    const double v = 0.58;
    Vector3d du_fd = (surface.evaluate(u + h, v) -
                      surface.evaluate(u - h, v)) / (2.0 * h);
    Vector3d dv_fd = (surface.evaluate(u, v + h) -
                      surface.evaluate(u, v - h)) / (2.0 * h);

    CHECK(near_vec(surface.derivative(u, v, 1, 0), du_fd, 1e-5),
          "surface u derivative matches finite difference");
    CHECK(near_vec(surface.derivative(u, v, 0, 1), dv_fd, 1e-5),
          "surface v derivative matches finite difference");
}

static void test_debug_exports() {
    cout << "\n=== Debug exports ===" << endl;
    BSplineCurve3D curve = make_test_curve();
    BSplineSurface3D surface = make_test_surface();

    bool curve_ok = export_bspline_curve_polyline_obj(
        "bspline_test_curve.obj", curve, 16);
    bool surface_ok = export_bspline_surface_mesh_obj(
        "bspline_test_surface.obj", surface, 8, 7);
    bool net_ok = export_bspline_surface_control_net_obj(
        "bspline_test_control_net.obj", surface);
    bool points_ok = export_control_points_obj(
        "bspline_test_control_points.obj", curve.control_points);

    CHECK(curve_ok && file_nonempty("bspline_test_curve.obj"),
          "curve polyline OBJ export succeeds");
    CHECK(surface_ok && file_nonempty("bspline_test_surface.obj"),
          "surface mesh OBJ export succeeds");
    CHECK(net_ok && file_nonempty("bspline_test_control_net.obj"),
          "surface control net OBJ export succeeds");
    CHECK(points_ok && file_nonempty("bspline_test_control_points.obj"),
          "control points OBJ export succeeds");
}

int main() {
    cout << "========================================" << endl;
    cout << "  B-spline Unit Tests" << endl;
    cout << "========================================" << endl;

    try {
        test_basis_properties();
        test_curve_endpoints();
        test_plane_surface();
        test_surface_edges();
        test_derivatives();
        test_debug_exports();
    } catch (const std::exception& e) {
        cerr << "  EXCEPTION: " << e.what() << endl;
        g_fail++;
    }

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
