#include "trimmed_bspline_surface.h"

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::cerr;
using std::cout;
using std::endl;
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

static bool file_nonempty(const std::string& filename) {
    std::ifstream fin(filename, std::ios::binary | std::ios::ate);
    return fin.is_open() && fin.tellg() > 0;
}

static BSplineSurface3D make_planar_surface() {
    int nu = 4;
    int nv = 4;
    vector<vector<Vector3d>> grid(nu, vector<Vector3d>(nv));
    for (int i = 0; i < nu; i++) {
        double u = (double)i / (double)(nu - 1);
        for (int j = 0; j < nv; j++) {
            double v = (double)j / (double)(nv - 1);
            grid[i][j] = Vector3d(u, v, 0.25 * u + 0.1 * v);
        }
    }
    return BSplineSurface3D(
        3,
        3,
        make_open_uniform_knot_vector(nu, 3),
        make_open_uniform_knot_vector(nv, 3),
        grid);
}

static vector<Vector2d> make_trim_loop() {
    vector<Vector2d> loop;
    for (int i = 0; i < 48; i++) {
        double t = 2.0 * 3.141592653589793 * (double)i / 48.0;
        loop.push_back(Vector2d(
            0.5 + 0.38 * std::cos(t),
            0.5 + 0.28 * std::sin(t)));
    }
    return loop;
}

static void test_trimmed_surface_sampling() {
    cout << "\n=== Trimmed B-spline surface sampling ===" << endl;
    TrimmedBSplineSurfacePatch patch;
    patch.surface = make_planar_surface();
    patch.outer_trim_polyline = make_trim_loop();
    patch.outer_trim_curve = fit_trim_curve_2d_from_polyline(
        patch.outer_trim_polyline, 12, 1e-6);
    patch.valid = true;
    patch.reason = "ok";

    MatrixXd V;
    MatrixXi F;
    MatrixXd UV;
    bool ok = sample_trimmed_bspline_surface(patch, 32, 32, V, F, &UV);
    CHECK(ok, "trimmed sampling succeeds");
    CHECK(V.rows() > 0 && F.rows() > 0, "trimmed mesh is non-empty");
    CHECK(UV.rows() == V.rows() && UV.cols() == 2, "UV output matches vertices");
    CHECK(patch.outer_trim_curve.valid, "2D trim curve fit succeeds");
    CHECK(export_trimmed_bspline_surface_debug(
              "trimmed_bspline_surface_test", patch, 32, 32),
          "debug export succeeds");
    CHECK(file_nonempty("trimmed_bspline_surface_test_sampled_surface.obj") &&
              file_nonempty("trimmed_bspline_surface_test_uv_triangulation.obj") &&
              file_nonempty("trimmed_bspline_surface_test_uv_trim_loop.obj"),
          "debug files are non-empty");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Trimmed B-spline Surface Tests" << endl;
    cout << "========================================" << endl;

    test_trimmed_surface_sampling();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;
    return g_fail;
}
