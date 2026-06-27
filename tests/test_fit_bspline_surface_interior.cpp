#include "fit_bspline_surface_interior.h"

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

static Vector3d target_surface(double u, double v) {
    double x = 4.0 * u;
    double y = 2.0 * v;
    double z = 0.10 * x + 0.05 * y + 0.35 * std::sin(3.141592653589793 * u) *
                                      std::sin(3.141592653589793 * v);
    return Vector3d(x, y, z);
}

static QuadLikeBoundary make_boundary() {
    QuadLikeBoundary boundary;
    boundary.valid = true;
    boundary.quality_score = 1.0;
    boundary.corner_loop_indices = {{0, 8, 16, 24}};
    for (int i = 0; i <= 8; i++) {
        double t = (double)i / 8.0;
        boundary.side_polylines[0].push_back(target_surface(t, 0.0));
    }
    for (int i = 0; i <= 8; i++) {
        double t = (double)i / 8.0;
        boundary.side_polylines[1].push_back(target_surface(1.0, t));
    }
    for (int i = 0; i <= 8; i++) {
        double t = (double)i / 8.0;
        boundary.side_polylines[2].push_back(target_surface(1.0 - t, 1.0));
    }
    for (int i = 0; i <= 8; i++) {
        double t = (double)i / 8.0;
        boundary.side_polylines[3].push_back(target_surface(0.0, 1.0 - t));
    }
    return boundary;
}

static vector<SurfaceFitSample> make_samples() {
    vector<SurfaceFitSample> samples;
    for (int i = 0; i <= 10; i++) {
        for (int j = 0; j <= 10; j++) {
            double u = (double)i / 10.0;
            double v = (double)j / 10.0;
            SurfaceFitSample s;
            s.uv = Eigen::Vector2d(u, v);
            s.position = target_surface(u, v);
            s.weight = (i == 0 || j == 0 || i == 10 || j == 10) ? 0.25 : 1.0;
            samples.push_back(s);
        }
    }
    return samples;
}

static bool boundary_control_points_equal(
    const BSplineSurface3D& a,
    const BSplineSurface3D& b,
    double tol) {
    int nu = (int)a.control_grid.size();
    int nv = (int)a.control_grid[0].size();
    for (int i = 0; i < nu; i++) {
        for (int j = 0; j < nv; j++) {
            if (!(i == 0 || j == 0 || i == nu - 1 || j == nv - 1)) continue;
            if ((a.control_grid[i][j] - b.control_grid[i][j]).norm() > tol) {
                return false;
            }
        }
    }
    return true;
}

static double interior_control_delta(
    const BSplineSurface3D& a,
    const BSplineSurface3D& b) {
    int nu = (int)a.control_grid.size();
    int nv = (int)a.control_grid[0].size();
    double max_delta = 0.0;
    for (int i = 1; i < nu - 1; i++) {
        for (int j = 1; j < nv - 1; j++) {
            max_delta = std::max(max_delta,
                (a.control_grid[i][j] - b.control_grid[i][j]).norm());
        }
    }
    return max_delta;
}

static void test_surface_interior_fit() {
    cout << "\n=== Surface interior control fit ===" << endl;
    QuadLikeBoundary boundary = make_boundary();
    InitialBSplineSurfaceConfig icfg;
    icfg.control_count_u = 5;
    icfg.control_count_v = 5;
    InitialBSplineSurfacePatch initial =
        build_initial_bspline_surface_from_quad_boundary(boundary, icfg);
    CHECK(initial.valid, "initial patch builds");

    vector<SurfaceFitSample> samples = make_samples();
    SurfaceInteriorFitConfig fcfg;
    fcfg.fairness_weight = 1e-6;
    fcfg.initial_weight = 1e-8;
    fcfg.enable_point_to_plane = false;
    SurfaceInteriorFitResult result =
        fit_bspline_surface_interior_control_points(initial, samples, fcfg);

    cout << "  reason=" << result.reason << endl;
    cout << "  before mean/rms/max="
         << result.before.mean_error << " / "
         << result.before.rms_error << " / "
         << result.before.max_error << endl;
    cout << "  after mean/rms/max="
         << result.after.mean_error << " / "
         << result.after.rms_error << " / "
         << result.after.max_error << endl;

    CHECK(result.valid, "interior fit succeeds");
    CHECK(result.after.rms_error < result.before.rms_error,
          "RMS error decreases");
    CHECK(result.after.mean_error < result.before.mean_error,
          "mean error decreases");
    CHECK(boundary_control_points_equal(initial.surface, result.patch.surface, 1e-12),
          "boundary control points are unchanged");
    CHECK(interior_control_delta(initial.surface, result.patch.surface) > 1e-4,
          "interior control points move");
    CHECK(!result.normal_flip_detected, "sampled surface has no normal flips");
    CHECK(!result.control_points_out_of_bounds, "control points stay near region bbox");
    CHECK(export_surface_interior_fit_debug(
              "surface_interior_fit", result, samples),
          "debug exports succeed");
    CHECK(file_nonempty("surface_interior_fit_control_net.obj") &&
              file_nonempty("surface_interior_fit_sampled_surface.obj") &&
              file_nonempty("surface_interior_fit_error_heatmap.obj") &&
              file_nonempty("surface_interior_fit_report.csv"),
          "debug export files are non-empty");
}

int main() {
    cout << "========================================" << endl;
    cout << "  B-spline Surface Interior Fit Tests" << endl;
    cout << "========================================" << endl;

    test_surface_interior_fit();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
