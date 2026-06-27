#include "initial_bspline_surface.h"

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

static Vector3d surface_like_point(double x, double y) {
    double z = 0.12 * x + 0.05 * y + 0.08 * std::sin(0.7 * x) * std::cos(0.9 * y);
    return Vector3d(x, y, z);
}

static QuadLikeBoundary make_quad_boundary() {
    QuadLikeBoundary boundary;
    boundary.valid = true;
    boundary.quality_score = 1.0;
    boundary.corner_loop_indices = {{0, 9, 18, 27}};

    for (int i = 0; i <= 9; i++) {
        double t = (double)i / 9.0;
        boundary.side_polylines[0].push_back(surface_like_point(4.0 * t, 0.0));
    }
    for (int i = 0; i <= 9; i++) {
        double t = (double)i / 9.0;
        boundary.side_polylines[1].push_back(surface_like_point(4.0, 2.0 * t));
    }
    for (int i = 0; i <= 9; i++) {
        double t = (double)i / 9.0;
        boundary.side_polylines[2].push_back(surface_like_point(4.0 * (1.0 - t), 2.0));
    }
    for (int i = 0; i <= 9; i++) {
        double t = (double)i / 9.0;
        boundary.side_polylines[3].push_back(surface_like_point(0.0, 2.0 * (1.0 - t)));
    }
    return boundary;
}

static double explicit_boundary_max_error(
    const InitialBSplineSurfacePatch& patch,
    int sample_count) {
    double max_error = 0.0;
    for (int i = 0; i < sample_count; i++) {
        double t = (double)i / (double)(sample_count - 1);
        max_error = std::max(max_error,
            (patch.surface.evaluate(t, 0.0) -
             patch.boundary_curves[0].evaluate(t)).norm());
        max_error = std::max(max_error,
            (patch.surface.evaluate(1.0, t) -
             patch.boundary_curves[1].evaluate(t)).norm());
        max_error = std::max(max_error,
            (patch.surface.evaluate(t, 1.0) -
             patch.boundary_curves[2].evaluate(1.0 - t)).norm());
        max_error = std::max(max_error,
            (patch.surface.evaluate(0.0, t) -
             patch.boundary_curves[3].evaluate(1.0 - t)).norm());
    }
    return max_error;
}

static void run_surface_case(
    const string& name,
    int control_count_u,
    int control_count_v) {
    cout << "\n=== Initial surface case: " << name << " ===" << endl;
    QuadLikeBoundary boundary = make_quad_boundary();
    InitialBSplineSurfaceConfig cfg;
    cfg.control_count_u = control_count_u;
    cfg.control_count_v = control_count_v;
    cfg.boundary_fairness_weight = 1e-7;
    cfg.coons_fit_samples_u = 14;
    cfg.coons_fit_samples_v = 13;
    cfg.boundary_check_samples = 25;

    InitialBSplineSurfacePatch patch =
        build_initial_bspline_surface_from_quad_boundary(boundary, cfg);

    cout << "  reason=" << patch.reason << endl;
    cout << "  boundary_mean_error=" << patch.boundary_mean_error
         << " boundary_max_error=" << patch.boundary_max_error << endl;
    cout << "  coons_fit_mean_error=" << patch.coons_fit_mean_error
         << " coons_fit_max_error=" << patch.coons_fit_max_error << endl;

    CHECK(patch.valid, name + ": patch builds successfully");
    CHECK(patch.surface.control_grid.size() == (size_t)control_count_u,
          name + ": surface has expected u control count");
    CHECK(patch.surface.control_grid[0].size() == (size_t)control_count_v,
          name + ": surface has expected v control count");

    double boundary_max = explicit_boundary_max_error(patch, 25);
    cout << "  explicit 100-sample boundary max=" << boundary_max << endl;
    CHECK(boundary_max < 1e-11,
          name + ": surface boundaries exactly match boundary curves");
    CHECK(patch.boundary_max_error < 1e-11,
          name + ": report boundary max is near floating point precision");

    string prefix = "initial_bspline_surface_" + name;
    CHECK(export_initial_bspline_surface_debug(prefix, patch, 12, 11),
          name + ": debug exports succeed");
    CHECK(file_nonempty(prefix + "_boundary_bottom.obj") &&
              file_nonempty(prefix + "_boundary_right.obj") &&
              file_nonempty(prefix + "_boundary_top.obj") &&
              file_nonempty(prefix + "_boundary_left.obj") &&
              file_nonempty(prefix + "_coons_surface.obj") &&
              file_nonempty(prefix + "_control_net.obj") &&
              file_nonempty(prefix + "_sampled_surface.obj") &&
              file_nonempty(prefix + "_boundary_report.csv"),
          name + ": debug export files are non-empty");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Initial B-spline Surface Tests" << endl;
    cout << "========================================" << endl;

    run_surface_case("grid4x4", 4, 4);
    run_surface_case("grid5x5", 5, 5);

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
