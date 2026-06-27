#include "bspline.h"

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

static const double kPi = 3.141592653589793238462643383279502884;

static bool near_vec(const Vector3d& a, const Vector3d& b, double tol) {
    return (a - b).norm() <= tol;
}

static bool file_nonempty(const string& filename) {
    std::ifstream fin(filename, std::ios::binary | std::ios::ate);
    return fin.is_open() && fin.tellg() > 0;
}

static vector<Vector3d> make_line_polyline() {
    vector<Vector3d> points;
    for (int i = 0; i < 32; i++) {
        double t = (double)i / 31.0;
        points.push_back(Vector3d(5.0 * t, -1.0 + 2.0 * t, 0.5 * t));
    }
    return points;
}

static vector<Vector3d> make_arc_polyline() {
    vector<Vector3d> points;
    for (int i = 0; i < 48; i++) {
        double t = (double)i / 47.0;
        double a = 0.15 * kPi + 0.7 * kPi * t;
        points.push_back(Vector3d(std::cos(a), std::sin(a), 0.25 * t));
    }
    return points;
}

static vector<Vector3d> make_noisy_smooth_polyline() {
    vector<Vector3d> points;
    for (int i = 0; i < 64; i++) {
        double t = (double)i / 63.0;
        double x = 4.0 * t;
        double y = std::sin(2.0 * kPi * t);
        double z = 0.4 * std::cos(1.5 * kPi * t);
        double noise = 0.015 * std::sin(31.0 * t) + 0.01 * std::cos(17.0 * t);
        points.push_back(Vector3d(x, y + noise, z - 0.5 * noise));
    }
    return points;
}

static vector<Vector3d> make_polyline_boundary() {
    vector<Vector3d> points;
    auto add_segment = [&](const Vector3d& a, const Vector3d& b, int count, bool skip_first) {
        for (int i = skip_first ? 1 : 0; i < count; i++) {
            double t = (double)i / (double)(count - 1);
            points.push_back((1.0 - t) * a + t * b);
        }
    };

    Vector3d p0(0.0, 0.0, 0.0);
    Vector3d p1(1.5, 0.0, 0.0);
    Vector3d p2(1.5, 0.8, 0.1);
    Vector3d p3(2.6, 0.8, 0.05);
    Vector3d p4(2.6, 1.8, 0.0);
    add_segment(p0, p1, 12, false);
    add_segment(p1, p2, 10, true);
    add_segment(p2, p3, 10, true);
    add_segment(p3, p4, 12, true);
    return points;
}

static bool export_fit_debug(
    const string& prefix,
    const vector<Vector3d>& polyline,
    const BSplineCurve3D& curve) {
    bool ok = true;
    ok = export_polyline_obj(prefix + "_polyline.obj", polyline) && ok;
    ok = export_bspline_curve_polyline_obj(prefix + "_fit_curve.obj", curve, 96) && ok;
    ok = export_bspline_curve_control_polygon_obj(prefix + "_control_polygon.obj", curve) && ok;
    ok = file_nonempty(prefix + "_polyline.obj") && ok;
    ok = file_nonempty(prefix + "_fit_curve.obj") && ok;
    ok = file_nonempty(prefix + "_control_polygon.obj") && ok;
    return ok;
}

static void run_case(
    const string& name,
    const vector<Vector3d>& polyline,
    int control_count,
    double fairness_weight,
    double mean_tol,
    double max_tol) {
    cout << "\n=== Fit case: " << name << " ===" << endl;

    BSplineCurveFitResult result = fit_cubic_bspline_curve_least_squares(
        polyline, control_count, fairness_weight);

    cout << "  solver: " << result.message << endl;
    cout << "  mean_error=" << result.mean_error
         << " max_error=" << result.max_error << endl;

    CHECK(result.success, name + ": fit succeeds");
    if (!result.success) return;

    CHECK(near_vec(result.curve.control_points.front(), polyline.front(), 1e-12),
          name + ": first control point is fixed");
    CHECK(near_vec(result.curve.control_points.back(), polyline.back(), 1e-12),
          name + ": last control point is fixed");
    CHECK(near_vec(result.curve.evaluate(0.0), polyline.front(), 1e-12),
          name + ": curve passes through polyline start");
    CHECK(near_vec(result.curve.evaluate(1.0), polyline.back(), 1e-12),
          name + ": curve passes through polyline end");
    CHECK(result.mean_error <= mean_tol,
          name + ": mean error under tolerance");
    CHECK(result.max_error <= max_tol,
          name + ": max error under tolerance");
    CHECK(export_fit_debug("bspline_fit_" + name, polyline, result.curve),
          name + ": debug OBJ export succeeds");
}

int main() {
    cout << "========================================" << endl;
    cout << "  B-spline Curve Fitting Tests" << endl;
    cout << "========================================" << endl;

    try {
        run_case("line", make_line_polyline(), 4, 1e-6, 1e-10, 1e-10);
        run_case("arc", make_arc_polyline(), 8, 1e-4, 0.004, 0.012);
        run_case("noisy_smooth", make_noisy_smooth_polyline(), 10, 0.05, 0.035, 0.09);
        run_case("polyline_boundary", make_polyline_boundary(), 12, 1e-5, 0.045, 0.14);
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
