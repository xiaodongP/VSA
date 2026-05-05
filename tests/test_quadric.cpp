#include <iostream>
#include <cmath>
#include <string>
#include "quadric_proxy.h"

using namespace Eigen;
using namespace std;

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
    } while(0)

#define CHECK_APPROX(val, expected, tol, msg) \
    do { \
        double _v = (val), _e = (expected), _t = (tol); \
        if (abs(_v - _e) > _t) { \
            cerr << "  FAIL: " << msg << " (got " << _v \
                 << ", expected " << _e << ", tol " << _t << ")" << endl; \
            g_fail++; \
        } else { \
            cout << "  OK:   " << msg << " (" << _v << ")" << endl; \
            g_pass++; \
        } \
    } while(0)

static QuadricProxy unit_sphere() {
    // f = x^2 + y^2 + z^2 - 1
    VectorXd c = VectorXd::Zero(10);
    c(0) = -1;
    c(4) = 1;
    c(7) = 1;
    c(9) = 1;
    return QuadricProxy(c);
}

static QuadricProxy plane_z() {
    // f = z
    VectorXd c = VectorXd::Zero(10);
    c(3) = 1;
    return QuadricProxy(c);
}

// ============================================================

void test_eval() {
    cout << "\n=== Test 1: eval (unit sphere) ===" << endl;
    QuadricProxy q = unit_sphere();

    CHECK_APPROX(q.eval(Vector3d(1, 0, 0)), 0.0, 1e-12, "f(1,0,0)=0");
    CHECK_APPROX(q.eval(Vector3d(0, 1, 0)), 0.0, 1e-12, "f(0,1,0)=0");
    CHECK_APPROX(q.eval(Vector3d(0, 0, 1)), 0.0, 1e-12, "f(0,0,1)=0");

    Vector3d p111 = Vector3d(1, 1, 1) / sqrt(3.0);
    CHECK_APPROX(q.eval(p111), 0.0, 1e-12, "f(1,1,1)/sqrt(3)=0");

    CHECK_APPROX(q.eval(Vector3d(2, 0, 0)), 3.0, 1e-12, "f(2,0,0)=3");
    CHECK_APPROX(q.eval(Vector3d(0.5, 0, 0)), -0.75, 1e-12, "f(0.5,0,0)=-0.75");

    cout << "\n--- eval (plane z=0) ---" << endl;
    QuadricProxy pl = plane_z();
    CHECK_APPROX(pl.eval(Vector3d(5, -3, 0)), 0.0, 1e-12, "f(5,-3,0)=0");
    CHECK_APPROX(pl.eval(Vector3d(0, 0, 7)), 7.0, 1e-12, "f(0,0,7)=7");
}

void test_grad() {
    cout << "\n=== Test 2: grad (unit sphere) ===" << endl;
    QuadricProxy q = unit_sphere();

    Vector3d g1 = q.grad(Vector3d(1, 0, 0));
    CHECK_APPROX(g1(0), 2.0, 1e-12, "grad_x(1,0,0)=2");
    CHECK_APPROX(g1(1), 0.0, 1e-12, "grad_y(1,0,0)=0");
    CHECK_APPROX(g1(2), 0.0, 1e-12, "grad_z(1,0,0)=0");

    Vector3d g2 = q.grad(Vector3d(1, 1, 1));
    CHECK_APPROX(g2(0), 2.0, 1e-12, "grad_x(1,1,1)=2");
    CHECK_APPROX(g2(1), 2.0, 1e-12, "grad_y(1,1,1)=2");
    CHECK_APPROX(g2(2), 2.0, 1e-12, "grad_z(1,1,1)=2");

    cout << "\n--- grad (plane z) ---" << endl;
    QuadricProxy pl = plane_z();
    Vector3d g3 = pl.grad(Vector3d(5, 5, 5));
    CHECK_APPROX(g3(0), 0.0, 1e-12, "grad_x(5,5,5)=0");
    CHECK_APPROX(g3(1), 0.0, 1e-12, "grad_y(5,5,5)=0");
    CHECK_APPROX(g3(2), 1.0, 1e-12, "grad_z(5,5,5)=1");
}

void test_matrices() {
    cout << "\n=== Test 3: quadraticMatrix & homogeneousMatrix ===" << endl;
    QuadricProxy q = unit_sphere();

    // Q should be identity for unit sphere
    Matrix3d Q = q.quadraticMatrix();
    CHECK_APPROX(Q(0,0), 1.0, 1e-12, "Q(0,0)=1");
    CHECK_APPROX(Q(1,1), 1.0, 1e-12, "Q(1,1)=1");
    CHECK_APPROX(Q(2,2), 1.0, 1e-12, "Q(2,2)=1");
    CHECK_APPROX(Q(0,1), 0.0, 1e-12, "Q(0,1)=0");

    // [1,2,3] Q [1,2,3]^T = 1+4+9 = 14
    Vector3d p(1, 2, 3);
    CHECK_APPROX(p.transpose() * Q * p, 14.0, 1e-12,
                 "[1,2,3] Q [1,2,3]^T = 14");

    // H: [1,2,3,1] H [1,2,3,1]^T should equal eval
    Matrix4d H = q.homogeneousMatrix();
    Vector4d p4(1, 2, 3, 1);
    double homo_val = p4.transpose() * H * p4;
    CHECK_APPROX(homo_val, q.eval(p), 1e-12,
                 "homo eval == direct eval");

    // Also verify homogeneousMatrix for plane z
    cout << "\n--- matrices (plane z) ---" << endl;
    QuadricProxy pl = plane_z();
    Matrix4d Hpl = pl.homogeneousMatrix();
    Vector4d pz(10, 20, 0, 1);
    CHECK_APPROX(pz.transpose() * Hpl * pz, 0.0, 1e-12,
                 "plane homo at z=0 gives 0");
    Vector4d pz2(10, 20, 3, 1);
    CHECK_APPROX(pz2.transpose() * Hpl * pz2, 3.0, 1e-12,
                 "plane homo at z=3 gives 3");
}

void test_point_distance() {
    cout << "\n=== Test 4: point_distance ===" << endl;
    QuadricProxy q = unit_sphere();

    // On sphere: distance ~ 0
    double d0 = q.point_distance(Vector3d(1, 0, 0));
    CHECK_APPROX(d0, 0.0, 1e-6, "dist((1,0,0))~0");

    // (1.1,0,0): Taubin approx is close to 0.1 but not exact
    double d1 = q.point_distance(Vector3d(1.1, 0, 0));
    CHECK(d1 > 0.05 && d1 < 0.15,
          "dist((1.1,0,0)) in [0.05, 0.15] (got " + to_string(d1) + ")");

    // (2,0,0): should be positive and reasonable
    double d2 = q.point_distance(Vector3d(2, 0, 0));
    CHECK(d2 > 0.3 && d2 < 1.5,
          "dist((2,0,0)) in [0.3, 1.5] (got " + to_string(d2) + ")");

    // Inside sphere: (0.5,0,0), distance should be positive
    double d3 = q.point_distance(Vector3d(0.5, 0, 0));
    CHECK(d3 >= 0.0, "dist((0.5,0,0)) >= 0");

    cout << "\n--- point_distance (plane z) ---" << endl;
    QuadricProxy pl = plane_z();
    // Exact for planes: curvature_norm = 0
    double dp1 = pl.point_distance(Vector3d(100, 200, 5));
    CHECK_APPROX(dp1, 5.0, 1e-10, "plane dist((100,200,5))=5");
    double dp2 = pl.point_distance(Vector3d(0, 0, 0));
    CHECK_APPROX(dp2, 0.0, 1e-12, "plane dist(origin)=0");
}

void test_triangle_error() {
    cout << "\n=== Test 5: triangle_error ===" << endl;
    QuadricProxy q = unit_sphere();

    // Triangle on the sphere surface
    Vector3d v0(1, 0, 0), v1(0, 1, 0), v2(0, 0, 1);
    double err1 = q.triangle_error(v0, v1, v2);
    CHECK(err1 >= 0.0, "on-sphere error >= 0");

    // The error is small (vertices are on sphere, barycenter is slightly off)
    CHECK(err1 < 0.1,
          "on-sphere error < 0.1 (got " + to_string(err1) + ")");

    // Far-from-sphere triangle: error should be larger
    Vector3d w0(3, 0, 0), w1(0, 3, 0), w2(0, 0, 3);
    double err2 = q.triangle_error(w0, w1, w2);
    CHECK(err2 > err1,
          "off-sphere error > on-sphere error (" + to_string(err2) + " > " + to_string(err1) + ")");

    cout << "\n--- triangle_error (plane) ---" << endl;
    QuadricProxy pl = plane_z();
    Vector3d p0(0, 0, 0), p1(1, 0, 0), p2(0, 1, 0);
    double err3 = pl.triangle_error(p0, p1, p2);
    CHECK_APPROX(err3, 0.0, 1e-12, "plane triangle on z=0 has zero error");
}

void test_fitting_sphere() {
    cout << "\n=== Test 6: Taubin fitting (UV sphere) ===" << endl;

    // UV sphere mesh — breaks axis symmetry that plagued octahedron
    const int n_lon = 12, n_lat = 6;
    const int n_verts = 2 + n_lon * n_lat;

    MatrixXd V(n_verts, 3);
    V.row(0) = Vector3d(0, 0, 1);  // north pole
    V.row(1) = Vector3d(0, 0, -1); // south pole

    int idx = 2;
    for (int i = 1; i <= n_lat; i++) {
        double phi = M_PI * i / (n_lat + 1);
        for (int j = 0; j < n_lon; j++) {
            double theta = 2.0 * M_PI * j / n_lon;
            V.row(idx++) = Vector3d(sin(phi)*cos(theta),
                                    sin(phi)*sin(theta),
                                    cos(phi));
        }
    }

    vector<Vector3i> faces;

    // North cap
    for (int j = 0; j < n_lon; j++) {
        int jn = (j + 1) % n_lon;
        faces.push_back(Vector3i(0, 2 + j, 2 + jn));
    }
    // Middle bands
    for (int i = 0; i < n_lat - 1; i++) {
        for (int j = 0; j < n_lon; j++) {
            int jn = (j + 1) % n_lon;
            int v00 = 2 + i * n_lon + j;
            int v01 = 2 + i * n_lon + jn;
            int v10 = 2 + (i+1) * n_lon + j;
            int v11 = 2 + (i+1) * n_lon + jn;
            faces.push_back(Vector3i(v00, v10, v01));
            faces.push_back(Vector3i(v01, v10, v11));
        }
    }
    // South cap
    int south = 2 + (n_lat - 1) * n_lon;
    for (int j = 0; j < n_lon; j++) {
        int jn = (j + 1) % n_lon;
        faces.push_back(Vector3i(1, south + jn, south + j));
    }

    int nf = (int)faces.size();
    MatrixXi F(nf, 3);
    for (int i = 0; i < nf; i++) F.row(i) = faces[i];
    MatrixXi R = MatrixXi::Zero(nf, 1);

    QuadricProxy q = fit_quadric_region(R, 0, F, V);

    cout << "  Coefficients: ";
    for (int i = 0; i < 10; i++) cout << q.coeffs(i) << " ";
    cout << endl;

    // Residuals at known sphere points
    double r1 = abs(q.eval(Vector3d(1, 0, 0)));
    double r2 = abs(q.eval(Vector3d(0, 1, 0)));
    double r3 = abs(q.eval(Vector3d(0, 0, 1)));
    double r4 = abs(q.eval(Vector3d(-1, 0, 0)));
    Vector3d p111 = Vector3d(1, 1, 1).normalized();
    double r5 = abs(q.eval(p111));

    cout << "  Residuals: " << r1 << " " << r2 << " " << r3
         << " " << r4 << " " << r5 << endl;

    CHECK(r1 < 0.02, "res(1,0,0) < 0.02");
    CHECK(r2 < 0.02, "res(0,1,0) < 0.02");
    CHECK(r3 < 0.02, "res(0,0,1) < 0.02");
    CHECK(r4 < 0.02, "res(-1,0,0) < 0.02");
    CHECK(r5 < 0.05, "res(1,1,1)/norm < 0.05");

    double reg_err = quadric_region_error(R, 0, F, V, q);
    cout << "  Region error: " << reg_err << endl;
    CHECK(reg_err < 0.1, "region error < 0.1");

    // Verify ||coeffs|| = 1
    CHECK_APPROX(q.coeffs.norm(), 1.0, 1e-10, "||coeffs|| = 1");
}

void test_fitting_plane() {
    cout << "\n=== Test 7: Taubin fitting (flat grid -> plane z=0) ===" << endl;

    // 3x3 grid in z=0 plane
    MatrixXd V(9, 3);
    V << 0, 0, 0,
         1, 0, 0,
         2, 0, 0,
         0, 1, 0,
         1, 1, 0,
         2, 1, 0,
         0, 2, 0,
         1, 2, 0,
         2, 2, 0;

    MatrixXi F(8, 3);
    F << 0, 1, 3,
         1, 4, 3,
         1, 2, 4,
         2, 5, 4,
         3, 4, 6,
         4, 7, 6,
         4, 5, 7,
         5, 8, 7;

    MatrixXi R = MatrixXi::Zero(8, 1);

    QuadricProxy q = fit_quadric_region(R, 0, F, V);

    cout << "  Coefficients: ";
    for (int i = 0; i < 10; i++) cout << q.coeffs(i) << " ";
    cout << endl;

    // Points on z=0 plane should have eval ~0
    double r1 = abs(q.eval(Vector3d(0.5, 0.5, 0)));
    double r2 = abs(q.eval(Vector3d(1.5, 1.5, 0)));
    double r3 = abs(q.eval(Vector3d(0, 0, 0)));

    cout << "  Residuals on plane: " << r1 << " " << r2 << " " << r3 << endl;

    CHECK(r1 < 0.01, "res(0.5,0.5,0) < 0.01");
    CHECK(r2 < 0.01, "res(1.5,1.5,0) < 0.01");
    CHECK(r3 < 0.01, "res(0,0,0) < 0.01");

    // Off-plane point should have non-zero eval
    double r4 = abs(q.eval(Vector3d(0, 0, 1)));
    CHECK(r4 > 0.1, "res(0,0,1) > 0.1 (off plane)");

    double reg_err = quadric_region_error(R, 0, F, V, q);
    cout << "  Region error: " << reg_err << endl;
    CHECK(reg_err < 0.01, "region error < 0.01");

    CHECK_APPROX(q.coeffs.norm(), 1.0, 1e-10, "||coeffs|| = 1");
}

// ============================================================

int main() {
    cout << "========================================" << endl;
    cout << "  QuadricProxy Unit Tests" << endl;
    cout << "========================================" << endl;

    test_eval();
    test_grad();
    test_matrices();
    test_point_distance();
    test_triangle_error();
    test_fitting_sphere();
    test_fitting_plane();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
