#ifndef QUADRIC_PROXY_HEADER
#define QUADRIC_PROXY_HEADER

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <cassert>

using namespace Eigen;
using namespace std;

// Quadric surface proxy:
//   f(x,y,z) = C0 + C1*x + C2*y + C3*z
//            + C4*x^2 + C5*x*y + C6*x*z
//            + C7*y^2 + C8*y*z + C9*z^2
//
// Coefficients vector: coeffs = [C0, C1, C2, C3, C4, C5, C6, C7, C8, C9]^T
struct QuadricProxy {
    VectorXd coeffs;

    QuadricProxy();
    explicit QuadricProxy(const VectorXd& c);

    double eval(const Vector3d& p) const;
    Vector3d grad(const Vector3d& p) const;

    // 3x3 symmetric Q: [x y z] Q [x y z]^T = quadratic terms
    Matrix3d quadraticMatrix() const;

    // 4x4 homogeneous H: [x y z 1] H [x y z 1]^T = f(x,y,z)
    Matrix4d homogeneousMatrix() const;

    // Taubin second-order approximate distance
    double point_distance(const Vector3d& p) const;

    // Triangle error: area * mean(d0^2, d1^2, d2^2, dbary^2)
    double triangle_error(const Vector3d& v0, const Vector3d& v1,
                          const Vector3d& v2) const;
};

// Taubin quadric fitting for one region
QuadricProxy fit_quadric_region(const MatrixXi& R, int region_id,
                                const MatrixXi& F, const MatrixXd& V);

// Region error: sum(triangle errors) / sum(areas)
double quadric_region_error(const MatrixXi& R, int region_id,
                            const MatrixXi& F, const MatrixXd& V,
                            const QuadricProxy& proxy);

#endif
