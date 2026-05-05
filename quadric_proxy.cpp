#include "quadric_proxy.h"

// ============================================================
// Helper
// ============================================================

static double tri_area(const Vector3d& v0, const Vector3d& v1, const Vector3d& v2) {
    return 0.5 * (v1 - v0).cross(v2 - v0).norm();
}

// ============================================================
// QuadricProxy struct
// ============================================================

QuadricProxy::QuadricProxy() : coeffs(VectorXd::Zero(10)) {}

QuadricProxy::QuadricProxy(const VectorXd& c) : coeffs(c) {
    if (c.size() != 10) {
        cerr << "QuadricProxy: expected 10 coefficients, got " << c.size() << endl;
        coeffs = VectorXd::Zero(10);
    }
}

double QuadricProxy::eval(const Vector3d& p) const {
    double x = p(0), y = p(1), z = p(2);
    return coeffs(0)
         + coeffs(1)*x + coeffs(2)*y + coeffs(3)*z
         + coeffs(4)*x*x + coeffs(5)*x*y + coeffs(6)*x*z
         + coeffs(7)*y*y + coeffs(8)*y*z + coeffs(9)*z*z;
}

Vector3d QuadricProxy::grad(const Vector3d& p) const {
    double x = p(0), y = p(1), z = p(2);
    Vector3d g;
    g(0) = coeffs(1) + 2*coeffs(4)*x + coeffs(5)*y + coeffs(6)*z;
    g(1) = coeffs(2) + coeffs(5)*x + 2*coeffs(7)*y + coeffs(8)*z;
    g(2) = coeffs(3) + coeffs(6)*x + coeffs(8)*y + 2*coeffs(9)*z;
    return g;
}

Matrix3d QuadricProxy::quadraticMatrix() const {
    Matrix3d Q;
    Q << coeffs(4),     coeffs(5)/2.0, coeffs(6)/2.0,
         coeffs(5)/2.0, coeffs(7),     coeffs(8)/2.0,
         coeffs(6)/2.0, coeffs(8)/2.0, coeffs(9);
    return Q;
}

Matrix4d QuadricProxy::homogeneousMatrix() const {
    Matrix4d H;
    H << coeffs(4),     coeffs(5)/2.0, coeffs(6)/2.0, coeffs(1)/2.0,
         coeffs(5)/2.0, coeffs(7),     coeffs(8)/2.0, coeffs(2)/2.0,
         coeffs(6)/2.0, coeffs(8)/2.0, coeffs(9),     coeffs(3)/2.0,
         coeffs(1)/2.0, coeffs(2)/2.0, coeffs(3)/2.0, coeffs(0);
    return H;
}

double QuadricProxy::point_distance(const Vector3d& p) const {
    const double eps = 1e-12;

    double c4 = coeffs(4), c5 = coeffs(5), c6 = coeffs(6);
    double c7 = coeffs(7), c8 = coeffs(8), c9 = coeffs(9);

    double curvature_norm = sqrt(c4*c4 + c7*c7 + c9*c9
                                 + 0.5*(c5*c5 + c6*c6 + c8*c8));

    double f_val = abs(eval(p));
    double grad_norm = grad(p).norm();

    double t;
    if (curvature_norm < eps) {
        t = f_val / max(grad_norm, eps);
    } else {
        double disc = grad_norm*grad_norm + 4.0*curvature_norm*f_val;
        t = (-grad_norm + sqrt(max(disc, 0.0))) / (2.0*curvature_norm);
    }
    return max(t, 0.0);
}

double QuadricProxy::triangle_error(const Vector3d& v0, const Vector3d& v1,
                                    const Vector3d& v2) const {
    double area = tri_area(v0, v1, v2);
    if (area < 1e-15) return 0.0;

    Vector3d bary = (v0 + v1 + v2) / 3.0;
    double d0 = point_distance(v0);
    double d1 = point_distance(v1);
    double d2 = point_distance(v2);
    double db = point_distance(bary);

    return area * (d0*d0 + d1*d1 + d2*d2 + db*db) / 4.0;
}

// ============================================================
// Taubin fitting helpers
// ============================================================

// Monomial vector: v(p) = [1, x, y, z, x^2, xy, xz, y^2, yz, z^2]^T
static VectorXd monomial_vec(const Vector3d& p) {
    double x = p(0), y = p(1), z = p(2);
    VectorXd v(10);
    v << 1.0, x, y, z, x*x, x*y, x*z, y*y, y*z, z*z;
    return v;
}

// Gradient matrix D(p): 3x10
// D =
// [0, 1, 0, 0, 2x, y, z, 0, 0, 0]
// [0, 0, 1, 0, 0, x, 0, 2y, z, 0]
// [0, 0, 0, 1, 0, 0, x, 0, y, 2z]
static MatrixXd gradient_mat(const Vector3d& p) {
    double x = p(0), y = p(1), z = p(2);
    MatrixXd D(3, 10);
    D.row(0) << 0, 1, 0, 0, 2*x, y, z,   0,   0, 0;
    D.row(1) << 0, 0, 1, 0, 0,   x, 0,   2*y, z, 0;
    D.row(2) << 0, 0, 0, 1, 0,   0, x,   0,   y, 2*z;
    return D;
}

// ============================================================
// Taubin quadric fitting
// ============================================================

QuadricProxy fit_quadric_region(const MatrixXi& R, int region_id,
                                const MatrixXi& F, const MatrixXd& V) {
    const double eps = 1e-10;

    MatrixXd M_mat = MatrixXd::Zero(10, 10);
    MatrixXd N_mat = MatrixXd::Zero(10, 10);

    for (int f = 0; f < R.rows(); f++) {
        if (R(f, 0) != region_id) continue;

        Vector3i face = F.row(f);
        Vector3d v0 = V.row(face(0));
        Vector3d v1 = V.row(face(1));
        Vector3d v2 = V.row(face(2));

        double area = tri_area(v0, v1, v2);
        double w = area / 4.0;

        Vector3d samples[4] = {v0, v1, v2, (v0 + v1 + v2) / 3.0};

        for (int s = 0; s < 4; s++) {
            VectorXd v = monomial_vec(samples[s]);
            MatrixXd D = gradient_mat(samples[s]);
            M_mat += w * v * v.transpose();
            N_mat += w * D.transpose() * D;
        }
    }

    // Regularize N to handle near-singular cases
    MatrixXd N_reg = N_mat + eps * MatrixXd::Identity(10, 10);

    // Solve generalized eigenvalue problem: M s = lambda N s
    GeneralizedSelfAdjointEigenSolver<MatrixXd> solver(M_mat, N_reg);

    VectorXd eigenvalues = solver.eigenvalues();
    MatrixXd eigenvectors = solver.eigenvectors();

    // Eigenvalues sorted in increasing order — col(0) is the minimum
    VectorXd s = eigenvectors.col(0);

    // Validate: reject NaN/Inf
    bool valid = true;
    for (int i = 0; i < 10; i++) {
        if (isnan(s(i)) || isinf(s(i))) { valid = false; break; }
    }
    if (!valid || s.norm() < eps) {
        cerr << "fit_quadric_region: degenerate solution, returning zero proxy" << endl;
        return QuadricProxy();
    }

    // Normalize ||s|| = 1
    s.normalize();
    return QuadricProxy(s);
}

// ============================================================
// Region error (area-weighted normalized)
// ============================================================

double quadric_region_error(const MatrixXi& R, int region_id,
                            const MatrixXi& F, const MatrixXd& V,
                            const QuadricProxy& proxy) {
    double total_error = 0.0;
    double total_area = 0.0;

    for (int f = 0; f < R.rows(); f++) {
        if (R(f, 0) != region_id) continue;

        Vector3i face = F.row(f);
        Vector3d v0 = V.row(face(0));
        Vector3d v1 = V.row(face(1));
        Vector3d v2 = V.row(face(2));

        double area = tri_area(v0, v1, v2);
        total_error += proxy.triangle_error(v0, v1, v2);
        total_area += area;
    }

    if (total_area < 1e-15) return 0.0;
    return total_error / total_area;
}
