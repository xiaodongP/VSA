#include "proxy_projection.h"
#include <cmath>
#include <fstream>
#include <algorithm>
#include <map>
#include <set>

static inline Vector3d v3(const Eigen::RowVectorXd& r) {
    return Vector3d(r(0), r(1), r(2));
}

// Projection result status
enum ProjStatus { PROJ_SUCCESS, PROJ_FALLBACK, PROJ_FAILURE };

// ============================================================
// Build vertex → set of regions mapping
// ============================================================

static vector<set<int>> build_vertex_regions(const MatrixXi& F, const MatrixXi& R,
                                              int num_vertices) {
    vector<set<int>> vr(num_vertices);
    for (int fi = 0; fi < F.rows(); fi++) {
        int reg = R(fi, 0);
        if (reg < 0) continue;
        for (int j = 0; j < 3; j++) {
            int vi = F(fi, j);
            if (vi >= 0 && vi < num_vertices)
                vr[vi].insert(reg);
        }
    }
    return vr;
}

// ============================================================
// Plane projection (exact orthogonal)
// ============================================================

static Vector3d project_onto_plane(const Vector3d& p, const MatrixXd& Proxies,
                                    int rid, int num_proxies) {
    Vector3d center = v3(Proxies.row(rid));
    Vector3d normal = v3(Proxies.row(num_proxies + rid));
    double d = normal.dot(p) - normal.dot(center);
    return p - d * normal;
}

// ============================================================
// Quadric projection: one Newton step (1st-order)
// ============================================================

static Vector3d quadric_first_order(const Vector3d& p, const QuadricProxy& proxy) {
    double f_val = proxy.eval(p);
    Vector3d g = proxy.grad(p);
    double g_norm_sq = g.squaredNorm();
    if (g_norm_sq < 1e-15) return p;
    return p - (f_val / g_norm_sq) * g;
}

// ============================================================
// Quadric projection: Newton iteration
// Returns projected point and status.
// ============================================================

static pair<Vector3d, ProjStatus> project_onto_quadric(
    const Vector3d& p, const QuadricProxy& proxy, ClassifiedType type,
    const ProjectionConfig& cfg)
{
    // TYPE_PLANE: one step is exact
    if (type == TYPE_PLANE) {
        Vector3d result = quadric_first_order(p, proxy);
        return {result, PROJ_SUCCESS};
    }

    // For SPHERE: use radial projection as initial guess
    // For CIRCULAR_CYLINDER: use axial+radial projection
    // For all others: start from p and use Newton

    Vector3d x = p;
    bool used_type_hint = false;

    if (type == TYPE_SPHERE) {
        Matrix3d Q = proxy.quadraticMatrix();
        Vector3d b(proxy.coeffs(1), proxy.coeffs(2), proxy.coeffs(3));
        bool invertible = false;
        Matrix3d Q_inv;
        Q.computeInverseWithCheck(Q_inv, invertible);
        if (invertible) {
            Vector3d center = -0.5 * Q_inv * b;
            Vector3d dir = p - center;
            double dir_norm = dir.norm();
            // Compute radius: f(center) + λ*R² = 0 → R = sqrt(-f(center)/λ)
            double f_center = proxy.eval(center);
            SelfAdjointEigenSolver<Matrix3d> eig(Q);
            double lambda = eig.eigenvalues()(1);  // middle eigenvalue ≈ all equal
            if (dir_norm > 1e-12 && lambda > 1e-12 && f_center < 0) {
                double R = sqrt(-f_center / lambda);
                x = center + R * dir / dir_norm;
                used_type_hint = true;
            }
        }
    }

    if (type == TYPE_CIRCULAR_CYLINDER) {
        Matrix3d Q = proxy.quadraticMatrix();
        SelfAdjointEigenSolver<Matrix3d> eig(Q);
        Vector3d evals = eig.eigenvalues();
        Matrix3d evecs = eig.eigenvectors();

        // Find zero eigenvalue index
        double max_abs = max({abs(evals(0)), abs(evals(1)), abs(evals(2))});
        int zero_idx = -1;
        for (int i = 0; i < 3; i++) {
            if (abs(evals(i)) < 0.1 * max_abs) { zero_idx = i; break; }
        }

        if (zero_idx >= 0) {
            Vector3d axis = evecs.col(zero_idx);
            Vector3d b(proxy.coeffs(1), proxy.coeffs(2), proxy.coeffs(3));
            bool invertible = false;
            Matrix3d Q_inv;
            Q.computeInverseWithCheck(Q_inv, invertible);
            if (invertible) {
                Vector3d center = -0.5 * Q_inv * b;
                // Project p onto axis
                double t = axis.dot(p - center);
                Vector3d foot = center + t * axis;
                Vector3d radial = p - foot;
                double radial_norm = radial.norm();

                // Get nonzero eigenvalue (the two equal ones)
                int nonzero_idx = (zero_idx == 0) ? 1 : 0;
                double lambda = evals(nonzero_idx);

                // Compute cylinder radius: f(foot + perp) = 0
                // perp is a unit vector perpendicular to axis
                // f(foot + r*perp) = λ*r² + f(foot) = 0 → r = sqrt(-f(foot)/λ)
                double f_foot = proxy.eval(foot);
                if (lambda > 1e-12 && f_foot < 0 && radial_norm > 1e-12) {
                    double R = sqrt(-f_foot / lambda);
                    x = foot + R * radial / radial_norm;
                    used_type_hint = true;
                }
            }
        }
    }

    // Newton iteration
    bool converged = false;
    for (int iter = 0; iter < cfg.max_newton_iter; iter++) {
        double f_val = proxy.eval(x);
        if (abs(f_val) < cfg.newton_tol) {
            converged = true;
            break;
        }
        Vector3d g = proxy.grad(x);
        double g_norm_sq = g.squaredNorm();
        if (g_norm_sq < 1e-15) break;  // degenerate gradient

        Vector3d step = -(f_val / g_norm_sq) * g;
        double step_norm = step.norm();
        if (step_norm > cfg.newton_step_max)
            step *= cfg.newton_step_max / step_norm;
        x += step;
    }

    if (converged) return {x, PROJ_SUCCESS};

    // Check final residual
    if (abs(proxy.eval(x)) < cfg.newton_tol * 100) {
        return {x, PROJ_SUCCESS};
    }

    // Fallback: 1st-order projection from original point
    Vector3d fallback = quadric_first_order(p, proxy);
    if (abs(proxy.eval(fallback)) < abs(proxy.eval(p))) {
        return {fallback, PROJ_FALLBACK};
    }

    // Complete failure: keep original
    return {p, PROJ_FAILURE};
}

// ============================================================
// Project all vertices
// ============================================================

MatrixXd project_vertices(const MatrixXd& V, const MatrixXi& F,
                           const MatrixXi& R, int num_proxies,
                           ProxyType proxy_type,
                           const vector<QuadricProxy>& QP,
                           const MatrixXd& Proxies,
                           const vector<ClassifiedType>& proxy_types,
                           const ProjectionConfig& cfg,
                           ProjectionLog& log_out)
{
    int nv = V.rows();
    MatrixXd V_out = V;  // start with original positions

    log_out = ProjectionLog();
    log_out.total_vertices = nv;

    // Build vertex → regions mapping
    auto vertex_regions = build_vertex_regions(F, R, nv);

    for (int vi = 0; vi < nv; vi++) {
        const auto& regions = vertex_regions[vi];
        if (regions.empty()) continue;

        if (regions.size() == 1)
            log_out.interior_count++;
        else
            log_out.boundary_count++;

        Vector3d p = v3(V.row(vi));
        Vector3d projected = Vector3d::Zero();
        bool any_success = false;

        for (int rid : regions) {
            if (rid < 0 || rid >= num_proxies) continue;

            Vector3d pt;
            ProjStatus status;

            if (proxy_type == PLANE_PROXY) {
                pt = project_onto_plane(p, Proxies, rid, num_proxies);
                status = PROJ_SUCCESS;
            } else {
                ClassifiedType ct = (rid < (int)proxy_types.size())
                                    ? proxy_types[rid] : TYPE_GENERAL_QUADRIC;
                auto result = project_onto_quadric(p, QP[rid], ct, cfg);
                pt = result.first;
                status = result.second;
            }

            if (status != PROJ_FAILURE) any_success = true;
            projected += pt;
        }

        if (!regions.empty()) {
            projected /= (double)regions.size();
            V_out.row(vi) = projected.transpose();
        }

        if (any_success)
            log_out.success_count++;
        else
            log_out.failure_count++;
    }

    return V_out;
}

// ============================================================
// OBJ export
// ============================================================

void export_obj(const MatrixXd& V, const MatrixXi& F, const string& filename) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Cannot write " << filename << endl;
        return;
    }
    fout << "# Projected mesh from VSA" << endl;
    fout << "# " << V.rows() << " vertices, " << F.rows() << " faces" << endl;

    for (int i = 0; i < V.rows(); i++) {
        fout << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << endl;
    }
    for (int i = 0; i < F.rows(); i++) {
        // OBJ uses 1-indexed vertices
        fout << "f " << (F(i, 0) + 1) << " " << (F(i, 1) + 1) << " " << (F(i, 2) + 1) << endl;
    }
    fout.close();
    cout << "Exported OBJ: " << filename << " (" << V.rows() << " verts, "
         << F.rows() << " faces)" << endl;
}

// ============================================================
// Projection log export
// ============================================================

void export_projection_log(const ProjectionLog& log, int num_proxies,
                            const string& filename) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Cannot write " << filename << endl;
        return;
    }
    fout << "total_vertices,interior_count,boundary_count,"
         << "success_count,fallback_count,failure_count,num_proxies" << endl;
    fout << log.total_vertices << ","
         << log.interior_count << ","
         << log.boundary_count << ","
         << log.success_count << ","
         << log.fallback_count << ","
         << log.failure_count << ","
         << num_proxies << endl;
    fout.close();
    cout << "Exported projection log: " << filename << endl;
}
