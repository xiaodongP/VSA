#include "proxy_validity.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

// Helper: row vector to Vector3d
static inline Vector3d v3(const Eigen::RowVectorXd& r) {
    return Vector3d(r(0), r(1), r(2));
}

// ============================================================
// ProxyValidityConfig defaults
// ============================================================

ProxyValidityConfig::ProxyValidityConfig() {
    enable_basic = false;
    enable_degeneracy = false;
    enable_classification = false;
    enable_two_sheet = false;

    coeff_norm_min   = 1e-6;
    h_cond_max       = 1e12;
    q_cond_max       = 1e12;
    region_error_max = 1e6;

    pop_sigma_ratio  = 1e-3;
    near_pop_ratio   = 1e-2;
    rank_sv_min      = 1e-6;

    eig_zero_tol     = 1e-6;
    eig_equal_tol    = 0.1;

    two_sheet_ratio  = 0.1;
}

// ============================================================
// Utility functions
// ============================================================

string quadric_type_name(QuadricType t) {
    switch (t) {
        case QUADRIC_UNKNOWN:               return "UNKNOWN";
        case QUADRIC_SPHERE:                return "SPHERE";
        case QUADRIC_ELLIPSOID:             return "ELLIPSOID";
        case QUADRIC_CYLINDER:              return "CYLINDER";
        case QUADRIC_HYPERBOLOID_ONE_SHEET:  return "HYPERBOLOID_ONE_SHEET";
        case QUADRIC_HYPERBOLOID_TWO_SHEETS: return "HYPERBOLOID_TWO_SHEETS";
        case QUADRIC_PARABOLOID:            return "PARABOLOID";
        case QUADRIC_GENERAL:               return "GENERAL_QUADRIC";
        case QUADRIC_DEGENERATE:            return "DEGENERATE";
        default:                            return "UNKNOWN";
    }
}

string invalid_reason_string(int reason) {
    if (reason == VALID_OK) return "";
    vector<string> parts;
    if (reason & INVALID_NAN_INF)        parts.push_back("nan_inf");
    if (reason & INVALID_COEFF_NORM)     parts.push_back("coeff_norm_too_small");
    if (reason & INVALID_H_ILL_COND)     parts.push_back("h_ill_conditioned");
    if (reason & INVALID_Q_ILL_COND)     parts.push_back("q_ill_conditioned");
    if (reason & INVALID_REGION_ERROR)   parts.push_back("region_error_too_large");
    if (reason & INVALID_PAIR_OF_PLANES) parts.push_back("pair_of_planes");
    if (reason & INVALID_NEAR_PLANES)    parts.push_back("near_pair_of_planes");
    if (reason & INVALID_RANK_DEFICIENT) parts.push_back("rank_deficient");
    if (reason & INVALID_TWO_SHEET_SPAN) parts.push_back("two_sheet_span");
    string s;
    for (size_t i = 0; i < parts.size(); i++) {
        if (i > 0) s += "; ";
        s += parts[i];
    }
    return s;
}

void print_proxy_validity(const ProxyValidityReport& rpt) {
    cout << "  proxy " << rpt.proxy_id
         << ": type=" << quadric_type_name(rpt.quadric_type)
         << " ||c||=" << rpt.coeff_norm
         << " cond_H=" << rpt.h_condition
         << " cond_Q=" << rpt.q_condition;
    if (!rpt.is_valid) {
        cout << " INVALID: " << rpt.invalid_reason_str;
    }
    cout << endl;
}

// ============================================================
// Layer 1: Basic numerical validity
// Returns bitmask of InvalidReason flags. Fills diagnostic outputs.
// ============================================================

static int check_basic_validity(const QuadricProxy& proxy,
                                 double region_error,
                                 const ProxyValidityConfig& cfg,
                                 double& out_coeff_norm,
                                 double& out_h_cond,
                                 double& out_q_cond,
                                 VectorXd& out_sv_H) {
    int reason = VALID_OK;
    out_coeff_norm = 0;
    out_h_cond = 0;
    out_q_cond = 0;
    out_sv_H = VectorXd::Zero(4);

    // NaN/Inf check
    bool has_bad = false;
    for (int i = 0; i < proxy.coeffs.size(); i++) {
        if (isnan(proxy.coeffs(i)) || isinf(proxy.coeffs(i))) {
            has_bad = true;
            break;
        }
    }
    if (has_bad) reason |= INVALID_NAN_INF;

    // Coefficient norm
    out_coeff_norm = proxy.coeffs.norm();
    if (out_coeff_norm < cfg.coeff_norm_min) reason |= INVALID_COEFF_NORM;

    // SVD of homogeneous matrix H
    Matrix4d H = proxy.homogeneousMatrix();
    JacobiSVD<Matrix4d> svd_H(H);
    out_sv_H = svd_H.singularValues();
    double sv_max_H = out_sv_H(0);
    double sv_min_H = out_sv_H(3);
    if (sv_min_H > 1e-15) {
        out_h_cond = sv_max_H / sv_min_H;
    } else {
        out_h_cond = 1e18;  // effectively infinite
    }
    if (out_h_cond > cfg.h_cond_max) reason |= INVALID_H_ILL_COND;

    // SVD of quadratic matrix Q
    Matrix3d Q = proxy.quadraticMatrix();
    JacobiSVD<Matrix3d> svd_Q(Q);
    VectorXd sv_Q = svd_Q.singularValues();
    double sv_max_Q = sv_Q(0);
    double sv_min_Q = sv_Q(2);
    if (sv_min_Q > 1e-15) {
        out_q_cond = sv_max_Q / sv_min_Q;
    } else {
        out_q_cond = 1e18;
    }
    if (out_q_cond > cfg.q_cond_max) reason |= INVALID_Q_ILL_COND;

    // Region error
    if (region_error > cfg.region_error_max) reason |= INVALID_REGION_ERROR;

    return reason;
}

// ============================================================
// Layer 2: Degeneracy detection
// Uses singular values of H from Layer 1.
// ============================================================

static int check_degeneracy(const VectorXd& sv_H,
                             const ProxyValidityConfig& cfg) {
    int reason = VALID_OK;
    // sv_H sorted descending: sv_H(0) >= sv_H(1) >= sv_H(2) >= sv_H(3)
    double sigma2 = sv_H(1);
    double sigma3 = sv_H(2);
    double sigma4 = sv_H(3);

    // Pair of planes: sigma4/sigma3 very small, sigma3 not negligible
    if (sigma2 > cfg.rank_sv_min) {
        double ratio = (sigma3 > 1e-15) ? sigma4 / sigma3 : 1e18;
        if (ratio < cfg.pop_sigma_ratio) {
            reason |= INVALID_PAIR_OF_PLANES;
        } else if (ratio < cfg.near_pop_ratio) {
            reason |= INVALID_NEAR_PLANES;
        }
    }

    // Rank-deficient: smallest singular value near zero
    if (sigma4 < cfg.rank_sv_min) {
        reason |= INVALID_RANK_DEFICIENT;
    }

    return reason;
}

// ============================================================
// Layer 3: Quadric type classification
// ============================================================

static QuadricType classify_quadric(const QuadricProxy& proxy,
                                     const ProxyValidityConfig& cfg,
                                     Vector3d& out_eigenvalues) {
    Matrix3d Q = proxy.quadraticMatrix();
    SelfAdjointEigenSolver<Matrix3d> eig(Q);

    if (eig.info() != Success) return QUADRIC_UNKNOWN;

    // Eigenvalues sorted ascending
    Vector3d evals = eig.eigenvalues();
    out_eigenvalues = evals;

    // Tolerance: relative to largest absolute eigenvalue
    double max_abs = max({abs(evals(0)), abs(evals(1)), abs(evals(2))});
    if (max_abs < 1e-15) return QUADRIC_DEGENERATE;  // all zero
    double tol = cfg.eig_zero_tol * max_abs;

    // Count zero eigenvalues
    int n_zero = 0;
    int zero_idx = -1;
    for (int i = 0; i < 3; i++) {
        if (abs(evals(i)) < tol) {
            n_zero++;
            zero_idx = i;
        }
    }

    // Count signs among nonzero
    int n_pos = 0, n_neg = 0;
    for (int i = 0; i < 3; i++) {
        if (evals(i) >= tol) n_pos++;
        else if (evals(i) <= -tol) n_neg++;
    }

    if (n_zero >= 2) return QUADRIC_DEGENERATE;

    if (n_zero == 1) {
        // One zero eigenvalue: cylinder or paraboloid
        // Check linear coefficient in the zero-eigenvalue direction
        Vector3d e_zero = eig.eigenvectors().col(zero_idx);
        Vector3d linear(proxy.coeffs(1), proxy.coeffs(2), proxy.coeffs(3));
        double lin_proj = abs(e_zero.dot(linear));
        if (lin_proj > tol) {
            return QUADRIC_PARABOLOID;
        }
        return QUADRIC_CYLINDER;
    }

    // n_zero == 0: all eigenvalues nonzero
    if (n_pos == 3 || n_neg == 3) {
        // All same sign: sphere or ellipsoid
        double min_abs = min({abs(evals(0)), abs(evals(1)), abs(evals(2))});
        double max_ratio = max_abs / min_abs;
        if (max_ratio < (1.0 + cfg.eig_equal_tol)) {
            return QUADRIC_SPHERE;
        }
        return QUADRIC_ELLIPSOID;
    }

    // Mixed signs: one-sheet or two-sheet hyperboloid
    // Determine minority sign
    int minority_sign = (n_pos == 1) ? 1 : -1;

    // Compute center: c = -0.5 * Q^{-1} * [C1,C2,C3]
    Vector3d b(proxy.coeffs(1), proxy.coeffs(2), proxy.coeffs(3));
    Vector3d center;
    bool invertible = false;
    Matrix3d Q_inv;
    Q.computeInverseWithCheck(Q_inv, invertible);
    if (!invertible) return QUADRIC_GENERAL;

    center = -0.5 * Q_inv * b;

    // Evaluate f at center: f(c) = C0 + b^T c + c^T Q c
    double f_center = proxy.coeffs(0) + b.dot(center) + center.transpose() * Q * center;

    // Two-sheet: f(center) has same sign as minority eigenvalue
    if ((minority_sign > 0 && f_center > 0) || (minority_sign < 0 && f_center < 0)) {
        return QUADRIC_HYPERBOLOID_TWO_SHEETS;
    }
    return QUADRIC_HYPERBOLOID_ONE_SHEET;
}

// ============================================================
// Layer 4: Two-sheet hyperboloid span check
// ============================================================

static bool check_two_sheet_span(const QuadricProxy& proxy,
                                  int region_id,
                                  const MatrixXi& R, const MatrixXi& F,
                                  const MatrixXd& V,
                                  const Vector3d& eigenvalues_Q,
                                  const ProxyValidityConfig& cfg) {
    Matrix3d Q = proxy.quadraticMatrix();
    SelfAdjointEigenSolver<Matrix3d> eig(Q);
    if (eig.info() != Success) return false;

    Vector3d evals = eig.eigenvalues();
    double max_abs = max({abs(evals(0)), abs(evals(1)), abs(evals(2))});
    if (max_abs < 1e-15) return false;
    double tol = cfg.eig_zero_tol * max_abs;

    // Find minority sign eigenvalue index
    int n_pos = 0, n_neg = 0;
    for (int i = 0; i < 3; i++) {
        if (evals(i) >= tol) n_pos++;
        else if (evals(i) <= -tol) n_neg++;
    }
    int minority_idx = -1;
    if (n_pos == 1) {
        for (int i = 0; i < 3; i++)
            if (evals(i) >= tol) { minority_idx = i; break; }
    } else if (n_neg == 1) {
        for (int i = 0; i < 3; i++)
            if (evals(i) <= -tol) { minority_idx = i; break; }
    }
    if (minority_idx < 0) return false;

    // Compute center
    Vector3d b(proxy.coeffs(1), proxy.coeffs(2), proxy.coeffs(3));
    bool invertible = false;
    Matrix3d Q_inv;
    Q.computeInverseWithCheck(Q_inv, invertible);
    if (!invertible) return false;

    Vector3d center = -0.5 * Q_inv * b;

    // Separating eigenvector
    Vector3d e_sep = eig.eigenvectors().col(minority_idx);
    // Rotation matrix (eigenvectors as columns)
    Matrix3d E = eig.eigenvectors();

    // Count samples on each side
    int n_positive = 0, n_negative = 0;
    for (int i = 0; i < F.rows(); i++) {
        if (R(i, 0) != region_id) continue;
        Vector3d centroid = (v3(V.row(F(i, 0))) + v3(V.row(F(i, 1))) + v3(V.row(F(i, 2)))) / 3.0;
        Vector3d canonical = E.transpose() * (centroid - center);
        double sep_coord = canonical(minority_idx);
        if (sep_coord > 0) n_positive++;
        else if (sep_coord < 0) n_negative++;
    }

    int total = n_positive + n_negative;
    if (total == 0) return false;

    double minority_fraction = (double)min(n_positive, n_negative) / (double)total;
    return minority_fraction > cfg.two_sheet_ratio;
}

// ============================================================
// Region error computation (reused from quadric_proxy)
// ============================================================

static double compute_region_error(const QuadricProxy& proxy, int region_id,
                                    const MatrixXi& R, const MatrixXi& F,
                                    const MatrixXd& V) {
    double total_err = 0, total_area = 0;
    for (int i = 0; i < F.rows(); i++) {
        if (R(i, 0) != region_id) continue;
        Vector3i f = F.row(i);
        Vector3d v0 = v3(V.row(f(0)));
        Vector3d v1 = v3(V.row(f(1)));
        Vector3d v2 = v3(V.row(f(2)));
        double area = 0.5 * (v1 - v0).cross(v2 - v0).norm();
        total_err += proxy.triangle_error(v0, v1, v2);
        total_area += area;
    }
    if (total_area < 1e-15) return 0;
    return total_err / total_area;
}

// ============================================================
// Main orchestrator
// ============================================================

ProxyValidityReport check_proxy_validity(const QuadricProxy& proxy,
    int proxy_id, const MatrixXi& R, const MatrixXi& F, const MatrixXd& V,
    const ProxyValidityConfig& cfg)
{
    ProxyValidityReport rpt;
    rpt.proxy_id = proxy_id;
    rpt.is_valid = true;
    rpt.invalid_reason = VALID_OK;
    rpt.quadric_type = QUADRIC_UNKNOWN;
    rpt.coeff_norm = 0;
    rpt.h_condition = 0;
    rpt.q_condition = 0;
    rpt.region_error = 0;
    rpt.sigma3_ratio = 0;
    rpt.singular_values_H = VectorXd::Zero(4);
    rpt.eigenvalues_Q = Vector3d::Zero(3);

    // Compute region error
    rpt.region_error = compute_region_error(proxy, proxy_id, R, F, V);

    // Layer 1: Basic numerical validity
    if (cfg.enable_basic) {
        int basic_reason = check_basic_validity(proxy, rpt.region_error, cfg,
            rpt.coeff_norm, rpt.h_condition, rpt.q_condition, rpt.singular_values_H);

        // Compute sigma3_ratio for diagnostics
        if (rpt.singular_values_H(2) > 1e-15) {
            rpt.sigma3_ratio = rpt.singular_values_H(3) / rpt.singular_values_H(2);
        }
        rpt.invalid_reason |= basic_reason;
    }

    // Layer 2: Degeneracy detection
    if (cfg.enable_degeneracy && rpt.singular_values_H.size() == 4) {
        int deg_reason = check_degeneracy(rpt.singular_values_H, cfg);
        rpt.invalid_reason |= deg_reason;
    }

    // Layer 3: Classification
    if (cfg.enable_classification) {
        rpt.quadric_type = classify_quadric(proxy, cfg, rpt.eigenvalues_Q);
    }

    // Layer 4: Two-sheet span check
    if (cfg.enable_two_sheet &&
        (rpt.quadric_type == QUADRIC_HYPERBOLOID_TWO_SHEETS)) {
        bool spans = check_two_sheet_span(proxy, proxy_id, R, F, V,
            rpt.eigenvalues_Q, cfg);
        if (spans) rpt.invalid_reason |= INVALID_TWO_SHEET_SPAN;
    }

    // Finalize
    rpt.is_valid = (rpt.invalid_reason == VALID_OK);
    rpt.invalid_reason_str = invalid_reason_string(rpt.invalid_reason);

    return rpt;
}

vector<ProxyValidityReport> check_all_proxies(
    const vector<QuadricProxy>& QP,
    const MatrixXi& R, const MatrixXi& F, const MatrixXd& V,
    const ProxyValidityConfig& cfg)
{
    vector<ProxyValidityReport> reports;
    for (int i = 0; i < (int)QP.size(); i++) {
        reports.push_back(check_proxy_validity(QP[i], i, R, F, V, cfg));
    }
    return reports;
}

// ============================================================
// Find worst face in region
// ============================================================

int find_worst_face_in_region(const QuadricProxy& proxy, int region_id,
    const MatrixXi& R, const MatrixXi& F, const MatrixXd& V) {
    int worst_face = -1;
    double worst_err = -1;
    for (int i = 0; i < F.rows(); i++) {
        if (R(i, 0) != region_id) continue;
        Vector3i f = F.row(i);
        double err = proxy.triangle_error(v3(V.row(f(0))), v3(V.row(f(1))), v3(V.row(f(2))));
        if (err > worst_err) {
            worst_err = err;
            worst_face = i;
        }
    }
    return worst_face;
}

// ============================================================
// CSV export
// ============================================================

void export_proxy_validity_log(const vector<ProxyValidityReport>& reports,
    const string& filename) {
    ofstream fout(filename);
    if (!fout.is_open()) { cerr << "Cannot write " << filename << endl; return; }

    fout << "proxy_id,is_valid,quadric_type,coeff_norm,h_condition,"
         << "q_condition,region_error,invalid_reason,"
         << "sigma_H_0,sigma_H_1,sigma_H_2,sigma_H_3,"
         << "lambda_Q_0,lambda_Q_1,lambda_Q_2,sigma3_ratio" << endl;

    for (const auto& r : reports) {
        fout << r.proxy_id << ","
             << (r.is_valid ? "true" : "false") << ","
             << quadric_type_name(r.quadric_type) << ","
             << r.coeff_norm << ","
             << r.h_condition << ","
             << r.q_condition << ","
             << r.region_error << ","
             << "\"" << r.invalid_reason_str << "\"";
        for (int i = 0; i < 4; i++)
            fout << "," << r.singular_values_H(i);
        for (int i = 0; i < 3; i++)
            fout << "," << r.eigenvalues_Q(i);
        fout << "," << r.sigma3_ratio << endl;
    }
    fout.close();
    cout << "Exported proxy validity log: " << filename << endl;
}
