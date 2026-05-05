#ifndef PROXY_VALIDITY_HEADER
#define PROXY_VALIDITY_HEADER

#include <Eigen/Dense>
#include <string>
#include <vector>
#include "quadric_proxy.h"

using namespace Eigen;
using namespace std;

enum QuadricType {
    QUADRIC_UNKNOWN = 0,
    QUADRIC_SPHERE,
    QUADRIC_ELLIPSOID,
    QUADRIC_CYLINDER,
    QUADRIC_HYPERBOLOID_ONE_SHEET,
    QUADRIC_HYPERBOLOID_TWO_SHEETS,
    QUADRIC_PARABOLOID,
    QUADRIC_GENERAL,
    QUADRIC_DEGENERATE
};

// Bitmask flags for invalid reasons
enum InvalidReason {
    VALID_OK                = 0,
    INVALID_NAN_INF         = 1,
    INVALID_COEFF_NORM      = 2,
    INVALID_H_ILL_COND      = 4,
    INVALID_Q_ILL_COND      = 8,
    INVALID_REGION_ERROR    = 16,
    INVALID_PAIR_OF_PLANES  = 32,
    INVALID_NEAR_PLANES     = 64,
    INVALID_RANK_DEFICIENT  = 128,
    INVALID_TWO_SHEET_SPAN  = 256
};

struct ProxyValidityReport {
    int proxy_id;
    bool is_valid;
    QuadricType quadric_type;
    double coeff_norm;
    double h_condition;
    double q_condition;
    double region_error;
    int invalid_reason;
    string invalid_reason_str;
    VectorXd singular_values_H;  // size 4
    Vector3d eigenvalues_Q;      // sorted ascending
    double sigma3_ratio;         // sigma3/sigma2 of H
};

struct ProxyValidityConfig {
    bool enable_basic;
    bool enable_degeneracy;
    bool enable_classification;
    bool enable_two_sheet;

    // Layer 1 thresholds
    double coeff_norm_min;       // default 1e-6
    double h_cond_max;           // default 1e12
    double q_cond_max;           // default 1e12
    double region_error_max;     // default 1e6

    // Layer 2 thresholds
    double pop_sigma_ratio;      // pair of planes: sigma3/sigma2 (default 1e-3)
    double near_pop_ratio;       // near pair of planes (default 1e-2)
    double rank_sv_min;          // min singular value for rank (default 1e-6)

    // Layer 3 thresholds
    double eig_zero_tol;         // eigenvalue zero tolerance (default 1e-6)
    double eig_equal_tol;        // eigenvalue equality tolerance (default 0.1)

    // Layer 4 thresholds
    double two_sheet_ratio;      // minority fraction threshold (default 0.1)

    ProxyValidityConfig();
};

// Full validity check for one proxy
ProxyValidityReport check_proxy_validity(const QuadricProxy& proxy,
    int proxy_id, const MatrixXi& R, const MatrixXi& F, const MatrixXd& V,
    const ProxyValidityConfig& cfg);

// Check all proxies
vector<ProxyValidityReport> check_all_proxies(
    const vector<QuadricProxy>& QP,
    const MatrixXi& R, const MatrixXi& F, const MatrixXd& V,
    const ProxyValidityConfig& cfg);

// Find face with maximum error in a region
int find_worst_face_in_region(const QuadricProxy& proxy, int region_id,
    const MatrixXi& R, const MatrixXi& F, const MatrixXd& V);

// Utility: human-readable names
string quadric_type_name(QuadricType t);
string invalid_reason_string(int reason);
void print_proxy_validity(const ProxyValidityReport& rpt);

// CSV export
void export_proxy_validity_log(const vector<ProxyValidityReport>& reports,
    const string& filename);

#endif
