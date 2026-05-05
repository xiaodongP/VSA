#ifndef BOUNDARY_SMOOTHING_HEADER
#define BOUNDARY_SMOOTHING_HEADER

#include <Eigen/Dense>
#include <string>
#include <vector>
#include "quadric_proxy.h"
#include "distance.h"

using namespace Eigen;
using namespace std;

#ifndef PROXY_TYPE_DEFINED
#define PROXY_TYPE_DEFINED
enum ProxyType { PLANE_PROXY, QUADRIC_PROXY };
#endif

struct SmoothLogEntry {
    int region_i, region_j;
    int fuzzy_count, v0_count, v1_count;
    double cut_cost;
    int relabeled_count;
    double boundary_length_before, boundary_length_after;
};

struct SmoothConfig {
    int ring;       // default 2
    double lambda;  // default 1.0
    SmoothConfig() : ring(2), lambda(1.0) {}
};

// Smooth region boundaries using graph cut.
// Modifies R in-place. Returns log entries per pair.
void smooth_boundaries(MatrixXi& R, const MatrixXi& F, const MatrixXd& V,
                       const MatrixXi& Ad, int num_proxies,
                       ProxyType proxy_type,
                       const vector<QuadricProxy>& QP,
                       const MatrixXd& Proxies, MetricMode metric,
                       const SmoothConfig& cfg,
                       vector<SmoothLogEntry>& log_out);

// Compute total boundary length for a region pair
double compute_pair_boundary_length(const MatrixXi& R, const MatrixXi& F,
                                     const MatrixXd& V, const MatrixXi& Ad,
                                     int ri, int rj);

// Export smoothing log as CSV
void export_smooth_log(const vector<SmoothLogEntry>& log, const string& filename);

#endif
