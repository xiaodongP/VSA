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
    int pass = 0;
    int region_i = -1, region_j = -1;
    int boundary_seed_count = 0;
    int fuzzy_count = 0, v0_count = 0, v1_count = 0;
    int graph_nodes = 0, graph_edges = 0;
    double cut_cost = 0.0;
    int relabeled_count = 0;
    int relabeled_i_to_j = 0;
    int relabeled_j_to_i = 0;
    int feature_fixed_i = 0;
    int feature_fixed_j = 0;
    double boundary_length_before = 0.0;
    double boundary_length_after = 0.0;
    double region_normal_angle_deg = 0.0;
    int feature_barrier_edge_count = 0;
    bool skipped_sharp = false;
    bool skipped_empty = false;
    bool skipped_constraints = false;
    bool skipped_empty_region = false;
    bool rejected_boundary_length_increase = false;
};

struct BoundarySmoothingResult {
    int region_pairs_processed = 0;
    int sharp_pairs_skipped = 0;
    int total_relabeled_faces = 0;
    int relabeled_i_to_j = 0;
    int relabeled_j_to_i = 0;
    int boundary_edge_count_before = 0;
    int boundary_edge_count_after = 0;
    double boundary_length_before = 0.0;
    double boundary_length_after = 0.0;
    int regions_becoming_empty = 0;
    int disconnected_tiny_components = 0;
    vector<SmoothLogEntry> pair_logs;
};

struct BoundarySmoothingOptions {
    int ring;                         // k-ring fuzzy band size
    int ringSize;                     // alias kept for the paper-style option name
    double lambda;
    double sharpAngleThresholdDeg;
    bool skipSharpBoundary;
    int maxPasses;
    double infCost;
    bool refitProxiesAfterSmoothing;
    bool enableDebugOutput;
    bool protectFeatureBoundaryFaces;
    bool rejectBoundaryLengthIncrease;

    BoundarySmoothingOptions()
        : ring(1),
          ringSize(1),
          lambda(1.0),
          sharpAngleThresholdDeg(45.0),
          skipSharpBoundary(true),
          maxPasses(1),
          infCost(1e12),
          refitProxiesAfterSmoothing(true),
          enableDebugOutput(true),
          protectFeatureBoundaryFaces(true),
          rejectBoundaryLengthIncrease(true) {}
};

using SmoothConfig = BoundarySmoothingOptions;

// Smooth region boundaries using graph cut.
// Modifies R in-place. Returns log entries per pair.
BoundarySmoothingResult smooth_boundaries(MatrixXi& R, const MatrixXi& F,
                       const MatrixXd& V, const MatrixXi& Ad, int num_proxies,
                       ProxyType proxy_type,
                       vector<QuadricProxy>& QP,
                       MatrixXd& Proxies, MetricMode metric,
                       const SmoothConfig& cfg,
                       vector<SmoothLogEntry>& log_out);

// Compute total boundary length for a region pair
double compute_pair_boundary_length(const MatrixXi& R, const MatrixXi& F,
                                     const MatrixXd& V, const MatrixXi& Ad,
                                     int ri, int rj);

// Export smoothing log as CSV
void export_smooth_log(const vector<SmoothLogEntry>& log, const string& filename);

#endif
