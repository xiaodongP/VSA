#ifndef VSA_BATCH_HEADER
#define VSA_BATCH_HEADER

#include <Eigen/Dense>
#include <string>
#include <vector>
#include "quadric_proxy.h"
#include "proxy_validity.h"
#include "boundary_smoothing.h"
#include "proxy_classification.h"
#include "proxy_projection.h"

using namespace Eigen;
using namespace std;

#ifndef PROXY_TYPE_DEFINED
#define PROXY_TYPE_DEFINED
enum ProxyType { PLANE_PROXY, QUADRIC_PROXY };
#endif

struct IterationStats {
    int iter;
    int num_proxies;
    double total_error;
    int changed_faces;
    double max_region_error;
    vector<int> region_face_counts;
    vector<double> region_errors; // area-normalized per region
};

struct InsertionStep {
    int step;
    int num_proxies_before;
    int split_region;
    double split_region_error;
    int seed_face;
    double seed_face_error;
    double total_error_after_lloyd;
};

struct MergeStep {
    int step;
    int region_i, region_j;
    double E_i, E_j;
    double E_t;
    double epsilon;
    bool accepted;
    string reject_reason;
};

// Run VSA in batch mode (fixed proxy count).
int run_vsa_batch(const string& model_name,
                  int num_proxies, ProxyType proxy_type,
                  int max_iter, unsigned int seed,
                  MatrixXi& R_out,
                  vector<IterationStats>& stats_out);

// Run VSA with progressive proxy insertion.
// Starts from init_proxies, inserts one proxy at a time until target reached or error threshold met.
int run_vsa_progressive(const string& model_name,
                        int init_proxies, ProxyType proxy_type,
                        int target_proxies, double error_threshold,
                        int lloyd_iter_per_insert, unsigned int seed,
                        MatrixXi& R_out,
                        vector<IterationStats>& stats_out,
                        vector<InsertionStep>& insertion_log,
                        vector<MergeStep>& merge_log,
                        vector<SmoothLogEntry>& smooth_log,
                        bool enable_merge = false,
                        bool enable_smooth = false,
                        SmoothConfig smooth_cfg = SmoothConfig(),
                        const ProxyValidityConfig& validity_cfg = ProxyValidityConfig(),
                        bool enable_classify = false,
                        double classify_eps = 0.1,
                        bool enable_projection = false,
                        ProjectionConfig proj_cfg = ProjectionConfig());

#endif
