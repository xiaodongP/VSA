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

    // Validity-guided fields
    string split_mode;               // "max_error", "invalid_proxy", "suspicious_proxy"
    string invalid_reason;
    string detected_quadric_type;
    int validity_priority;
    double condition_H;
    double condition_Q;
    double max_region_error_before;
    double max_region_error_after;
    int invalid_proxy_count_before;
    int invalid_proxy_count_after;
    string stop_reason;

    InsertionStep() : step(0), num_proxies_before(0), split_region(-1),
        split_region_error(0), seed_face(-1), seed_face_error(0),
        total_error_after_lloyd(0), split_mode("max_error"),
        validity_priority(0), condition_H(0), condition_Q(0),
        max_region_error_before(0), max_region_error_after(0),
        invalid_proxy_count_before(0), invalid_proxy_count_after(0) {}
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
                        ProjectionConfig proj_cfg = ProjectionConfig(),
                        bool validity_guided = false,
                        int max_validity_split_attempts = 20,
                        int min_faces_to_split = 4,
                        bool export_validity_each_step = false);

// Run merge pass: iteratively merge adjacent region pairs.
// Returns number of merged pairs.
int run_merge_pass(MatrixXi& R, vector<QuadricProxy>& QP, MatrixXd& Proxies,
                   int& num_proxies, ProxyType proxy_type, MetricMode metric,
                   const MatrixXi& F, const MatrixXd& V, const MatrixXi& Ad,
                   double relative_threshold, int max_iterations);

#endif
