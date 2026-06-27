#ifndef FEATURE_BARRIER_HEADER
#define FEATURE_BARRIER_HEADER

#include <Eigen/Dense>
#include <set>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>
#include <map>

using namespace Eigen;
using namespace std;

struct EdgeKey {
    int v0, v1;  // sorted: v0 < v1
    EdgeKey() : v0(-1), v1(-1) {}
    EdgeKey(int a, int b) : v0(min(a,b)), v1(max(a,b)) {}
    bool operator<(const EdgeKey& o) const {
        return v0 < o.v0 || (v0 == o.v0 && v1 < o.v1);
    }
    bool operator==(const EdgeKey& o) const {
        return v0 == o.v0 && v1 == o.v1;
    }
};

// Global feature-edge state
extern set<EdgeKey> g_feature_edges;
extern bool g_feature_barrier_enabled;
extern double g_feature_angle_threshold;

// Compute feature edges via dihedral angle threshold
// Uses Face_normal from distance.cpp
void compute_feature_edges(const MatrixXi& F, const MatrixXi& Ad,
                           double angle_threshold_deg);

// Feature pre-partition: BFS groups of faces that don't cross feature edges
VectorXi build_feature_groups(const MatrixXi& F, const MatrixXi& Ad,
                              const set<EdgeKey>& feature_edges);

// Inline hot-path: check if the shared edge between face fi and its k-th neighbor is a feature edge
inline bool is_feature_barrier(int fi, int k, const MatrixXi& F, const MatrixXi& Ad) {
    if (!g_feature_barrier_enabled || g_feature_edges.empty()) return false;
    int nb = Ad(fi, k);
    if (nb < 0 || nb >= F.rows()) return false;
    // Compute shared edge (sorted vertex pair) via vertex intersection
    int a = F(fi,0), b = F(fi,1), c = F(fi,2);
    int na = F(nb,0), nbv = F(nb,1), nc = F(nb,2);
    EdgeKey ek;
    if ((a==na||a==nbv||a==nc) && (b==na||b==nbv||b==nc))
        ek = EdgeKey(a,b);
    else if ((a==na||a==nbv||a==nc) && (c==na||c==nbv||c==nc))
        ek = EdgeKey(a,c);
    else
        ek = EdgeKey(b,c);
    return g_feature_edges.count(ek) > 0;
}

// Count feature edges on the boundary between two regions
int count_feature_edges_on_boundary(const MatrixXi& R, const MatrixXi& F,
                                     const MatrixXi& Ad, int ri, int rj);

// Get 3D points of all feature edges (for viewer visualization)
void get_feature_edge_points(const MatrixXi& F, const MatrixXd& V,
                              const set<EdgeKey>& feature_edges,
                              MatrixXd& P1, MatrixXd& P2);

// Export feature edges log
void export_feature_edges_log(const set<EdgeKey>& feature_edges,
                               const MatrixXi& F, const MatrixXd& V,
                               const string& filename);

struct FeatureBarrierEnforceReport {
    int old_region_count = 0;
    int new_region_count = 0;
    int split_region_count = 0;
    int added_region_count = 0;
    int violating_feature_edges_before = 0;
    int violating_feature_edges_after = 0;
};

struct BoundaryNormalRelabelReport {
    int boundary_face_count = 0;
    int relabeled_face_count = 0;
    int iterations = 0;
};

// Final cleanup for boundary outlier faces. Reassigns a boundary face to an
// adjacent region only when the face normal is clearly more consistent with
// that neighboring region's adjacent faces.
BoundaryNormalRelabelReport relabel_boundary_faces_by_normal(
    MatrixXi& R,
    const MatrixXi& F,
    const MatrixXi& Ad,
    int num_regions,
    double min_improvement = 0.25,
    double min_candidate_alignment = 0.75,
    int max_iterations = 2,
    bool verbose = true);

// Final hard barrier pass: split any region that spans multiple components
// separated by feature edges. May increase num_regions.
FeatureBarrierEnforceReport enforce_feature_barrier_final(
    MatrixXi& R,
    const MatrixXi& F,
    const MatrixXi& Ad,
    int& num_regions,
    bool verbose = true);

#endif
