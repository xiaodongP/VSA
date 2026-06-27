#ifndef TRIMMED_REGION_INPUT_HEADER
#define TRIMMED_REGION_INPUT_HEADER

#include "feature_barrier.h"

#include <Eigen/Dense>

#include <set>
#include <string>
#include <vector>

struct RegionFaceSet {
    int region_id = -1;
    std::vector<int> face_ids;
};

struct DirectedBoundaryEdge {
    int from = -1;
    int to = -1;
    EdgeKey key;
    int region_face_id = -1;
    int adjacent_region_id = -1;
    bool is_mesh_boundary = false;
    bool is_feature_barrier = false;
    bool is_user_marker = false;
};

struct AuthoritativeBoundaryLoop {
    int id = -1;
    bool is_perimeter = false;
    bool closed = false;
    std::vector<int> vertex_ids;
    std::vector<Eigen::Vector3d> positions;
    std::vector<DirectedBoundaryEdge> directed_edges;
    double projected_abs_area = 0.0;
};

struct BoundarySegment {
    int id = -1;
    int loop_id = -1;
    int adjacent_region_id = -1;
    std::vector<int> authoritative_vertex_ids;
    std::vector<Eigen::Vector3d> authoritative_positions;
    std::vector<Eigen::Vector3d> guide_positions;
    std::vector<EdgeKey> edge_keys;
    double length = 0.0;
    Eigen::Vector3d tangent_begin = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangent_end = Eigen::Vector3d::Zero();
    bool touches_feature_barrier = false;
    bool touches_mesh_boundary = false;
    bool touches_user_marker = false;
};

struct BoundarySegmentationConfig {
    std::set<EdgeKey> feature_edges;
    std::set<EdgeKey> marker_edges;
    std::set<int> marker_vertex_ids;
    int guide_smoothing_iterations = 0;
    double guide_smoothing_weight = 0.25;
};

struct BoundarySegmentationResult {
    RegionFaceSet region;
    std::vector<AuthoritativeBoundaryLoop> loops;
    int perimeter_loop_index = -1;
    std::vector<BoundarySegment> perimeter_segments;
    bool valid = false;
    std::string reason;
    bool region_connected = false;
    bool manifold = false;
    bool orientation_valid = false;
    bool boundary_coverage_valid = false;
};

BoundarySegmentationResult build_trimmed_region_input(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& face_region_ids,
    int target_region_id,
    const BoundarySegmentationConfig& config = BoundarySegmentationConfig());

BoundarySegmentationResult build_trimmed_region_input_from_face_set(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& face_region_ids,
    const RegionFaceSet& region,
    const BoundarySegmentationConfig& config = BoundarySegmentationConfig());

bool export_trimmed_region_input_debug_obj(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const BoundarySegmentationResult& result);

#endif
