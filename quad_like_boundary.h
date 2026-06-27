#ifndef QUAD_LIKE_BOUNDARY_HEADER
#define QUAD_LIKE_BOUNDARY_HEADER

#include "region_boundary.h"

#include <Eigen/Dense>

#include <array>
#include <string>
#include <vector>

struct QuadLikeBoundary {
    std::array<int, 4> corner_loop_indices;
    std::array<std::vector<Eigen::Vector3d>, 4> side_polylines;
    double quality_score;
    bool valid;

    QuadLikeBoundary();
};

struct QuadLikeBoundaryConfig {
    int max_corner_candidates;
    double min_corner_separation_fraction;
    double corner_angle_weight;
    double length_balance_weight;
    double opposite_parallel_weight;
    double adjacent_orthogonal_weight;
    double side_bending_weight;
    double min_quality_score;
    bool use_manual_corners;
    std::array<int, 4> manual_corner_loop_indices;

    QuadLikeBoundaryConfig();
};

struct QuadLikeBoundaryCandidate {
    int loop_index;
    double turn_angle;
    double score;
    Eigen::Vector2d pca_position;
};

struct QuadLikeBoundaryDebugInfo {
    std::vector<Eigen::Vector2d> pca_positions;
    std::vector<QuadLikeBoundaryCandidate> candidates;
    std::array<int, 4> selected_corner_loop_indices;
    std::string reason;
};

struct QuadLikeBoundaryResult {
    QuadLikeBoundary boundary;
    QuadLikeBoundaryDebugInfo debug;
    bool used_manual_fallback;
    bool success;
    std::string reason;

    QuadLikeBoundaryResult();
};

bool load_quad_like_boundary_manual_config(
    const std::string& filename,
    std::array<int, 4>& corner_loop_indices,
    std::string& reason);

QuadLikeBoundaryResult split_quad_like_boundary(
    const RegionBoundaryLoop& loop,
    const QuadLikeBoundaryConfig& cfg = QuadLikeBoundaryConfig());

bool export_quad_like_boundary_pca_debug_obj(
    const std::string& filename,
    const QuadLikeBoundaryDebugInfo& debug);

#endif
