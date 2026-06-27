#ifndef TRIMMED_LABELING_HEADER
#define TRIMMED_LABELING_HEADER

#include "trimmed_region_input.h"

#include <Eigen/Dense>

#include <array>
#include <string>
#include <vector>

enum class LabelRejectionReason {
    None,
    Concave,
    HighEnergy
};

enum class VirtualCornerType {
    Concave,
    Parallel,
    WeakConvex,
    StrongConvex
};

struct AutomaticLabelingConfig {
    bool enable_candidate_rejection = true;
    bool enable_candidate_concatenation = true;
    bool enable_reduction = true;
    bool export_debug = true;
    std::string debug_output_dir = "labeling_debug";

    double concave_turn_threshold = 15.0 * 3.14159265358979323846 / 180.0;
    double parallel_angle_threshold = 12.0 * 3.14159265358979323846 / 180.0;
    double weak_convex_angle_threshold = 115.0 * 3.14159265358979323846 / 180.0;
    double obtuse_angle_threshold = 90.0 * 3.14159265358979323846 / 180.0;
    double label_energy_threshold = 2.5;
    double high_energy_merge_bias = 0.25;
    double w_dist = 0.35;
    double w_tri = 0.15;
    double tie_epsilon = 1e-8;
    int max_reduction_rounds = 16;
    int min_supported_label_count = 0;

    bool enable_vsa_orientation = false;
    bool use_supplied_orientation_axes = false;
    Eigen::Vector3d orientation_axis_u = Eigen::Vector3d::UnitX();
    Eigen::Vector3d orientation_axis_v = Eigen::Vector3d::UnitY();
};

struct LabelCandidateMetrics {
    int group_id = -1;
    std::vector<int> segment_ids;
    double raw_energy = 0.0;
    double label_energy = 0.0;
    double phi_begin = 0.0;
    double phi_end = 0.0;
    bool begin_concave = false;
    bool end_concave = false;
    bool rejected = false;
    LabelRejectionReason rejection_reason = LabelRejectionReason::None;
};

struct VirtualCorner {
    int id = -1;
    int prev_group_id = -1;
    int next_group_id = -1;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d closest_prev = Eigen::Vector3d::Zero();
    Eigen::Vector3d closest_next = Eigen::Vector3d::Zero();
    VirtualCornerType type = VirtualCornerType::StrongConvex;
    double theta = 0.0;
    double signed_turn = 0.0;
    double e_ang = 0.0;
    double e_dist = 0.0;
    double e_tri = 0.0;
    double e_cvx = 0.0;
    double e_cor = 0.0;
};

struct LabelGroup {
    int id = -1;
    bool active = true;
    std::vector<int> segment_ids;
    std::vector<Eigen::Vector3d> guide_polyline;
};

struct AbstractSide {
    int side_index = -1;
    std::vector<int> label_group_ids;
    std::vector<int> segment_ids;
    Eigen::Vector3d average_tangent = Eigen::Vector3d::Zero();
};

struct OrientationResolution {
    bool valid = false;
    bool used_vsa_axes = false;
    int rotation = 0;
    bool reflected = false;
    double mean_direction_error = 0.0;
    double confidence = 0.0;
    std::array<std::string, 4> side_to_cardinal;
};

struct AutomaticLabelingResult {
    bool valid = false;
    bool ambiguous = false;
    std::string reason;
    std::vector<LabelCandidateMetrics> initial_candidates;
    std::vector<LabelGroup> final_label_groups;
    std::vector<VirtualCorner> final_corners;
    std::vector<std::vector<VirtualCorner>> corner_rounds;
    std::vector<AbstractSide> abstract_sides;
    OrientationResolution orientation;
    std::vector<std::string> operation_log;
    int final_label_count = 0;
    int final_corner_count = 0;
};

AutomaticLabelingResult run_automatic_labeling(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingConfig& config = AutomaticLabelingConfig());

bool export_labeling_debug_artifacts(
    const std::string& directory,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& result);

const char* to_string(LabelRejectionReason reason);
const char* to_string(VirtualCornerType type);

#endif
