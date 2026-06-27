#ifndef CONSTRAINED_ARAP_PARAMETERIZATION_HEADER
#define CONSTRAINED_ARAP_PARAMETERIZATION_HEADER

#include "bezier_guiding_frame.h"
#include "trimmed_labeling.h"
#include "trimmed_region_input.h"

#include <Eigen/Dense>

#include <array>
#include <string>
#include <vector>

enum class ParameterSideRole {
    South,
    East,
    North,
    West,
    Unassigned
};

struct ConstrainedArapTriangleStats {
    int local_face_id = -1;
    int global_face_id = -1;
    double signed_uv_area = 0.0;
    double abs_uv_area = 0.0;
    double area_3d = 0.0;
    double area_ratio = 0.0;
    double conformal_distortion = 0.0;
    double arap_residual = 0.0;
    int orientation = 0;
    bool flipped = false;
};

struct ConstrainedArapConfig {
    int max_iterations = 10;
    double convergence_tolerance = 1e-7;
    double min_signed_area = 1e-14;
    bool fail_on_flips = true;
    bool use_labeling_orientation = true;
    bool export_debug = true;
    std::string debug_output_prefix = "constrained_arap";
    std::array<ParameterSideRole, 4> fallback_side_roles = {
        ParameterSideRole::South,
        ParameterSideRole::East,
        ParameterSideRole::North,
        ParameterSideRole::West};
};

struct ConstrainedArapResult {
    bool valid = false;
    std::string reason;
    Eigen::MatrixXd UV;
    std::vector<int> region_vertex_ids;
    Eigen::MatrixXi local_faces;
    std::vector<int> local_face_to_global_face;
    std::vector<ConstrainedArapTriangleStats> triangle_stats;
    std::vector<double> iteration_residuals;
    std::vector<int> constrained_vertex_ids;
    std::vector<Eigen::Vector2d> constrained_uvs;
    double width = 1.0;
    double height = 1.0;
    double mean_area_ratio = 0.0;
    double max_area_ratio = 0.0;
    double mean_conformal_distortion = 0.0;
    double max_conformal_distortion = 0.0;
    double mean_arap_residual = 0.0;
    double max_arap_residual = 0.0;
    int flipped_triangle_count = 0;
};

ConstrainedArapResult parameterize_constrained_arap_mvp(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling,
    const BezierGuidingFrameResult& guiding_frame,
    const ConstrainedArapConfig& config = ConstrainedArapConfig());

bool export_constrained_arap_debug(
    const std::string& prefix,
    const Eigen::MatrixXd& V,
    const ConstrainedArapResult& result);

const char* to_string(ParameterSideRole role);

#endif
