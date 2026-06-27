#ifndef ROTATION_ANGLE_PARAMETERIZATION_HEADER
#define ROTATION_ANGLE_PARAMETERIZATION_HEADER

#include "bezier_guiding_frame.h"
#include "constrained_arap_parameterization.h"
#include "trimmed_labeling.h"
#include "trimmed_region_input.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

struct RotationConstraintRowReport {
    std::string type;
    int vertex_id = -1;
    int face_a = -1;
    int face_b = -1;
    double rhs = 0.0;
    double residual = 0.0;
};

struct RotationAngleInitializationConfig {
    bool fail_on_multiply_connected = true;
    bool include_boundary_angle_defects = true;
    bool include_label_orientation_constraints = true;
    double label_orientation_weight = 25.0;
    double singular_value_tolerance = 1e-10;
    bool use_sparse_linear_solvers = true;
    bool print_progress_to_console = false;
    bool export_debug = true;
    std::string debug_prefix = "rotation_angle";
};

struct RotationAngleInitializationResult {
    bool valid = false;
    bool unsupported_multiply_connected = false;
    std::string reason;
    std::vector<int> region_face_ids;
    Eigen::VectorXd dual_omega;
    Eigen::VectorXd face_rotation_angles;
    std::vector<Eigen::Matrix2d> face_rotations;
    Eigen::MatrixXd triangle_soup_uv;
    Eigen::MatrixXi triangle_soup_faces;
    std::vector<RotationConstraintRowReport> constraint_reports;
    double max_fan_closure_error = 0.0;
    double max_noncontractible_loop_error = 0.0;
    double mean_label_orientation_error = 0.0;
    double max_label_orientation_error = 0.0;
};

struct KktGlobalStitchingConfig {
    bool enable_smoothed_arap = false;
    double lambda_scale = 1e-3;
    double lambda_smooth = 0.0;
    double rank_tolerance = 1e-10;
    double min_signed_area = 1e-14;
    bool fail_on_flips = true;
    bool enable_dense_diagnostics = false;
    bool print_progress_to_console = false;
    bool export_debug = true;
    std::string debug_prefix = "kkt_stitching";
};

struct KktGlobalStitchingResult {
    bool valid = false;
    std::string reason;
    Eigen::MatrixXd UV;
    std::vector<int> region_vertex_ids;
    Eigen::MatrixXi local_faces;
    std::vector<int> local_face_to_global_face;
    std::vector<double> alpha_values;
    std::vector<ConstrainedArapTriangleStats> before_triangle_stats;
    std::vector<ConstrainedArapTriangleStats> after_triangle_stats;
    int constraint_rank = 0;
    int constraint_count = 0;
    bool redundant_constraints = false;
    int flipped_triangle_count = 0;
    double mean_arap_residual = 0.0;
    double max_arap_residual = 0.0;
    double adjacent_residual_variation = 0.0;
    double max_label_coordinate_error = 0.0;
    double max_length_ratio_error = 0.0;
    std::vector<double> smallest_kkt_singular_values;
    std::vector<std::string> conflicting_constraints;
};

RotationAngleInitializationResult initialize_rotation_angles_section421(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling,
    const RotationAngleInitializationConfig& config = RotationAngleInitializationConfig());

KktGlobalStitchingResult parameterize_kkt_global_stitching(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling,
    const BezierGuidingFrameResult& guiding_frame,
    const RotationAngleInitializationResult& rotation_init,
    const KktGlobalStitchingConfig& config = KktGlobalStitchingConfig());

bool export_rotation_angle_initialization_debug(
    const std::string& prefix,
    const RotationAngleInitializationResult& result);

bool export_kkt_global_stitching_debug(
    const std::string& prefix,
    const KktGlobalStitchingResult& result);

#endif
