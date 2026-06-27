#ifndef TRIMMED_BSPLINE_PIPELINE_HEADER
#define TRIMMED_BSPLINE_PIPELINE_HEADER

#include "tensor_product_bspline_fitter.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

struct TrimmedBSplinePipelineConfig {
    std::string output_dir = "trimmed_bspline_output";
    int control_count_u = 8;
    int control_count_v = 8;
    int spline_degree = 3;
    double fitting_regularization = 1e-7;
    double fitting_initial_weight = 1e-4;
    int surface_sample_u = 50;
    int surface_sample_v = 50;
    int extension_sample_u = 18;
    int extension_sample_v = 18;
    double extension_margin = 0.02;
    bool enable_smoothed_arap = true;
    double arap_smoothing_weight = 1e-4;
    bool enable_extension_fairness = true;
    double extension_fairness_weight = 0.35;
    double extension_isocurve_weight = 0.5;
    double max_extension_bbox_growth = 1.25;
    double max_extension_area_growth = 1.35;
    double boundary_fit_weight = 1000.0;
    int trim_curve_control_count = 12;
    double trim_curve_fairness_weight = 1e-7;
    bool apply_harmonic_boundary_correction = true;
    bool snap_output_boundary_to_authoritative = false;
    double max_boundary_error_ratio = 0.025;
    double max_boundary_error_absolute = 1e-7;
    bool allow_non_labeling_surface_fallback = false;
    bool export_debug_artifacts = true;
    bool run_ablation_baselines = true;
    bool estimate_condition_number = true;
    bool print_progress_to_console = true;
};

struct TrimmedBSplinePipelineMetrics {
    bool valid = false;
    std::string reason;
    std::string labeling_configuration;
    bool ambiguous = false;
    int flipped_triangle_count = 0;
    double mean_arap_distortion = 0.0;
    double max_arap_distortion = 0.0;
    double label_coordinate_constraint_error = 0.0;
    double guiding_frame_length_ratio_error = 0.0;
    double original_region_rms_error = 0.0;
    double original_region_max_error = 0.0;
    double boundary_rms_error = 0.0;
    double boundary_max_error = 0.0;
    double artificial_extension_curvature_mean = 0.0;
    double artificial_extension_curvature_max = 0.0;
    double surface_area_growth_ratio = 0.0;
    double bounding_box_growth_ratio = 0.0;
    int weak_control_point_count = 0;
    double linear_system_condition_estimate = 0.0;
};

struct TrimmedBSplinePipelineResult {
    bool valid = false;
    std::string reason;
    TrimmedBSplinePipelineMetrics metrics;
    std::vector<TensorProductBSplineFitResult> baseline_results;
};

TrimmedBSplinePipelineResult run_single_region_trimmed_bspline_pipeline(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& face_region_ids,
    int target_region_id,
    const TrimmedBSplinePipelineConfig& config = TrimmedBSplinePipelineConfig());

bool write_trimmed_bspline_reproduction_doc(const std::string& filename);

#endif
