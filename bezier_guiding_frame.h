#ifndef BEZIER_GUIDING_FRAME_HEADER
#define BEZIER_GUIDING_FRAME_HEADER

#include "trimmed_labeling.h"
#include "trimmed_region_input.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

struct BezierCurve3D {
    int degree = 1;
    std::vector<Eigen::Vector3d> control_points;

    Eigen::Vector3d evaluate(double t) const;
    Eigen::Vector3d derivative(double t) const;
    std::vector<Eigen::Vector3d> sample(int count) const;
};

struct BezierFrameIterationReport {
    int curve_index = -1;
    int side_index = -1;
    int degree = 1;
    int iteration = 0;
    double mean_error = 0.0;
    double rms_error = 0.0;
    double max_error = 0.0;
    double condition_estimate = 0.0;
};

struct BezierGuidingFrameConfig {
    int initial_degree = 1;
    int max_degree = 5;
    int max_iterations_per_degree = 12;
    int projection_samples = 80;
    int fairness_quadrature_samples = 16;
    double lambda_frame = 1e-4;
    double fitting_tolerance = 1e-4;
    double convergence_tolerance = 1e-7;
    bool use_virtual_corners = true;
    bool export_debug = true;
    std::string debug_output_dir = "bezier_frame_debug";
};

struct BezierGuidingFrameResult {
    bool valid = false;
    std::string reason;
    std::vector<int> side_indices;
    std::vector<std::vector<Eigen::Vector3d>> labeled_samples;
    std::vector<Eigen::Vector3d> shared_corners;
    std::vector<BezierCurve3D> curves;
    std::vector<BezierFrameIterationReport> reports;
    int final_degree = 1;
    double mean_error = 0.0;
    double rms_error = 0.0;
    double max_error = 0.0;
};

BezierGuidingFrameResult build_bezier_guiding_frame(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling,
    const BezierGuidingFrameConfig& config = BezierGuidingFrameConfig());

BezierGuidingFrameResult build_bezier_guiding_frame_from_samples(
    const std::vector<std::vector<Eigen::Vector3d>>& labeled_samples,
    const std::vector<Eigen::Vector3d>& optional_virtual_corners,
    const BezierGuidingFrameConfig& config = BezierGuidingFrameConfig());

bool export_bezier_guiding_frame_debug(
    const std::string& directory,
    const BezierGuidingFrameResult& result);

#endif
