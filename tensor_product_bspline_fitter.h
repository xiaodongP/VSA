#ifndef TENSOR_PRODUCT_BSPLINE_FITTER_HEADER
#define TENSOR_PRODUCT_BSPLINE_FITTER_HEADER

#include "bspline.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

enum class TensorProductFitBaseline {
    TrimmedOnly,
    ArapOnly,
    ExtensionOnly,
    LabelingExtension
};

struct TensorProductBSplineFitInput {
    Eigen::MatrixXd UV;
    Eigen::MatrixXi F;
    Eigen::MatrixXd positions;
    std::vector<bool> original_vertex_mask;
    std::vector<bool> original_face_mask;
    std::vector<double> sample_weights;
    TensorProductFitBaseline baseline = TensorProductFitBaseline::ExtensionOnly;
    std::string label;
};

struct TensorProductBSplineFitConfig {
    int degree_u = 3;
    int degree_v = 3;
    int control_count_u = 6;
    int control_count_v = 6;
    double control_net_fairness_weight = 1e-6;
    double control_net_initial_weight = 0.0;
    double weak_support_threshold = 1e-4;
    double normal_equation_regularization = 1e-10;
    int sample_u = 40;
    int sample_v = 40;
    bool fit_original_vertices_only = false;
    bool estimate_condition_number = true;
    bool export_debug = true;
    std::string debug_prefix = "tensor_product_bspline_fit";
};

struct TensorProductBSplineFitStats {
    double mean_error = 0.0;
    double rms_error = 0.0;
    double max_error = 0.0;
};

struct WeakSupportStats {
    int control_point_count = 0;
    int weak_control_point_count = 0;
    double min_support = 0.0;
    double mean_support = 0.0;
    double max_support = 0.0;
};

struct TensorProductBSplineFitResult {
    bool valid = false;
    std::string reason;
    TensorProductFitBaseline baseline = TensorProductFitBaseline::ExtensionOnly;
    BSplineSurface3D surface;
    Eigen::Vector2d uv_min = Eigen::Vector2d::Zero();
    Eigen::Vector2d uv_max = Eigen::Vector2d::Ones();
    TensorProductBSplineFitStats all_vertex_error;
    TensorProductBSplineFitStats original_region_error;
    double extension_region_smoothness = 0.0;
    double condition_estimate = 0.0;
    WeakSupportStats weak_support;
    int sample_count = 0;
    int fit_sample_count = 0;
    int normal_matrix_nonzeros = 0;
};

const char* to_string(TensorProductFitBaseline baseline);

TensorProductBSplineFitResult fit_tensor_product_cubic_bspline_surface(
    const TensorProductBSplineFitInput& input,
    const TensorProductBSplineFitConfig& config = TensorProductBSplineFitConfig());

bool export_tensor_product_bspline_fit_debug(
    const std::string& prefix,
    const TensorProductBSplineFitResult& result);

#endif
