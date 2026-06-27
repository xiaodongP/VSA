#ifndef RECTANGULAR_DOMAIN_EXTENSION_HEADER
#define RECTANGULAR_DOMAIN_EXTENSION_HEADER

#include "rotation_angle_parameterization.h"
#include "trimmed_region_input.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

struct TrimLoop2D {
    bool is_perimeter = false;
    std::vector<int> vertex_ids;
    std::vector<Eigen::Vector2d> uv_polyline;
};

struct RectangularDomainExtensionInput {
    Eigen::MatrixXd UV;
    std::vector<int> region_vertex_ids;
    Eigen::MatrixXi local_faces;
    std::vector<int> local_face_to_global_face;
    std::vector<TrimLoop2D> trim_loops;
};

struct RectangularDomainExtensionConfig {
    double margin = 0.0;
    int sample_count_u = 0;
    int sample_count_v = 0;
    int min_sample_count = 4;
    int max_sample_count = 80;
    double dedup_epsilon = 1e-9;
    double min_constraint_edge_length = 1e-8;
    bool export_debug = true;
    std::string debug_prefix = "rectangular_domain_extension";
};

struct RectangularDomainExtensionResult {
    bool valid = false;
    std::string reason;
    Eigen::MatrixXd full_uv_vertices;
    Eigen::MatrixXi full_faces;
    std::vector<bool> original_vertex_mask;
    std::vector<bool> original_face_mask;
    std::vector<TrimLoop2D> authoritative_trim_loops;
    Eigen::Vector2d rectangle_min = Eigen::Vector2d::Zero();
    Eigen::Vector2d rectangle_max = Eigen::Vector2d::Ones();
    int grid_sample_u = 0;
    int grid_sample_v = 0;
    int original_vertex_count = 0;
    int artificial_vertex_count = 0;
    int original_face_count = 0;
    int artificial_face_count = 0;
};

RectangularDomainExtensionInput make_rectangular_extension_input(
    const BoundarySegmentationResult& boundary,
    const KktGlobalStitchingResult& parameterization);

RectangularDomainExtensionResult build_rectangular_domain_extension(
    const RectangularDomainExtensionInput& input,
    const RectangularDomainExtensionConfig& config = RectangularDomainExtensionConfig());

bool export_rectangular_domain_extension_debug(
    const std::string& prefix,
    const RectangularDomainExtensionResult& result);

#endif
