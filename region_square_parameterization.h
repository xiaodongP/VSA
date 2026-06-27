#ifndef REGION_SQUARE_PARAMETERIZATION_HEADER
#define REGION_SQUARE_PARAMETERIZATION_HEADER

#include "quad_like_boundary.h"
#include "region_boundary.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

struct ParameterizationTriangleStats {
    int global_face_id;
    double signed_uv_area;
    double abs_uv_area;
    double area_3d;
    double area_distortion;
    double angle_distortion;
    bool flipped;

    ParameterizationTriangleStats();
};

struct RegionSquareParameterizationConfig {
    double uv_range_tolerance;
    double min_signed_area;
    bool fail_on_flips;

    RegionSquareParameterizationConfig();
};

struct RegionSquareParameterizationResult {
    Eigen::MatrixXd UV;
    std::vector<int> region_vertex_ids;
    Eigen::MatrixXi local_faces;
    std::vector<int> local_face_to_global_face;
    std::vector<ParameterizationTriangleStats> triangle_stats;
    double min_signed_area;
    double max_signed_area;
    double mean_area_distortion;
    double max_area_distortion;
    double mean_angle_distortion;
    double max_angle_distortion;
    bool valid;
    bool has_flips;
    bool uv_in_range;
    bool orientation_reflected;
    std::string reason;

    RegionSquareParameterizationResult();
};

RegionSquareParameterizationResult parameterize_region_to_square(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& face_region_ids,
    int target_region_id,
    const RegionBoundaryLoop& boundary_loop,
    const QuadLikeBoundary& quad_boundary,
    const RegionSquareParameterizationConfig& cfg = RegionSquareParameterizationConfig());

bool export_region_square_parameterization_debug(
    const std::string& prefix,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const RegionSquareParameterizationResult& result);

bool export_region_uv_mesh_obj(
    const std::string& filename,
    const RegionSquareParameterizationResult& result);

bool export_region_uv_signed_area_csv(
    const std::string& filename,
    const RegionSquareParameterizationResult& result);

bool export_region_uv_distortion_csv(
    const std::string& filename,
    const RegionSquareParameterizationResult& result);

bool export_region_uv_correspondence_obj(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const RegionSquareParameterizationResult& result);

#endif
