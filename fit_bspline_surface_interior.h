#ifndef FIT_BSPLINE_SURFACE_INTERIOR_HEADER
#define FIT_BSPLINE_SURFACE_INTERIOR_HEADER

#include "initial_bspline_surface.h"
#include "region_square_parameterization.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

struct SurfaceFitSample {
    Eigen::Vector2d uv;
    Eigen::Vector3d position;
    Eigen::Vector3d normal;
    double weight;
    bool has_normal;

    SurfaceFitSample();
};

struct SurfaceInteriorFitConfig {
    double fairness_weight;
    double initial_weight;
    double point_to_plane_weight;
    bool enable_point_to_plane;
    double control_bbox_padding_factor;
    int surface_sample_u;
    int surface_sample_v;

    SurfaceInteriorFitConfig();
};

struct SurfaceFitErrorStats {
    double mean_error;
    double rms_error;
    double max_error;

    SurfaceFitErrorStats();
};

struct SurfaceInteriorFitResult {
    InitialBSplineSurfacePatch patch;
    SurfaceFitErrorStats before;
    SurfaceFitErrorStats after;
    bool valid;
    bool normal_flip_detected;
    bool control_points_out_of_bounds;
    std::string reason;

    SurfaceInteriorFitResult();
};

std::vector<SurfaceFitSample> make_region_vertex_fit_samples(
    const Eigen::MatrixXd& V,
    const RegionSquareParameterizationResult& parameterization);

SurfaceInteriorFitResult fit_bspline_surface_interior_control_points(
    const InitialBSplineSurfacePatch& initial_patch,
    const std::vector<SurfaceFitSample>& samples,
    const SurfaceInteriorFitConfig& cfg = SurfaceInteriorFitConfig());

bool export_surface_interior_fit_debug(
    const std::string& prefix,
    const SurfaceInteriorFitResult& result,
    const std::vector<SurfaceFitSample>& samples);

bool export_surface_fit_error_heatmap_obj(
    const std::string& filename,
    const BSplineSurface3D& surface,
    const std::vector<SurfaceFitSample>& samples);

bool export_surface_fit_report_csv(
    const std::string& filename,
    const SurfaceInteriorFitResult& result);

#endif
