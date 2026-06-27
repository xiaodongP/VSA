#ifndef REUSABLE_TRIMMED_BSPLINE_SURFACE_HEADER
#define REUSABLE_TRIMMED_BSPLINE_SURFACE_HEADER

#include "bspline.h"
#include "rectangular_domain_extension.h"
#include "trimmed_bspline_surface.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

struct ReusableTrimLoop2D {
    bool is_perimeter = false;
    std::vector<int> source_vertex_ids;
    std::vector<Eigen::Vector2d> uv_polyline;
    std::vector<Eigen::Vector3d> spatial_polyline;
    TrimCurve2D fitted_curve;
};

struct ReusableTrimmedBSplineSurface {
    BSplineSurface3D surface;
    std::vector<ReusableTrimLoop2D> trim_loops;
    Eigen::Vector2d source_uv_min = Eigen::Vector2d::Zero();
    Eigen::Vector2d source_uv_max = Eigen::Vector2d::Ones();
    int source_region_id = -1;
    std::string generator;
    bool valid = false;
    std::string reason;
};

struct BoundaryRibbonSurface {
    int loop_id = -1;
    BSplineSurface3D surface;
    std::vector<Eigen::Vector2d> boundary_uv;
    std::vector<Eigen::Vector3d> boundary_xyz;
    double ribbon_width_uv = 0.0;
    bool valid = false;
    std::string reason;
};

ReusableTrimmedBSplineSurface make_reusable_trimmed_bspline_surface(
    const BSplineSurface3D& surface,
    const std::vector<TrimLoop2D>& normalized_trim_loops,
    const Eigen::Vector2d& source_uv_min,
    const Eigen::Vector2d& source_uv_max,
    int source_region_id,
    int trim_curve_control_count = 12,
    double trim_curve_fairness_weight = 1e-7,
    const Eigen::MatrixXd* source_vertices = nullptr);

bool export_reusable_trimmed_bspline_surface_json(
    const std::string& filename,
    const ReusableTrimmedBSplineSurface& asset);

bool sample_reusable_trimmed_bspline_surface(
    const ReusableTrimmedBSplineSurface& asset,
    int sample_u,
    int sample_v,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd* UV = nullptr);

bool sample_abc_boundary_controlled_trimmed_surface(
    const ReusableTrimmedBSplineSurface& asset,
    int sample_u,
    int sample_v,
    double ribbon_width,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd* UV = nullptr);

std::vector<BoundaryRibbonSurface> build_g0_boundary_ribbon_surfaces(
    const ReusableTrimmedBSplineSurface& asset,
    int row_count = 4,
    double ribbon_width = 0.06);

bool export_boundary_ribbon_surfaces_debug(
    const std::string& output_dir,
    const std::vector<BoundaryRibbonSurface>& ribbons,
    int sample_u_per_edge = 3,
    int sample_v = 8);

bool export_reusable_trimmed_bspline_surface_debug(
    const std::string& output_dir,
    const ReusableTrimmedBSplineSurface& asset,
    int sample_u,
    int sample_v);

#endif
