#ifndef TRIMMED_BSPLINE_SURFACE_HEADER
#define TRIMMED_BSPLINE_SURFACE_HEADER

#include "bspline.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

struct TrimCurve2D {
    int degree;
    std::vector<double> knots;
    std::vector<Eigen::Vector2d> control_points;
    std::vector<Eigen::Vector2d> polyline;
    bool valid;
    std::string reason;

    TrimCurve2D();
};

struct TrimmedBSplineSurfacePatch {
    BSplineSurface3D surface;
    std::vector<Eigen::Vector2d> outer_trim_polyline;
    TrimCurve2D outer_trim_curve;
    bool valid;
    std::string reason;

    TrimmedBSplineSurfacePatch();
};

TrimCurve2D fit_trim_curve_2d_from_polyline(
    const std::vector<Eigen::Vector2d>& polyline,
    int control_count,
    double fairness_weight);

bool sample_trimmed_bspline_surface(
    const TrimmedBSplineSurfacePatch& patch,
    int grid_resolution_u,
    int grid_resolution_v,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd* UV = nullptr);

bool export_trimmed_bspline_surface_debug(
    const std::string& prefix,
    const TrimmedBSplineSurfacePatch& patch,
    int grid_resolution_u,
    int grid_resolution_v);

bool export_uv_trim_loop_obj(
    const std::string& filename,
    const std::vector<Eigen::Vector2d>& trim_polyline);

bool export_trim_curve_control_polygon_obj(
    const std::string& filename,
    const TrimCurve2D& curve);

#endif
