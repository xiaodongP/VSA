#ifndef INITIAL_BSPLINE_SURFACE_HEADER
#define INITIAL_BSPLINE_SURFACE_HEADER

#include "bspline.h"
#include "quad_like_boundary.h"

#include <array>
#include <string>

struct InitialBSplineSurfaceConfig {
    int control_count_u;
    int control_count_v;
    double boundary_fairness_weight;
    int coons_fit_samples_u;
    int coons_fit_samples_v;
    int boundary_check_samples;

    InitialBSplineSurfaceConfig();
};

struct InitialBSplineSurfacePatch {
    std::array<BSplineCurve3D, 4> boundary_curves;
    BSplineSurface3D surface;
    double boundary_mean_error;
    double boundary_max_error;
    double coons_fit_mean_error;
    double coons_fit_max_error;
    bool valid;
    std::string reason;

    InitialBSplineSurfacePatch();
};

InitialBSplineSurfacePatch build_initial_bspline_surface_from_quad_boundary(
    const QuadLikeBoundary& boundary,
    const InitialBSplineSurfaceConfig& cfg = InitialBSplineSurfaceConfig());

Eigen::Vector3d evaluate_coons_patch(
    const std::array<BSplineCurve3D, 4>& boundary_curves,
    double u,
    double v);

bool export_initial_bspline_surface_debug(
    const std::string& prefix,
    const InitialBSplineSurfacePatch& patch,
    int surface_sample_u,
    int surface_sample_v);

bool export_coons_patch_obj(
    const std::string& filename,
    const std::array<BSplineCurve3D, 4>& boundary_curves,
    int sample_u,
    int sample_v);

bool export_boundary_consistency_report(
    const std::string& filename,
    const InitialBSplineSurfacePatch& patch);

#endif
