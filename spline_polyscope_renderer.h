#ifndef SPLINE_POLYSCOPE_RENDERER_HEADER
#define SPLINE_POLYSCOPE_RENDERER_HEADER

#include "initial_bspline_surface.h"

#include <string>

struct SplinePolyscopeRenderConfig {
    int sample_u;
    int sample_v;
    int boundary_sample_count;
    bool show_control_net;
    bool show_boundary_curves;

    SplinePolyscopeRenderConfig();
};

bool polyscope_spline_renderer_available();

bool show_initial_bspline_surface_in_polyscope(
    const InitialBSplineSurfacePatch& patch,
    const SplinePolyscopeRenderConfig& cfg,
    std::string& message);

#endif
