#ifndef QUAD_LIKE_BOUNDARY_VIEWER_HEADER
#define QUAD_LIKE_BOUNDARY_VIEWER_HEADER

#include "quad_like_boundary.h"

#include <igl/opengl/glfw/Viewer.h>

void show_quad_like_boundary_in_viewer(
    igl::opengl::glfw::Viewer& viewer,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& face_region_ids,
    int target_region_id,
    const QuadLikeBoundary& boundary);

#endif
