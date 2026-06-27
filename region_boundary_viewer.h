#ifndef REGION_BOUNDARY_VIEWER_HEADER
#define REGION_BOUNDARY_VIEWER_HEADER

#include "region_boundary.h"

#include <igl/opengl/glfw/Viewer.h>

void show_region_boundary_in_viewer(
    igl::opengl::glfw::Viewer& viewer,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& face_region_ids,
    int target_region_id,
    const RegionBoundaryExtractionResult& result);

#endif
