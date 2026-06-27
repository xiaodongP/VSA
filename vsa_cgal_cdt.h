#pragma once

#include <Eigen/Core>

#include <vector>

bool vsa_cgal_constrained_delaunay_2d(
    const std::vector<Eigen::Vector2d>& input_points,
    int boundary_count,
    std::vector<Eigen::Vector3i>& out_triangles);
