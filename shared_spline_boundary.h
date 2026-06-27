#ifndef SHARED_SPLINE_BOUNDARY_HEADER
#define SHARED_SPLINE_BOUNDARY_HEADER

#include "bspline.h"

#include <Eigen/Dense>

#include <array>
#include <string>
#include <vector>

struct SharedSplineBoundary {
    int id;
    int region_a;
    int region_b;
    BSplineCurve3D curve;
    std::vector<int> control_point_ids;

    SharedSplineBoundary();
};

struct GlobalControlPointPool {
    std::vector<Eigen::Vector3d> values;
};

struct BSplinePatchTopology {
    std::vector<std::vector<int>> control_point_ids;
};

struct SharedSplinePatch {
    int region_id;
    int degree_u;
    int degree_v;
    std::vector<double> knots_u;
    std::vector<double> knots_v;
    BSplinePatchTopology topology;
    std::array<int, 4> shared_boundary_ids;
    std::array<bool, 4> shared_boundary_reversed;

    SharedSplinePatch();
};

struct SharedSplineAssembly {
    GlobalControlPointPool pool;
    std::vector<SharedSplinePatch> patches;
    std::vector<SharedSplineBoundary> shared_boundaries;
    bool valid;
    std::string reason;

    SharedSplineAssembly();
};

struct SharedSplineSampledMesh {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXi patch_ids;
    std::vector<std::vector<int>> shared_boundary_vertex_ids;
};

struct SharedSplineAdjacencyReport {
    double max_boundary_sample_difference;
    double mean_boundary_sample_difference;
    int shared_boundary_edge_count;
    int shared_boundary_edges_with_two_faces;
    int outer_boundary_edge_count;
    int nonmanifold_edge_count;
    bool watertight_across_shared_boundary;
    bool valid;
    std::string reason;

    SharedSplineAdjacencyReport();
};

SharedSplineAssembly build_two_patch_shared_boundary_assembly(
    const BSplineSurface3D& patch_a,
    int region_a,
    int shared_side_a,
    const BSplineSurface3D& patch_b,
    int region_b,
    int shared_side_b,
    bool opposite_orientation);

BSplineSurface3D materialize_patch_surface(
    const SharedSplinePatch& patch,
    const GlobalControlPointPool& pool);

Eigen::Vector3d evaluate_shared_patch_surface(
    const SharedSplinePatch& patch,
    const GlobalControlPointPool& pool,
    double u,
    double v);

bool sample_shared_spline_assembly(
    const SharedSplineAssembly& assembly,
    int sample_u,
    int sample_v,
    SharedSplineSampledMesh& out_mesh,
    SharedSplineAdjacencyReport& out_report);

SharedSplineAdjacencyReport check_shared_spline_adjacency(
    const SharedSplineAssembly& assembly,
    const SharedSplineSampledMesh& mesh,
    int sample_u,
    int sample_v);

bool export_shared_spline_debug(
    const std::string& prefix,
    const SharedSplineAssembly& assembly,
    const SharedSplineSampledMesh& mesh,
    const SharedSplineAdjacencyReport& report);

bool export_shared_spline_control_point_ids_csv(
    const std::string& filename,
    const SharedSplineAssembly& assembly);

bool export_shared_spline_adjacency_report_csv(
    const std::string& filename,
    const SharedSplineAdjacencyReport& report);

bool export_shared_spline_colored_ply(
    const std::string& filename,
    const SharedSplineSampledMesh& mesh);

#endif
