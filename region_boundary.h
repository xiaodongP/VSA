#ifndef REGION_BOUNDARY_HEADER
#define REGION_BOUNDARY_HEADER

#include <Eigen/Dense>

#include <array>
#include <string>
#include <vector>

struct RegionBoundaryLoop {
    std::vector<int> vertex_ids;
    std::vector<Eigen::Vector3d> positions;
    bool closed;

    RegionBoundaryLoop();
};

struct RegionBoundaryExtractionResult {
    RegionBoundaryLoop loop;
    std::vector<std::array<int, 2>> boundary_edges;
    std::vector<int> region_face_ids;
    bool success;
    std::string reason;
    int boundary_loop_count;
    bool region_connected;
    bool nonmanifold_boundary;
    bool broken_chain;
    bool duplicate_edge;

    RegionBoundaryExtractionResult();
};

RegionBoundaryExtractionResult extract_region_boundary_loop(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& face_region_ids,
    int target_region_id);

bool export_region_boundary_debug_obj(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<int>& face_region_ids,
    int target_region_id,
    const RegionBoundaryExtractionResult& result);

#endif
