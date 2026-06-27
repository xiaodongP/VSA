#ifndef TRIMMED_MESH_VALIDATION_HEADER
#define TRIMMED_MESH_VALIDATION_HEADER

#include <Eigen/Dense>

#include <string>
#include <vector>

struct MeshValidationReport {
    int vertex_count = 0;
    int face_count = 0;
    int exact_duplicate_faces = 0;
    int geometric_duplicate_faces = 0;
    int degenerate_faces = 0;
    int near_degenerate_faces = 0;
    int inconsistent_winding_edges = 0;
    int normal_jump_edges = 0;
    int nonmanifold_edges = 0;
    int isolated_vertices = 0;
    double min_double_area = 0.0;
    double min_quality = 0.0;

    std::vector<int> exact_duplicate_face_ids;
    std::vector<int> geometric_duplicate_face_ids;
    std::vector<int> degenerate_face_ids;
    std::vector<int> near_degenerate_face_ids;
    std::vector<int> low_quality_face_ids;
    std::vector<int> bad_winding_face_ids;
    std::vector<int> normal_jump_face_ids;
    std::vector<int> nonmanifold_face_ids;
    std::vector<int> isolated_vertex_ids;
};

MeshValidationReport validate_trimmed_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd* UV = nullptr);

bool orient_mesh_faces_consistently(
    const Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    const Eigen::Vector3d& target_normal);

bool remove_degenerate_faces(
    const Eigen::MatrixXd& V,
    Eigen::MatrixXi& F);

bool write_mesh_validation_json(
    const std::string& filename,
    const MeshValidationReport& report,
    const Eigen::MatrixXd* UV = nullptr);

bool export_mesh_validation_issue_objs(
    const std::string& output_dir,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const MeshValidationReport& report);

#endif
