#include "region_boundary_viewer.h"

#include <set>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::RowVector3d;

void show_region_boundary_in_viewer(
    igl::opengl::glfw::Viewer& viewer,
    const MatrixXd& V,
    const MatrixXi& F,
    const std::vector<int>& face_region_ids,
    int target_region_id,
    const RegionBoundaryExtractionResult& result) {
    viewer.data().clear();
    viewer.data().set_mesh(V, F);

    MatrixXd colors(F.rows(), 3);
    for (int fi = 0; fi < F.rows(); fi++) {
        if (fi < (int)face_region_ids.size() &&
            face_region_ids[fi] == target_region_id) {
            colors.row(fi) = RowVector3d(0.20, 0.55, 1.00);
        } else {
            colors.row(fi) = RowVector3d(0.78, 0.78, 0.78);
        }
    }
    viewer.data().set_colors(colors);
    viewer.data().show_lines = true;

    if (!result.boundary_edges.empty()) {
        MatrixXd P1(result.boundary_edges.size(), 3);
        MatrixXd P2(result.boundary_edges.size(), 3);
        MatrixXd EC(result.boundary_edges.size(), 3);
        for (int i = 0; i < (int)result.boundary_edges.size(); i++) {
            int a = result.boundary_edges[i][0];
            int b = result.boundary_edges[i][1];
            P1.row(i) = V.row(a);
            P2.row(i) = V.row(b);
            EC.row(i) = result.success
                      ? RowVector3d(1.00, 0.10, 0.10)
                      : RowVector3d(1.00, 0.75, 0.05);
        }
        viewer.data().add_edges(P1, P2, EC);
    }

    if (!result.loop.vertex_ids.empty()) {
        MatrixXd P(result.loop.vertex_ids.size(), 3);
        MatrixXd C(result.loop.vertex_ids.size(), 3);
        for (int i = 0; i < (int)result.loop.vertex_ids.size(); i++) {
            P.row(i) = V.row(result.loop.vertex_ids[i]);
            C.row(i) = RowVector3d(0.05, 1.00, 0.25);
        }
        viewer.data().add_points(P, C);
    }

    viewer.data().show_overlay = true;
    viewer.data().line_width = 2.0;
}
