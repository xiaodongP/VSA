#include "quad_like_boundary_viewer.h"

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::RowVector3d;

void show_quad_like_boundary_in_viewer(
    igl::opengl::glfw::Viewer& viewer,
    const MatrixXd& V,
    const MatrixXi& F,
    const std::vector<int>& face_region_ids,
    int target_region_id,
    const QuadLikeBoundary& boundary) {
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

    if (!boundary.valid) return;

    RowVector3d side_colors[4] = {
        RowVector3d(1.00, 0.10, 0.10),
        RowVector3d(0.10, 0.75, 0.20),
        RowVector3d(0.10, 0.35, 1.00),
        RowVector3d(1.00, 0.65, 0.05)
    };

    for (int s = 0; s < 4; s++) {
        const auto& side = boundary.side_polylines[s];
        if (side.size() < 2) continue;
        MatrixXd P1(side.size() - 1, 3);
        MatrixXd P2(side.size() - 1, 3);
        MatrixXd C(side.size() - 1, 3);
        for (int i = 0; i + 1 < (int)side.size(); i++) {
            P1.row(i) = side[i].transpose();
            P2.row(i) = side[i + 1].transpose();
            C.row(i) = side_colors[s];
        }
        viewer.data().add_edges(P1, P2, C);
    }

    MatrixXd corner_points(4, 3);
    MatrixXd corner_colors(4, 3);
    for (int s = 0; s < 4; s++) {
        corner_points.row(s) = boundary.side_polylines[s].front().transpose();
        corner_colors.row(s) = RowVector3d(1.0, 1.0, 1.0);
    }
    viewer.data().add_points(corner_points, corner_colors);
    viewer.data().show_overlay = true;
    viewer.data().line_width = 3.0;
}
