#include "spline_polyscope_renderer.h"

#include "bspline.h"

#include <Eigen/Dense>

#include <algorithm>
#include <vector>

#ifdef VSA_ENABLE_POLYSCOPE
#include <polyscope/curve_network.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#endif

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector3d;
using std::string;
using std::vector;

SplinePolyscopeRenderConfig::SplinePolyscopeRenderConfig()
    : sample_u(48),
      sample_v(48),
      boundary_sample_count(96),
      show_control_net(true),
      show_boundary_curves(true) {}

bool polyscope_spline_renderer_available() {
#ifdef VSA_ENABLE_POLYSCOPE
    return true;
#else
    return false;
#endif
}

#ifdef VSA_ENABLE_POLYSCOPE

namespace {

static void build_control_net_edges(
    const BSplineSurface3D& surface,
    MatrixXd& nodes,
    MatrixXi& edges) {
    int nu = (int)surface.control_grid.size();
    int nv = nu > 0 ? (int)surface.control_grid[0].size() : 0;
    nodes.resize(nu * nv, 3);
    for (int i = 0; i < nu; i++) {
        for (int j = 0; j < nv; j++) {
            nodes.row(i * nv + j) = surface.control_grid[i][j].transpose();
        }
    }

    vector<Eigen::Vector2i> edge_list;
    for (int i = 0; i < nu; i++) {
        for (int j = 0; j < nv; j++) {
            int id = i * nv + j;
            if (i + 1 < nu) edge_list.push_back(Eigen::Vector2i(id, (i + 1) * nv + j));
            if (j + 1 < nv) edge_list.push_back(Eigen::Vector2i(id, i * nv + j + 1));
        }
    }
    edges.resize((int)edge_list.size(), 2);
    for (int i = 0; i < (int)edge_list.size(); i++) {
        edges.row(i) = edge_list[i];
    }
}

static void build_boundary_curve_network(
    const InitialBSplineSurfacePatch& patch,
    int sample_count,
    MatrixXd& nodes,
    MatrixXi& edges) {
    int n = std::max(2, sample_count);
    vector<Vector3d> pts;
    vector<Eigen::Vector2i> edge_list;
    for (int side = 0; side < 4; side++) {
        vector<Vector3d> side_pts = patch.boundary_curves[side].sample(n);
        int offset = (int)pts.size();
        for (const Vector3d& p : side_pts) pts.push_back(p);
        for (int k = 0; k + 1 < (int)side_pts.size(); k++) {
            edge_list.push_back(Eigen::Vector2i(offset + k, offset + k + 1));
        }
    }
    nodes.resize((int)pts.size(), 3);
    for (int i = 0; i < (int)pts.size(); i++) nodes.row(i) = pts[i].transpose();
    edges.resize((int)edge_list.size(), 2);
    for (int i = 0; i < (int)edge_list.size(); i++) edges.row(i) = edge_list[i];
}

} // namespace

#endif

bool show_initial_bspline_surface_in_polyscope(
    const InitialBSplineSurfacePatch& patch,
    const SplinePolyscopeRenderConfig& cfg,
    string& message) {
    if (!patch.valid) {
        message = "B-spline patch is invalid: " + patch.reason;
        return false;
    }

#ifndef VSA_ENABLE_POLYSCOPE
    message =
        "Polyscope renderer is not enabled. Reconfigure with -DVSA_ENABLE_POLYSCOPE=ON.";
    return false;
#else
    MatrixXd V;
    MatrixXi F;
    sample_bspline_surface(
        patch.surface,
        std::max(4, cfg.sample_u),
        std::max(4, cfg.sample_v),
        V,
        F);

    polyscope::init();
    polyscope::options::programName = "VSA B-spline Surface";

    auto* surface = polyscope::registerSurfaceMesh("B-spline sampled surface", V, F);
    surface->setSmoothShade(true);

    if (cfg.show_control_net) {
        MatrixXd control_nodes;
        MatrixXi control_edges;
        build_control_net_edges(patch.surface, control_nodes, control_edges);
        polyscope::registerCurveNetwork(
            "B-spline control net",
            control_nodes,
            control_edges);
    }

    if (cfg.show_boundary_curves) {
        MatrixXd boundary_nodes;
        MatrixXi boundary_edges;
        build_boundary_curve_network(
            patch,
            std::max(2, cfg.boundary_sample_count),
            boundary_nodes,
            boundary_edges);
        polyscope::registerCurveNetwork(
            "B-spline boundary curves",
            boundary_nodes,
            boundary_edges);
    }

    message = "Opened B-spline patch in Polyscope.";
    polyscope::show();
    return true;
#endif
}
