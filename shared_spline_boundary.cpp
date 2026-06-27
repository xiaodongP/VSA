#include "shared_spline_boundary.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector3d;
using std::array;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::tuple;
using std::vector;

namespace {

static bool valid_side_id(int side) {
    return side >= 0 && side < 4;
}

static bool same_knot_vector(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size()) return false;
    for (int i = 0; i < (int)a.size(); i++) {
        if (std::abs(a[i] - b[i]) > 1e-12) return false;
    }
    return true;
}

static int side_control_count(const BSplineSurface3D& surface, int side) {
    int nu = (int)surface.control_grid.size();
    int nv = (int)surface.control_grid[0].size();
    return (side == 0 || side == 2) ? nu : nv;
}

static int side_degree(const BSplineSurface3D& surface, int side) {
    return (side == 0 || side == 2) ? surface.degree_u : surface.degree_v;
}

static vector<double> side_knots(const BSplineSurface3D& surface, int side) {
    return (side == 0 || side == 2) ? surface.knots_u : surface.knots_v;
}

static vector<Vector3d> side_control_points_parameter_order(
    const BSplineSurface3D& surface,
    int side) {
    vector<Vector3d> cps;
    int nu = (int)surface.control_grid.size();
    int nv = (int)surface.control_grid[0].size();
    if (side == 0) {
        for (int i = 0; i < nu; i++) cps.push_back(surface.control_grid[i][0]);
    } else if (side == 1) {
        for (int j = 0; j < nv; j++) cps.push_back(surface.control_grid[nu - 1][j]);
    } else if (side == 2) {
        for (int i = 0; i < nu; i++) cps.push_back(surface.control_grid[i][nv - 1]);
    } else if (side == 3) {
        for (int j = 0; j < nv; j++) cps.push_back(surface.control_grid[0][j]);
    }
    return cps;
}

static vector<int> side_control_ids_parameter_order(
    const SharedSplinePatch& patch,
    int side) {
    vector<int> ids;
    int nu = (int)patch.topology.control_point_ids.size();
    int nv = (int)patch.topology.control_point_ids[0].size();
    if (side == 0) {
        for (int i = 0; i < nu; i++) ids.push_back(patch.topology.control_point_ids[i][0]);
    } else if (side == 1) {
        for (int j = 0; j < nv; j++) ids.push_back(patch.topology.control_point_ids[nu - 1][j]);
    } else if (side == 2) {
        for (int i = 0; i < nu; i++) ids.push_back(patch.topology.control_point_ids[i][nv - 1]);
    } else if (side == 3) {
        for (int j = 0; j < nv; j++) ids.push_back(patch.topology.control_point_ids[0][j]);
    }
    return ids;
}

static bool is_on_side(int i, int j, int su, int sv, int side) {
    if (side == 0) return j == 0;
    if (side == 1) return i == su - 1;
    if (side == 2) return j == sv - 1;
    if (side == 3) return i == 0;
    return false;
}

static int side_sample_index(int i, int j, int side) {
    return (side == 0 || side == 2) ? i : j;
}

static int side_sample_count(int su, int sv, int side) {
    return (side == 0 || side == 2) ? su : sv;
}

static pair<int, int> edge_key(int a, int b) {
    return std::make_pair(std::min(a, b), std::max(a, b));
}

static int add_pool_point(GlobalControlPointPool& pool, const Vector3d& p) {
    int id = (int)pool.values.size();
    pool.values.push_back(p);
    return id;
}

static string mesh_vertex_key_for_sample(
    const SharedSplinePatch& patch,
    int patch_index,
    int i,
    int j,
    int su,
    int sv) {
    for (int side = 0; side < 4; side++) {
        int shared_id = patch.shared_boundary_ids[side];
        if (shared_id < 0) continue;
        if (!is_on_side(i, j, su, sv, side)) continue;
        int k = side_sample_index(i, j, side);
        int count = side_sample_count(su, sv, side);
        if (patch.shared_boundary_reversed[side]) k = count - 1 - k;
        return "s:" + std::to_string(shared_id) + ":" + std::to_string(k);
    }
    return "p:" + std::to_string(patch_index) + ":" +
           std::to_string(i) + ":" + std::to_string(j);
}

static bool export_mesh_obj(
    const string& filename,
    const SharedSplineSampledMesh& mesh) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "# Shared spline sampled watertight mesh\n";
    for (int i = 0; i < mesh.V.rows(); i++) {
        fout << "v " << mesh.V(i, 0) << " " << mesh.V(i, 1) << " " << mesh.V(i, 2) << "\n";
    }
    int current_patch = -999999;
    for (int i = 0; i < mesh.F.rows(); i++) {
        int pid = mesh.patch_ids.rows() == mesh.F.rows() ? mesh.patch_ids(i, 0) : -1;
        if (pid != current_patch) {
            fout << "g patch_" << pid << "\n";
            current_patch = pid;
        }
        fout << "f " << (mesh.F(i, 0) + 1)
             << " " << (mesh.F(i, 1) + 1)
             << " " << (mesh.F(i, 2) + 1) << "\n";
    }
    return true;
}

static bool export_shared_boundary_obj(
    const string& filename,
    const SharedSplineAssembly& assembly) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "# Shared boundary curves\n";
    int offset = 1;
    for (const SharedSplineBoundary& b : assembly.shared_boundaries) {
        vector<Vector3d> pts = b.curve.sample(64);
        fout << "g shared_boundary_" << b.id << "\n";
        for (const Vector3d& p : pts) {
            fout << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
        }
        for (int i = 0; i + 1 < (int)pts.size(); i++) {
            fout << "l " << (offset + i) << " " << (offset + i + 1) << "\n";
        }
        offset += (int)pts.size();
    }
    return true;
}

} // namespace

SharedSplineBoundary::SharedSplineBoundary()
    : id(-1),
      region_a(-1),
      region_b(-1) {}

SharedSplinePatch::SharedSplinePatch()
    : region_id(-1),
      degree_u(3),
      degree_v(3),
      shared_boundary_ids({{-1, -1, -1, -1}}),
      shared_boundary_reversed({{false, false, false, false}}) {}

SharedSplineAssembly::SharedSplineAssembly()
    : valid(false) {}

SharedSplineAdjacencyReport::SharedSplineAdjacencyReport()
    : max_boundary_sample_difference(0.0),
      mean_boundary_sample_difference(0.0),
      shared_boundary_edge_count(0),
      shared_boundary_edges_with_two_faces(0),
      outer_boundary_edge_count(0),
      nonmanifold_edge_count(0),
      watertight_across_shared_boundary(false),
      valid(false) {}

SharedSplineAssembly build_two_patch_shared_boundary_assembly(
    const BSplineSurface3D& patch_a,
    int region_a,
    int shared_side_a,
    const BSplineSurface3D& patch_b,
    int region_b,
    int shared_side_b,
    bool opposite_orientation) {
    SharedSplineAssembly assembly;
    if (!valid_side_id(shared_side_a) || !valid_side_id(shared_side_b)) {
        assembly.reason = "shared side id must be in [0,3]";
        return assembly;
    }
    if (patch_a.control_grid.empty() || patch_b.control_grid.empty() ||
        patch_a.control_grid[0].empty() || patch_b.control_grid[0].empty()) {
        assembly.reason = "patch control grids must be non-empty";
        return assembly;
    }
    int count_a = side_control_count(patch_a, shared_side_a);
    int count_b = side_control_count(patch_b, shared_side_b);
    if (count_a != count_b) {
        assembly.reason = "shared sides must have the same control count";
        return assembly;
    }
    if (side_degree(patch_a, shared_side_a) != side_degree(patch_b, shared_side_b) ||
        !same_knot_vector(side_knots(patch_a, shared_side_a),
                          side_knots(patch_b, shared_side_b))) {
        assembly.reason = "shared sides must have identical degree and knots";
        return assembly;
    }

    vector<Vector3d> a_side = side_control_points_parameter_order(patch_a, shared_side_a);
    vector<Vector3d> b_side = side_control_points_parameter_order(patch_b, shared_side_b);
    for (int k = 0; k < count_a; k++) {
        int bk = opposite_orientation ? count_b - 1 - k : k;
        if ((a_side[k] - b_side[bk]).norm() > 1e-9) {
            assembly.reason = "declared shared side control points are not geometrically identical";
            return assembly;
        }
    }

    SharedSplinePatch topo_a;
    topo_a.region_id = region_a;
    topo_a.degree_u = patch_a.degree_u;
    topo_a.degree_v = patch_a.degree_v;
    topo_a.knots_u = patch_a.knots_u;
    topo_a.knots_v = patch_a.knots_v;
    int nua = (int)patch_a.control_grid.size();
    int nva = (int)patch_a.control_grid[0].size();
    topo_a.topology.control_point_ids.assign(nua, vector<int>(nva, -1));
    for (int i = 0; i < nua; i++) {
        for (int j = 0; j < nva; j++) {
            topo_a.topology.control_point_ids[i][j] =
                add_pool_point(assembly.pool, patch_a.control_grid[i][j]);
        }
    }
    topo_a.shared_boundary_ids[shared_side_a] = 0;
    topo_a.shared_boundary_reversed[shared_side_a] = false;

    vector<int> a_side_ids = side_control_ids_parameter_order(topo_a, shared_side_a);

    SharedSplinePatch topo_b;
    topo_b.region_id = region_b;
    topo_b.degree_u = patch_b.degree_u;
    topo_b.degree_v = patch_b.degree_v;
    topo_b.knots_u = patch_b.knots_u;
    topo_b.knots_v = patch_b.knots_v;
    int nub = (int)patch_b.control_grid.size();
    int nvb = (int)patch_b.control_grid[0].size();
    topo_b.topology.control_point_ids.assign(nub, vector<int>(nvb, -1));

    vector<pair<int, int>> b_side_coords;
    if (shared_side_b == 0) {
        for (int i = 0; i < nub; i++) b_side_coords.push_back({i, 0});
    } else if (shared_side_b == 1) {
        for (int j = 0; j < nvb; j++) b_side_coords.push_back({nub - 1, j});
    } else if (shared_side_b == 2) {
        for (int i = 0; i < nub; i++) b_side_coords.push_back({i, nvb - 1});
    } else {
        for (int j = 0; j < nvb; j++) b_side_coords.push_back({0, j});
    }
    set<pair<int, int>> b_shared_coord_set(b_side_coords.begin(), b_side_coords.end());

    for (int i = 0; i < nub; i++) {
        for (int j = 0; j < nvb; j++) {
            if (b_shared_coord_set.count({i, j})) continue;
            topo_b.topology.control_point_ids[i][j] =
                add_pool_point(assembly.pool, patch_b.control_grid[i][j]);
        }
    }

    for (int k = 0; k < count_b; k++) {
        int ak = opposite_orientation ? count_a - 1 - k : k;
        int i = b_side_coords[k].first;
        int j = b_side_coords[k].second;
        topo_b.topology.control_point_ids[i][j] = a_side_ids[ak];
    }
    topo_b.shared_boundary_ids[shared_side_b] = 0;
    topo_b.shared_boundary_reversed[shared_side_b] = opposite_orientation;

    SharedSplineBoundary shared;
    shared.id = 0;
    shared.region_a = region_a;
    shared.region_b = region_b;
    shared.control_point_ids = a_side_ids;
    shared.curve = BSplineCurve3D(
        side_degree(patch_a, shared_side_a),
        side_knots(patch_a, shared_side_a),
        a_side);

    assembly.patches.push_back(topo_a);
    assembly.patches.push_back(topo_b);
    assembly.shared_boundaries.push_back(shared);
    assembly.valid = true;
    assembly.reason = "ok";
    return assembly;
}

BSplineSurface3D materialize_patch_surface(
    const SharedSplinePatch& patch,
    const GlobalControlPointPool& pool) {
    int nu = (int)patch.topology.control_point_ids.size();
    int nv = nu > 0 ? (int)patch.topology.control_point_ids[0].size() : 0;
    vector<vector<Vector3d>> grid(nu, vector<Vector3d>(nv, Vector3d::Zero()));
    for (int i = 0; i < nu; i++) {
        for (int j = 0; j < nv; j++) {
            int id = patch.topology.control_point_ids[i][j];
            if (id >= 0 && id < (int)pool.values.size()) grid[i][j] = pool.values[id];
        }
    }
    return BSplineSurface3D(
        patch.degree_u, patch.degree_v,
        patch.knots_u, patch.knots_v,
        grid);
}

Vector3d evaluate_shared_patch_surface(
    const SharedSplinePatch& patch,
    const GlobalControlPointPool& pool,
    double u,
    double v) {
    return materialize_patch_surface(patch, pool).evaluate(u, v);
}

bool sample_shared_spline_assembly(
    const SharedSplineAssembly& assembly,
    int sample_u,
    int sample_v,
    SharedSplineSampledMesh& out_mesh,
    SharedSplineAdjacencyReport& out_report) {
    out_mesh = SharedSplineSampledMesh();
    out_report = SharedSplineAdjacencyReport();
    if (!assembly.valid || assembly.patches.empty()) {
        out_report.reason = "assembly is invalid";
        return false;
    }
    int su = std::max(2, sample_u);
    int sv = std::max(2, sample_v);

    vector<Vector3d> vertices;
    vector<Eigen::Vector3i> faces;
    vector<int> patch_ids;
    map<string, int> key_to_vertex;

    for (int pi = 0; pi < (int)assembly.patches.size(); pi++) {
        const SharedSplinePatch& patch = assembly.patches[pi];
        BSplineSurface3D surface = materialize_patch_surface(patch, assembly.pool);
        vector<vector<int>> sample_ids(su, vector<int>(sv, -1));
        for (int i = 0; i < su; i++) {
            double u = (double)i / (double)(su - 1);
            for (int j = 0; j < sv; j++) {
                double v = (double)j / (double)(sv - 1);
                string key = mesh_vertex_key_for_sample(patch, pi, i, j, su, sv);
                auto it = key_to_vertex.find(key);
                if (it == key_to_vertex.end()) {
                    int id = (int)vertices.size();
                    key_to_vertex[key] = id;
                    vertices.push_back(surface.evaluate(u, v));
                    sample_ids[i][j] = id;
                } else {
                    sample_ids[i][j] = it->second;
                }
            }
        }

        for (int i = 0; i < su - 1; i++) {
            for (int j = 0; j < sv - 1; j++) {
                int a = sample_ids[i][j];
                int b = sample_ids[i + 1][j];
                int c = sample_ids[i][j + 1];
                int d = sample_ids[i + 1][j + 1];
                faces.push_back(Eigen::Vector3i(a, b, c));
                patch_ids.push_back(patch.region_id);
                faces.push_back(Eigen::Vector3i(b, d, c));
                patch_ids.push_back(patch.region_id);
            }
        }
    }

    out_mesh.V.resize((int)vertices.size(), 3);
    for (int i = 0; i < (int)vertices.size(); i++) out_mesh.V.row(i) = vertices[i].transpose();
    out_mesh.F.resize((int)faces.size(), 3);
    out_mesh.patch_ids.resize((int)faces.size(), 1);
    for (int i = 0; i < (int)faces.size(); i++) {
        out_mesh.F.row(i) = faces[i];
        out_mesh.patch_ids(i, 0) = patch_ids[i];
    }

    out_mesh.shared_boundary_vertex_ids.clear();
    out_mesh.shared_boundary_vertex_ids.resize(assembly.shared_boundaries.size());
    for (int bi = 0; bi < (int)assembly.shared_boundaries.size(); bi++) {
        int sid = assembly.shared_boundaries[bi].id;
        int count = -1;
        for (const SharedSplinePatch& patch : assembly.patches) {
            for (int side = 0; side < 4; side++) {
                if (patch.shared_boundary_ids[side] != sid) continue;
                int c = side_sample_count(su, sv, side);
                if (count < 0) count = c;
                else count = std::min(count, c);
            }
        }
        if (count < 0) continue;
        out_mesh.shared_boundary_vertex_ids[bi].resize(count, -1);
        for (int k = 0; k < count; k++) {
            string key = "s:" + std::to_string(sid) + ":" + std::to_string(k);
            auto it = key_to_vertex.find(key);
            if (it != key_to_vertex.end()) {
                out_mesh.shared_boundary_vertex_ids[bi][k] = it->second;
            }
        }
    }

    out_report = check_shared_spline_adjacency(assembly, out_mesh, su, sv);
    return out_report.valid;
}

SharedSplineAdjacencyReport check_shared_spline_adjacency(
    const SharedSplineAssembly& assembly,
    const SharedSplineSampledMesh& mesh,
    int sample_u,
    int sample_v) {
    SharedSplineAdjacencyReport report;
    if (!assembly.valid || assembly.patches.size() < 2 || assembly.shared_boundaries.empty()) {
        report.reason = "assembly does not contain a shared boundary";
        return report;
    }

    int shared_id = assembly.shared_boundaries[0].id;
    vector<pair<int, bool>> patch_side_reversal;
    vector<int> patch_side;
    for (const SharedSplinePatch& patch : assembly.patches) {
        for (int s = 0; s < 4; s++) {
            if (patch.shared_boundary_ids[s] == shared_id) {
                patch_side.push_back(s);
                patch_side_reversal.push_back({s, patch.shared_boundary_reversed[s]});
            }
        }
    }
    if (patch_side.size() != 2) {
        report.reason = "shared boundary must be referenced by exactly two patch sides";
        return report;
    }

    int sample_count = side_sample_count(sample_u, sample_v, patch_side[0]);
    double sum = 0.0;
    double max_diff = 0.0;
    for (int k = 0; k < sample_count; k++) {
        Vector3d pts[2];
        for (int p = 0; p < 2; p++) {
            const SharedSplinePatch& patch = assembly.patches[p];
            int side = patch_side[p];
            int local_k = patch.shared_boundary_reversed[side] ? sample_count - 1 - k : k;
            double t = (double)local_k / (double)(sample_count - 1);
            double u = (side == 0 || side == 2) ? t : (side == 1 ? 1.0 : 0.0);
            double v = (side == 1 || side == 3) ? t : (side == 2 ? 1.0 : 0.0);
            pts[p] = evaluate_shared_patch_surface(patch, assembly.pool, u, v);
        }
        double d = (pts[0] - pts[1]).norm();
        sum += d;
        max_diff = std::max(max_diff, d);
    }
    report.mean_boundary_sample_difference = sum / (double)sample_count;
    report.max_boundary_sample_difference = max_diff;

    map<pair<int, int>, int> edge_count;
    for (int fi = 0; fi < mesh.F.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            int a = mesh.F(fi, k);
            int b = mesh.F(fi, (k + 1) % 3);
            edge_count[edge_key(a, b)]++;
        }
    }
    for (const auto& kv : edge_count) {
        if (kv.second == 1) report.outer_boundary_edge_count++;
        else if (kv.second > 2) report.nonmanifold_edge_count++;
    }

    report.shared_boundary_edge_count = sample_count - 1;
    report.shared_boundary_edges_with_two_faces = 0;
    if (!mesh.shared_boundary_vertex_ids.empty()) {
        const vector<int>& ids = mesh.shared_boundary_vertex_ids[0];
        report.shared_boundary_edge_count = std::max(0, (int)ids.size() - 1);
        for (int k = 0; k + 1 < (int)ids.size(); k++) {
            if (ids[k] < 0 || ids[k + 1] < 0) continue;
            auto it = edge_count.find(edge_key(ids[k], ids[k + 1]));
            if (it != edge_count.end() && it->second == 2) {
                report.shared_boundary_edges_with_two_faces++;
            }
        }
    }

    report.watertight_across_shared_boundary =
        report.max_boundary_sample_difference < 1e-10 &&
        report.shared_boundary_edges_with_two_faces == report.shared_boundary_edge_count &&
        report.nonmanifold_edge_count == 0;
    report.valid = report.watertight_across_shared_boundary;
    report.reason = report.valid ? "ok" : "shared boundary adjacency check failed";
    return report;
}

bool export_shared_spline_control_point_ids_csv(
    const string& filename,
    const SharedSplineAssembly& assembly) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "patch_index,region_id,i,j,control_point_id,x,y,z\n";
    for (int pi = 0; pi < (int)assembly.patches.size(); pi++) {
        const SharedSplinePatch& patch = assembly.patches[pi];
        for (int i = 0; i < (int)patch.topology.control_point_ids.size(); i++) {
            for (int j = 0; j < (int)patch.topology.control_point_ids[i].size(); j++) {
                int id = patch.topology.control_point_ids[i][j];
                const Vector3d& p = assembly.pool.values[id];
                fout << pi << "," << patch.region_id << ","
                     << i << "," << j << "," << id << ","
                     << p.x() << "," << p.y() << "," << p.z() << "\n";
            }
        }
    }
    return true;
}

bool export_shared_spline_adjacency_report_csv(
    const string& filename,
    const SharedSplineAdjacencyReport& report) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "valid," << (report.valid ? 1 : 0) << "\n";
    fout << "reason," << report.reason << "\n";
    fout << "max_boundary_sample_difference," << report.max_boundary_sample_difference << "\n";
    fout << "mean_boundary_sample_difference," << report.mean_boundary_sample_difference << "\n";
    fout << "shared_boundary_edge_count," << report.shared_boundary_edge_count << "\n";
    fout << "shared_boundary_edges_with_two_faces," << report.shared_boundary_edges_with_two_faces << "\n";
    fout << "outer_boundary_edge_count," << report.outer_boundary_edge_count << "\n";
    fout << "nonmanifold_edge_count," << report.nonmanifold_edge_count << "\n";
    fout << "watertight_across_shared_boundary," << (report.watertight_across_shared_boundary ? 1 : 0) << "\n";
    return true;
}

bool export_shared_spline_colored_ply(
    const string& filename,
    const SharedSplineSampledMesh& mesh) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "ply\nformat ascii 1.0\n";
    fout << "element vertex " << mesh.V.rows() << "\n";
    fout << "property float x\nproperty float y\nproperty float z\n";
    fout << "element face " << mesh.F.rows() << "\n";
    fout << "property list uchar int vertex_indices\n";
    fout << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    fout << "end_header\n";
    for (int i = 0; i < mesh.V.rows(); i++) {
        fout << mesh.V(i, 0) << " " << mesh.V(i, 1) << " " << mesh.V(i, 2) << "\n";
    }
    for (int i = 0; i < mesh.F.rows(); i++) {
        int pid = mesh.patch_ids.rows() == mesh.F.rows() ? mesh.patch_ids(i, 0) : 0;
        int r = (pid % 2 == 0) ? 60 : 230;
        int g = (pid % 2 == 0) ? 130 : 150;
        int b = (pid % 2 == 0) ? 240 : 60;
        fout << "3 " << mesh.F(i, 0) << " " << mesh.F(i, 1) << " " << mesh.F(i, 2)
             << " " << r << " " << g << " " << b << "\n";
    }
    return true;
}

bool export_shared_spline_debug(
    const string& prefix,
    const SharedSplineAssembly& assembly,
    const SharedSplineSampledMesh& mesh,
    const SharedSplineAdjacencyReport& report) {
    bool ok = true;
    ok = export_mesh_obj(prefix + "_watertight_mesh.obj", mesh) && ok;
    ok = export_shared_spline_colored_ply(prefix + "_colored_patches.ply", mesh) && ok;
    ok = export_shared_boundary_obj(prefix + "_shared_boundary.obj", assembly) && ok;
    ok = export_shared_spline_control_point_ids_csv(prefix + "_control_point_ids.csv", assembly) && ok;
    ok = export_shared_spline_adjacency_report_csv(prefix + "_adjacency_report.csv", report) && ok;
    return ok;
}
