#include "region_boundary.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <queue>
#include <set>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector3d;
using std::array;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::vector;

namespace {

static pair<int, int> edge_key(int a, int b) {
    return std::make_pair(std::min(a, b), std::max(a, b));
}

static bool valid_mesh_inputs(
    const MatrixXd& V,
    const MatrixXi& F,
    const vector<int>& face_region_ids,
    string& reason) {
    if (V.cols() != 3) {
        reason = "V must have 3 columns";
        return false;
    }
    if (F.cols() != 3) {
        reason = "F must have 3 columns";
        return false;
    }
    if ((int)face_region_ids.size() != F.rows()) {
        reason = "face_region_ids size must equal F.rows()";
        return false;
    }
    for (int fi = 0; fi < F.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            int vid = F(fi, k);
            if (vid < 0 || vid >= V.rows()) {
                reason = "F contains an invalid vertex index";
                return false;
            }
        }
    }
    return true;
}

static bool check_target_region_connected(
    const vector<int>& region_face_ids,
    const MatrixXi& F,
    const vector<int>& face_region_ids,
    int target_region_id,
    bool& connected) {
    connected = true;
    if (region_face_ids.empty()) return true;

    map<pair<int, int>, vector<int>> edge_to_faces;
    for (int fi : region_face_ids) {
        for (int k = 0; k < 3; k++) {
            edge_to_faces[edge_key(F(fi, k), F(fi, (k + 1) % 3))].push_back(fi);
        }
    }

    map<int, vector<int>> face_adj;
    for (const auto& kv : edge_to_faces) {
        const vector<int>& faces = kv.second;
        for (int i = 0; i < (int)faces.size(); i++) {
            for (int j = i + 1; j < (int)faces.size(); j++) {
                if (face_region_ids[faces[i]] == target_region_id &&
                    face_region_ids[faces[j]] == target_region_id) {
                    face_adj[faces[i]].push_back(faces[j]);
                    face_adj[faces[j]].push_back(faces[i]);
                }
            }
        }
    }

    set<int> target_set(region_face_ids.begin(), region_face_ids.end());
    set<int> visited;
    std::queue<int> q;
    q.push(region_face_ids.front());
    visited.insert(region_face_ids.front());

    while (!q.empty()) {
        int fi = q.front();
        q.pop();
        for (int nb : face_adj[fi]) {
            if (!target_set.count(nb)) continue;
            if (visited.insert(nb).second) q.push(nb);
        }
    }

    connected = visited.size() == region_face_ids.size();
    return true;
}

static int count_boundary_components(
    const map<int, vector<int>>& vertex_adj) {
    set<int> visited;
    int components = 0;
    for (const auto& kv : vertex_adj) {
        int start = kv.first;
        if (visited.count(start)) continue;
        components++;
        vector<int> stack(1, start);
        visited.insert(start);
        while (!stack.empty()) {
            int v = stack.back();
            stack.pop_back();
            auto it = vertex_adj.find(v);
            if (it == vertex_adj.end()) continue;
            for (int nb : it->second) {
                if (visited.insert(nb).second) stack.push_back(nb);
            }
        }
    }
    return components;
}

static void fill_loop_positions(
    RegionBoundaryLoop& loop,
    const MatrixXd& V) {
    loop.positions.clear();
    loop.positions.reserve(loop.vertex_ids.size());
    for (int vid : loop.vertex_ids) {
        loop.positions.push_back(V.row(vid).transpose());
    }
}

} // namespace

RegionBoundaryLoop::RegionBoundaryLoop()
    : closed(false) {}

RegionBoundaryExtractionResult::RegionBoundaryExtractionResult()
    : success(false),
      boundary_loop_count(0),
      region_connected(false),
      nonmanifold_boundary(false),
      broken_chain(false),
      duplicate_edge(false) {}

RegionBoundaryExtractionResult extract_region_boundary_loop(
    const MatrixXd& V,
    const MatrixXi& F,
    const vector<int>& face_region_ids,
    int target_region_id) {
    RegionBoundaryExtractionResult result;

    string reason;
    if (!valid_mesh_inputs(V, F, face_region_ids, reason)) {
        result.reason = reason;
        return result;
    }

    for (int fi = 0; fi < F.rows(); fi++) {
        if (face_region_ids[fi] == target_region_id) {
            result.region_face_ids.push_back(fi);
        }
    }
    if (result.region_face_ids.empty()) {
        result.reason = "target region has no faces";
        return result;
    }

    map<pair<int, int>, vector<int>> edge_to_faces;
    set<pair<pair<int, int>, int>> seen_face_edges;
    for (int fi = 0; fi < F.rows(); fi++) {
        set<pair<int, int>> local_edges;
        for (int k = 0; k < 3; k++) {
            pair<int, int> e = edge_key(F(fi, k), F(fi, (k + 1) % 3));
            if (e.first == e.second || !local_edges.insert(e).second) {
                result.duplicate_edge = true;
            }
            pair<pair<int, int>, int> face_edge(e, fi);
            if (!seen_face_edges.insert(face_edge).second) {
                result.duplicate_edge = true;
            }
            edge_to_faces[e].push_back(fi);
        }
    }
    if (result.duplicate_edge) {
        result.reason = "duplicate or degenerate mesh edge in a face";
        return result;
    }

    check_target_region_connected(
        result.region_face_ids, F, face_region_ids,
        target_region_id, result.region_connected);
    if (!result.region_connected) {
        result.reason = "target region is not edge-connected";
        return result;
    }

    for (const auto& kv : edge_to_faces) {
        const pair<int, int>& e = kv.first;
        const vector<int>& faces = kv.second;
        int target_count = 0;
        int outside_count = 0;
        for (int fi : faces) {
            if (face_region_ids[fi] == target_region_id) target_count++;
            else outside_count++;
        }

        if ((int)faces.size() > 2 && target_count > 0) {
            result.nonmanifold_boundary = true;
            result.reason = "target region touches a non-manifold mesh edge";
            return result;
        }

        bool is_boundary = false;
        if (faces.size() == 1) {
            is_boundary = target_count == 1;
        } else {
            is_boundary = target_count > 0 && outside_count > 0;
        }

        if (is_boundary) {
            result.boundary_edges.push_back({e.first, e.second});
        }
    }

    if (result.boundary_edges.empty()) {
        result.reason = "target region has no boundary edges";
        return result;
    }

    map<int, vector<int>> vertex_adj;
    set<pair<int, int>> unique_boundary_edges;
    for (const array<int, 2>& e : result.boundary_edges) {
        pair<int, int> key = edge_key(e[0], e[1]);
        if (!unique_boundary_edges.insert(key).second) {
            result.duplicate_edge = true;
            result.reason = "duplicate boundary edge";
            return result;
        }
        vertex_adj[e[0]].push_back(e[1]);
        vertex_adj[e[1]].push_back(e[0]);
    }

    result.boundary_loop_count = count_boundary_components(vertex_adj);
    if (result.boundary_loop_count > 1) {
        result.reason = "target region has multiple boundary loops";
        return result;
    }

    for (const auto& kv : vertex_adj) {
        int degree = (int)kv.second.size();
        if (degree != 2) {
            if (degree <= 1) {
                result.broken_chain = true;
                result.reason = "boundary is an open or broken chain";
            } else {
                result.nonmanifold_boundary = true;
                result.reason = "boundary vertex has degree greater than 2";
            }
            return result;
        }
    }

    int start = vertex_adj.begin()->first;
    int prev = -1;
    int cur = start;
    set<int> visited_vertices;
    vector<int> ordered;
    ordered.reserve(vertex_adj.size());

    for (;;) {
        if (visited_vertices.count(cur)) {
            result.reason = "boundary walk revisited a vertex before closing";
            return result;
        }
        visited_vertices.insert(cur);
        ordered.push_back(cur);

        vector<int> nbrs = vertex_adj[cur];
        std::sort(nbrs.begin(), nbrs.end());
        int next = -1;
        if (prev < 0) {
            next = nbrs[0];
        } else {
            next = (nbrs[0] == prev) ? nbrs[1] : nbrs[0];
        }

        if (next == start) {
            break;
        }
        prev = cur;
        cur = next;

        if ((int)ordered.size() > (int)vertex_adj.size()) {
            result.reason = "boundary walk exceeded boundary vertex count";
            return result;
        }
    }

    if (visited_vertices.size() != vertex_adj.size()) {
        result.reason = "boundary loop did not visit all boundary vertices";
        return result;
    }
    if (ordered.size() < 3 || result.boundary_edges.size() != ordered.size()) {
        result.reason = "boundary is not a single simple closed loop";
        return result;
    }

    result.loop.vertex_ids = ordered;
    result.loop.closed = true;
    fill_loop_positions(result.loop, V);
    result.success = true;
    result.reason = "ok";
    return result;
}

bool export_region_boundary_debug_obj(
    const string& filename,
    const MatrixXd& V,
    const MatrixXi& F,
    const vector<int>& face_region_ids,
    int target_region_id,
    const RegionBoundaryExtractionResult& result) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;

    fout << "# Region boundary debug OBJ\n";
    fout << "# target_region_id " << target_region_id << "\n";
    fout << "# success " << (result.success ? 1 : 0) << "\n";
    fout << "# reason " << result.reason << "\n";

    for (int i = 0; i < V.rows(); i++) {
        fout << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
    }

    fout << "g region_triangles\n";
    for (int fi = 0; fi < F.rows(); fi++) {
        if (face_region_ids[fi] != target_region_id) continue;
        fout << "f " << (F(fi, 0) + 1)
             << " " << (F(fi, 1) + 1)
             << " " << (F(fi, 2) + 1) << "\n";
    }

    fout << "g boundary_edges\n";
    for (const array<int, 2>& e : result.boundary_edges) {
        fout << "l " << (e[0] + 1) << " " << (e[1] + 1) << "\n";
    }

    fout << "g boundary_vertices\n";
    for (int vid : result.loop.vertex_ids) {
        fout << "p " << (vid + 1) << "\n";
    }

    return true;
}
