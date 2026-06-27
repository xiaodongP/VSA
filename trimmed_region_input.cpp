#include "trimmed_region_input.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <queue>
#include <set>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SelfAdjointEigenSolver;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::map;
using std::set;
using std::string;
using std::vector;

namespace {

struct EdgeFaces {
    vector<int> faces;
};

static bool valid_inputs(
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
        set<int> local_vertices;
        for (int k = 0; k < 3; k++) {
            int vid = F(fi, k);
            if (vid < 0 || vid >= V.rows()) {
                reason = "F contains an invalid vertex index";
                return false;
            }
            if (!local_vertices.insert(vid).second) {
                reason = "F contains a degenerate face with duplicate vertices";
                return false;
            }
        }
    }
    return true;
}

static map<EdgeKey, EdgeFaces> build_edge_faces(const MatrixXi& F) {
    map<EdgeKey, EdgeFaces> edge_faces;
    for (int fi = 0; fi < F.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            edge_faces[EdgeKey(F(fi, k), F(fi, (k + 1) % 3))].faces.push_back(fi);
        }
    }
    return edge_faces;
}

static bool region_is_connected(
    const MatrixXi& F,
    const vector<int>& face_ids,
    const map<EdgeKey, EdgeFaces>& edge_faces) {
    if (face_ids.empty()) return true;

    set<int> face_set(face_ids.begin(), face_ids.end());
    set<int> visited;
    std::queue<int> q;
    q.push(face_ids.front());
    visited.insert(face_ids.front());

    while (!q.empty()) {
        int fi = q.front();
        q.pop();
        for (int k = 0; k < 3; k++) {
            EdgeKey key(F(fi, k), F(fi, (k + 1) % 3));
            auto it = edge_faces.find(key);
            if (it == edge_faces.end()) continue;
            for (int nb : it->second.faces) {
                if (!face_set.count(nb)) continue;
                if (visited.insert(nb).second) q.push(nb);
            }
        }
    }

    return visited.size() == face_ids.size();
}

static double polyline_length(const vector<Vector3d>& points) {
    double length = 0.0;
    for (int i = 1; i < (int)points.size(); i++) {
        length += (points[i] - points[i - 1]).norm();
    }
    return length;
}

static Vector3d normalized_or_zero(const Vector3d& v) {
    double n = v.norm();
    if (n <= 1e-14) return Vector3d::Zero();
    return v / n;
}

static double projected_loop_abs_area(const vector<Vector3d>& positions) {
    if (positions.size() < 3) return 0.0;

    Vector3d centroid = Vector3d::Zero();
    for (const Vector3d& p : positions) centroid += p;
    centroid /= (double)positions.size();

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const Vector3d& p : positions) {
        Vector3d d = p - centroid;
        cov += d * d.transpose();
    }

    SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if (solver.info() != Eigen::Success) return 0.0;
    Vector3d axis0 = solver.eigenvectors().col(2);
    Vector3d axis1 = solver.eigenvectors().col(1);

    vector<Vector2d> uv;
    uv.reserve(positions.size());
    for (const Vector3d& p : positions) {
        Vector3d d = p - centroid;
        uv.push_back(Vector2d(d.dot(axis0), d.dot(axis1)));
    }

    double area2 = 0.0;
    for (int i = 0; i < (int)uv.size(); i++) {
        const Vector2d& a = uv[i];
        const Vector2d& b = uv[(i + 1) % uv.size()];
        area2 += a.x() * b.y() - a.y() * b.x();
    }
    return std::abs(0.5 * area2);
}

static void smooth_guide_positions(
    BoundarySegment& segment,
    const BoundarySegmentationConfig& config) {
    segment.guide_positions = segment.authoritative_positions;
    if (config.guide_smoothing_iterations <= 0 ||
        segment.guide_positions.size() < 4) {
        return;
    }

    const double w = std::max(0.0, std::min(1.0, config.guide_smoothing_weight));
    vector<Vector3d> current = segment.guide_positions;
    vector<Vector3d> next = current;
    for (int iter = 0; iter < config.guide_smoothing_iterations; iter++) {
        for (int i = 1; i + 1 < (int)current.size(); i++) {
            Vector3d avg = 0.5 * (current[i - 1] + current[i + 1]);
            next[i] = (1.0 - w) * current[i] + w * avg;
        }
        current.swap(next);
    }
    segment.guide_positions = current;
}

static bool same_segmentation_state(
    const DirectedBoundaryEdge& a,
    const DirectedBoundaryEdge& b) {
    return a.adjacent_region_id == b.adjacent_region_id &&
           a.is_mesh_boundary == b.is_mesh_boundary &&
           a.is_feature_barrier == b.is_feature_barrier &&
           a.is_user_marker == b.is_user_marker;
}

static bool build_directed_boundary_edges(
    const MatrixXi& F,
    const vector<int>& face_region_ids,
    const RegionFaceSet& region,
    const BoundarySegmentationConfig& config,
    const map<EdgeKey, EdgeFaces>& edge_faces,
    vector<DirectedBoundaryEdge>& boundary_edges,
    string& reason) {
    set<int> region_face_set(region.face_ids.begin(), region.face_ids.end());
    for (int fi : region.face_ids) {
        for (int k = 0; k < 3; k++) {
            int from = F(fi, k);
            int to = F(fi, (k + 1) % 3);
            EdgeKey key(from, to);
            auto it = edge_faces.find(key);
            if (it == edge_faces.end()) {
                reason = "internal error: mesh edge missing from edge map";
                return false;
            }

            const vector<int>& faces = it->second.faces;
            if (faces.size() > 2) {
                reason = "region touches a non-manifold mesh edge";
                return false;
            }

            int region_count = 0;
            int adjacent_face = -1;
            for (int ef : faces) {
                if (region_face_set.count(ef)) {
                    region_count++;
                } else {
                    adjacent_face = ef;
                }
            }

            if (region_count > 1) continue;

            DirectedBoundaryEdge edge;
            edge.from = from;
            edge.to = to;
            edge.key = key;
            edge.region_face_id = fi;
            edge.is_mesh_boundary = adjacent_face < 0;
            edge.adjacent_region_id =
                edge.is_mesh_boundary ? -1 : face_region_ids[adjacent_face];
            edge.is_feature_barrier = config.feature_edges.count(key) > 0;
            edge.is_user_marker = config.marker_edges.count(key) > 0 ||
                                  config.marker_vertex_ids.count(from) > 0 ||
                                  config.marker_vertex_ids.count(to) > 0;
            boundary_edges.push_back(edge);
        }
    }

    if (boundary_edges.empty()) {
        reason = "region has no boundary edges";
        return false;
    }
    return true;
}

static bool order_boundary_loops(
    const MatrixXd& V,
    const vector<DirectedBoundaryEdge>& boundary_edges,
    vector<AuthoritativeBoundaryLoop>& loops,
    string& reason) {
    map<int, int> outgoing;
    map<int, int> incoming;
    for (int i = 0; i < (int)boundary_edges.size(); i++) {
        const DirectedBoundaryEdge& e = boundary_edges[i];
        if (outgoing.count(e.from)) {
            reason = "non-manifold boundary: a boundary vertex has multiple outgoing edges";
            return false;
        }
        if (incoming.count(e.to)) {
            reason = "non-manifold boundary: a boundary vertex has multiple incoming edges";
            return false;
        }
        outgoing[e.from] = i;
        incoming[e.to] = i;
    }

    for (const auto& kv : outgoing) {
        if (!incoming.count(kv.first)) {
            reason = "broken boundary chain: missing incoming edge";
            return false;
        }
    }
    for (const auto& kv : incoming) {
        if (!outgoing.count(kv.first)) {
            reason = "broken boundary chain: missing outgoing edge";
            return false;
        }
    }

    vector<char> visited(boundary_edges.size(), 0);
    for (int start_edge = 0; start_edge < (int)boundary_edges.size(); start_edge++) {
        if (visited[start_edge]) continue;

        AuthoritativeBoundaryLoop loop;
        loop.id = (int)loops.size();
        int edge_index = start_edge;
        int start_vertex = boundary_edges[start_edge].from;
        set<int> vertices_seen;

        for (;;) {
            if (visited[edge_index]) {
                reason = "boundary walk revisited an edge before closing";
                return false;
            }
            const DirectedBoundaryEdge& edge = boundary_edges[edge_index];
            visited[edge_index] = 1;
            if (!vertices_seen.insert(edge.from).second) {
                reason = "boundary loop self-intersects at a repeated vertex";
                return false;
            }
            loop.vertex_ids.push_back(edge.from);
            loop.positions.push_back(V.row(edge.from).transpose());
            loop.directed_edges.push_back(edge);

            int next_vertex = edge.to;
            if (next_vertex == start_vertex) {
                loop.closed = true;
                break;
            }
            auto it = outgoing.find(next_vertex);
            if (it == outgoing.end()) {
                reason = "broken boundary chain during traversal";
                return false;
            }
            edge_index = it->second;
        }

        if (loop.vertex_ids.size() < 3 ||
            loop.vertex_ids.size() != loop.directed_edges.size()) {
            reason = "boundary loop has fewer than three edges";
            return false;
        }
        loop.projected_abs_area = projected_loop_abs_area(loop.positions);
        loops.push_back(loop);
    }

    int visited_count = 0;
    for (char v : visited) {
        if (v) visited_count++;
    }
    if (visited_count != (int)boundary_edges.size()) {
        reason = "boundary coverage failed: not all boundary edges were visited";
        return false;
    }
    return true;
}

static vector<int> perimeter_break_indices(
    const AuthoritativeBoundaryLoop& loop,
    const BoundarySegmentationConfig& config) {
    const int n = (int)loop.directed_edges.size();
    vector<int> breaks;
    for (int i = 0; i < n; i++) {
        const DirectedBoundaryEdge& prev = loop.directed_edges[(i + n - 1) % n];
        const DirectedBoundaryEdge& cur = loop.directed_edges[i];
        bool is_break = !same_segmentation_state(prev, cur);
        is_break = is_break || prev.is_feature_barrier || cur.is_feature_barrier;
        is_break = is_break || prev.is_user_marker || cur.is_user_marker;
        is_break = is_break || config.marker_vertex_ids.count(loop.vertex_ids[i]) > 0;
        if (is_break) breaks.push_back(i);
    }
    if (breaks.empty()) breaks.push_back(0);
    return breaks;
}

static BoundarySegment make_segment(
    const MatrixXd& V,
    const AuthoritativeBoundaryLoop& loop,
    int id,
    int start_edge,
    int end_edge,
    bool full_loop,
    const BoundarySegmentationConfig& config) {
    BoundarySegment segment;
    segment.id = id;
    segment.loop_id = loop.id;
    segment.adjacent_region_id = loop.directed_edges[start_edge].adjacent_region_id;

    int n = (int)loop.directed_edges.size();
    int edge_index = start_edge;
    segment.authoritative_vertex_ids.push_back(loop.directed_edges[start_edge].from);
    segment.authoritative_positions.push_back(
        V.row(loop.directed_edges[start_edge].from).transpose());

    for (;;) {
        const DirectedBoundaryEdge& edge = loop.directed_edges[edge_index];
        segment.edge_keys.push_back(edge.key);
        segment.authoritative_vertex_ids.push_back(edge.to);
        segment.authoritative_positions.push_back(V.row(edge.to).transpose());
        segment.touches_feature_barrier =
            segment.touches_feature_barrier || edge.is_feature_barrier;
        segment.touches_mesh_boundary =
            segment.touches_mesh_boundary || edge.is_mesh_boundary;
        segment.touches_user_marker =
            segment.touches_user_marker || edge.is_user_marker ||
            config.marker_vertex_ids.count(edge.from) > 0 ||
            config.marker_vertex_ids.count(edge.to) > 0;

        edge_index = (edge_index + 1) % n;
        if (full_loop) {
            if (edge_index == start_edge) break;
        } else {
            if (edge_index == end_edge) break;
        }
    }

    segment.length = polyline_length(segment.authoritative_positions);
    if (segment.authoritative_positions.size() >= 2) {
        segment.tangent_begin = normalized_or_zero(
            segment.authoritative_positions[1] - segment.authoritative_positions[0]);
        int last = (int)segment.authoritative_positions.size() - 1;
        segment.tangent_end = normalized_or_zero(
            segment.authoritative_positions[last] -
            segment.authoritative_positions[last - 1]);
    }
    smooth_guide_positions(segment, config);
    return segment;
}

static bool build_perimeter_segments(
    const MatrixXd& V,
    const AuthoritativeBoundaryLoop& perimeter,
    const BoundarySegmentationConfig& config,
    vector<BoundarySegment>& segments,
    string& reason) {
    vector<int> breaks = perimeter_break_indices(perimeter, config);
    if (breaks.empty()) {
        reason = "internal error: no segmentation break point was produced";
        return false;
    }

    if (breaks.size() == 1) {
        segments.push_back(make_segment(
            V, perimeter, 0, breaks[0], breaks[0], true, config));
        return true;
    }

    for (int bi = 0; bi < (int)breaks.size(); bi++) {
        int start = breaks[bi];
        int end = breaks[(bi + 1) % breaks.size()];
        segments.push_back(make_segment(
            V, perimeter, (int)segments.size(), start, end, false, config));
    }

    for (const BoundarySegment& segment : segments) {
        if (segment.edge_keys.empty()) {
            reason = "empty boundary segment produced";
            return false;
        }
    }
    return true;
}

static bool write_point(std::ofstream& fout, const Vector3d& p) {
    if (!fout.good()) return false;
    fout << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
    return true;
}

} // namespace

BoundarySegmentationResult build_trimmed_region_input(
    const MatrixXd& V,
    const MatrixXi& F,
    const vector<int>& face_region_ids,
    int target_region_id,
    const BoundarySegmentationConfig& config) {
    RegionFaceSet region;
    region.region_id = target_region_id;
    for (int fi = 0; fi < (int)face_region_ids.size(); fi++) {
        if (face_region_ids[fi] == target_region_id) region.face_ids.push_back(fi);
    }
    return build_trimmed_region_input_from_face_set(
        V, F, face_region_ids, region, config);
}

BoundarySegmentationResult build_trimmed_region_input_from_face_set(
    const MatrixXd& V,
    const MatrixXi& F,
    const vector<int>& face_region_ids,
    const RegionFaceSet& region,
    const BoundarySegmentationConfig& config) {
    BoundarySegmentationResult result;
    result.region = region;

    string reason;
    if (!valid_inputs(V, F, face_region_ids, reason)) {
        result.reason = reason;
        return result;
    }

    if (region.face_ids.empty()) {
        result.reason = "region face set is empty";
        return result;
    }

    set<int> unique_faces;
    for (int fi : region.face_ids) {
        if (fi < 0 || fi >= F.rows()) {
            result.reason = "region face set contains an invalid face id";
            return result;
        }
        if (!unique_faces.insert(fi).second) {
            result.reason = "region face set contains duplicate face ids";
            return result;
        }
    }

    map<EdgeKey, EdgeFaces> edge_faces = build_edge_faces(F);
    result.manifold = true;
    for (const auto& kv : edge_faces) {
        if (kv.second.faces.size() > 2) {
            result.manifold = false;
            result.reason = "mesh contains a non-manifold edge";
            return result;
        }
    }

    result.region_connected = region_is_connected(F, region.face_ids, edge_faces);
    if (!result.region_connected) {
        result.reason = "region face set is not edge-connected";
        return result;
    }

    vector<DirectedBoundaryEdge> boundary_edges;
    if (!build_directed_boundary_edges(
            F, face_region_ids, region, config, edge_faces, boundary_edges, reason)) {
        result.reason = reason;
        result.manifold = false;
        return result;
    }

    if (!order_boundary_loops(V, boundary_edges, result.loops, reason)) {
        result.reason = reason;
        result.manifold = false;
        return result;
    }
    result.boundary_coverage_valid = true;
    result.orientation_valid = true;

    double best_area = -1.0;
    for (int i = 0; i < (int)result.loops.size(); i++) {
        if (result.loops[i].projected_abs_area > best_area) {
            best_area = result.loops[i].projected_abs_area;
            result.perimeter_loop_index = i;
        }
    }

    if (result.perimeter_loop_index < 0) {
        result.reason = "failed to identify perimeter loop";
        return result;
    }
    for (int i = 0; i < (int)result.loops.size(); i++) {
        result.loops[i].is_perimeter = i == result.perimeter_loop_index;
    }

    if (!build_perimeter_segments(
            V,
            result.loops[result.perimeter_loop_index],
            config,
            result.perimeter_segments,
            reason)) {
        result.reason = reason;
        return result;
    }

    result.valid = true;
    result.reason = "ok";
    return result;
}

bool export_trimmed_region_input_debug_obj(
    const string& filename,
    const MatrixXd& V,
    const MatrixXi& F,
    const BoundarySegmentationResult& result) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;

    fout << "# Trimmed region input debug OBJ\n";
    fout << "# valid " << (result.valid ? 1 : 0) << "\n";
    fout << "# reason " << result.reason << "\n";
    fout << "# region_id " << result.region.region_id << "\n";
    fout << "# loop_count " << result.loops.size() << "\n";
    fout << "# perimeter_loop_index " << result.perimeter_loop_index << "\n";
    fout << "# segment_count " << result.perimeter_segments.size() << "\n";

    for (int i = 0; i < V.rows(); i++) {
        fout << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
    }

    fout << "g region_faces\n";
    for (int fi : result.region.face_ids) {
        fout << "f " << (F(fi, 0) + 1)
             << " " << (F(fi, 1) + 1)
             << " " << (F(fi, 2) + 1) << "\n";
    }

    for (const AuthoritativeBoundaryLoop& loop : result.loops) {
        fout << "g loop_" << loop.id
             << (loop.is_perimeter ? "_perimeter" : "_inner") << "\n";
        for (const DirectedBoundaryEdge& edge : loop.directed_edges) {
            fout << "l " << (edge.from + 1) << " " << (edge.to + 1) << "\n";
        }
    }

    for (const BoundarySegment& segment : result.perimeter_segments) {
        fout << "g segment_" << segment.id
             << "_adj_" << segment.adjacent_region_id << "\n";
        for (int i = 1; i < (int)segment.authoritative_vertex_ids.size(); i++) {
            fout << "l " << (segment.authoritative_vertex_ids[i - 1] + 1)
                 << " " << (segment.authoritative_vertex_ids[i] + 1) << "\n";
        }
    }

    int base = (int)V.rows();
    fout << "g guide_positions\n";
    for (const BoundarySegment& segment : result.perimeter_segments) {
        for (const Vector3d& p : segment.guide_positions) {
            write_point(fout, p);
        }
    }

    int cursor = base + 1;
    for (const BoundarySegment& segment : result.perimeter_segments) {
        if (segment.guide_positions.empty()) continue;
        fout << "g segment_" << segment.id << "_guide_polyline\n";
        for (int i = 1; i < (int)segment.guide_positions.size(); i++) {
            fout << "l " << (cursor + i - 1) << " " << (cursor + i) << "\n";
        }
        cursor += (int)segment.guide_positions.size();
    }

    return fout.good();
}
