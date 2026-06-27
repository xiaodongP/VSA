#include "rotation_angle_parameterization.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <set>

using Eigen::Matrix2d;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::vector;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 1e-12;

struct LocalTri {
    Vector2d x[3];
    double area = 0.0;
    bool valid = false;
};

struct DualEdge {
    int face_a = -1;
    int face_b = -1;
    int va = -1;
    int vb = -1;
};

struct RegionMesh {
    vector<int> region_vertices;
    map<int, int> global_to_local_vertex;
    map<int, int> global_face_to_local_face;
    MatrixXi local_faces;
    vector<int> local_face_to_global_face;
};

struct BoundaryUvConstraintData {
    vector<int> active_side_indices;
    vector<ParameterSideRole> roles;
    map<int, Vector2d> fixed_uv;
    vector<int> side_of_vertex;
    vector<vector<int>> side_vertices;
    vector<vector<double>> side_params;
    double width = 1.0;
    double height = 1.0;
};

static pair<int, int> edge_key(int a, int b) {
    return std::make_pair(std::min(a, b), std::max(a, b));
}

static double cross2(const Vector2d& a, const Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

static double signed_area_uv(const Vector2d& a, const Vector2d& b, const Vector2d& c) {
    return 0.5 * cross2(b - a, c - a);
}

static double tri_area_3d(const Vector3d& a, const Vector3d& b, const Vector3d& c) {
    return 0.5 * (b - a).cross(c - a).norm();
}

static double angle_between_edges(const Vector3d& a, const Vector3d& b) {
    double denom = std::max(kEps, a.norm() * b.norm());
    double c = std::max(-1.0, std::min(1.0, a.dot(b) / denom));
    return std::acos(c);
}

static double normalize_angle_pi(double a) {
    while (a > kPi) a -= 2.0 * kPi;
    while (a < -kPi) a += 2.0 * kPi;
    return a;
}

static Matrix2d rotation2(double a) {
    double c = std::cos(a);
    double s = std::sin(a);
    Matrix2d R;
    R << c, -s, s, c;
    return R;
}

static ParameterSideRole role_from_cardinal(const string& name) {
    if (name == "South") return ParameterSideRole::South;
    if (name == "East") return ParameterSideRole::East;
    if (name == "North") return ParameterSideRole::North;
    if (name == "West") return ParameterSideRole::West;
    return ParameterSideRole::Unassigned;
}

static vector<ParameterSideRole> resolve_roles(const AutomaticLabelingResult& labeling) {
    vector<ParameterSideRole> roles(labeling.abstract_sides.size(), ParameterSideRole::Unassigned);
    for (int i = 0; i < (int)roles.size(); i++) {
        if (labeling.orientation.valid && i < 4) {
            roles[i] = role_from_cardinal(labeling.orientation.side_to_cardinal[i]);
        }
        if (roles[i] == ParameterSideRole::Unassigned) {
            if (i == 0) roles[i] = ParameterSideRole::South;
            else if (i == 1) roles[i] = ParameterSideRole::East;
            else if (i == 2) roles[i] = ParameterSideRole::North;
            else if (i == 3) roles[i] = ParameterSideRole::West;
        }
    }
    return roles;
}

static double role_axis_angle(ParameterSideRole role) {
    switch (role) {
    case ParameterSideRole::South: return 0.0;
    case ParameterSideRole::East: return 0.5 * kPi;
    case ParameterSideRole::North: return kPi;
    case ParameterSideRole::West: return -0.5 * kPi;
    case ParameterSideRole::Unassigned: return 0.0;
    }
    return 0.0;
}

static Vector2d role_start_uv(ParameterSideRole role, double w, double h) {
    switch (role) {
    case ParameterSideRole::South: return Vector2d(0.0, 0.0);
    case ParameterSideRole::East: return Vector2d(w, 0.0);
    case ParameterSideRole::North: return Vector2d(w, h);
    case ParameterSideRole::West: return Vector2d(0.0, h);
    case ParameterSideRole::Unassigned: return Vector2d(0.0, 0.0);
    }
    return Vector2d(0.0, 0.0);
}

static Vector2d role_end_uv(ParameterSideRole role, double w, double h) {
    switch (role) {
    case ParameterSideRole::South: return Vector2d(w, 0.0);
    case ParameterSideRole::East: return Vector2d(w, h);
    case ParameterSideRole::North: return Vector2d(0.0, h);
    case ParameterSideRole::West: return Vector2d(0.0, 0.0);
    case ParameterSideRole::Unassigned: return Vector2d(0.0, 0.0);
    }
    return Vector2d(0.0, 0.0);
}

static bool collect_region_mesh(
    const MatrixXi& F,
    const BoundarySegmentationResult& input,
    RegionMesh& mesh,
    string& reason) {
    set<int> vertices;
    mesh.local_face_to_global_face = input.region.face_ids;
    if (mesh.local_face_to_global_face.empty()) {
        reason = "region has no faces";
        return false;
    }
    for (int fi : mesh.local_face_to_global_face) {
        if (fi < 0 || fi >= F.rows()) {
            reason = "region has invalid face id";
            return false;
        }
        mesh.global_face_to_local_face[fi] = (int)mesh.global_face_to_local_face.size();
        for (int k = 0; k < 3; k++) vertices.insert(F(fi, k));
    }
    mesh.region_vertices.assign(vertices.begin(), vertices.end());
    for (int i = 0; i < (int)mesh.region_vertices.size(); i++) {
        mesh.global_to_local_vertex[mesh.region_vertices[i]] = i;
    }
    mesh.local_faces.resize((int)mesh.local_face_to_global_face.size(), 3);
    for (int li = 0; li < (int)mesh.local_face_to_global_face.size(); li++) {
        int fi = mesh.local_face_to_global_face[li];
        for (int k = 0; k < 3; k++) {
            mesh.local_faces(li, k) = mesh.global_to_local_vertex[F(fi, k)];
        }
    }
    return true;
}

static LocalTri local_triangle(const Vector3d& p0, const Vector3d& p1, const Vector3d& p2) {
    LocalTri tri;
    double l01 = (p1 - p0).norm();
    double l02 = (p2 - p0).norm();
    if (l01 <= kEps || l02 <= kEps) return tri;
    double x = (p2 - p0).dot(p1 - p0) / l01;
    double y2 = l02 * l02 - x * x;
    if (y2 <= kEps) return tri;
    tri.x[0] = Vector2d(0, 0);
    tri.x[1] = Vector2d(l01, 0);
    tri.x[2] = Vector2d(x, std::sqrt(y2));
    tri.area = 0.5 * l01 * tri.x[2].y();
    tri.valid = true;
    return tri;
}

static bool local_face_edge_angle(
    const MatrixXd& V,
    const MatrixXi& F,
    int gface,
    int from,
    int to,
    double& angle) {
    if (gface < 0 || gface >= F.rows()) return false;
    int ia = -1;
    int ib = -1;
    for (int k = 0; k < 3; k++) {
        if (F(gface, k) == from) ia = k;
        if (F(gface, k) == to) ib = k;
    }
    if (ia < 0 || ib < 0) return false;
    LocalTri tri = local_triangle(V.row(F(gface, 0)),
                                  V.row(F(gface, 1)),
                                  V.row(F(gface, 2)));
    if (!tri.valid) return false;
    Vector2d e = tri.x[ib] - tri.x[ia];
    if (e.norm() <= kEps) return false;
    angle = std::atan2(e.y(), e.x());
    return true;
}

static vector<LocalTri> build_local_triangles(
    const MatrixXd& V,
    const MatrixXi& F,
    const RegionMesh& mesh) {
    vector<LocalTri> tris(mesh.local_face_to_global_face.size());
    for (int li = 0; li < (int)mesh.local_face_to_global_face.size(); li++) {
        int fi = mesh.local_face_to_global_face[li];
        tris[li] = local_triangle(V.row(F(fi, 0)), V.row(F(fi, 1)), V.row(F(fi, 2)));
    }
    return tris;
}

static map<pair<int, int>, vector<int>> build_edge_to_local_faces(const RegionMesh& mesh) {
    map<pair<int, int>, vector<int>> edge_faces;
    for (int fi = 0; fi < mesh.local_faces.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            int a = mesh.local_faces(fi, k);
            int b = mesh.local_faces(fi, (k + 1) % 3);
            edge_faces[edge_key(a, b)].push_back(fi);
        }
    }
    return edge_faces;
}

static vector<DualEdge> build_dual_edges(const RegionMesh& mesh) {
    vector<DualEdge> duals;
    auto edge_faces = build_edge_to_local_faces(mesh);
    for (const auto& kv : edge_faces) {
        if (kv.second.size() != 2) continue;
        DualEdge e;
        e.face_a = std::min(kv.second[0], kv.second[1]);
        e.face_b = std::max(kv.second[0], kv.second[1]);
        e.va = kv.first.first;
        e.vb = kv.first.second;
        duals.push_back(e);
    }
    return duals;
}

static int find_dual_edge(
    const map<pair<int, int>, int>& dual_index,
    int f0,
    int f1,
    double& sign) {
    int a = std::min(f0, f1);
    int b = std::max(f0, f1);
    auto it = dual_index.find(std::make_pair(a, b));
    if (it == dual_index.end()) return -1;
    sign = (f0 < f1) ? 1.0 : -1.0;
    return it->second;
}

static double face_corner_angle(
    const MatrixXd& V,
    const MatrixXi& F,
    int global_face,
    int global_vertex) {
    int ids[3] = {F(global_face, 0), F(global_face, 1), F(global_face, 2)};
    int k = -1;
    for (int i = 0; i < 3; i++) {
        if (ids[i] == global_vertex) k = i;
    }
    if (k < 0) return 0.0;
    Vector3d p = V.row(ids[k]);
    Vector3d a = V.row(ids[(k + 1) % 3]);
    Vector3d b = V.row(ids[(k + 2) % 3]);
    return angle_between_edges(a - p, b - p);
}

static bool order_incident_faces(
    const MatrixXi& local_faces,
    int local_vertex,
    const vector<int>& incident,
    vector<int>& ordered,
    bool& closed) {
    if (incident.empty()) return false;
    map<int, vector<int>> adj;
    set<int> inc(incident.begin(), incident.end());
    for (int fi : incident) {
        for (int fj : incident) {
            if (fi >= fj) continue;
            int shared = 0;
            bool has_v = false;
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    if (local_faces(fi, a) == local_faces(fj, b)) {
                        shared++;
                        if (local_faces(fi, a) == local_vertex) has_v = true;
                    }
                }
            }
            if (shared == 2 && has_v) {
                adj[fi].push_back(fj);
                adj[fj].push_back(fi);
            }
        }
    }
    int start = incident[0];
    closed = true;
    for (int fi : incident) {
        int d = (int)adj[fi].size();
        if (d == 1) {
            start = fi;
            closed = false;
            break;
        }
        if (d != 2 && incident.size() > 1) closed = false;
    }
    ordered.clear();
    set<int> visited;
    int prev = -1;
    int cur = start;
    for (;;) {
        ordered.push_back(cur);
        visited.insert(cur);
        int next = -1;
        for (int nb : adj[cur]) {
            if (nb != prev && !visited.count(nb)) {
                next = nb;
                break;
            }
        }
        if (next < 0) break;
        prev = cur;
        cur = next;
    }
    return ordered.size() == incident.size();
}

static void add_dense_row_to_triplets(
    const vector<pair<int, double>>& cols,
    int row,
    vector<Triplet<double>>& triplets) {
    for (const auto& c : cols) {
        if (std::abs(c.second) > 1e-14) triplets.emplace_back(row, c.first, c.second);
    }
}

static VectorXd least_norm_solution(
    const SparseMatrix<double>& Fmat,
    const VectorXd& K,
    double tol,
    bool use_sparse) {
    if (use_sparse) {
        SparseMatrix<double> gram = Fmat * Fmat.transpose();
        vector<Triplet<double>> reg;
        reg.reserve(gram.rows());
        double lambda = std::max(tol, 1e-12);
        for (int i = 0; i < gram.rows(); i++) reg.emplace_back(i, i, lambda);
        SparseMatrix<double> R(gram.rows(), gram.cols());
        R.setFromTriplets(reg.begin(), reg.end());
        gram = gram + R;

        Eigen::SimplicialLDLT<SparseMatrix<double>> ldlt;
        ldlt.compute(gram);
        if (ldlt.info() == Eigen::Success) {
            VectorXd y = ldlt.solve(K);
            if (ldlt.info() == Eigen::Success) return Fmat.transpose() * y;
        }

        Eigen::SparseLU<SparseMatrix<double>> lu;
        lu.compute(gram);
        if (lu.info() == Eigen::Success) {
            VectorXd y = lu.solve(K);
            if (lu.info() == Eigen::Success) return Fmat.transpose() * y;
        }
    }
    MatrixXd Fd(Fmat);
    MatrixXd gram = Fd * Fd.transpose();
    Eigen::JacobiSVD<MatrixXd> svd(gram, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXd lambda = svd.solve(K);
    for (int i = 0; i < svd.singularValues().size(); i++) {
        if (svd.singularValues()(i) < tol) {
            // JacobiSVD::solve already damps tiny singular values relative to threshold.
        }
    }
    return Fmat.transpose() * lambda;
}

static VectorXd solve_face_angles(
    const MatrixXd& V,
    const MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling,
    const RegionMesh& mesh,
    const vector<DualEdge>& duals,
    const VectorXd& omega,
    double label_weight,
    double regularization,
    bool use_sparse) {
    int nf = (int)mesh.local_face_to_global_face.size();
    vector<Triplet<double>> triplets;
    vector<double> rhs;
    int row = 0;
    for (int i = 0; i < (int)duals.size(); i++) {
        triplets.emplace_back(row, duals[i].face_b, 1.0);
        triplets.emplace_back(row, duals[i].face_a, -1.0);
        rhs.push_back(omega(i));
        row++;
    }
    triplets.emplace_back(row, 0, 1.0);
    rhs.push_back(0.0);
    row++;

    vector<ParameterSideRole> roles = resolve_roles(labeling);
    for (int si = 0; si < (int)labeling.abstract_sides.size(); si++) {
        if (si >= (int)roles.size() || roles[si] == ParameterSideRole::Unassigned) continue;
        double target_angle = role_axis_angle(roles[si]);
        for (int segment_id : labeling.abstract_sides[si].segment_ids) {
            const BoundarySegment* seg = nullptr;
            for (const BoundarySegment& s : input.perimeter_segments) {
                if (s.id == segment_id) seg = &s;
            }
            if (!seg) continue;
            for (int vi = 1; vi < (int)seg->authoritative_vertex_ids.size(); vi++) {
                int a = seg->authoritative_vertex_ids[vi - 1];
                int b = seg->authoritative_vertex_ids[vi];
                int gface = -1;
                for (const DirectedBoundaryEdge& edge : input.loops[input.perimeter_loop_index].directed_edges) {
                    if ((edge.from == a && edge.to == b) || (edge.from == b && edge.to == a)) {
                        gface = edge.region_face_id;
                        break;
                    }
                }
                auto fit = mesh.global_face_to_local_face.find(gface);
                if (fit == mesh.global_face_to_local_face.end()) continue;
                double edge_angle = 0.0;
                if (!local_face_edge_angle(V, F, gface, a, b, edge_angle)) continue;
                triplets.emplace_back(row, fit->second, label_weight);
                rhs.push_back(label_weight * normalize_angle_pi(target_angle - edge_angle));
                row++;
            }
        }
    }

    SparseMatrix<double> A(row, nf);
    A.setFromTriplets(triplets.begin(), triplets.end());
    VectorXd b(row);
    for (int i = 0; i < row; i++) b(i) = rhs[i];
    if (use_sparse) {
        SparseMatrix<double> normal = A.transpose() * A;
        VectorXd nb = A.transpose() * b;
        vector<Triplet<double>> reg;
        reg.reserve(normal.rows());
        double lambda = std::max(regularization, 1e-12);
        for (int i = 0; i < normal.rows(); i++) reg.emplace_back(i, i, lambda);
        SparseMatrix<double> R(normal.rows(), normal.cols());
        R.setFromTriplets(reg.begin(), reg.end());
        normal = normal + R;

        Eigen::SimplicialLDLT<SparseMatrix<double>> ldlt;
        ldlt.compute(normal);
        if (ldlt.info() == Eigen::Success) {
            VectorXd x = ldlt.solve(nb);
            if (ldlt.info() == Eigen::Success) return x;
        }

        Eigen::SparseLU<SparseMatrix<double>> lu;
        lu.compute(normal);
        if (lu.info() == Eigen::Success) {
            VectorXd x = lu.solve(nb);
            if (lu.info() == Eigen::Success) return x;
        }
    }
    MatrixXd dense(A);
    Eigen::JacobiSVD<MatrixXd> svd(dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return svd.solve(b);
}

static double curve_arc_length(const BezierCurve3D& curve) {
    vector<Vector3d> pts = curve.sample(64);
    double len = 0.0;
    for (int i = 1; i < (int)pts.size(); i++) len += (pts[i] - pts[i - 1]).norm();
    return len;
}

static double side_frame_length(const BezierGuidingFrameResult& frame, int side, double fallback) {
    for (int i = 0; i < (int)frame.side_indices.size(); i++) {
        if (frame.side_indices[i] == side && i < (int)frame.curves.size()) {
            return std::max(curve_arc_length(frame.curves[i]), 1e-8);
        }
    }
    return std::max(fallback, 1e-8);
}

static const BoundarySegment* find_segment(const BoundarySegmentationResult& input, int id) {
    for (const BoundarySegment& s : input.perimeter_segments) {
        if (s.id == id) return &s;
    }
    return nullptr;
}

static bool build_boundary_uv_data(
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling,
    const BezierGuidingFrameResult& frame,
    BoundaryUvConstraintData& data,
    string& reason) {
    data.roles = resolve_roles(labeling);
    data.side_of_vertex.assign(0, -1);
    double south = 0, north = 0, east = 0, west = 0;
    int sc = 0, nc = 0, ec = 0, wc = 0;
    for (int si = 0; si < (int)labeling.abstract_sides.size(); si++) {
        const AbstractSide& side = labeling.abstract_sides[si];
        if (side.segment_ids.empty()) continue;
        double fallback = 0.0;
        for (int sid : side.segment_ids) {
            const BoundarySegment* s = find_segment(input, sid);
            if (s) fallback += s->length;
        }
        double len = side_frame_length(frame, si, fallback);
        if (data.roles[si] == ParameterSideRole::South) { south += len; sc++; }
        else if (data.roles[si] == ParameterSideRole::North) { north += len; nc++; }
        else if (data.roles[si] == ParameterSideRole::East) { east += len; ec++; }
        else if (data.roles[si] == ParameterSideRole::West) { west += len; wc++; }
    }
    data.width = (sc + nc) ? (south + north) / (double)(sc + nc) : 1.0;
    data.height = (ec + wc) ? (east + west) / (double)(ec + wc) : 1.0;
    data.width = std::max(data.width, 1e-8);
    data.height = std::max(data.height, 1e-8);

    int max_vid = 0;
    for (const BoundarySegment& s : input.perimeter_segments) {
        for (int vid : s.authoritative_vertex_ids) max_vid = std::max(max_vid, vid);
    }
    data.side_of_vertex.assign(max_vid + 1, -1);
    data.side_vertices.resize(labeling.abstract_sides.size());
    data.side_params.resize(labeling.abstract_sides.size());

    for (int si = 0; si < (int)labeling.abstract_sides.size(); si++) {
        const AbstractSide& side = labeling.abstract_sides[si];
        if (side.segment_ids.empty()) continue;
        vector<int> ids;
        vector<Vector3d> pts;
        for (int sid : side.segment_ids) {
            const BoundarySegment* s = find_segment(input, sid);
            if (!s) continue;
            for (int i = 0; i < (int)s->authoritative_vertex_ids.size(); i++) {
                if (!ids.empty() && i == 0 && ids.back() == s->authoritative_vertex_ids[i]) continue;
                ids.push_back(s->authoritative_vertex_ids[i]);
                pts.push_back(s->authoritative_positions[i]);
            }
        }
        if (ids.size() < 2) continue;
        vector<double> params(ids.size(), 0.0);
        double total = 0.0;
        for (int i = 1; i < (int)pts.size(); i++) {
            total += (pts[i] - pts[i - 1]).norm();
            params[i] = total;
        }
        if (total <= kEps) total = 1.0;
        for (double& t : params) t /= total;
        Vector2d a = role_start_uv(data.roles[si], data.width, data.height);
        Vector2d b = role_end_uv(data.roles[si], data.width, data.height);
        for (int i = 0; i < (int)ids.size(); i++) {
            Vector2d uv = (1.0 - params[i]) * a + params[i] * b;
            auto it = data.fixed_uv.find(ids[i]);
            if (it != data.fixed_uv.end() && (it->second - uv).norm() > 1e-7) {
                reason = "conflicting hard boundary UV constraints";
                return false;
            }
            data.fixed_uv[ids[i]] = uv;
            data.side_of_vertex[ids[i]] = si;
        }
        data.side_vertices[si] = ids;
        data.side_params[si] = params;
        data.active_side_indices.push_back(si);
    }
    return true;
}

static int estimate_rank(const SparseMatrix<double>& C, double tol) {
    (void)tol;
    if (C.rows() == 0 || C.cols() == 0) return 0;
    vector<vector<int>> row_columns(C.rows());
    for (int k = 0; k < C.outerSize(); k++) {
        for (SparseMatrix<double>::InnerIterator it(C, k); it; ++it) {
            if (std::abs(it.value()) > 1e-14) row_columns[it.row()].push_back(it.col());
        }
    }
    vector<int> column_match(C.cols(), -1);
    std::function<bool(int, vector<char>&)> augment =
        [&](int row, vector<char>& seen) {
            for (int col : row_columns[row]) {
                if (seen[col]) continue;
                seen[col] = 1;
                if (column_match[col] < 0 || augment(column_match[col], seen)) {
                    column_match[col] = row;
                    return true;
                }
            }
            return false;
        };
    int structural_rank = 0;
    for (int row = 0; row < C.rows(); row++) {
        vector<char> seen(C.cols(), 0);
        if (augment(row, seen)) structural_rank++;
    }
    return structural_rank;
}

static bool solve_kkt(
    const SparseMatrix<double>& Q,
    const VectorXd& q,
    const SparseMatrix<double>& C,
    const VectorXd& d,
    VectorXd& x,
    string& reason,
    vector<double>& smallest_singular_values,
    vector<string>& diagnostics,
    bool allow_dense_fallback) {
    int n = Q.rows();
    int m = C.rows();
    vector<Triplet<double>> trips;
    for (int k = 0; k < Q.outerSize(); k++) {
        for (SparseMatrix<double>::InnerIterator it(Q, k); it; ++it) {
            trips.emplace_back(it.row(), it.col(), it.value());
        }
    }
    for (int k = 0; k < C.outerSize(); k++) {
        for (SparseMatrix<double>::InnerIterator it(C, k); it; ++it) {
            trips.emplace_back(n + it.row(), it.col(), it.value());
            trips.emplace_back(it.col(), n + it.row(), it.value());
        }
    }
    SparseMatrix<double> K(n + m, n + m);
    K.setFromTriplets(trips.begin(), trips.end());
    VectorXd rhs(n + m);
    rhs.head(n) = q;
    rhs.tail(m) = d;
    Eigen::SparseLU<SparseMatrix<double>> solver;
    solver.compute(K);
    if (solver.info() != Eigen::Success) {
        vector<Triplet<double>> reg_trips;
        reg_trips.reserve(n + m);
        const double eps = 1e-12;
        for (int i = 0; i < n; i++) reg_trips.emplace_back(i, i, eps);
        for (int i = 0; i < m; i++) reg_trips.emplace_back(n + i, n + i, -eps);
        SparseMatrix<double> R(n + m, n + m);
        R.setFromTriplets(reg_trips.begin(), reg_trips.end());
        K = K + R;
        solver.compute(K);
        if (solver.info() == Eigen::Success) {
            diagnostics.push_back("Sparse KKT factorization used 1e-12 saddle regularization");
        }
    }
    if (solver.info() != Eigen::Success) {
        if (!allow_dense_fallback) {
            diagnostics.push_back("Sparse KKT factorization failed; dense fallback disabled");
            reason = "sparse KKT factorization failed";
            return false;
        }
        MatrixXd dense(K);
        Eigen::JacobiSVD<MatrixXd> svd(dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
        smallest_singular_values.clear();
        int count = std::min<int>(6, svd.singularValues().size());
        for (int i = 0; i < count; i++) {
            int idx = (int)svd.singularValues().size() - 1 - i;
            smallest_singular_values.push_back(svd.singularValues()(idx));
        }
        diagnostics.push_back("Sparse KKT factorization failed; dense SVD fallback was used");
        VectorXd sol = svd.solve(rhs);
        x = sol.head(n);
        reason = "KKT solved with dense SVD fallback after sparse factorization failed";
        return true;
    }
    VectorXd sol = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        if (!allow_dense_fallback) {
            diagnostics.push_back("Sparse KKT solve failed; dense fallback disabled");
            reason = "sparse KKT solve failed";
            return false;
        }
        MatrixXd dense(K);
        Eigen::JacobiSVD<MatrixXd> svd(dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
        smallest_singular_values.clear();
        int count = std::min<int>(6, svd.singularValues().size());
        for (int i = 0; i < count; i++) {
            int idx = (int)svd.singularValues().size() - 1 - i;
            smallest_singular_values.push_back(svd.singularValues()(idx));
        }
        diagnostics.push_back("Sparse KKT solve failed; dense SVD fallback was used");
        sol = svd.solve(rhs);
        x = sol.head(n);
        reason = "KKT solved with dense SVD fallback after sparse solve failed";
        return true;
    }
    x = sol.head(n);
    return true;
}

static void add_energy_row(
    const vector<pair<int, double>>& coeffs,
    double rhs,
    double weight,
    vector<Triplet<double>>& qtrips,
    VectorXd& qdiag_dummy,
    map<pair<int, int>, double>& Qmap,
    VectorXd& q) {
    (void)qtrips;
    (void)qdiag_dummy;
    double w2 = weight * weight;
    for (const auto& a : coeffs) {
        q(a.first) += w2 * a.second * rhs;
        for (const auto& b : coeffs) {
            Qmap[std::make_pair(a.first, b.first)] += w2 * a.second * b.second;
        }
    }
}

static bool triangle_gradients(const LocalTri& tri, Vector2d grad[3]) {
    if (!tri.valid) return false;
    Matrix2d X;
    X.col(0) = tri.x[1] - tri.x[0];
    X.col(1) = tri.x[2] - tri.x[0];
    if (std::abs(X.determinant()) <= kEps) return false;
    Matrix2d inv = X.inverse();
    grad[1] = inv.row(0).transpose();
    grad[2] = inv.row(1).transpose();
    grad[0] = -grad[1] - grad[2];
    return true;
}

static Matrix2d triangle_jacobian(
    const LocalTri& tri,
    const MatrixXi& local_faces,
    int fi,
    const MatrixXd& local_uv) {
    Matrix2d J = Matrix2d::Zero();
    Vector2d grad[3];
    if (!triangle_gradients(tri, grad)) return J;
    for (int k = 0; k < 3; k++) {
        int v = local_faces(fi, k);
        double u = local_uv(v, 0);
        double vv = local_uv(v, 1);
        J(0, 0) += u * grad[k].x();
        J(0, 1) += u * grad[k].y();
        J(1, 0) += vv * grad[k].x();
        J(1, 1) += vv * grad[k].y();
    }
    return J;
}

static void add_jacobian_component_row(
    int n_vertices,
    const MatrixXi& local_faces,
    int fi,
    const Vector2d grad[3],
    int uv_row,
    int xy_col,
    double rhs,
    double weight,
    map<pair<int, int>, double>& Qmap,
    VectorXd& q) {
    vector<pair<int, double>> coeffs;
    for (int k = 0; k < 3; k++) {
        int v = local_faces(fi, k);
        int col = uv_row == 0 ? v : n_vertices + v;
        coeffs.push_back({col, grad[k](xy_col)});
    }
    vector<Triplet<double>> dummy;
    VectorXd dummyv;
    add_energy_row(coeffs, rhs, weight, dummy, dummyv, Qmap, q);
}

static void add_triangle_arap_energy(
    int n_vertices,
    const MatrixXi& local_faces,
    const vector<LocalTri>& tris,
    const vector<Matrix2d>& rotations,
    double lambda_smooth,
    map<pair<int, int>, double>& Qmap,
    VectorXd& q) {
    double base = std::max(0.0, 1.0 - lambda_smooth);
    if (base <= 0.0) return;
    for (int fi = 0; fi < local_faces.rows(); fi++) {
        Vector2d grad[3];
        if (!triangle_gradients(tris[fi], grad)) continue;
        double weight = std::sqrt(std::max(tris[fi].area, 1e-16) * base);
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                add_jacobian_component_row(
                    n_vertices, local_faces, fi, grad,
                    r, c, rotations[fi](r, c), weight, Qmap, q);
            }
        }
    }
}

static void add_residual_smoothing_energy(
    const MatrixXd& V,
    const RegionMesh& mesh,
    const vector<LocalTri>& tris,
    const vector<Matrix2d>& rotations,
    double lambda_smooth,
    map<pair<int, int>, double>& Qmap,
    VectorXd& q) {
    if (lambda_smooth <= 0.0) return;
    auto edge_faces = build_edge_to_local_faces(mesh);
    int n = (int)mesh.region_vertices.size();
    for (const auto& kv : edge_faces) {
        if (kv.second.size() != 2) continue;
        int f0 = kv.second[0];
        int f1 = kv.second[1];
        Vector2d g0[3], g1[3];
        if (!triangle_gradients(tris[f0], g0) ||
            !triangle_gradients(tris[f1], g1)) {
            continue;
        }
        int ev0 = kv.first.first;
        int ev1 = kv.first.second;
        Vector3d p0 = V.row(mesh.region_vertices[ev0]);
        Vector3d p1 = V.row(mesh.region_vertices[ev1]);
        double edge_len = std::max((p1 - p0).norm(), 1e-8);
        double weight = std::sqrt(lambda_smooth * edge_len);
        double a0 = tris[f0].area;
        double a1 = tris[f1].area;
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                vector<pair<int, double>> coeffs;
                for (int k = 0; k < 3; k++) {
                    int v = mesh.local_faces(f0, k);
                    int col = r == 0 ? v : n + v;
                    coeffs.push_back({col, a0 * g0[k](c)});
                }
                for (int k = 0; k < 3; k++) {
                    int v = mesh.local_faces(f1, k);
                    int col = r == 0 ? v : n + v;
                    coeffs.push_back({col, -a1 * g1[k](c)});
                }
                double rhs = a0 * rotations[f0](r, c) - a1 * rotations[f1](r, c);
                vector<Triplet<double>> dummy;
                VectorXd dummyv;
                add_energy_row(coeffs, rhs, weight, dummy, dummyv, Qmap, q);
            }
        }
    }
}

static void assemble_stitching_energy(
    const MatrixXd& V,
    const RegionMesh& mesh,
    const vector<LocalTri>& tris,
    const vector<Matrix2d>& rotations,
    int side_count,
    const KktGlobalStitchingConfig& config,
    bool use_smoothing,
    SparseMatrix<double>& Q,
    VectorXd& q) {
    int n = (int)mesh.region_vertices.size();
    int unknowns = 2 * n + side_count;
    q = VectorXd::Zero(unknowns);
    map<pair<int, int>, double> Qmap;
    double lambda = use_smoothing ? std::max(0.0, std::min(1.0, config.lambda_smooth)) : 0.0;
    add_triangle_arap_energy(n, mesh.local_faces, tris, rotations, lambda, Qmap, q);
    if (use_smoothing && config.enable_smoothed_arap) {
        add_residual_smoothing_energy(V, mesh, tris, rotations, lambda, Qmap, q);
    }
    for (int ai = 0; ai < side_count; ai++) {
        int col = 2 * n + ai;
        vector<Triplet<double>> dummy;
        VectorXd dummyv;
        add_energy_row(
            {{col, 1.0}},
            1.0,
            std::sqrt(config.lambda_scale),
            dummy,
            dummyv,
            Qmap,
            q);
    }
    vector<Triplet<double>> qtrips;
    for (const auto& kv : Qmap) {
        qtrips.emplace_back(kv.first.first, kv.first.second, kv.second);
    }
    Q.resize(unknowns, unknowns);
    Q.setFromTriplets(qtrips.begin(), qtrips.end());
}

static void compute_stats(
    const MatrixXd& V,
    const MatrixXi& F,
    const RegionMesh& mesh,
    const vector<LocalTri>& tris,
    const vector<Matrix2d>& rotations,
    const MatrixXd& local_uv,
    const KktGlobalStitchingConfig& cfg,
    vector<ConstrainedArapTriangleStats>& stats,
    double& mean_residual,
    double& max_residual,
    int& flipped_count) {
    stats.clear();
    mean_residual = 0.0;
    max_residual = 0.0;
    flipped_count = 0;
    int positive_orientation_count = 0;
    int negative_orientation_count = 0;
    for (int fi = 0; fi < mesh.local_faces.rows(); fi++) {
        int a = mesh.local_faces(fi, 0);
        int b = mesh.local_faces(fi, 1);
        int c = mesh.local_faces(fi, 2);
        Vector2d uv[3] = {local_uv.row(a), local_uv.row(b), local_uv.row(c)};
        int gfi = mesh.local_face_to_global_face[fi];
        ConstrainedArapTriangleStats st;
        st.local_face_id = fi;
        st.global_face_id = gfi;
        st.signed_uv_area = signed_area_uv(uv[0], uv[1], uv[2]);
        st.abs_uv_area = std::abs(st.signed_uv_area);
        st.area_3d = tri_area_3d(V.row(F(gfi, 0)), V.row(F(gfi, 1)), V.row(F(gfi, 2)));
        st.area_ratio = st.area_3d > kEps ? st.abs_uv_area / st.area_3d : 0.0;
        st.orientation = st.signed_uv_area > cfg.min_signed_area ? 1 :
                         (st.signed_uv_area < -cfg.min_signed_area ? -1 : 0);
        if (st.orientation > 0) positive_orientation_count++;
        if (st.orientation < 0) negative_orientation_count++;
        st.conformal_distortion = 0.0;
        if (tris[fi].valid) {
            Matrix2d J = triangle_jacobian(tris[fi], mesh.local_faces, fi, local_uv);
            Eigen::JacobiSVD<Matrix2d> svd(J);
            double s0 = svd.singularValues()(0);
            double s1 = svd.singularValues()(1);
            st.conformal_distortion =
                s1 > kEps ? s0 / s1 : std::numeric_limits<double>::infinity();
            st.arap_residual = (J - rotations[fi]).norm();
        }
        stats.push_back(st);
        mean_residual += st.arap_residual;
        max_residual = std::max(max_residual, st.arap_residual);
    }
    int dominant_orientation =
        negative_orientation_count > positive_orientation_count ? -1 : 1;
    for (ConstrainedArapTriangleStats& st : stats) {
        st.flipped =
            st.orientation == 0 ||
            (positive_orientation_count > 0 &&
             negative_orientation_count > 0 &&
             st.orientation != dominant_orientation);
        if (st.flipped) flipped_count++;
    }
    if (!stats.empty()) mean_residual /= (double)stats.size();
}

} // namespace

RotationAngleInitializationResult initialize_rotation_angles_section421(
    const MatrixXd& V,
    const MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling,
    const RotationAngleInitializationConfig& config) {
    RotationAngleInitializationResult result;
    auto start_time = std::chrono::steady_clock::now();
    auto progress = [&](const string& message) {
        if (!config.print_progress_to_console) return;
        double t = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        std::cout << "[rotation init] [" << t << "s] "
                  << message << std::endl;
    };
    if (!input.valid) {
        result.reason = "boundary input is invalid: " + input.reason;
        return result;
    }
    if (input.loops.size() > 1) {
        result.unsupported_multiply_connected = true;
        result.reason = "multiply-connected regions require non-contractible loop basis; unsupported in this stage";
        return result;
    }
    RegionMesh mesh;
    if (!collect_region_mesh(F, input, mesh, result.reason)) return result;
    progress("collected region mesh: vertices=" +
             std::to_string(mesh.region_vertices.size()) +
             ", faces=" + std::to_string(mesh.local_face_to_global_face.size()));
    result.region_face_ids = mesh.local_face_to_global_face;
    vector<LocalTri> tris = build_local_triangles(V, F, mesh);
    vector<DualEdge> duals = build_dual_edges(mesh);
    progress("built local triangles and dual graph: dual_edges=" +
             std::to_string(duals.size()));
    map<pair<int, int>, int> dual_index;
    for (int i = 0; i < (int)duals.size(); i++) {
        dual_index[std::make_pair(duals[i].face_a, duals[i].face_b)] = i;
    }

    vector<vector<int>> incident(mesh.region_vertices.size());
    for (int lfi = 0; lfi < mesh.local_faces.rows(); lfi++) {
        for (int k = 0; k < 3; k++) incident[mesh.local_faces(lfi, k)].push_back(lfi);
    }

    vector<Triplet<double>> trips;
    vector<double> Kvals;
    int row = 0;
    for (int lv = 0; lv < (int)incident.size(); lv++) {
        vector<int> order;
        bool closed = false;
        if (!order_incident_faces(mesh.local_faces, lv, incident[lv], order, closed)) continue;
        if (!closed && !config.include_boundary_angle_defects) continue;
        vector<pair<int, double>> coeffs;
        for (int i = 0; i + 1 < (int)order.size(); i++) {
            double sign = 1.0;
            int di = find_dual_edge(dual_index, order[i], order[i + 1], sign);
            if (di >= 0) coeffs.push_back({di, sign});
        }
        if (closed && order.size() > 1) {
            double sign = 1.0;
            int di = find_dual_edge(dual_index, order.back(), order.front(), sign);
            if (di >= 0) coeffs.push_back({di, sign});
        }
        if (coeffs.empty()) continue;
        double sum_angles = 0.0;
        int gv = mesh.region_vertices[lv];
        for (int lfi : incident[lv]) {
            int gfi = mesh.local_face_to_global_face[lfi];
            sum_angles += face_corner_angle(V, F, gfi, gv);
        }
        double K = (closed ? 2.0 * kPi : kPi) - sum_angles;
        add_dense_row_to_triplets(coeffs, row, trips);
        Kvals.push_back(K);
        RotationConstraintRowReport rep;
        rep.type = closed ? "interior_fan_closure" : "boundary_angle_defect";
        rep.vertex_id = gv;
        rep.rhs = K;
        result.constraint_reports.push_back(rep);
        row++;
    }

    SparseMatrix<double> Fmat(row, (int)duals.size());
    Fmat.setFromTriplets(trips.begin(), trips.end());
    VectorXd K(row);
    for (int i = 0; i < row; i++) K(i) = Kvals[i];
    progress("solving dual omega: constraints=" + std::to_string(Fmat.rows()) +
             ", unknowns=" + std::to_string(Fmat.cols()) +
             (config.use_sparse_linear_solvers ? ", sparse" : ", dense"));
    result.dual_omega = least_norm_solution(
        Fmat, K, config.singular_value_tolerance,
        config.use_sparse_linear_solvers);
    progress("solved dual omega");
    VectorXd residual = Fmat * result.dual_omega - K;
    for (int i = 0; i < residual.size(); i++) {
        result.constraint_reports[i].residual = residual(i);
        if (result.constraint_reports[i].type.find("fan") != string::npos) {
            result.max_fan_closure_error =
                std::max(result.max_fan_closure_error, std::abs(residual(i)));
        }
    }

    progress("solving per-face rotation angles");
    result.face_rotation_angles = solve_face_angles(
        V, F, input, labeling, mesh, duals, result.dual_omega,
        config.include_label_orientation_constraints ? config.label_orientation_weight : 0.0,
        config.singular_value_tolerance,
        config.use_sparse_linear_solvers);
    progress("solved per-face rotation angles");
    result.face_rotations.resize(result.face_rotation_angles.size());
    for (int i = 0; i < result.face_rotation_angles.size(); i++) {
        result.face_rotations[i] = rotation2(result.face_rotation_angles(i));
    }

    result.triangle_soup_uv.resize(mesh.local_faces.rows() * 3, 2);
    result.triangle_soup_faces.resize(mesh.local_faces.rows(), 3);
    for (int fi = 0; fi < mesh.local_faces.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            Vector2d p = result.face_rotations[fi] * tris[fi].x[k];
            result.triangle_soup_uv.row(3 * fi + k) = p.transpose();
            result.triangle_soup_faces(fi, k) = 3 * fi + k;
        }
    }

    double label_sum = 0.0;
    int label_count = 0;
    vector<ParameterSideRole> roles = resolve_roles(labeling);
    for (int si = 0; si < (int)labeling.abstract_sides.size(); si++) {
        if (si >= (int)roles.size() || roles[si] == ParameterSideRole::Unassigned) continue;
        double target = role_axis_angle(roles[si]);
        for (int sid : labeling.abstract_sides[si].segment_ids) {
            const BoundarySegment* seg = find_segment(input, sid);
            if (!seg) continue;
            for (int vi = 1; vi < (int)seg->authoritative_vertex_ids.size(); vi++) {
                int a = seg->authoritative_vertex_ids[vi - 1];
                int b = seg->authoritative_vertex_ids[vi];
                int gface = -1;
                for (const DirectedBoundaryEdge& e : input.loops[input.perimeter_loop_index].directed_edges) {
                    if ((e.from == a && e.to == b) || (e.from == b && e.to == a)) {
                        gface = e.region_face_id;
                        break;
                    }
                }
                auto fit = mesh.global_face_to_local_face.find(gface);
                if (fit == mesh.global_face_to_local_face.end()) continue;
                double edge_angle = 0.0;
                if (!local_face_edge_angle(V, F, gface, a, b, edge_angle)) continue;
                double err = std::abs(normalize_angle_pi(result.face_rotation_angles(fit->second) + edge_angle - target));
                label_sum += err;
                result.max_label_orientation_error = std::max(result.max_label_orientation_error, err);
                label_count++;
            }
        }
    }
    result.mean_label_orientation_error = label_count ? label_sum / (double)label_count : 0.0;

    result.valid = true;
    result.reason = "ok";
    if (config.export_debug) export_rotation_angle_initialization_debug(config.debug_prefix, result);
    return result;
}

KktGlobalStitchingResult parameterize_kkt_global_stitching(
    const MatrixXd& V,
    const MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling,
    const BezierGuidingFrameResult& guiding_frame,
    const RotationAngleInitializationResult& rotation_init,
    const KktGlobalStitchingConfig& config) {
    KktGlobalStitchingResult result;
    auto start_time = std::chrono::steady_clock::now();
    auto progress = [&](const string& message) {
        if (!config.print_progress_to_console) return;
        double t = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        std::cout << "[KKT stitching] [" << t << "s] "
                  << message << std::endl;
    };
    if (!rotation_init.valid) {
        result.reason = "rotation initialization invalid: " + rotation_init.reason;
        return result;
    }
    RegionMesh mesh;
    if (!collect_region_mesh(F, input, mesh, result.reason)) return result;
    progress("collected region mesh: vertices=" +
             std::to_string(mesh.region_vertices.size()) +
             ", faces=" + std::to_string(mesh.local_face_to_global_face.size()));
    vector<LocalTri> tris = build_local_triangles(V, F, mesh);
    BoundaryUvConstraintData bdata;
    if (!build_boundary_uv_data(input, labeling, guiding_frame, bdata, result.reason)) return result;
    progress("built boundary UV constraints: active_sides=" +
             std::to_string(bdata.active_side_indices.size()) +
             ", fixed_vertices=" + std::to_string(bdata.fixed_uv.size()));

    result.region_vertex_ids = mesh.region_vertices;
    result.local_faces = mesh.local_faces;
    result.local_face_to_global_face = mesh.local_face_to_global_face;
    result.UV = MatrixXd::Constant(V.rows(), 2, std::numeric_limits<double>::quiet_NaN());

    int n = (int)mesh.region_vertices.size();
    int side_count = (int)bdata.active_side_indices.size();
    int unknowns = 2 * n + side_count;
    map<int, int> side_to_alpha;
    for (int i = 0; i < side_count; i++) side_to_alpha[bdata.active_side_indices[i]] = i;

    SparseMatrix<double> Q;
    VectorXd q;
    progress("assembling sparse ARAP energy");
    assemble_stitching_energy(
        V,
        mesh,
        tris,
        rotation_init.face_rotations,
        side_count,
        config,
        config.enable_smoothed_arap && config.lambda_smooth > 0.0,
        Q,
        q);
    progress("assembled Q: " + std::to_string(Q.rows()) + "x" +
             std::to_string(Q.cols()) + ", nnz=" +
             std::to_string(Q.nonZeros()));

    vector<Triplet<double>> ctrips;
    vector<double> dvals;
    auto add_constraint = [&](const vector<pair<int, double>>& coeffs, double rhs) {
        int r = (int)dvals.size();
        for (const auto& c : coeffs) ctrips.emplace_back(r, c.first, c.second);
        dvals.push_back(rhs);
    };
    if (!bdata.fixed_uv.empty()) {
        const auto& anchor = *bdata.fixed_uv.begin();
        int li = mesh.global_to_local_vertex[anchor.first];
        add_constraint({{li, 1.0}}, anchor.second.x());
        add_constraint({{n + li, 1.0}}, anchor.second.y());
    }
    set<pair<int, int>> coordinate_constraints;
    for (int si : bdata.active_side_indices) {
        auto ait = side_to_alpha.find(si);
        if (ait == side_to_alpha.end()) continue;
        int alpha_col = 2 * n + ait->second;
        ParameterSideRole role = bdata.roles[si];
        const vector<int>& ids = bdata.side_vertices[si];
        const vector<double>& params = bdata.side_params[si];
        for (int gid : ids) {
            int li = mesh.global_to_local_vertex[gid];
            Vector2d target = bdata.fixed_uv[gid];
            if (role == ParameterSideRole::South || role == ParameterSideRole::North) {
                pair<int, int> key(n + li, 1);
                if (coordinate_constraints.insert(key).second) {
                    add_constraint({{n + li, 1.0}}, target.y());
                }
            } else if (role == ParameterSideRole::East || role == ParameterSideRole::West) {
                pair<int, int> key(li, 0);
                if (coordinate_constraints.insert(key).second) {
                    add_constraint({{li, 1.0}}, target.x());
                }
            }
        }
        for (int i = 1; i < (int)ids.size(); i++) {
            int a = mesh.global_to_local_vertex[ids[i - 1]];
            int b = mesh.global_to_local_vertex[ids[i]];
            double delta = params[i] - params[i - 1];
            if (role == ParameterSideRole::South || role == ParameterSideRole::North) {
                add_constraint({{b, 1.0}, {a, -1.0},
                                {alpha_col, -delta * bdata.width}}, 0.0);
            } else if (role == ParameterSideRole::East || role == ParameterSideRole::West) {
                add_constraint({{n + b, 1.0}, {n + a, -1.0},
                                {alpha_col, -delta * bdata.height}}, 0.0);
            }
        }
    }

    SparseMatrix<double> C((int)dvals.size(), unknowns);
    C.setFromTriplets(ctrips.begin(), ctrips.end());
    VectorXd d(dvals.size());
    for (int i = 0; i < (int)dvals.size(); i++) d(i) = dvals[i];
    result.constraint_count = C.rows();
    progress("assembled C: " + std::to_string(C.rows()) + "x" +
             std::to_string(C.cols()) + ", nnz=" +
             std::to_string(C.nonZeros()));
    progress("estimating sparse structural constraint rank");
    result.constraint_rank = estimate_rank(C, config.rank_tolerance);
    result.redundant_constraints =
        result.constraint_rank >= 0 && result.constraint_rank < result.constraint_count;
    progress("constraint rank=" + std::to_string(result.constraint_rank) +
             "/" + std::to_string(result.constraint_count) +
             ", redundant=" + string(result.redundant_constraints ? "true" : "false"));

    VectorXd x;
    MatrixXd before_local_uv;
    if (config.enable_smoothed_arap && config.lambda_smooth > 0.0) {
        progress("assembling and solving base sparse KKT for before-smoothing metrics");
        SparseMatrix<double> Qbase;
        VectorXd qbase;
        assemble_stitching_energy(
            V,
            mesh,
            tris,
            rotation_init.face_rotations,
            side_count,
            config,
            false,
            Qbase,
            qbase);
        VectorXd xbase;
        string base_reason;
        vector<double> base_singulars;
        vector<string> base_diagnostics;
        if (solve_kkt(
                Qbase,
                qbase,
                C,
                d,
                xbase,
                base_reason,
                base_singulars,
                base_diagnostics,
                config.enable_dense_diagnostics)) {
            before_local_uv.resize(n, 2);
            for (int i = 0; i < n; i++) {
                before_local_uv(i, 0) = xbase(i);
                before_local_uv(i, 1) = xbase(n + i);
            }
        }
        progress("base sparse KKT stage finished");
    }

    progress("factorizing final sparse KKT: dimension=" +
             std::to_string(Q.rows() + C.rows()) +
             ", estimated_nnz=" +
             std::to_string(Q.nonZeros() + 2 * C.nonZeros()));
    if (!solve_kkt(
            Q,
            q,
            C,
            d,
            x,
            result.reason,
            result.smallest_kkt_singular_values,
            result.conflicting_constraints,
            config.enable_dense_diagnostics)) {
        result.conflicting_constraints.push_back("KKT solve failed; inspect redundant constraints and boundary labels");
        return result;
    }
    progress("final sparse KKT solve finished");
    for (const string& diagnostic : result.conflicting_constraints) {
        progress("solver diagnostic: " + diagnostic);
    }

    MatrixXd local_uv(n, 2);
    for (int i = 0; i < n; i++) {
        local_uv(i, 0) = x(i);
        local_uv(i, 1) = x(n + i);
        result.UV.row(mesh.region_vertices[i]) = local_uv.row(i);
    }
    result.alpha_values.resize(side_count);
    for (int ai = 0; ai < side_count; ai++) result.alpha_values[ai] = x(2 * n + ai);

    double before_mean = 0.0;
    double before_max = 0.0;
    int before_flips = 0;
    if (before_local_uv.rows() == n) {
        compute_stats(V, F, mesh, tris, rotation_init.face_rotations, before_local_uv, config,
                      result.before_triangle_stats, before_mean, before_max, before_flips);
    } else {
        compute_stats(V, F, mesh, tris, rotation_init.face_rotations, local_uv, config,
                      result.before_triangle_stats, before_mean, before_max, before_flips);
    }
    compute_stats(V, F, mesh, tris, rotation_init.face_rotations, local_uv, config,
                  result.after_triangle_stats, result.mean_arap_residual,
                  result.max_arap_residual, result.flipped_triangle_count);
    progress("computed distortion statistics: flips=" +
             std::to_string(result.flipped_triangle_count) +
             ", mean_arap=" + std::to_string(result.mean_arap_residual));

    result.max_label_coordinate_error = 0.0;
    if (!bdata.fixed_uv.empty()) {
        const auto& anchor = *bdata.fixed_uv.begin();
        int li = mesh.global_to_local_vertex[anchor.first];
        result.max_label_coordinate_error =
            std::max(result.max_label_coordinate_error,
                     (local_uv.row(li).transpose() - anchor.second).norm());
    }
    for (int si : bdata.active_side_indices) {
        ParameterSideRole role = bdata.roles[si];
        for (int gid : bdata.side_vertices[si]) {
            int li = mesh.global_to_local_vertex[gid];
            Vector2d target = bdata.fixed_uv[gid];
            double err = 0.0;
            if (role == ParameterSideRole::South || role == ParameterSideRole::North) {
                err = std::abs(local_uv(li, 1) - target.y());
            } else if (role == ParameterSideRole::East || role == ParameterSideRole::West) {
                err = std::abs(local_uv(li, 0) - target.x());
            }
            result.max_label_coordinate_error =
                std::max(result.max_label_coordinate_error, err);
        }
    }
    result.max_length_ratio_error = 0.0;
    for (int si : bdata.active_side_indices) {
        auto ait = side_to_alpha.find(si);
        if (ait == side_to_alpha.end()) continue;
        double alpha = result.alpha_values[ait->second];
        ParameterSideRole role = bdata.roles[si];
        const vector<int>& ids = bdata.side_vertices[si];
        const vector<double>& params = bdata.side_params[si];
        for (int i = 1; i < (int)ids.size(); i++) {
            int a = mesh.global_to_local_vertex[ids[i - 1]];
            int b = mesh.global_to_local_vertex[ids[i]];
            double delta = params[i] - params[i - 1];
            double residual = 0.0;
            if (role == ParameterSideRole::South || role == ParameterSideRole::North) {
                residual = (local_uv(b, 0) - local_uv(a, 0)) -
                           alpha * delta * bdata.width;
            } else if (role == ParameterSideRole::East || role == ParameterSideRole::West) {
                residual = (local_uv(b, 1) - local_uv(a, 1)) -
                           alpha * delta * bdata.height;
            }
            result.max_length_ratio_error =
                std::max(result.max_length_ratio_error, std::abs(residual));
        }
    }

    auto edge_faces = build_edge_to_local_faces(mesh);
    double sum_var = 0.0;
    int cnt = 0;
    for (const auto& kv : edge_faces) {
        if (kv.second.size() != 2) continue;
        double r0 = result.after_triangle_stats[kv.second[0]].arap_residual;
        double r1 = result.after_triangle_stats[kv.second[1]].arap_residual;
        sum_var += std::abs(r0 - r1);
        cnt++;
    }
    result.adjacent_residual_variation = cnt ? sum_var / (double)cnt : 0.0;

    if (result.flipped_triangle_count > 0 && config.fail_on_flips) {
        result.reason = "KKT stitching produced flipped or degenerate triangles";
        if (config.export_debug) {
            progress("exporting failure diagnostics");
            export_kkt_global_stitching_debug(config.debug_prefix, result);
        }
        return result;
    }
    result.valid = true;
    result.reason = "ok";
    if (config.export_debug) {
        progress("exporting KKT diagnostics");
        export_kkt_global_stitching_debug(config.debug_prefix, result);
    }
    progress("done");
    return result;
}

bool export_rotation_angle_initialization_debug(
    const string& prefix,
    const RotationAngleInitializationResult& result) {
    bool ok = true;
    {
        std::ofstream out(prefix + "_constraints.csv");
        ok = out.is_open() && ok;
        out << "type,vertex_id,face_a,face_b,rhs,residual\n";
        for (const auto& r : result.constraint_reports) {
            out << r.type << "," << r.vertex_id << "," << r.face_a << ","
                << r.face_b << "," << r.rhs << "," << r.residual << "\n";
        }
    }
    {
        std::ofstream out(prefix + "_face_rotations.csv");
        ok = out.is_open() && ok;
        out << "local_face,global_face,angle\n";
        for (int i = 0; i < result.face_rotation_angles.size(); i++) {
            int gf = i < (int)result.region_face_ids.size() ? result.region_face_ids[i] : -1;
            out << i << "," << gf << "," << result.face_rotation_angles(i) << "\n";
        }
    }
    {
        std::ofstream out(prefix + "_triangle_soup.obj");
        ok = out.is_open() && ok;
        for (int i = 0; i < result.triangle_soup_uv.rows(); i++) {
            out << "v " << result.triangle_soup_uv(i, 0) << " "
                << result.triangle_soup_uv(i, 1) << " 0\n";
        }
        for (int i = 0; i < result.triangle_soup_faces.rows(); i++) {
            out << "f " << (result.triangle_soup_faces(i, 0) + 1) << " "
                << (result.triangle_soup_faces(i, 1) + 1) << " "
                << (result.triangle_soup_faces(i, 2) + 1) << "\n";
        }
    }
    return ok;
}

bool export_kkt_global_stitching_debug(
    const string& prefix,
    const KktGlobalStitchingResult& result) {
    bool ok = true;
    {
        std::ofstream out(prefix + "_uv.obj");
        ok = out.is_open() && ok;
        for (int vid : result.region_vertex_ids) {
            out << "v " << result.UV(vid, 0) << " " << result.UV(vid, 1) << " 0\n";
        }
        for (int i = 0; i < result.local_faces.rows(); i++) {
            out << "f " << (result.local_faces(i, 0) + 1) << " "
                << (result.local_faces(i, 1) + 1) << " "
                << (result.local_faces(i, 2) + 1) << "\n";
        }
    }
    auto write_stats = [&](const string& name, const vector<ConstrainedArapTriangleStats>& stats) {
        std::ofstream out(prefix + "_" + name + "_distortion.csv");
        ok = out.is_open() && ok;
        out << "global_face_id,area_ratio,conformal_distortion,arap_residual,orientation,flipped\n";
        for (const auto& s : stats) {
            out << s.global_face_id << "," << s.area_ratio << ","
                << s.conformal_distortion << "," << s.arap_residual << ","
                << s.orientation << "," << (s.flipped ? 1 : 0) << "\n";
        }
    };
    write_stats("before", result.before_triangle_stats);
    write_stats("after", result.after_triangle_stats);
    {
        std::ofstream out(prefix + "_summary.csv");
        ok = out.is_open() && ok;
        out << "valid,reason,constraint_rank,constraint_count,redundant,flips,mean_arap,max_arap,label_coord_error,length_ratio_error,adjacent_residual_variation\n";
        out << (result.valid ? 1 : 0) << "," << result.reason << ","
            << result.constraint_rank << "," << result.constraint_count << ","
            << (result.redundant_constraints ? 1 : 0) << ","
            << result.flipped_triangle_count << ","
            << result.mean_arap_residual << "," << result.max_arap_residual << ","
            << result.max_label_coordinate_error << ","
            << result.max_length_ratio_error << ","
            << result.adjacent_residual_variation << "\n";
    }
    {
        std::ofstream out(prefix + "_kkt_diagnostics.csv");
        ok = out.is_open() && ok;
        out << "kind,value\n";
        for (double s : result.smallest_kkt_singular_values) {
            out << "smallest_singular_value," << s << "\n";
        }
        for (const string& msg : result.conflicting_constraints) {
            out << "diagnostic,\"" << msg << "\"\n";
        }
    }
    return ok;
}
