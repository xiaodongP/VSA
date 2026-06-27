#include "constrained_arap_parameterization.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>

using Eigen::Matrix2d;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::vector;

namespace {

constexpr double kEps = 1e-12;

struct LocalTriangleGeometry {
    Vector2d x[3];
    double area = 0.0;
    bool valid = false;
};

struct BoundaryConstraintBuilder {
    map<int, Vector2d> fixed_uv;
    vector<int> constrained_vertex_ids;
    vector<Vector2d> constrained_uvs;
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

static double triangle_area_3d(const Vector3d& a, const Vector3d& b, const Vector3d& c) {
    return 0.5 * (b - a).cross(c - a).norm();
}

static Vector2d role_start_uv(ParameterSideRole role, double width, double height) {
    switch (role) {
    case ParameterSideRole::South: return Vector2d(0.0, 0.0);
    case ParameterSideRole::East: return Vector2d(width, 0.0);
    case ParameterSideRole::North: return Vector2d(width, height);
    case ParameterSideRole::West: return Vector2d(0.0, height);
    case ParameterSideRole::Unassigned: return Vector2d(0.0, 0.0);
    }
    return Vector2d(0.0, 0.0);
}

static Vector2d role_end_uv(ParameterSideRole role, double width, double height) {
    switch (role) {
    case ParameterSideRole::South: return Vector2d(width, 0.0);
    case ParameterSideRole::East: return Vector2d(width, height);
    case ParameterSideRole::North: return Vector2d(0.0, height);
    case ParameterSideRole::West: return Vector2d(0.0, 0.0);
    case ParameterSideRole::Unassigned: return Vector2d(0.0, 0.0);
    }
    return Vector2d(0.0, 0.0);
}

static ParameterSideRole role_from_cardinal(const string& name) {
    if (name == "South") return ParameterSideRole::South;
    if (name == "East") return ParameterSideRole::East;
    if (name == "North") return ParameterSideRole::North;
    if (name == "West") return ParameterSideRole::West;
    return ParameterSideRole::Unassigned;
}

static vector<ParameterSideRole> resolve_roles(
    const AutomaticLabelingResult& labeling,
    const ConstrainedArapConfig& config) {
    vector<ParameterSideRole> roles(labeling.abstract_sides.size(), ParameterSideRole::Unassigned);
    for (int i = 0; i < (int)roles.size(); i++) {
        if (config.use_labeling_orientation && labeling.orientation.valid && i < 4) {
            roles[i] = role_from_cardinal(labeling.orientation.side_to_cardinal[i]);
        }
        if (roles[i] == ParameterSideRole::Unassigned && i < 4) {
            roles[i] = config.fallback_side_roles[i];
        }
    }
    return roles;
}

static bool collect_region_submesh(
    const MatrixXi& F,
    const RegionFaceSet& region,
    vector<int>& region_vertices,
    map<int, int>& global_to_local,
    MatrixXi& local_faces,
    vector<int>& local_face_to_global_face,
    string& reason) {
    set<int> vertex_set;
    local_face_to_global_face.clear();
    for (int fi : region.face_ids) {
        if (fi < 0 || fi >= F.rows()) {
            reason = "region contains invalid face id";
            return false;
        }
        local_face_to_global_face.push_back(fi);
        for (int k = 0; k < 3; k++) vertex_set.insert(F(fi, k));
    }
    if (vertex_set.empty() || local_face_to_global_face.empty()) {
        reason = "region has no faces or vertices";
        return false;
    }
    region_vertices.assign(vertex_set.begin(), vertex_set.end());
    global_to_local.clear();
    for (int i = 0; i < (int)region_vertices.size(); i++) {
        global_to_local[region_vertices[i]] = i;
    }
    local_faces.resize((int)local_face_to_global_face.size(), 3);
    for (int li = 0; li < (int)local_face_to_global_face.size(); li++) {
        int fi = local_face_to_global_face[li];
        for (int k = 0; k < 3; k++) {
            auto it = global_to_local.find(F(fi, k));
            if (it == global_to_local.end()) {
                reason = "failed to map face vertex into local region";
                return false;
            }
            local_faces(li, k) = it->second;
        }
    }
    return true;
}

static const BoundarySegment* find_segment(
    const BoundarySegmentationResult& input,
    int segment_id) {
    for (const BoundarySegment& segment : input.perimeter_segments) {
        if (segment.id == segment_id) return &segment;
    }
    return nullptr;
}

static double curve_arc_length(const BezierCurve3D& curve, int samples = 64) {
    vector<Vector3d> pts = curve.sample(samples);
    double length = 0.0;
    for (int i = 1; i < (int)pts.size(); i++) length += (pts[i] - pts[i - 1]).norm();
    return length;
}

static double side_sample_length(
    const BoundarySegmentationResult& input,
    const AbstractSide& side) {
    double length = 0.0;
    for (int segment_id : side.segment_ids) {
        const BoundarySegment* segment = find_segment(input, segment_id);
        if (!segment) continue;
        length += segment->length;
    }
    return length;
}

static double frame_side_length(
    const BezierGuidingFrameResult& frame,
    int side_index,
    double fallback) {
    for (int i = 0; i < (int)frame.side_indices.size(); i++) {
        if (frame.side_indices[i] == side_index && i < (int)frame.curves.size()) {
            return std::max(curve_arc_length(frame.curves[i]), 1e-8);
        }
    }
    return std::max(fallback, 1e-8);
}

static bool add_fixed_uv(
    BoundaryConstraintBuilder& builder,
    int vid,
    const Vector2d& uv,
    string& reason) {
    auto it = builder.fixed_uv.find(vid);
    if (it != builder.fixed_uv.end()) {
        if ((it->second - uv).norm() > 1e-7) {
            reason = "inconsistent UV constraints for boundary vertex";
            return false;
        }
        return true;
    }
    builder.fixed_uv[vid] = uv;
    builder.constrained_vertex_ids.push_back(vid);
    builder.constrained_uvs.push_back(uv);
    return true;
}

static bool build_boundary_constraints(
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling,
    const BezierGuidingFrameResult& frame,
    const ConstrainedArapConfig& config,
    BoundaryConstraintBuilder& builder,
    string& reason) {
    vector<ParameterSideRole> roles = resolve_roles(labeling, config);
    double south_len = 0.0, north_len = 0.0, east_len = 0.0, west_len = 0.0;
    int south_count = 0, north_count = 0, east_count = 0, west_count = 0;
    for (int si = 0; si < (int)labeling.abstract_sides.size(); si++) {
        if (labeling.abstract_sides[si].segment_ids.empty()) continue;
        double fallback = side_sample_length(input, labeling.abstract_sides[si]);
        double len = frame_side_length(frame, si, fallback);
        if (roles[si] == ParameterSideRole::South) {
            south_len += len;
            south_count++;
        } else if (roles[si] == ParameterSideRole::North) {
            north_len += len;
            north_count++;
        } else if (roles[si] == ParameterSideRole::East) {
            east_len += len;
            east_count++;
        } else if (roles[si] == ParameterSideRole::West) {
            west_len += len;
            west_count++;
        }
    }
    builder.width = 1.0;
    builder.height = 1.0;
    if (south_count + north_count > 0) {
        builder.width = (south_len + north_len) / (double)(south_count + north_count);
    }
    if (east_count + west_count > 0) {
        builder.height = (east_len + west_len) / (double)(east_count + west_count);
    }
    builder.width = std::max(builder.width, 1e-8);
    builder.height = std::max(builder.height, 1e-8);

    for (int si = 0; si < (int)labeling.abstract_sides.size(); si++) {
        const AbstractSide& side = labeling.abstract_sides[si];
        if (side.segment_ids.empty()) continue;
        ParameterSideRole role = roles[si];
        if (role == ParameterSideRole::Unassigned) continue;

        vector<int> ids;
        vector<Vector3d> points;
        for (int segment_id : side.segment_ids) {
            const BoundarySegment* segment = find_segment(input, segment_id);
            if (!segment) continue;
            for (int i = 0; i < (int)segment->authoritative_vertex_ids.size(); i++) {
                if (!ids.empty() && i == 0 &&
                    ids.back() == segment->authoritative_vertex_ids[i]) {
                    continue;
                }
                ids.push_back(segment->authoritative_vertex_ids[i]);
                points.push_back(segment->authoritative_positions[i]);
            }
        }
        if (ids.size() < 2) continue;

        vector<double> params(ids.size(), 0.0);
        double total = 0.0;
        for (int i = 1; i < (int)points.size(); i++) {
            total += (points[i] - points[i - 1]).norm();
            params[i] = total;
        }
        if (total <= kEps) {
            for (int i = 0; i < (int)params.size(); i++) {
                params[i] = (double)i / (double)std::max(1, (int)params.size() - 1);
            }
        } else {
            for (double& t : params) t /= total;
        }

        Vector2d start = role_start_uv(role, builder.width, builder.height);
        Vector2d end = role_end_uv(role, builder.width, builder.height);
        for (int i = 0; i < (int)ids.size(); i++) {
            Vector2d uv = (1.0 - params[i]) * start + params[i] * end;
            if (!add_fixed_uv(builder, ids[i], uv, reason)) return false;
        }
    }

    if (builder.fixed_uv.size() < 2) {
        reason = "not enough boundary constraints for ARAP parameterization";
        return false;
    }
    return true;
}

static vector<set<int>> build_adjacency(const MatrixXi& local_faces) {
    vector<set<int>> adjacency;
    int n = 0;
    for (int i = 0; i < local_faces.rows(); i++) {
        for (int k = 0; k < 3; k++) n = std::max(n, local_faces(i, k) + 1);
    }
    adjacency.assign(n, set<int>());
    for (int fi = 0; fi < local_faces.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            int a = local_faces(fi, k);
            int b = local_faces(fi, (k + 1) % 3);
            adjacency[a].insert(b);
            adjacency[b].insert(a);
        }
    }
    return adjacency;
}

static SparseMatrix<double> build_uniform_laplacian(const vector<set<int>>& adjacency) {
    vector<Triplet<double>> triplets;
    for (int i = 0; i < (int)adjacency.size(); i++) {
        triplets.emplace_back(i, i, (double)adjacency[i].size());
        for (int nb : adjacency[i]) triplets.emplace_back(i, nb, -1.0);
    }
    SparseMatrix<double> L((int)adjacency.size(), (int)adjacency.size());
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

static bool solve_kkt_scalar(
    const SparseMatrix<double>& A,
    const VectorXd& rhs,
    const vector<int>& constrained_locals,
    const vector<double>& constrained_values,
    VectorXd& x,
    string& reason) {
    int n = A.rows();
    int m = (int)constrained_locals.size();
    vector<Triplet<double>> triplets;
    triplets.reserve(A.nonZeros() + 2 * m);
    for (int k = 0; k < A.outerSize(); k++) {
        for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
        }
    }
    for (int ci = 0; ci < m; ci++) {
        int v = constrained_locals[ci];
        triplets.emplace_back(v, n + ci, 1.0);
        triplets.emplace_back(n + ci, v, 1.0);
    }
    SparseMatrix<double> K(n + m, n + m);
    K.setFromTriplets(triplets.begin(), triplets.end());
    VectorXd b = VectorXd::Zero(n + m);
    b.head(n) = rhs;
    for (int ci = 0; ci < m; ci++) b(n + ci) = constrained_values[ci];

    Eigen::SparseLU<SparseMatrix<double>> solver;
    solver.compute(K);
    if (solver.info() != Eigen::Success) {
        reason = "KKT factorization failed";
        return false;
    }
    VectorXd sol = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        reason = "KKT solve failed";
        return false;
    }
    x = sol.head(n);
    return true;
}

static bool solve_kkt_uv(
    const SparseMatrix<double>& A,
    const MatrixXd& rhs,
    const vector<int>& constrained_locals,
    const vector<Vector2d>& constrained_values,
    MatrixXd& UV,
    string& reason) {
    vector<double> cu(constrained_values.size()), cv(constrained_values.size());
    for (int i = 0; i < (int)constrained_values.size(); i++) {
        cu[i] = constrained_values[i].x();
        cv[i] = constrained_values[i].y();
    }
    VectorXd xu, xv;
    if (!solve_kkt_scalar(A, rhs.col(0), constrained_locals, cu, xu, reason)) return false;
    if (!solve_kkt_scalar(A, rhs.col(1), constrained_locals, cv, xv, reason)) return false;
    UV.resize(A.rows(), 2);
    UV.col(0) = xu;
    UV.col(1) = xv;
    return true;
}

static LocalTriangleGeometry local_triangle_geometry(
    const Vector3d& p0,
    const Vector3d& p1,
    const Vector3d& p2) {
    LocalTriangleGeometry g;
    double l01 = (p1 - p0).norm();
    double l02 = (p2 - p0).norm();
    if (l01 <= kEps || l02 <= kEps) return g;
    double x2 = (p2 - p0).dot(p1 - p0) / l01;
    double y2_sq = l02 * l02 - x2 * x2;
    if (y2_sq <= kEps) return g;
    g.x[0] = Vector2d(0.0, 0.0);
    g.x[1] = Vector2d(l01, 0.0);
    g.x[2] = Vector2d(x2, std::sqrt(y2_sq));
    g.area = 0.5 * l01 * g.x[2].y();
    g.valid = true;
    return g;
}

static Matrix2d best_rotation_2d(
    const Vector2d src[3],
    const Vector2d dst[3]) {
    Matrix2d cov = Matrix2d::Zero();
    const int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    for (const auto& e : edges) {
        Vector2d a = src[e[0]] - src[e[1]];
        Vector2d b = dst[e[0]] - dst[e[1]];
        cov += b * a.transpose();
    }
    Eigen::JacobiSVD<Matrix2d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix2d R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0.0) {
        Matrix2d U = svd.matrixU();
        U.col(1) *= -1.0;
        R = U * svd.matrixV().transpose();
    }
    return R;
}

static bool arap_global_solve(
    const MatrixXd& V,
    const MatrixXi& F,
    const MatrixXi& local_faces,
    const vector<int>& region_vertex_ids,
    const vector<int>& local_face_to_global_face,
    const vector<LocalTriangleGeometry>& local_geo,
    const vector<Matrix2d>& rotations,
    const vector<int>& constrained_locals,
    const vector<Vector2d>& constrained_values,
    MatrixXd& local_uv,
    string& reason) {
    int n = (int)region_vertex_ids.size();
    vector<Triplet<double>> triplets;
    MatrixXd rhs = MatrixXd::Zero(n, 2);
    map<pair<int, int>, int> edge_seen;
    (void)V;
    (void)F;
    (void)local_face_to_global_face;

    auto add_edge = [&](int a, int b, const Vector2d& target) {
        triplets.emplace_back(a, a, 1.0);
        triplets.emplace_back(b, b, 1.0);
        triplets.emplace_back(a, b, -1.0);
        triplets.emplace_back(b, a, -1.0);
        rhs.row(a) += target.transpose();
        rhs.row(b) -= target.transpose();
    };

    for (int fi = 0; fi < local_faces.rows(); fi++) {
        if (!local_geo[fi].valid) continue;
        const int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
        for (const auto& e : edges) {
            int ia = e[0];
            int ib = e[1];
            int a = local_faces(fi, ia);
            int b = local_faces(fi, ib);
            Vector2d xij = local_geo[fi].x[ia] - local_geo[fi].x[ib];
            Vector2d target = rotations[fi] * xij;
            add_edge(a, b, target);
            edge_seen[edge_key(a, b)]++;
        }
    }

    SparseMatrix<double> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return solve_kkt_uv(A, rhs, constrained_locals, constrained_values, local_uv, reason);
}

static void compute_triangle_stats(
    const MatrixXd& V,
    const MatrixXi& F,
    const MatrixXd& local_uv,
    const MatrixXi& local_faces,
    const vector<int>& local_face_to_global_face,
    const vector<LocalTriangleGeometry>& local_geo,
    const vector<Matrix2d>& rotations,
    const ConstrainedArapConfig& config,
    ConstrainedArapResult& result) {
    result.triangle_stats.clear();
    result.mean_area_ratio = 0.0;
    result.max_area_ratio = 0.0;
    result.mean_conformal_distortion = 0.0;
    result.max_conformal_distortion = 0.0;
    result.mean_arap_residual = 0.0;
    result.max_arap_residual = 0.0;
    result.flipped_triangle_count = 0;
    int positive_orientation_count = 0;
    int negative_orientation_count = 0;

    for (int fi = 0; fi < local_faces.rows(); fi++) {
        int gfi = local_face_to_global_face[fi];
        int a = local_faces(fi, 0);
        int b = local_faces(fi, 1);
        int c = local_faces(fi, 2);
        Vector2d uv0 = local_uv.row(a);
        Vector2d uv1 = local_uv.row(b);
        Vector2d uv2 = local_uv.row(c);
        Vector3d p0 = V.row(F(gfi, 0));
        Vector3d p1 = V.row(F(gfi, 1));
        Vector3d p2 = V.row(F(gfi, 2));

        ConstrainedArapTriangleStats st;
        st.local_face_id = fi;
        st.global_face_id = gfi;
        st.signed_uv_area = signed_area_uv(uv0, uv1, uv2);
        st.abs_uv_area = std::abs(st.signed_uv_area);
        st.area_3d = triangle_area_3d(p0, p1, p2);
        st.area_ratio = st.area_3d > kEps ? st.abs_uv_area / st.area_3d : 0.0;
        st.orientation = st.signed_uv_area > config.min_signed_area ? 1 :
                         (st.signed_uv_area < -config.min_signed_area ? -1 : 0);
        if (st.orientation > 0) positive_orientation_count++;
        if (st.orientation < 0) negative_orientation_count++;

        st.conformal_distortion = std::numeric_limits<double>::infinity();
        if (local_geo[fi].valid) {
            Matrix2d X, U;
            X.col(0) = local_geo[fi].x[1] - local_geo[fi].x[0];
            X.col(1) = local_geo[fi].x[2] - local_geo[fi].x[0];
            U.col(0) = uv1 - uv0;
            U.col(1) = uv2 - uv0;
            if (std::abs(X.determinant()) > kEps) {
                Matrix2d J = U * X.inverse();
                Eigen::JacobiSVD<Matrix2d> svd(J);
                double s0 = svd.singularValues()(0);
                double s1 = svd.singularValues()(1);
                if (s1 > kEps) st.conformal_distortion = s0 / s1;
            }

            const int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
            Vector2d uv[3] = {uv0, uv1, uv2};
            for (const auto& e : edges) {
                Vector2d target =
                    rotations[fi] * (local_geo[fi].x[e[0]] - local_geo[fi].x[e[1]]);
                Vector2d actual = uv[e[0]] - uv[e[1]];
                st.arap_residual += (actual - target).squaredNorm();
            }
            st.arap_residual = std::sqrt(st.arap_residual / 3.0);
        }

        result.triangle_stats.push_back(st);
        result.mean_area_ratio += st.area_ratio;
        result.max_area_ratio = std::max(result.max_area_ratio, st.area_ratio);
        if (std::isfinite(st.conformal_distortion)) {
            result.mean_conformal_distortion += st.conformal_distortion;
            result.max_conformal_distortion =
                std::max(result.max_conformal_distortion, st.conformal_distortion);
        }
        result.mean_arap_residual += st.arap_residual;
        result.max_arap_residual = std::max(result.max_arap_residual, st.arap_residual);
    }

    int dominant_orientation =
        negative_orientation_count > positive_orientation_count ? -1 : 1;
    for (ConstrainedArapTriangleStats& st : result.triangle_stats) {
        st.flipped =
            st.orientation == 0 ||
            (positive_orientation_count > 0 &&
             negative_orientation_count > 0 &&
             st.orientation != dominant_orientation);
        if (st.flipped) result.flipped_triangle_count++;
    }

    double n = (double)std::max(1, (int)result.triangle_stats.size());
    result.mean_area_ratio /= n;
    result.mean_conformal_distortion /= n;
    result.mean_arap_residual /= n;
}

static bool export_uv_obj(const string& filename, const ConstrainedArapResult& result) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << "# Constrained ARAP UV mesh\n";
    for (int vid : result.region_vertex_ids) {
        out << "v " << result.UV(vid, 0) << " " << result.UV(vid, 1) << " 0\n";
    }
    for (int fi = 0; fi < result.local_faces.rows(); fi++) {
        out << "f " << (result.local_faces(fi, 0) + 1)
            << " " << (result.local_faces(fi, 1) + 1)
            << " " << (result.local_faces(fi, 2) + 1) << "\n";
    }
    return true;
}

static bool export_stats_csv(const string& filename, const ConstrainedArapResult& result) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << "global_face_id,signed_uv_area,abs_uv_area,area_3d,area_ratio,conformal_distortion,arap_residual,orientation,flipped\n";
    for (const ConstrainedArapTriangleStats& st : result.triangle_stats) {
        out << st.global_face_id << ","
            << st.signed_uv_area << ","
            << st.abs_uv_area << ","
            << st.area_3d << ","
            << st.area_ratio << ","
            << st.conformal_distortion << ","
            << st.arap_residual << ","
            << st.orientation << ","
            << (st.flipped ? 1 : 0) << "\n";
    }
    return true;
}

static bool export_constraints_csv(const string& filename, const ConstrainedArapResult& result) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << "vertex_id,u,v\n";
    for (int i = 0; i < (int)result.constrained_vertex_ids.size(); i++) {
        out << result.constrained_vertex_ids[i] << ","
            << result.constrained_uvs[i].x() << ","
            << result.constrained_uvs[i].y() << "\n";
    }
    return true;
}

static bool export_iterations_csv(const string& filename, const ConstrainedArapResult& result) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << "iteration,mean_arap_residual\n";
    for (int i = 0; i < (int)result.iteration_residuals.size(); i++) {
        out << i << "," << result.iteration_residuals[i] << "\n";
    }
    return true;
}

} // namespace

const char* to_string(ParameterSideRole role) {
    switch (role) {
    case ParameterSideRole::South: return "South";
    case ParameterSideRole::East: return "East";
    case ParameterSideRole::North: return "North";
    case ParameterSideRole::West: return "West";
    case ParameterSideRole::Unassigned: return "Unassigned";
    }
    return "Unknown";
}

ConstrainedArapResult parameterize_constrained_arap_mvp(
    const MatrixXd& V,
    const MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling,
    const BezierGuidingFrameResult& guiding_frame,
    const ConstrainedArapConfig& config) {
    ConstrainedArapResult result;
    result.UV = MatrixXd::Constant(V.rows(), 2, std::numeric_limits<double>::quiet_NaN());
    if (!input.valid) {
        result.reason = "boundary input is invalid: " + input.reason;
        return result;
    }
    if (!labeling.valid) {
        result.reason = "labeling input is invalid: " + labeling.reason;
        return result;
    }
    if (!guiding_frame.valid) {
        result.reason = "guiding frame input is invalid: " + guiding_frame.reason;
        return result;
    }

    map<int, int> global_to_local;
    if (!collect_region_submesh(
            F, input.region, result.region_vertex_ids, global_to_local,
            result.local_faces, result.local_face_to_global_face, result.reason)) {
        return result;
    }

    BoundaryConstraintBuilder constraints;
    if (!build_boundary_constraints(
            input, labeling, guiding_frame, config, constraints, result.reason)) {
        return result;
    }
    result.width = constraints.width;
    result.height = constraints.height;
    result.constrained_vertex_ids = constraints.constrained_vertex_ids;
    result.constrained_uvs = constraints.constrained_uvs;

    vector<int> constrained_locals;
    vector<Vector2d> constrained_values;
    for (const auto& kv : constraints.fixed_uv) {
        auto it = global_to_local.find(kv.first);
        if (it == global_to_local.end()) {
            result.reason = "boundary constraint vertex is not in region submesh";
            return result;
        }
        constrained_locals.push_back(it->second);
        constrained_values.push_back(kv.second);
    }

    vector<set<int>> adjacency = build_adjacency(result.local_faces);
    SparseMatrix<double> L = build_uniform_laplacian(adjacency);
    MatrixXd zero_rhs = MatrixXd::Zero((int)result.region_vertex_ids.size(), 2);
    MatrixXd local_uv;
    if (!solve_kkt_uv(
            L, zero_rhs, constrained_locals, constrained_values,
            local_uv, result.reason)) {
        return result;
    }

    vector<LocalTriangleGeometry> local_geo(result.local_faces.rows());
    for (int fi = 0; fi < result.local_faces.rows(); fi++) {
        int gfi = result.local_face_to_global_face[fi];
        local_geo[fi] = local_triangle_geometry(
            V.row(F(gfi, 0)), V.row(F(gfi, 1)), V.row(F(gfi, 2)));
    }

    vector<Matrix2d> rotations(result.local_faces.rows(), Matrix2d::Identity());
    MatrixXd best_injective_uv;
    vector<ConstrainedArapTriangleStats> best_triangle_stats;
    double best_mean_area_ratio = 0.0;
    double best_max_area_ratio = 0.0;
    double best_mean_conformal_distortion = 0.0;
    double best_max_conformal_distortion = 0.0;
    double best_mean_arap_residual = 0.0;
    double best_max_arap_residual = 0.0;
    auto remember_injective_state = [&]() {
        if (result.flipped_triangle_count != 0) return;
        best_injective_uv = local_uv;
        best_triangle_stats = result.triangle_stats;
        best_mean_area_ratio = result.mean_area_ratio;
        best_max_area_ratio = result.max_area_ratio;
        best_mean_conformal_distortion = result.mean_conformal_distortion;
        best_max_conformal_distortion = result.max_conformal_distortion;
        best_mean_arap_residual = result.mean_arap_residual;
        best_max_arap_residual = result.max_arap_residual;
    };

    compute_triangle_stats(
        V, F, local_uv, result.local_faces,
        result.local_face_to_global_face, local_geo,
        rotations, config, result);
    remember_injective_state();

    double previous = std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < std::max(0, config.max_iterations); iter++) {
        for (int fi = 0; fi < result.local_faces.rows(); fi++) {
            if (!local_geo[fi].valid) continue;
            Vector2d dst[3] = {
                local_uv.row(result.local_faces(fi, 0)),
                local_uv.row(result.local_faces(fi, 1)),
                local_uv.row(result.local_faces(fi, 2))};
            rotations[fi] = best_rotation_2d(local_geo[fi].x, dst);
        }

        MatrixXd next_uv;
        if (!arap_global_solve(
                V, F, result.local_faces, result.region_vertex_ids,
                result.local_face_to_global_face, local_geo, rotations,
                constrained_locals, constrained_values, next_uv, result.reason)) {
            return result;
        }
        local_uv = next_uv;

        compute_triangle_stats(
            V, F, local_uv, result.local_faces,
            result.local_face_to_global_face, local_geo,
            rotations, config, result);
        result.iteration_residuals.push_back(result.mean_arap_residual);
        remember_injective_state();
        if (std::abs(previous - result.mean_arap_residual) <=
            config.convergence_tolerance) {
            break;
        }
        previous = result.mean_arap_residual;
    }

    if (result.iteration_residuals.empty()) {
        compute_triangle_stats(
            V, F, local_uv, result.local_faces,
            result.local_face_to_global_face, local_geo,
            rotations, config, result);
    }

    if (result.flipped_triangle_count > 0 && best_injective_uv.rows() == local_uv.rows()) {
        local_uv = best_injective_uv;
        result.triangle_stats = best_triangle_stats;
        result.mean_area_ratio = best_mean_area_ratio;
        result.max_area_ratio = best_max_area_ratio;
        result.mean_conformal_distortion = best_mean_conformal_distortion;
        result.max_conformal_distortion = best_max_conformal_distortion;
        result.mean_arap_residual = best_mean_arap_residual;
        result.max_arap_residual = best_max_arap_residual;
        result.flipped_triangle_count = 0;
    }

    for (int li = 0; li < (int)result.region_vertex_ids.size(); li++) {
        int gid = result.region_vertex_ids[li];
        result.UV.row(gid) = local_uv.row(li);
    }

    if (result.flipped_triangle_count > 0 && config.fail_on_flips) {
        result.reason = "constrained ARAP produced flipped or degenerate triangles";
        if (config.export_debug) export_constrained_arap_debug(config.debug_output_prefix, V, result);
        return result;
    }

    result.valid = true;
    result.reason = "ok";
    if (config.export_debug) export_constrained_arap_debug(config.debug_output_prefix, V, result);
    return result;
}

bool export_constrained_arap_debug(
    const string& prefix,
    const MatrixXd&,
    const ConstrainedArapResult& result) {
    bool ok = true;
    ok = export_uv_obj(prefix + "_uv.obj", result) && ok;
    ok = export_stats_csv(prefix + "_triangle_stats.csv", result) && ok;
    ok = export_constraints_csv(prefix + "_constraints.csv", result) && ok;
    ok = export_iterations_csv(prefix + "_iterations.csv", result) && ok;
    return ok;
}
