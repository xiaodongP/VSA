#include "region_square_parameterization.h"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <fstream>
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
using std::map;
using std::pair;
using std::set;
using std::string;
using std::vector;

namespace {

static double nan_value() {
    return std::numeric_limits<double>::quiet_NaN();
}

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

static double angle_between(double a, double b, double c) {
    double denom = std::max(1e-30, a * b);
    double cosv = (a * a + b * b - c * c) / (2.0 * denom);
    cosv = std::max(-1.0, std::min(1.0, cosv));
    return std::acos(cosv);
}

static void triangle_angles_2d(
    const Vector2d& p0,
    const Vector2d& p1,
    const Vector2d& p2,
    double out[3]) {
    double l01 = (p1 - p0).norm();
    double l12 = (p2 - p1).norm();
    double l20 = (p0 - p2).norm();
    out[0] = angle_between(l01, l20, l12);
    out[1] = angle_between(l12, l01, l20);
    out[2] = angle_between(l20, l12, l01);
}

static void triangle_angles_3d(
    const Vector3d& p0,
    const Vector3d& p1,
    const Vector3d& p2,
    double out[3]) {
    double l01 = (p1 - p0).norm();
    double l12 = (p2 - p1).norm();
    double l20 = (p0 - p2).norm();
    out[0] = angle_between(l01, l20, l12);
    out[1] = angle_between(l12, l01, l20);
    out[2] = angle_between(l20, l12, l01);
}

static double rms_angle_difference(const double a[3], const double b[3]) {
    double s = 0.0;
    for (int i = 0; i < 3; i++) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s / 3.0);
}

static bool collect_region_submesh(
    const MatrixXi& F,
    const vector<int>& face_region_ids,
    int target_region_id,
    vector<int>& region_vertices,
    map<int, int>& global_to_local,
    MatrixXi& local_faces,
    vector<int>& local_face_to_global_face) {
    if ((int)face_region_ids.size() != F.rows()) return false;

    set<int> vertex_set;
    local_face_to_global_face.clear();
    for (int fi = 0; fi < F.rows(); fi++) {
        if (face_region_ids[fi] != target_region_id) continue;
        local_face_to_global_face.push_back(fi);
        for (int k = 0; k < 3; k++) vertex_set.insert(F(fi, k));
    }
    if (local_face_to_global_face.empty() || vertex_set.empty()) return false;

    region_vertices.assign(vertex_set.begin(), vertex_set.end());
    global_to_local.clear();
    for (int i = 0; i < (int)region_vertices.size(); i++) {
        global_to_local[region_vertices[i]] = i;
    }

    local_faces.resize((int)local_face_to_global_face.size(), 3);
    for (int i = 0; i < (int)local_face_to_global_face.size(); i++) {
        int fi = local_face_to_global_face[i];
        for (int k = 0; k < 3; k++) {
            local_faces(i, k) = global_to_local[F(fi, k)];
        }
    }
    return true;
}

static vector<int> loop_side_indices(int start, int end, int n) {
    vector<int> ids;
    int cur = start;
    for (;;) {
        ids.push_back(cur);
        if (cur == end) break;
        cur = (cur + 1) % n;
        if ((int)ids.size() > n + 1) break;
    }
    return ids;
}

static double side_chord_length(
    const RegionBoundaryLoop& loop,
    const vector<int>& side_indices) {
    double total = 0.0;
    for (int i = 1; i < (int)side_indices.size(); i++) {
        total += (loop.positions[side_indices[i]] -
                  loop.positions[side_indices[i - 1]]).norm();
    }
    return total;
}

static bool assign_boundary_uvs(
    const RegionBoundaryLoop& loop,
    const QuadLikeBoundary& quad_boundary,
    map<int, Vector2d>& boundary_uv,
    string& reason) {
    boundary_uv.clear();
    if (!loop.closed || loop.vertex_ids.size() != loop.positions.size()) {
        reason = "boundary loop must be closed and have matching ids/positions";
        return false;
    }
    if (!quad_boundary.valid) {
        reason = "quad-like boundary is not valid";
        return false;
    }
    int n = (int)loop.vertex_ids.size();
    for (int c : quad_boundary.corner_loop_indices) {
        if (c < 0 || c >= n) {
            reason = "corner loop index out of range";
            return false;
        }
    }

    for (int s = 0; s < 4; s++) {
        int a = quad_boundary.corner_loop_indices[s];
        int b = quad_boundary.corner_loop_indices[(s + 1) % 4];
        vector<int> side = loop_side_indices(a, b, n);
        if (side.size() < 2 || side.size() > (size_t)n + 1) {
            reason = "failed to trace boundary side";
            return false;
        }

        double total = side_chord_length(loop, side);
        double accum = 0.0;
        for (int i = 0; i < (int)side.size(); i++) {
            if (i > 0) {
                accum += (loop.positions[side[i]] -
                          loop.positions[side[i - 1]]).norm();
            }
            double t = total > 1e-14 ? accum / total
                                     : (double)i / (double)std::max(1, (int)side.size() - 1);
            Vector2d uv;
            if (s == 0) uv = Vector2d(t, 0.0);
            else if (s == 1) uv = Vector2d(1.0, t);
            else if (s == 2) uv = Vector2d(1.0 - t, 1.0);
            else uv = Vector2d(0.0, 1.0 - t);

            int vid = loop.vertex_ids[side[i]];
            auto it = boundary_uv.find(vid);
            if (it != boundary_uv.end() && (it->second - uv).norm() > 1e-8) {
                reason = "inconsistent boundary UV assignment at a corner";
                return false;
            }
            boundary_uv[vid] = uv;
        }
    }
    return true;
}

static bool build_region_adjacency(
    const MatrixXi& local_faces,
    vector<set<int>>& adjacency) {
    int vertex_count = 0;
    for (int i = 0; i < local_faces.rows(); i++) {
        for (int k = 0; k < 3; k++) vertex_count = std::max(vertex_count, local_faces(i, k) + 1);
    }
    adjacency.assign(vertex_count, set<int>());
    for (int i = 0; i < local_faces.rows(); i++) {
        for (int k = 0; k < 3; k++) {
            int a = local_faces(i, k);
            int b = local_faces(i, (k + 1) % 3);
            if (a == b) return false;
            adjacency[a].insert(b);
            adjacency[b].insert(a);
        }
    }
    return true;
}

static bool solve_harmonic_uv(
    const vector<int>& region_vertex_ids,
    const map<int, int>& global_to_local,
    const vector<set<int>>& adjacency,
    const map<int, Vector2d>& boundary_uv,
    MatrixXd& global_uv,
    string& reason) {
    vector<int> interior_locals;
    map<int, int> local_to_unknown;
    for (int local = 0; local < (int)region_vertex_ids.size(); local++) {
        int global = region_vertex_ids[local];
        auto bit = boundary_uv.find(global);
        if (bit != boundary_uv.end()) {
            global_uv.row(global) = bit->second.transpose();
        } else {
            int unknown = (int)interior_locals.size();
            interior_locals.push_back(local);
            local_to_unknown[local] = unknown;
        }
    }

    if (interior_locals.empty()) return true;

    vector<Triplet<double>> triplets;
    MatrixXd B = MatrixXd::Zero((int)interior_locals.size(), 2);
    triplets.reserve(interior_locals.size() * 7);

    for (int row = 0; row < (int)interior_locals.size(); row++) {
        int local = interior_locals[row];
        double degree = (double)adjacency[local].size();
        if (degree <= 0.0) {
            reason = "interior vertex has no adjacency";
            return false;
        }
        triplets.push_back(Triplet<double>(row, row, degree));
        for (int nb_local : adjacency[local]) {
            int nb_global = region_vertex_ids[nb_local];
            auto bit = boundary_uv.find(nb_global);
            if (bit != boundary_uv.end()) {
                B.row(row) += bit->second.transpose();
            } else {
                auto uit = local_to_unknown.find(nb_local);
                if (uit == local_to_unknown.end()) {
                    reason = "neighbor is neither boundary nor unknown";
                    return false;
                }
                triplets.push_back(Triplet<double>(row, uit->second, -1.0));
            }
        }
    }

    SparseMatrix<double> A((int)interior_locals.size(), (int)interior_locals.size());
    A.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        reason = "harmonic linear system factorization failed";
        return false;
    }
    MatrixXd X = solver.solve(B);
    if (solver.info() != Eigen::Success) {
        reason = "harmonic linear system solve failed";
        return false;
    }

    for (int row = 0; row < (int)interior_locals.size(); row++) {
        int global = region_vertex_ids[interior_locals[row]];
        global_uv.row(global) = X.row(row);
    }
    (void)global_to_local;
    return true;
}

static void compute_triangle_stats(
    const MatrixXd& V,
    const MatrixXi& F,
    RegionSquareParameterizationResult& result,
    const RegionSquareParameterizationConfig& cfg) {
    result.triangle_stats.clear();
    result.min_signed_area = std::numeric_limits<double>::infinity();
    result.max_signed_area = -std::numeric_limits<double>::infinity();
    result.mean_area_distortion = 0.0;
    result.max_area_distortion = 0.0;
    result.mean_angle_distortion = 0.0;
    result.max_angle_distortion = 0.0;
    result.has_flips = false;

    for (int i = 0; i < result.local_faces.rows(); i++) {
        int gfi = result.local_face_to_global_face[i];
        int v0 = F(gfi, 0);
        int v1 = F(gfi, 1);
        int v2 = F(gfi, 2);
        Vector2d uv0 = result.UV.row(v0).transpose();
        Vector2d uv1 = result.UV.row(v1).transpose();
        Vector2d uv2 = result.UV.row(v2).transpose();
        Vector3d p0 = V.row(v0).transpose();
        Vector3d p1 = V.row(v1).transpose();
        Vector3d p2 = V.row(v2).transpose();

        ParameterizationTriangleStats st;
        st.global_face_id = gfi;
        st.signed_uv_area = signed_area_uv(uv0, uv1, uv2);
        st.abs_uv_area = std::abs(st.signed_uv_area);
        st.area_3d = triangle_area_3d(p0, p1, p2);
        st.area_distortion = st.area_3d > 1e-30 ? st.abs_uv_area / st.area_3d : 0.0;
        double angles_uv[3], angles_3d[3];
        triangle_angles_2d(uv0, uv1, uv2, angles_uv);
        triangle_angles_3d(p0, p1, p2, angles_3d);
        st.angle_distortion = rms_angle_difference(angles_uv, angles_3d);
        st.flipped = st.signed_uv_area <= cfg.min_signed_area;

        result.triangle_stats.push_back(st);
        result.min_signed_area = std::min(result.min_signed_area, st.signed_uv_area);
        result.max_signed_area = std::max(result.max_signed_area, st.signed_uv_area);
        result.mean_area_distortion += st.area_distortion;
        result.max_area_distortion = std::max(result.max_area_distortion, st.area_distortion);
        result.mean_angle_distortion += st.angle_distortion;
        result.max_angle_distortion = std::max(result.max_angle_distortion, st.angle_distortion);
        if (st.flipped) result.has_flips = true;
    }

    double n = (double)std::max(1, (int)result.triangle_stats.size());
    result.mean_area_distortion /= n;
    result.mean_angle_distortion /= n;
    if (result.triangle_stats.empty()) {
        result.min_signed_area = 0.0;
        result.max_signed_area = 0.0;
    }
}

static bool check_uv_range(
    const MatrixXd& UV,
    const vector<int>& region_vertex_ids,
    double tol) {
    for (int vid : region_vertex_ids) {
        double u = UV(vid, 0);
        double v = UV(vid, 1);
        if (!std::isfinite(u) || !std::isfinite(v)) return false;
        if (u < -tol || u > 1.0 + tol || v < -tol || v > 1.0 + tol) return false;
    }
    return true;
}

static int flipped_triangle_count(
    const vector<ParameterizationTriangleStats>& stats) {
    int count = 0;
    for (const ParameterizationTriangleStats& st : stats) {
        if (st.flipped) count++;
    }
    return count;
}

static void reflect_region_uv_u(
    MatrixXd& UV,
    const vector<int>& region_vertex_ids) {
    for (int vid : region_vertex_ids) {
        UV(vid, 0) = 1.0 - UV(vid, 0);
    }
}

static void mark_global_orientation_reversal_as_valid(
    RegionSquareParameterizationResult& result) {
    result.has_flips = false;
    result.min_signed_area = std::numeric_limits<double>::infinity();
    result.max_signed_area = -std::numeric_limits<double>::infinity();
    for (ParameterizationTriangleStats& st : result.triangle_stats) {
        st.signed_uv_area = std::abs(st.signed_uv_area);
        st.abs_uv_area = st.signed_uv_area;
        st.flipped = false;
        result.min_signed_area = std::min(result.min_signed_area, st.signed_uv_area);
        result.max_signed_area = std::max(result.max_signed_area, st.signed_uv_area);
    }
    if (result.triangle_stats.empty()) {
        result.min_signed_area = 0.0;
        result.max_signed_area = 0.0;
    }
}

} // namespace

ParameterizationTriangleStats::ParameterizationTriangleStats()
    : global_face_id(-1),
      signed_uv_area(0.0),
      abs_uv_area(0.0),
      area_3d(0.0),
      area_distortion(0.0),
      angle_distortion(0.0),
      flipped(false) {}

RegionSquareParameterizationConfig::RegionSquareParameterizationConfig()
    : uv_range_tolerance(1e-8),
      min_signed_area(1e-14),
      fail_on_flips(true) {}

RegionSquareParameterizationResult::RegionSquareParameterizationResult()
    : min_signed_area(0.0),
      max_signed_area(0.0),
      mean_area_distortion(0.0),
      max_area_distortion(0.0),
      mean_angle_distortion(0.0),
      max_angle_distortion(0.0),
      valid(false),
      has_flips(false),
      uv_in_range(false),
      orientation_reflected(false) {}

RegionSquareParameterizationResult parameterize_region_to_square(
    const MatrixXd& V,
    const MatrixXi& F,
    const vector<int>& face_region_ids,
    int target_region_id,
    const RegionBoundaryLoop& boundary_loop,
    const QuadLikeBoundary& quad_boundary,
    const RegionSquareParameterizationConfig& cfg) {
    RegionSquareParameterizationResult result;
    result.UV = MatrixXd::Constant(V.rows(), 2, nan_value());

    if (V.cols() != 3 || F.cols() != 3) {
        result.reason = "V/F dimensions are invalid";
        return result;
    }

    map<int, int> global_to_local;
    if (!collect_region_submesh(
            F, face_region_ids, target_region_id,
            result.region_vertex_ids, global_to_local,
            result.local_faces, result.local_face_to_global_face)) {
        result.reason = "failed to collect target region submesh";
        return result;
    }

    map<int, Vector2d> boundary_uv;
    if (!assign_boundary_uvs(boundary_loop, quad_boundary, boundary_uv, result.reason)) {
        return result;
    }
    for (const auto& kv : boundary_uv) {
        if (!global_to_local.count(kv.first)) {
            result.reason = "boundary vertex is not part of target region";
            return result;
        }
    }

    vector<set<int>> adjacency;
    if (!build_region_adjacency(result.local_faces, adjacency)) {
        result.reason = "failed to build region adjacency";
        return result;
    }

    if (!solve_harmonic_uv(
            result.region_vertex_ids, global_to_local, adjacency,
            boundary_uv, result.UV, result.reason)) {
        return result;
    }

    result.uv_in_range = check_uv_range(
        result.UV, result.region_vertex_ids, cfg.uv_range_tolerance);
    compute_triangle_stats(V, F, result, cfg);

    if (result.has_flips) {
        MatrixXd original_uv = result.UV;
        reflect_region_uv_u(result.UV, result.region_vertex_ids);
        compute_triangle_stats(V, F, result, cfg);
        if (!result.has_flips) {
            result.UV = original_uv;
            compute_triangle_stats(V, F, result, cfg);
            mark_global_orientation_reversal_as_valid(result);
            result.orientation_reflected = true;
        } else {
            result.UV = original_uv;
            compute_triangle_stats(V, F, result, cfg);
        }
    }

    if (!result.uv_in_range) {
        result.reason = "UV coordinates are outside the expected square range";
        return result;
    }
    if (result.has_flips && cfg.fail_on_flips) {
        int flipped = flipped_triangle_count(result.triangle_stats);
        result.reason = "UV parameterization contains flipped or degenerate triangles"
            " (count=" + std::to_string(flipped) +
            ", min_signed_area=" + std::to_string(result.min_signed_area) +
            ", max_signed_area=" + std::to_string(result.max_signed_area) + ")";
        return result;
    }

    result.valid = true;
    result.reason = result.orientation_reflected ?
        "ok (UV orientation globally reversed)" : "ok";
    return result;
}

bool export_region_uv_mesh_obj(
    const string& filename,
    const RegionSquareParameterizationResult& result) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "# Region UV mesh OBJ\n";
    fout << "# valid " << (result.valid ? 1 : 0) << "\n";
    fout << "# reason " << result.reason << "\n";

    for (int vid : result.region_vertex_ids) {
        fout << "v " << result.UV(vid, 0) << " " << result.UV(vid, 1) << " 0\n";
    }
    for (int i = 0; i < result.local_faces.rows(); i++) {
        fout << "f " << (result.local_faces(i, 0) + 1)
             << " " << (result.local_faces(i, 1) + 1)
             << " " << (result.local_faces(i, 2) + 1) << "\n";
    }
    return true;
}

bool export_region_uv_signed_area_csv(
    const string& filename,
    const RegionSquareParameterizationResult& result) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "global_face_id,signed_uv_area,abs_uv_area,flipped\n";
    for (const ParameterizationTriangleStats& st : result.triangle_stats) {
        fout << st.global_face_id << ","
             << st.signed_uv_area << ","
             << st.abs_uv_area << ","
             << (st.flipped ? 1 : 0) << "\n";
    }
    return true;
}

bool export_region_uv_distortion_csv(
    const string& filename,
    const RegionSquareParameterizationResult& result) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "global_face_id,area_3d,uv_area,area_distortion,angle_distortion,flipped\n";
    for (const ParameterizationTriangleStats& st : result.triangle_stats) {
        fout << st.global_face_id << ","
             << st.area_3d << ","
             << st.abs_uv_area << ","
             << st.area_distortion << ","
             << st.angle_distortion << ","
             << (st.flipped ? 1 : 0) << "\n";
    }
    return true;
}

bool export_region_uv_correspondence_obj(
    const string& filename,
    const MatrixXd& V,
    const RegionSquareParameterizationResult& result) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout << "# 3D mesh to UV correspondence OBJ\n";
    double scale = 1.0;
    double offset = 1.5;

    for (int vid : result.region_vertex_ids) {
        fout << "v " << V(vid, 0) << " " << V(vid, 1) << " " << V(vid, 2) << "\n";
    }
    for (int vid : result.region_vertex_ids) {
        fout << "v " << (offset + scale * result.UV(vid, 0))
             << " " << (scale * result.UV(vid, 1))
             << " 0\n";
    }
    int n = (int)result.region_vertex_ids.size();
    for (int i = 0; i < n; i++) {
        fout << "l " << (i + 1) << " " << (n + i + 1) << "\n";
    }
    return true;
}

bool export_region_square_parameterization_debug(
    const string& prefix,
    const MatrixXd& V,
    const MatrixXi& F,
    const RegionSquareParameterizationResult& result) {
    bool ok = true;
    ok = export_region_uv_mesh_obj(prefix + "_uv.obj", result) && ok;
    ok = export_region_uv_signed_area_csv(prefix + "_signed_area.csv", result) && ok;
    ok = export_region_uv_distortion_csv(prefix + "_distortion_heatmap.csv", result) && ok;
    ok = export_region_uv_correspondence_obj(prefix + "_correspondence.obj", V, result) && ok;
    (void)F;
    return ok;
}
