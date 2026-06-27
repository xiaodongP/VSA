#include "vsa_reconstruction.h"

#include <igl/AABB.h>
#include <igl/Hit.h>

#ifdef VSA_USE_CGAL_CDT
#include "vsa_cgal_cdt.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <tuple>

using namespace Eigen;
using namespace std;

using Clock = std::chrono::steady_clock;

struct RegionRayProjector {
    MatrixXi localF;
    vector<int> local_face_to_global_face;
    igl::AABB<MatrixXd, 3> tree;
    bool valid;

    RegionRayProjector() : valid(false) {}
};

static double elapsed_ms(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

string regionFallbackTypeName(RegionFallbackType type) {
    switch (type) {
        case RegionFallbackType::None: return "None";
        case RegionFallbackType::SkipRegion: return "SkipRegion";
        case RegionFallbackType::OriginalRegionMesh: return "OriginalRegionMesh";
        case RegionFallbackType::FanTriangulation: return "FanTriangulation";
        case RegionFallbackType::GridTriangulation: return "GridTriangulation";
        case RegionFallbackType::ConstrainedTriangulation: return "ConstrainedTriangulation";
        case RegionFallbackType::InvalidChart: return "InvalidChart";
        case RegionFallbackType::InvalidPolygon: return "InvalidPolygon";
        case RegionFallbackType::TooFewInteriorSamples: return "TooFewInteriorSamples";
        case RegionFallbackType::TooManyFailedLifts: return "TooManyFailedLifts";
    }
    return "Unknown";
}

string projectionDropReasonName(ProjectionDropReason reason) {
    switch (reason) {
        case ProjectionDropReason::None: return "None";
        case ProjectionDropReason::ProjectionFailed: return "ProjectionFailed";
        case ProjectionDropReason::NaN: return "NaN";
        case ProjectionDropReason::DegenerateArea: return "DegenerateArea";
        case ProjectionDropReason::FlippedNormal: return "FlippedNormal";
        case ProjectionDropReason::OutsideDomain: return "OutsideDomain";
        case ProjectionDropReason::SharpBoundaryInvalid: return "SharpBoundaryInvalid";
        case ProjectionDropReason::RegionMismatch: return "RegionMismatch";
    }
    return "Unknown";
}

static void count_drop_reason(RegionReconstructionDebugInfo& dbg,
                              ProjectionDropReason reason) {
    if (reason == ProjectionDropReason::None) return;
    dbg.dropped_triangles++;
    switch (reason) {
        case ProjectionDropReason::ProjectionFailed: dbg.dropped_projection_failed++; break;
        case ProjectionDropReason::NaN: dbg.dropped_nan++; break;
        case ProjectionDropReason::DegenerateArea: dbg.dropped_degenerate_area++; break;
        case ProjectionDropReason::FlippedNormal: dbg.dropped_flipped_normal++; break;
        case ProjectionDropReason::OutsideDomain: dbg.dropped_outside_domain++; break;
        case ProjectionDropReason::SharpBoundaryInvalid: dbg.dropped_sharp_boundary_invalid++; break;
        case ProjectionDropReason::RegionMismatch: dbg.dropped_region_mismatch++; break;
        case ProjectionDropReason::None: break;
    }
}

static inline Vector3d vertex3(const MatrixXd& V, int i) {
    return Vector3d(V(i, 0), V(i, 1), V(i, 2));
}

static bool finite_double(double x) {
    return std::isfinite(x);
}

static bool finite_vec3(const Vector3d& v) {
    return finite_double(v(0)) && finite_double(v(1)) && finite_double(v(2));
}

static bool finite_vec2(const Vector2d& v) {
    return finite_double(v(0)) && finite_double(v(1));
}

static double triangle_area3(const Vector3d& a, const Vector3d& b, const Vector3d& c) {
    return 0.5 * (b - a).cross(c - a).norm();
}

static bool inside_expanded_region_bbox(const RegionPatch& patch,
                                        const Vector3d& p) {
    if (patch.bbox_3d_diag <= 1e-12 || !finite_vec3(p)) return true;
    double pad = max(1e-8, 2.0 * max(patch.sample_spacing,
                                     0.02 * patch.bbox_3d_diag));
    for (int i = 0; i < 3; i++) {
        if (p(i) < patch.bbox_3d_min(i) - pad ||
            p(i) > patch.bbox_3d_max(i) + pad) {
            return false;
        }
    }
    return true;
}

static bool push_oriented_region_face(const RegionPatch& patch,
                                      int a,
                                      int b,
                                      int c,
                                      const vector<Vector3d>& vertices,
                                      vector<Vector3i>& faces,
                                      vector<int>& labels) {
    if (a == b || b == c || c == a) return false;
    if (a < 0 || b < 0 || c < 0 ||
        a >= (int)vertices.size() ||
        b >= (int)vertices.size() ||
        c >= (int)vertices.size()) {
        return false;
    }

    const Vector3d& pa = vertices[a];
    const Vector3d& pb = vertices[b];
    const Vector3d& pc = vertices[c];
    Vector3d n = (pb - pa).cross(pc - pa);
    if (n.norm() < 2e-15 || !finite_vec3(n)) return false;

    if (patch.normal.norm() > 1e-12 && n.dot(patch.normal) < 0.0) {
        swap(b, c);
    }

    faces.push_back(Vector3i(a, b, c));
    labels.push_back(patch.region_id);
    return true;
}

static Vector3d average_region_normal(const RegionPatch& patch,
                                      const MatrixXd& V,
                                      const MatrixXi& F) {
    Vector3d n = Vector3d::Zero();
    for (int fi : patch.face_ids) {
        Vector3d a = vertex3(V, F(fi, 0));
        Vector3d b = vertex3(V, F(fi, 1));
        Vector3d c = vertex3(V, F(fi, 2));
        n += (b - a).cross(c - a);
    }
    if (n.norm() > 1e-12) n.normalize();
    return n;
}

static vector<int> collect_region_vertices(const RegionPatch& patch,
                                           const MatrixXi& F) {
    set<int> verts;
    for (int fi : patch.face_ids) {
        for (int k = 0; k < 3; k++) verts.insert(F(fi, k));
    }
    return vector<int>(verts.begin(), verts.end());
}

static void bbox_diag(const vector<int>& vids, const MatrixXd& V,
                      Vector3d& bb_min, Vector3d& bb_max, double& diag) {
    bb_min = Vector3d::Constant(numeric_limits<double>::infinity());
    bb_max = Vector3d::Constant(-numeric_limits<double>::infinity());
    for (int vid : vids) {
        Vector3d p = vertex3(V, vid);
        bb_min = bb_min.cwiseMin(p);
        bb_max = bb_max.cwiseMax(p);
    }
    diag = (bb_max - bb_min).norm();
}

static bool build_pca_normal(const vector<int>& vids,
                             const MatrixXd& V,
                             const Vector3d& center,
                             Vector3d& out_normal) {
    if (vids.size() < 3) return false;

    Matrix3d cov = Matrix3d::Zero();
    for (int vid : vids) {
        Vector3d d = vertex3(V, vid) - center;
        cov += d * d.transpose();
    }

    SelfAdjointEigenSolver<Matrix3d> eig(cov);
    if (eig.info() != Success) return false;
    out_normal = eig.eigenvectors().col(0);
    return out_normal.norm() > 1e-12 && finite_vec3(out_normal);
}

static bool make_tangent_frame(const Vector3d& normal,
                               Vector3d& tangent_u,
                               Vector3d& tangent_v) {
    if (normal.norm() < 1e-12 || !finite_vec3(normal)) return false;

    Vector3d n = normal.normalized();
    Vector3d axis = (abs(n.dot(Vector3d::UnitZ())) < 0.9)
                  ? Vector3d::UnitZ()
                  : Vector3d::UnitY();
    tangent_u = axis.cross(n);
    if (tangent_u.norm() < 1e-12) return false;
    tangent_u.normalize();
    tangent_v = n.cross(tangent_u);
    if (tangent_v.norm() < 1e-12) return false;
    tangent_v.normalize();
    return true;
}

static Vector2d project_to_chart(const RegionPatch& patch, const Vector3d& p) {
    Vector3d d = p - patch.center;
    return Vector2d(d.dot(patch.tangent_u), d.dot(patch.tangent_v));
}

static bool build_chart(RegionPatch& patch,
                        const MatrixXd& V,
                        const MatrixXi& F,
                        const ReconstructionConfig& cfg) {
    vector<int> vids = collect_region_vertices(patch, F);
    if (vids.empty()) return false;

    patch.center = Vector3d::Zero();
    for (int vid : vids) patch.center += vertex3(V, vid);
    patch.center /= (double)vids.size();

    Vector3d bb_min, bb_max;
    double diag = 0.0;
    bbox_diag(vids, V, bb_min, bb_max, diag);
    patch.bbox_3d_min = bb_min;
    patch.bbox_3d_max = bb_max;
    patch.bbox_3d_diag = diag;
    patch.max_lift_distance = max(1e-8, cfg.max_lift_distance_factor * max(diag, 1e-8));
    patch.min_lift_normal_alignment = cfg.min_lift_normal_alignment;

    Vector3d g = patch.quadric.grad(patch.center);
    if (g.norm() > 1e-12 && finite_vec3(g)) {
        double f = patch.quadric.eval(patch.center);
        Vector3d step = -(f / g.squaredNorm()) * g;
        if (finite_vec3(step) && step.norm() <= max(diag, 1e-8)) {
            patch.center += step;
        }
    }

    patch.normal = patch.quadric.grad(patch.center);
    bool has_quadric_normal = patch.normal.norm() > 1e-12 && finite_vec3(patch.normal);
    if (!has_quadric_normal) {
        if (!build_pca_normal(vids, V, patch.center, patch.normal)) {
            return false;
        }
    }
    patch.normal.normalize();

    Vector3d avg_n = average_region_normal(patch, V, F);
    if (avg_n.norm() > 1e-12 && patch.normal.dot(avg_n) < 0.0) {
        patch.normal = -patch.normal;
    }

    if (!make_tangent_frame(patch.normal, patch.tangent_u, patch.tangent_v)) {
        return false;
    }

    double min_signed = numeric_limits<double>::infinity();
    double max_signed = -numeric_limits<double>::infinity();
    for (int vid : vids) {
        double s = (vertex3(V, vid) - patch.center).dot(patch.normal);
        if (!finite_double(s)) continue;
        min_signed = min(min_signed, s);
        max_signed = max(max_signed, s);
    }
    if (finite_double(min_signed) && finite_double(max_signed)) {
        double normal_extent = max_signed - min_signed;
        patch.normal_extent = max(0.0, normal_extent);
        double thickness_limit = max(2.5 * normal_extent, 0.08 * max(diag, 1e-8));
        patch.max_lift_distance = min(patch.max_lift_distance,
                                      max(thickness_limit, 1e-8));
    }

    patch.valid_chart = true;
    return true;
}

static bool order_boundary_loop(RegionPatch& patch) {
    patch.ordered_boundary_vertex_ids.clear();
    patch.simple_boundary_loop = false;

    if (patch.boundary_edges.size() < 3) return false;

    map<int, vector<int>> adj;
    for (const EdgeKey& e : patch.boundary_edges) {
        adj[e.v0].push_back(e.v1);
        adj[e.v1].push_back(e.v0);
    }

    if (adj.size() < 3 || adj.size() != patch.boundary_edges.size()) {
        return false;
    }
    for (const auto& kv : adj) {
        if (kv.second.size() != 2) return false;
    }

    int start = adj.begin()->first;
    int prev = -1;
    int cur = start;
    set<int> visited;

    for (;;) {
        if (visited.count(cur)) return false;
        visited.insert(cur);
        patch.ordered_boundary_vertex_ids.push_back(cur);

        const vector<int>& nbrs = adj[cur];
        int next = (nbrs[0] == prev) ? nbrs[1] : nbrs[0];
        if (next == start) break;

        prev = cur;
        cur = next;
        if ((int)patch.ordered_boundary_vertex_ids.size() > (int)adj.size()) {
            return false;
        }
    }

    if (visited.size() != adj.size()) return false;
    patch.simple_boundary_loop = true;
    return true;
}

static int count_boundary_loops(const vector<EdgeKey>& boundary_edges) {
    if (boundary_edges.empty()) return 0;

    map<int, vector<int>> adj;
    for (const EdgeKey& e : boundary_edges) {
        adj[e.v0].push_back(e.v1);
        adj[e.v1].push_back(e.v0);
    }

    set<int> visited;
    int loops = 0;
    for (const auto& kv : adj) {
        int start = kv.first;
        if (visited.count(start)) continue;

        vector<int> stack;
        stack.push_back(start);
        visited.insert(start);
        int vertex_count = 0;
        int degree_sum = 0;
        bool all_degree_two = true;

        while (!stack.empty()) {
            int v = stack.back();
            stack.pop_back();
            vertex_count++;
            degree_sum += (int)adj[v].size();
            if (adj[v].size() != 2) all_degree_two = false;
            for (int nb : adj[v]) {
                if (!visited.count(nb)) {
                    visited.insert(nb);
                    stack.push_back(nb);
                }
            }
        }

        int edge_count = degree_sum / 2;
        if (all_degree_two && vertex_count >= 3 && edge_count == vertex_count) {
            loops++;
        }
    }
    return loops;
}

static double polygon_signed_area(const vector<Vector2d>& poly) {
    double a = 0.0;
    int n = (int)poly.size();
    for (int i = 0; i < n; i++) {
        const Vector2d& p = poly[i];
        const Vector2d& q = poly[(i + 1) % n];
        a += p(0) * q(1) - q(0) * p(1);
    }
    return 0.5 * a;
}

static double cross2(const Vector2d& a, const Vector2d& b) {
    return a(0) * b(1) - a(1) * b(0);
}

static bool point_on_segment_2d(const Vector2d& p,
                                const Vector2d& a,
                                const Vector2d& b) {
    Vector2d ab = b - a;
    Vector2d ap = p - a;
    double len2 = ab.squaredNorm();
    if (len2 < 1e-20) return (p - a).norm() < 1e-10;
    double t = ap.dot(ab) / len2;
    if (t < -1e-10 || t > 1.0 + 1e-10) return false;
    return abs(cross2(ab, ap)) <= 1e-10 * max(1.0, ab.norm());
}

static int orient2d_sign(const Vector2d& a,
                         const Vector2d& b,
                         const Vector2d& c) {
    double v = cross2(b - a, c - a);
    if (v > 1e-12) return 1;
    if (v < -1e-12) return -1;
    return 0;
}

static bool segments_intersect_2d(const Vector2d& a,
                                  const Vector2d& b,
                                  const Vector2d& c,
                                  const Vector2d& d) {
    int o1 = orient2d_sign(a, b, c);
    int o2 = orient2d_sign(a, b, d);
    int o3 = orient2d_sign(c, d, a);
    int o4 = orient2d_sign(c, d, b);

    if (o1 == 0 && point_on_segment_2d(c, a, b)) return true;
    if (o2 == 0 && point_on_segment_2d(d, a, b)) return true;
    if (o3 == 0 && point_on_segment_2d(a, c, d)) return true;
    if (o4 == 0 && point_on_segment_2d(b, c, d)) return true;

    return o1 != o2 && o3 != o4;
}

static bool polygon_has_self_intersections(const vector<Vector2d>& poly) {
    int n = (int)poly.size();
    if (n < 4) return false;
    for (int i = 0; i < n; i++) {
        int i_next = (i + 1) % n;
        for (int j = i + 1; j < n; j++) {
            int j_next = (j + 1) % n;
            bool adjacent = (i_next == j) || (j_next == i);
            if (adjacent) continue;

            if (segments_intersect_2d(poly[i], poly[i_next], poly[j], poly[j_next])) {
                return true;
            }
        }
    }
    return false;
}

static bool point_in_polygon(const Vector2d& p, const vector<Vector2d>& poly) {
    int n = (int)poly.size();
    if (n < 3) return false;

    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const Vector2d& a = poly[i];
        const Vector2d& b = poly[j];

        if (point_on_segment_2d(p, a, b)) return true;

        bool intersect = ((a(1) > p(1)) != (b(1) > p(1))) &&
            (p(0) < (b(0) - a(0)) * (p(1) - a(1)) / (b(1) - a(1) + 1e-30) + a(0));
        if (intersect) inside = !inside;
    }
    return inside;
}

static Vector2d closest_point_on_segment_2d(const Vector2d& p,
                                            const Vector2d& a,
                                            const Vector2d& b,
                                            double& out_t) {
    Vector2d ab = b - a;
    double len2 = ab.squaredNorm();
    if (len2 < 1e-20) {
        out_t = 0.0;
        return a;
    }
    out_t = max(0.0, min(1.0, (p - a).dot(ab) / len2));
    return a + out_t * ab;
}

static double distance_to_polygon_boundary_2d(const Vector2d& p,
                                              const vector<Vector2d>& poly) {
    if (poly.empty()) return numeric_limits<double>::infinity();
    double best = numeric_limits<double>::infinity();
    int n = (int)poly.size();
    for (int i = 0; i < n; i++) {
        double t = 0.0;
        Vector2d q = closest_point_on_segment_2d(p, poly[i], poly[(i + 1) % n], t);
        best = min(best, (p - q).norm());
    }
    return best;
}

static bool nearest_polygon_boundary_segment_2d(const Vector2d& p,
                                                const vector<Vector2d>& poly,
                                                double& out_distance,
                                                int& out_segment,
                                                double& out_t) {
    out_distance = numeric_limits<double>::infinity();
    out_segment = -1;
    out_t = 0.0;
    if (poly.empty()) return false;

    int n = (int)poly.size();
    for (int i = 0; i < n; i++) {
        double t = 0.0;
        Vector2d q = closest_point_on_segment_2d(p, poly[i], poly[(i + 1) % n], t);
        double d = (p - q).norm();
        if (d < out_distance) {
            out_distance = d;
            out_segment = i;
            out_t = t;
        }
    }

    return out_segment >= 0 && finite_double(out_distance);
}

static double smoothstep01(double x) {
    x = max(0.0, min(1.0, x));
    return x * x * (3.0 - 2.0 * x);
}

static void map_boundary_uv(RegionPatch& patch, const MatrixXd& V) {
    patch.boundary_uv.clear();
    for (int vid : patch.ordered_boundary_vertex_ids) {
        patch.boundary_uv.push_back(project_to_chart(patch, vertex3(V, vid)));
    }
}

static double compute_bbox_sample_spacing(RegionPatch& patch,
                                          const ReconstructionConfig& cfg) {
    if (patch.boundary_uv.empty()) return 0.0;
    Vector2d mn = patch.boundary_uv[0];
    Vector2d mx = patch.boundary_uv[0];
    for (const Vector2d& uv : patch.boundary_uv) {
        mn = mn.cwiseMin(uv);
        mx = mx.cwiseMax(uv);
    }

    patch.bbox_uv_min = mn;
    patch.bbox_uv_max = mx;
    Vector2d ext = mx - mn;
    double max_dim = max(ext(0), ext(1));
    if (max_dim < 1e-12) return 0.0;
    int base_res = max(1, cfg.grid_resolution);
    if (cfg.target_sample_spacing > 1e-12) {
        patch.sample_spacing = cfg.target_sample_spacing;
    } else {
        patch.sample_spacing = max_dim / (double)base_res;
    }
    return patch.sample_spacing;
}

static double compute_global_sample_spacing(vector<RegionPatch>& patches,
                                            const ReconstructionConfig& cfg) {
    if (cfg.target_sample_spacing > 1e-12) return cfg.target_sample_spacing;
    if (!cfg.enable_global_sample_spacing) return 0.0;

    vector<double> region_max_dims;
    region_max_dims.reserve(patches.size());
    for (RegionPatch& patch : patches) {
        if (!patch.valid_chart || !patch.simple_boundary_loop || patch.boundary_uv.size() < 3) {
            continue;
        }
        Vector2d mn = patch.boundary_uv[0];
        Vector2d mx = patch.boundary_uv[0];
        for (const Vector2d& uv : patch.boundary_uv) {
            mn = mn.cwiseMin(uv);
            mx = mx.cwiseMax(uv);
        }
        patch.bbox_uv_min = mn;
        patch.bbox_uv_max = mx;
        double max_dim = max((mx - mn)(0), (mx - mn)(1));
        if (max_dim > 1e-12 && finite_double(max_dim)) {
            region_max_dims.push_back(max_dim);
        }
    }

    if (region_max_dims.empty()) return 0.0;
    sort(region_max_dims.begin(), region_max_dims.end());
    double median_dim = region_max_dims[region_max_dims.size() / 2];
    double spacing = median_dim / (double)max(1, cfg.grid_resolution);
    return finite_double(spacing) && spacing > 1e-12 ? spacing : 0.0;
}

static double directed_t_as_edge_t(int a, int b, double t_directed) {
    EdgeKey e(a, b);
    return (a == e.v0) ? t_directed : (1.0 - t_directed);
}

static int edge_segment_count(const EdgeKey& e,
                              double spacing,
                              const MatrixXd& V,
                              const ReconstructionConfig& cfg) {
    if (!cfg.enable_boundary_subdivision || spacing <= 1e-12) return 1;
    double target = max(1e-12, cfg.boundary_subdivision_spacing_factor * spacing);
    double len = (vertex3(V, e.v1) - vertex3(V, e.v0)).norm();
    int n = max(1, (int)ceil(len / target));
    return min(n, max(1, cfg.max_boundary_subdivisions_per_edge));
}

static void build_subdivided_boundary_samples(
    RegionPatch& patch,
    const MatrixXd& V,
    const map<EdgeKey, int>& global_edge_segments) {
    patch.boundary_samples.clear();
    patch.boundary_uv.clear();
    if (patch.ordered_boundary_vertex_ids.size() < 3) return;

    int n = (int)patch.ordered_boundary_vertex_ids.size();
    for (int i = 0; i < n; i++) {
        int a = patch.ordered_boundary_vertex_ids[i];
        int b = patch.ordered_boundary_vertex_ids[(i + 1) % n];
        EdgeKey e(a, b);

        int seg_count = 1;
        auto it = global_edge_segments.find(e);
        if (it != global_edge_segments.end()) seg_count = max(1, it->second);

        for (int s = 0; s < seg_count; s++) {
            double t_directed = (double)s / (double)seg_count;
            double t_edge = directed_t_as_edge_t(a, b, t_directed);
            Vector3d pa = vertex3(V, a);
            Vector3d pb = vertex3(V, b);
            Vector3d pos = (1.0 - t_directed) * pa + t_directed * pb;

            RegionPatch::BoundarySample bs;
            bs.edge = e;
            bs.segment_index = (int)llround(t_edge * seg_count);
            bs.segment_count = seg_count;
            bs.t_from_edge_v0 = t_edge;
            bs.position = pos;
            bs.uv = project_to_chart(patch, pos);
            if (s == 0) bs.original_vertex_id = a;

            patch.boundary_samples.push_back(bs);
            patch.boundary_uv.push_back(bs.uv);
        }
    }
}

static bool validate_polygon(RegionPatch& patch) {
    patch.valid_polygon = false;
    patch.polygon_signed_area = 0.0;
    patch.polygon_abs_area = 0.0;
    patch.polygon_self_intersect = false;
    if (patch.boundary_uv.size() < 3) return false;
    for (const Vector2d& uv : patch.boundary_uv) {
        if (!finite_vec2(uv)) return false;
    }

    double area = polygon_signed_area(patch.boundary_uv);
    patch.polygon_signed_area = area;
    patch.polygon_abs_area = abs(area);
    patch.polygon_self_intersect = polygon_has_self_intersections(patch.boundary_uv);
    if (!finite_double(area) || abs(area) < 1e-14) return false;

    if (area < 0.0) {
        reverse(patch.boundary_uv.begin(), patch.boundary_uv.end());
        reverse(patch.ordered_boundary_vertex_ids.begin(), patch.ordered_boundary_vertex_ids.end());
        reverse(patch.boundary_samples.begin(), patch.boundary_samples.end());
        patch.polygon_signed_area = -area;
    }
    patch.valid_polygon = true;
    return true;
}

static void generate_grid_samples(RegionPatch& patch,
                                  const ReconstructionConfig& cfg) {
    patch.interior_uv_samples.clear();
    patch.interior_sample_meta.clear();
    patch.candidate_uv_samples.clear();
    patch.bbox_grid_resolution_x = 0;
    patch.bbox_grid_resolution_y = 0;
    if (!patch.valid_polygon || patch.boundary_uv.empty()) return;

    Vector2d mn = patch.bbox_uv_min;
    Vector2d mx = patch.bbox_uv_max;
    if (patch.sample_spacing <= 1e-12) {
        compute_bbox_sample_spacing(patch, cfg);
        mn = patch.bbox_uv_min;
        mx = patch.bbox_uv_max;
    }
    Vector2d ext = mx - mn;
    double max_dim = max(ext(0), ext(1));
    if (max_dim < 1e-12) return;

    double spacing = patch.sample_spacing > 1e-12
                   ? patch.sample_spacing
                   : max_dim / (double)max(1, cfg.grid_resolution);
    int nx = max(1, (int)ceil(ext(0) / spacing));
    int ny = max(1, (int)ceil(ext(1) / spacing));
    patch.bbox_grid_resolution_x = nx;
    patch.bbox_grid_resolution_y = ny;

    set<pair<long long, long long>> accepted_keys;
    double fast_boundary_band = max(0.0, cfg.fast_triangulation_boundary_band_factor) * spacing;

    auto add_sample = [&](const Vector2d& uv,
                          bool count_candidate,
                          int ix,
                          int iy,
                          bool is_grid_sample) {
        if (count_candidate) patch.candidate_uv_samples.push_back(uv);
        if (!point_in_polygon(uv, patch.boundary_uv)) return;
        double eps = max(1e-10, spacing * 1e-4);
        pair<long long, long long> key(
            (long long)llround(uv(0) / eps),
            (long long)llround(uv(1) / eps));
        if (!accepted_keys.insert(key).second) return;
        if ((int)patch.interior_uv_samples.size() < cfg.max_samples_per_region) {
            double bd = numeric_limits<double>::infinity();
            int boundary_segment = -1;
            double boundary_t = 0.0;
            nearest_polygon_boundary_segment_2d(
                uv, patch.boundary_uv, bd, boundary_segment, boundary_t);
            patch.interior_uv_samples.push_back(uv);
            patch.interior_sample_meta.push_back(
                RegionPatch::InteriorSampleMeta(ix,
                                                iy,
                                                is_grid_sample,
                                                bd <= fast_boundary_band,
                                                bd,
                                                boundary_segment,
                                                boundary_t));
        }
    };

    if (cfg.enable_boundary_ring_sampling && cfg.boundary_ring_layers > 0) {
        int n = (int)patch.boundary_uv.size();
        double ring_step = max(0.1, cfg.boundary_ring_spacing_factor) * spacing;
        for (int layer = 1; layer <= cfg.boundary_ring_layers; layer++) {
            double offset_distance = ring_step * (double)layer;
            for (int i = 0; i < n; i++) {
                const Vector2d& a = patch.boundary_uv[i];
                const Vector2d& b = patch.boundary_uv[(i + 1) % n];
                Vector2d edge = b - a;
                double len = edge.norm();
                if (len < 1e-12) continue;

                Vector2d midpoint = 0.5 * (a + b);
                Vector2d inward(-edge(1) / len, edge(0) / len);
                Vector2d uv = midpoint + offset_distance * inward;
                if (!point_in_polygon(uv, patch.boundary_uv)) {
                    uv = midpoint - offset_distance * inward;
                }
                add_sample(uv, true, -1, -1, false);
            }
        }
    }

    auto add_leaf_sample = [&](const Vector2d& uv,
                               int ix,
                               int iy,
                               bool is_grid_sample,
                               double local_spacing) {
        add_sample(uv, true, ix, iy, is_grid_sample);

        if (cfg.enable_adaptive_boundary_sampling &&
            !cfg.enable_boundary_ring_sampling) {
            double band = cfg.adaptive_boundary_band_factor * local_spacing;
            double bd = numeric_limits<double>::infinity();
            int boundary_segment = -1;
            double boundary_t = 0.0;
            nearest_polygon_boundary_segment_2d(
                uv, patch.boundary_uv, bd, boundary_segment, boundary_t);
            if (band > 0.0 && bd <= band) {
                double offset = 0.25 * local_spacing;
                add_sample(uv + Vector2d( offset,  offset), true, ix, iy, false);
                add_sample(uv + Vector2d(-offset,  offset), true, ix, iy, false);
                add_sample(uv + Vector2d( offset, -offset), true, ix, iy, false);
                add_sample(uv + Vector2d(-offset, -offset), true, ix, iy, false);
            }
        }
    };

    auto lifted_point = [&](const Vector2d& uv, Vector3d& out_pos) {
        if (!point_in_polygon(uv, patch.boundary_uv)) return false;
        return liftToQuadric(patch, uv, out_pos);
    };

    auto lifted_direction_derivative = [&](const Vector2d& c,
                                           const Vector2d& dir,
                                           double h,
                                           Vector3d& out_derivative) {
        out_derivative = Vector3d::Zero();
        if (h <= 1e-12 || dir.squaredNorm() < 1e-20) return false;

        Vector3d pc, pp, pm;
        bool has_center = lifted_point(c, pc);
        bool has_plus = lifted_point(c + h * dir, pp);
        bool has_minus = lifted_point(c - h * dir, pm);

        if (has_plus && has_minus) {
            out_derivative = (pp - pm) / (2.0 * h);
            return finite_vec3(out_derivative);
        }
        if (has_center && has_plus) {
            out_derivative = (pp - pc) / h;
            return finite_vec3(out_derivative);
        }
        if (has_center && has_minus) {
            out_derivative = (pc - pm) / h;
            return finite_vec3(out_derivative);
        }
        return false;
    };

    auto estimate_proxy_surface_metric = [&](const Vector2d& c,
                                             double du,
                                             double dv,
                                             double& out_lambda_max) {
        out_lambda_max = 0.0;
        Vector3d xu, xv;
        bool has_u = lifted_direction_derivative(c, Vector2d::UnitX(), du, xu);
        bool has_v = lifted_direction_derivative(c, Vector2d::UnitY(), dv, xv);
        if (!has_u && !has_v) return false;

        if (!has_u) xu = patch.tangent_u;
        if (!has_v) xv = patch.tangent_v;
        if (!finite_vec3(xu) || !finite_vec3(xv)) return false;

        double e = xu.dot(xu);
        double f = xu.dot(xv);
        double g = xv.dot(xv);
        if (!finite_double(e) || !finite_double(f) || !finite_double(g)) return false;
        if (e < 1e-20 && g < 1e-20) return false;

        double trace = e + g;
        double det_term = sqrt(max(0.0, (e - g) * (e - g) + 4.0 * f * f));
        out_lambda_max = 0.5 * (trace + det_term);
        return finite_double(out_lambda_max) && out_lambda_max > 1e-20;
    };

    auto cell_needs_surface_refinement = [&](const Vector2d& cell_min,
                                             const Vector2d& cell_max,
                                             int depth) {
        if (!cfg.enable_surface_metric_sampling ||
            depth >= cfg.surface_metric_refinement_passes) {
            return false;
        }

        Vector2d c = 0.5 * (cell_min + cell_max);
        if (!point_in_polygon(c, patch.boundary_uv)) return false;

        double target = cfg.surface_sample_spacing_factor * spacing;
        if (target <= 1e-12 || !finite_double(target)) return false;
        double threshold = max(1.0, cfg.surface_metric_refine_factor) * target;

        Vector2d cell_ext = cell_max - cell_min;
        double du = max(1e-8, 0.25 * max(cell_ext(0), spacing / pow(2.0, depth + 1)));
        double dv = max(1e-8, 0.25 * max(cell_ext(1), spacing / pow(2.0, depth + 1)));
        double lambda_max = 0.0;
        if (!estimate_proxy_surface_metric(c, du, dv, lambda_max)) {
            return false;
        }

        double uv_edge = max(cell_ext(0), cell_ext(1));
        double estimated_surface_edge = sqrt(lambda_max) * uv_edge;
        return finite_double(estimated_surface_edge) &&
               estimated_surface_edge > threshold;
    };

    struct SampleCell {
        Vector2d mn;
        Vector2d mx;
        int ix;
        int iy;
        int depth;
    };

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            Vector2d cell_min(mn(0) + (double)ix * ext(0) / (double)nx,
                              mn(1) + (double)iy * ext(1) / (double)ny);
            Vector2d cell_max(mn(0) + (double)(ix + 1) * ext(0) / (double)nx,
                              mn(1) + (double)(iy + 1) * ext(1) / (double)ny);

            vector<SampleCell> stack;
            stack.push_back({cell_min, cell_max, ix, iy, 0});
            while (!stack.empty()) {
                SampleCell cell = stack.back();
                stack.pop_back();

                if ((int)patch.interior_uv_samples.size() >= cfg.max_samples_per_region) {
                    return;
                }

                if (cell_needs_surface_refinement(cell.mn, cell.mx, cell.depth)) {
                    Vector2d mid = 0.5 * (cell.mn + cell.mx);
                    int next_depth = cell.depth + 1;
                    stack.push_back({Vector2d(cell.mn(0), cell.mn(1)),
                                     Vector2d(mid(0), mid(1)),
                                     cell.ix, cell.iy, next_depth});
                    stack.push_back({Vector2d(mid(0), cell.mn(1)),
                                     Vector2d(cell.mx(0), mid(1)),
                                     cell.ix, cell.iy, next_depth});
                    stack.push_back({Vector2d(cell.mn(0), mid(1)),
                                     Vector2d(mid(0), cell.mx(1)),
                                     cell.ix, cell.iy, next_depth});
                    stack.push_back({Vector2d(mid(0), mid(1)),
                                     Vector2d(cell.mx(0), cell.mx(1)),
                                     cell.ix, cell.iy, next_depth});
                    continue;
                }

                Vector2d uv = 0.5 * (cell.mn + cell.mx);
                double local_spacing = max((cell.mx - cell.mn).norm() / sqrt(2.0),
                                           spacing / pow(2.0, cell.depth));
                add_leaf_sample(uv, cell.ix, cell.iy, cell.depth == 0, local_spacing);
            }
        }
    }
}

static bool select_fan_sample(const RegionPatch& patch, Vector2d& out_uv) {
    Vector2d origin = Vector2d::Zero();
    if (point_in_polygon(origin, patch.boundary_uv)) {
        out_uv = origin;
        return true;
    }

    if (!patch.interior_uv_samples.empty()) {
        double best = numeric_limits<double>::infinity();
        Vector2d best_uv = patch.interior_uv_samples[0];
        for (const Vector2d& uv : patch.interior_uv_samples) {
            double d = uv.squaredNorm();
            if (d < best) {
                best = d;
                best_uv = uv;
            }
        }
        out_uv = best_uv;
        return true;
    }

    Vector2d centroid = Vector2d::Zero();
    for (const Vector2d& uv : patch.boundary_uv) centroid += uv;
    centroid /= (double)patch.boundary_uv.size();
    if (point_in_polygon(centroid, patch.boundary_uv)) {
        out_uv = centroid;
        return true;
    }

    return false;
}

bool NullDisplacementQuery::query(
    int region_id,
    const Vector3d& q,
    const Vector3d& normal,
    Vector3d& out_reference_point) {
    (void)region_id;
    (void)q;
    (void)normal;
    (void)out_reference_point;
    return false;
}

class OriginalMeshAABBDisplacementQuery : public DisplacementQuery {
public:
    OriginalMeshAABBDisplacementQuery(const MatrixXd& V, const MatrixXi& F)
        : V_(V), F_(F) {
        tree_.init(V_, F_);
    }

    bool query(
        int region_id,
        const Vector3d& q,
        const Vector3d& normal,
        Vector3d& out_reference_point) override {
        (void)region_id;
        (void)normal;

        if (!finite_vec3(q)) return false;

        int fid = -1;
        RowVector3d closest = RowVector3d::Zero();
        RowVector3d rq = q.transpose();
        double sqr_d = tree_.squared_distance(V_, F_, rq, fid, closest);
        if (fid < 0 || !finite_double(sqr_d)) return false;

        out_reference_point = closest.transpose();
        return finite_vec3(out_reference_point);
    }

private:
    const MatrixXd& V_;
    const MatrixXi& F_;
    igl::AABB<MatrixXd, 3> tree_;
};

vector<RegionPatch> extractRegionPatches(
    const MatrixXd& V,
    const MatrixXi& F,
    const MatrixXi& R,
    const vector<QuadricProxy>& QP,
    int num_regions,
    const ReconstructionConfig& cfg) {

    vector<RegionPatch> patches(num_regions);
    vector<set<int>> region_vertices(num_regions);
    vector<set<int>> boundary_vertices(num_regions);
    vector<set<EdgeKey>> seen_boundary_edges(num_regions);

    for (int rid = 0; rid < num_regions; rid++) {
        patches[rid].region_id = rid;
        if (rid < (int)QP.size()) patches[rid].quadric = QP[rid];
    }

    for (int fi = 0; fi < F.rows(); fi++) {
        int rid = R(fi, 0);
        if (rid < 0 || rid >= num_regions) continue;

        patches[rid].face_ids.push_back(fi);
        for (int k = 0; k < 3; k++) {
            region_vertices[rid].insert(F(fi, k));
        }
    }

    map<EdgeKey, vector<int>> edge_to_faces;
    for (int fi = 0; fi < F.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            EdgeKey ek(F(fi, k), F(fi, (k + 1) % 3));
            edge_to_faces[ek].push_back(fi);
        }
    }

    int edge_id = 0;
    for (const auto& kv : edge_to_faces) {
        const EdgeKey& ek = kv.first;
        const vector<int>& faces = kv.second;

        set<int> boundary_regions;
        if (faces.size() == 1) {
            int rid = R(faces[0], 0);
            if (rid >= 0 && rid < num_regions) boundary_regions.insert(rid);
        } else if (faces.size() == 2) {
            int r0 = R(faces[0], 0);
            int r1 = R(faces[1], 0);
            if (r0 != r1) {
                if (r0 >= 0 && r0 < num_regions) boundary_regions.insert(r0);
                if (r1 >= 0 && r1 < num_regions) boundary_regions.insert(r1);
            }
        } else {
            for (int fi : faces) {
                int rid = R(fi, 0);
                if (rid >= 0 && rid < num_regions) boundary_regions.insert(rid);
            }
        }

        for (int rid : boundary_regions) {
            if (seen_boundary_edges[rid].insert(ek).second) {
                patches[rid].boundary_edge_ids.push_back(edge_id);
                patches[rid].boundary_edges.push_back(ek);
                boundary_vertices[rid].insert(ek.v0);
                boundary_vertices[rid].insert(ek.v1);
            }
        }
        edge_id++;
    }

    map<EdgeKey, int> global_edge_segments;

    for (int rid = 0; rid < num_regions; rid++) {
        patches[rid].boundary_vertex_ids.assign(
            boundary_vertices[rid].begin(), boundary_vertices[rid].end());

        for (int vid : region_vertices[rid]) {
            if (!boundary_vertices[rid].count(vid)) {
                patches[rid].interior_vertex_ids.push_back(vid);
            }
        }

        if (!patches[rid].face_ids.empty()) {
            patches[rid].boundary_loop_count = count_boundary_loops(patches[rid].boundary_edges);
            auto chart_start = Clock::now();
            build_chart(patches[rid], V, F, cfg);
            auto chart_end = Clock::now();
            patches[rid].elapsed_chart_ms = elapsed_ms(chart_start, chart_end);
            if (patches[rid].valid_chart && order_boundary_loop(patches[rid])) {
                map_boundary_uv(patches[rid], V);
                compute_bbox_sample_spacing(patches[rid], cfg);
            }
        }
    }

    double global_spacing = compute_global_sample_spacing(patches, cfg);
    if (global_spacing > 1e-12) {
        for (RegionPatch& patch : patches) {
            if (patch.valid_chart && patch.simple_boundary_loop) {
                patch.sample_spacing = global_spacing;
            }
        }
        if (cfg.verbose) {
            cout << "[RegionRecon] global_sample_spacing=" << global_spacing
                 << " grid_reference=" << cfg.grid_resolution << endl;
        }
    }

    for (int rid = 0; rid < num_regions; rid++) {
        if (!patches[rid].valid_chart ||
            !patches[rid].simple_boundary_loop ||
            patches[rid].sample_spacing <= 1e-12) {
            continue;
        }

        int n = (int)patches[rid].ordered_boundary_vertex_ids.size();
        for (int i = 0; i < n; i++) {
            int a = patches[rid].ordered_boundary_vertex_ids[i];
            int b = patches[rid].ordered_boundary_vertex_ids[(i + 1) % n];
            EdgeKey e(a, b);
            int segs = edge_segment_count(e, patches[rid].sample_spacing, V, cfg);
            auto it = global_edge_segments.find(e);
            if (it == global_edge_segments.end()) {
                global_edge_segments[e] = segs;
            } else {
                it->second = max(it->second, segs);
            }
        }
    }

    for (int rid = 0; rid < num_regions; rid++) {
        if (patches[rid].face_ids.empty() ||
            !patches[rid].valid_chart ||
            !patches[rid].simple_boundary_loop) {
            continue;
        }

        build_subdivided_boundary_samples(patches[rid], V, global_edge_segments);
        if (validate_polygon(patches[rid])) {
            if (patches[rid].sample_spacing <= 1e-12) {
                compute_bbox_sample_spacing(patches[rid], cfg);
            }
            auto sampling_start = Clock::now();
            generate_grid_samples(patches[rid], cfg);
            auto sampling_end = Clock::now();
            patches[rid].elapsed_sampling_ms = elapsed_ms(sampling_start, sampling_end);
        }
    }

    return patches;
}

bool liftToQuadric(
    const RegionPatch& patch,
    const Vector2d& uv,
    Vector3d& out_pos) {

    if (!patch.valid_chart || !finite_vec2(uv)) return false;

    Vector3d y = patch.center + uv(0) * patch.tangent_u + uv(1) * patch.tangent_v;
    Vector3d n = patch.normal;
    if (n.norm() < 1e-12 || !finite_vec3(n)) return false;
    n.normalize();

    Matrix3d A = patch.quadric.quadraticMatrix();
    Vector3d b(patch.quadric.coeffs(1), patch.quadric.coeffs(2), patch.quadric.coeffs(3));

    double qa = n.dot(A * n);
    double qb = 2.0 * y.dot(A * n) + b.dot(n);
    double qc = patch.quadric.eval(y);

    vector<double> roots;
    const double eps = 1e-12;
    if (abs(qa) < eps) {
        if (abs(qb) < eps) {
            if (abs(qc) < 1e-10) {
                roots.push_back(0.0);
            } else {
                return false;
            }
        } else {
            roots.push_back(-qc / qb);
        }
    } else {
        double disc = qb * qb - 4.0 * qa * qc;
        if (disc < -1e-12 || !finite_double(disc)) return false;
        disc = max(0.0, disc);
        double root_disc = sqrt(disc);
        roots.push_back((-qb - root_disc) / (2.0 * qa));
        roots.push_back((-qb + root_disc) / (2.0 * qa));
    }

    bool found = false;
    double best_s = 0.0;
    double best_score = numeric_limits<double>::infinity();
    double min_alignment = max(0.0, patch.min_lift_normal_alignment);
    for (double s : roots) {
        if (!finite_double(s)) continue;
        if (abs(s) > patch.max_lift_distance) continue;

        Vector3d candidate = y + s * n;
        if (!finite_vec3(candidate)) continue;
        if (!inside_expanded_region_bbox(patch, candidate)) continue;

        Vector3d g_candidate = patch.quadric.grad(candidate);
        if (g_candidate.norm() < 1e-12 || !finite_vec3(g_candidate)) continue;
        g_candidate.normalize();

        // The sign of an implicit quadric gradient is arbitrary because the
        // fitted coefficients may be multiplied by -1. Use absolute alignment
        // so a correct sheet is not rejected only due to normal orientation.
        double alignment = abs(g_candidate.dot(n));
        if (!finite_double(alignment) ||
            alignment < min_alignment) {
            continue;
        }

        double abs_s = abs(s);
        double normalized_s = abs_s / max(1e-12, patch.max_lift_distance);
        double score = normalized_s + 0.05 * (1.0 - min(1.0, alignment));
        if (score < best_score) {
            best_score = score;
            best_s = s;
            found = true;
        }
    }

    if (!found) return false;

    double safe_lift_distance = patch.max_lift_distance;
    if (patch.sample_spacing > 1e-12 || patch.normal_extent > 1e-12) {
        double local_limit = 0.0;
        if (patch.sample_spacing > 1e-12) {
            local_limit = max(local_limit, 0.75 * patch.sample_spacing);
        }
        if (patch.normal_extent > 1e-12) {
            local_limit = max(local_limit, 3.0 * patch.normal_extent);
        }
        if (local_limit > 1e-12) {
            safe_lift_distance = min(safe_lift_distance, local_limit);
        }
    }

    if (abs(best_s) > safe_lift_distance) return false;

    out_pos = y + best_s * n;
    return finite_vec3(out_pos);
}

static bool lifted_direction_derivative_for_metric(const RegionPatch& patch,
                                                   const Vector2d& uv,
                                                   const Vector2d& dir,
                                                   double h,
                                                   Vector3d& out_derivative) {
    out_derivative = Vector3d::Zero();
    if (h <= 1e-12 || dir.squaredNorm() < 1e-20) return false;

    Vector3d pc, pp, pm;
    bool has_center = liftToQuadric(patch, uv, pc);
    bool has_plus = point_in_polygon(uv + h * dir, patch.boundary_uv) &&
                    liftToQuadric(patch, uv + h * dir, pp);
    bool has_minus = point_in_polygon(uv - h * dir, patch.boundary_uv) &&
                     liftToQuadric(patch, uv - h * dir, pm);

    if (has_plus && has_minus) {
        out_derivative = (pp - pm) / (2.0 * h);
        return finite_vec3(out_derivative);
    }
    if (has_center && has_plus) {
        out_derivative = (pp - pc) / h;
        return finite_vec3(out_derivative);
    }
    if (has_center && has_minus) {
        out_derivative = (pc - pm) / h;
        return finite_vec3(out_derivative);
    }
    return false;
}

static bool estimate_proxy_metric_at_uv(const RegionPatch& patch,
                                        const Vector2d& uv,
                                        double h,
                                        Matrix2d& out_metric) {
    out_metric.setZero();
    if (!point_in_polygon(uv, patch.boundary_uv)) return false;

    Vector3d xu, xv;
    bool has_u = lifted_direction_derivative_for_metric(
        patch, uv, Vector2d::UnitX(), h, xu);
    bool has_v = lifted_direction_derivative_for_metric(
        patch, uv, Vector2d::UnitY(), h, xv);
    if (!has_u && !has_v) return false;

    if (!has_u) xu = patch.tangent_u;
    if (!has_v) xv = patch.tangent_v;
    if (!finite_vec3(xu) || !finite_vec3(xv)) return false;

    out_metric(0, 0) = xu.dot(xu);
    out_metric(0, 1) = xu.dot(xv);
    out_metric(1, 0) = out_metric(0, 1);
    out_metric(1, 1) = xv.dot(xv);
    return finite_double(out_metric(0, 0)) &&
           finite_double(out_metric(0, 1)) &&
           finite_double(out_metric(1, 1)) &&
           out_metric.trace() > 1e-20;
}

static bool compute_metric_warp_transform(const RegionPatch& patch,
                                          const ReconstructionConfig& cfg,
                                          const vector<Vector2d>& sample_uvs,
                                          Matrix2d& out_transform,
                                          Vector2d& out_origin) {
    out_transform.setIdentity();
    out_origin = 0.5 * (patch.bbox_uv_min + patch.bbox_uv_max);
    if (!cfg.enable_metric_warped_triangulation ||
        !patch.valid_chart ||
        !patch.valid_polygon ||
        sample_uvs.empty()) {
        return false;
    }

    Matrix2d metric_sum = Matrix2d::Zero();
    int count = 0;
    int max_samples = min((int)sample_uvs.size(), 96);
    int stride = max(1, (int)sample_uvs.size() / max(1, max_samples));
    double h = max(1e-6, 0.15 * max(1e-6, patch.sample_spacing));

    for (int i = 0; i < (int)sample_uvs.size(); i += stride) {
        Matrix2d g;
        if (!estimate_proxy_metric_at_uv(patch, sample_uvs[i], h, g)) continue;
        metric_sum += g;
        count++;
    }

    if (count < 2) return false;
    Matrix2d avg = metric_sum / (double)count;
    avg = 0.5 * (avg + avg.transpose());

    SelfAdjointEigenSolver<Matrix2d> solver(avg);
    if (solver.info() != Success) return false;

    Vector2d eval = solver.eigenvalues();
    Matrix2d evec = solver.eigenvectors();
    double l0 = max(eval(0), 1e-10);
    double l1 = max(eval(1), 1e-10);
    if (!finite_double(l0) || !finite_double(l1)) return false;

    double clamp_ratio = max(1.0, cfg.metric_warp_anisotropy_clamp);
    if (l1 / l0 > clamp_ratio) {
        l0 = l1 / clamp_ratio;
    }

    Matrix2d sqrt_diag = Matrix2d::Zero();
    sqrt_diag(0, 0) = sqrt(l0);
    sqrt_diag(1, 1) = sqrt(l1);
    out_transform = evec * sqrt_diag * evec.transpose();

    double scale = sqrt(max(1e-20, out_transform.determinant()));
    if (finite_double(scale) && scale > 1e-12) {
        out_transform /= scale;
    }
    return finite_double(out_transform(0, 0)) &&
           finite_double(out_transform(0, 1)) &&
           finite_double(out_transform(1, 0)) &&
           finite_double(out_transform(1, 1));
}

static void build_region_ray_projectors(const MatrixXd& V,
                                        const MatrixXi& F,
                                        const vector<RegionPatch>& patches,
                                        vector<RegionRayProjector>& projectors) {
    projectors.clear();
    projectors.resize(patches.size());

    for (int rid = 0; rid < (int)patches.size(); rid++) {
        const RegionPatch& patch = patches[rid];
        RegionRayProjector& rp = projectors[rid];
        if (patch.face_ids.empty()) continue;

        rp.localF.resize((int)patch.face_ids.size(), 3);
        rp.local_face_to_global_face.clear();
        rp.local_face_to_global_face.reserve(patch.face_ids.size());
        for (int i = 0; i < (int)patch.face_ids.size(); i++) {
            int fi = patch.face_ids[i];
            rp.localF.row(i) = F.row(fi);
            rp.local_face_to_global_face.push_back(fi);
        }

        if (rp.localF.rows() > 0) {
            rp.tree.init(V, rp.localF);
            rp.valid = true;
        }
    }
}

static bool ray_project_one_direction(const MatrixXd& V,
                                      const RegionRayProjector& projector,
                                      const Vector3d& origin,
                                      const Vector3d& dir,
                                      double max_distance,
                                      Vector3d& out_pos,
                                      double& out_abs_t) {
    if (!projector.valid || projector.localF.rows() <= 0) return false;
    if (!finite_vec3(origin) || !finite_vec3(dir) || dir.norm() < 1e-12) return false;

    RowVector3d o = origin.transpose();
    RowVector3d d = dir.normalized().transpose();
    igl::Hit<double> hit;
    if (!projector.tree.intersect_ray(V, projector.localF, o, d, hit)) return false;
    if (!finite_double(hit.t) || hit.t < -1e-10) return false;

    double abs_t = abs(hit.t);
    if (finite_double(max_distance) && max_distance > 0.0 && abs_t > max_distance) {
        return false;
    }

    Vector3d p = origin + hit.t * dir.normalized();
    if (!finite_vec3(p)) return false;
    out_pos = p;
    out_abs_t = abs_t;
    return true;
}

static bool projectProxyPointToOriginalMeshRay(const RegionPatch& patch,
                                               const RegionRayProjector& projector,
                                               const MatrixXd& V,
                                               const ReconstructionConfig& cfg,
                                               const Vector3d& proxy_pos,
                                               Vector3d& out_pos) {
    if (!patch.valid_chart || !finite_vec3(proxy_pos)) return false;
    Vector3d n = patch.quadric.grad(proxy_pos);
    if (!finite_vec3(n) || n.norm() < 1e-12) {
        n = patch.normal;
    }
    if (!finite_vec3(n) || n.norm() < 1e-12) return false;
    n.normalize();

    Vector3d origin = proxy_pos;
    if (!finite_vec3(origin)) return false;

    double local_scale = max(patch.max_lift_distance,
                             max(patch.sample_spacing, patch.normal_extent));
    if (local_scale <= 1e-12 || !finite_double(local_scale)) {
        local_scale = max(1e-8, patch.bbox_3d_diag);
    }
    double max_distance = max(1e-8, cfg.max_ray_projection_distance_factor * local_scale);

    Vector3d best = Vector3d::Zero();
    double best_t = numeric_limits<double>::infinity();
    Vector3d candidate;
    double t = 0.0;
    if (ray_project_one_direction(V, projector, origin, n, max_distance, candidate, t)) {
        best = candidate;
        best_t = t;
    }
    if (ray_project_one_direction(V, projector, origin, -n, max_distance, candidate, t) &&
        t < best_t) {
        best = candidate;
        best_t = t;
    }

    if (!finite_double(best_t)) return false;
    out_pos = best;
    return true;
}

static bool projectUvToOriginalMeshRay(const RegionPatch& patch,
                                       const RegionRayProjector& projector,
                                       const MatrixXd& V,
                                       const ReconstructionConfig& cfg,
                                       const Vector2d& uv,
                                       Vector3d& out_pos) {
    if (!patch.valid_chart || !finite_vec2(uv)) return false;

    Vector3d proxy_pos;
    if (!liftToQuadric(patch, uv, proxy_pos)) {
        return false;
    }
    return projectProxyPointToOriginalMeshRay(
        patch, projector, V, cfg, proxy_pos, out_pos);
}

static int append_vertex(vector<Vector3d>& vertices, const Vector3d& p) {
    int id = (int)vertices.size();
    vertices.push_back(p);
    return id;
}

static int get_boundary_vertex(int original_vid,
                               const MatrixXd& V,
                               vector<Vector3d>& vertices,
                               map<int, int>& boundary_vertex_map) {
    auto it = boundary_vertex_map.find(original_vid);
    if (it != boundary_vertex_map.end()) return it->second;

    int rid = append_vertex(vertices, vertex3(V, original_vid));
    boundary_vertex_map[original_vid] = rid;
    return rid;
}

using BoundarySampleVertexKey = tuple<int, int, int, int>;

static int get_boundary_sample_vertex(
    const RegionPatch::BoundarySample& sample,
    vector<Vector3d>& vertices,
    map<BoundarySampleVertexKey, int>& boundary_sample_vertex_map) {
    if (sample.segment_index <= 0 ||
        sample.segment_index >= sample.segment_count) {
        return -1;
    }

    BoundarySampleVertexKey key(
        sample.edge.v0,
        sample.edge.v1,
        sample.segment_index,
        sample.segment_count);
    auto it = boundary_sample_vertex_map.find(key);
    if (it != boundary_sample_vertex_map.end()) return it->second;

    int rid = append_vertex(vertices, sample.position);
    boundary_sample_vertex_map[key] = rid;
    return rid;
}

static int get_fallback_original_vertex(int original_vid,
                                        const MatrixXd& V,
                                        vector<Vector3d>& vertices,
                                        map<int, int>& fallback_vertex_map) {
    auto it = fallback_vertex_map.find(original_vid);
    if (it != fallback_vertex_map.end()) return it->second;

    int rid = append_vertex(vertices, vertex3(V, original_vid));
    fallback_vertex_map[original_vid] = rid;
    return rid;
}

static int append_original_region_mesh(const RegionPatch& patch,
                                       const MatrixXd& V,
                                       const MatrixXi& F,
                                       vector<Vector3d>& vertices,
                                       vector<Vector3i>& faces,
                                       vector<int>& labels,
                                       map<int, int>& boundary_vertex_map,
                                       map<int, int>& fallback_vertex_map) {
    set<int> boundary_set(patch.boundary_vertex_ids.begin(), patch.boundary_vertex_ids.end());
    int before = (int)faces.size();

    for (int fi : patch.face_ids) {
        int ids[3];
        for (int k = 0; k < 3; k++) {
            int vid = F(fi, k);
            if (boundary_set.count(vid)) {
                ids[k] = get_boundary_vertex(vid, V, vertices, boundary_vertex_map);
            } else {
                ids[k] = get_fallback_original_vertex(vid, V, vertices, fallback_vertex_map);
            }
        }

        if (ids[0] == ids[1] || ids[1] == ids[2] || ids[2] == ids[0]) continue;
        faces.push_back(Vector3i(ids[0], ids[1], ids[2]));
        labels.push_back(patch.region_id);
    }

    return (int)faces.size() - before;
}

static bool apply_optional_displacement(const RegionPatch& patch,
                                        const ReconstructionConfig& cfg,
                                        DisplacementQuery* displacement_query,
                                        Vector3d& q,
                                        double& out_abs_displacement,
                                        bool& out_query_failed) {
    out_abs_displacement = 0.0;
    out_query_failed = false;

    if (!cfg.enable_displacement || displacement_query == nullptr) return true;

    Vector3d n = patch.quadric.grad(q);
    if (n.norm() < 1e-12 || !finite_vec3(n)) n = patch.normal;
    if (n.norm() < 1e-12 || !finite_vec3(n)) {
        out_query_failed = true;
        return true;
    }
    n.normalize();
    if (n.dot(patch.normal) < 0.0) n = -n;

    Vector3d ref = q;
    if (!displacement_query->query(patch.region_id, q, n, ref)) {
        out_query_failed = true;
        return true;
    }

    double d = (ref - q).dot(n);
    double max_d = max(1e-8, cfg.max_displacement_factor * patch.max_lift_distance);
    if (!finite_double(d) || abs(d) > max_d) {
        out_query_failed = true;
        return true;
    }

    q += d * n;
    out_abs_displacement = abs(d);
    return true;
}

static bool boundary_ribbon_point(const RegionPatch& patch,
                                  const Vector2d& uv,
                                  double& out_distance,
                                  Vector3d& out_position) {
    out_distance = numeric_limits<double>::infinity();
    out_position = Vector3d::Zero();

    const vector<RegionPatch::BoundarySample>& samples = patch.boundary_samples;
    if (samples.size() < 2) return false;

    int n = (int)samples.size();
    for (int i = 0; i < n; i++) {
        const auto& a = samples[i];
        const auto& b = samples[(i + 1) % n];
        double t = 0.0;
        Vector2d closest_uv = closest_point_on_segment_2d(uv, a.uv, b.uv, t);
        double d = (uv - closest_uv).norm();
        if (d >= out_distance) continue;

        Vector3d closest_boundary_pos = (1.0 - t) * a.position + t * b.position;
        Vector2d offset_uv = uv - closest_uv;
        Vector3d ribbon_pos = closest_boundary_pos
            + offset_uv(0) * patch.tangent_u
            + offset_uv(1) * patch.tangent_v;

        out_distance = d;
        out_position = ribbon_pos;
    }

    return finite_double(out_distance) && finite_vec3(out_position);
}

static bool boundary_ribbon_point_from_meta(const RegionPatch& patch,
                                            const Vector2d& uv,
                                            const RegionPatch::InteriorSampleMeta& meta,
                                            double& out_distance,
                                            Vector3d& out_position) {
    out_distance = meta.boundary_distance;
    out_position = Vector3d::Zero();

    const vector<RegionPatch::BoundarySample>& samples = patch.boundary_samples;
    if (samples.size() < 2 ||
        meta.nearest_boundary_segment < 0 ||
        meta.nearest_boundary_segment >= (int)samples.size() ||
        !finite_double(meta.boundary_distance)) {
        return false;
    }

    int i = meta.nearest_boundary_segment;
    const auto& a = samples[i];
    const auto& b = samples[(i + 1) % (int)samples.size()];
    double t = max(0.0, min(1.0, meta.nearest_boundary_t));
    Vector2d closest_uv = (1.0 - t) * a.uv + t * b.uv;
    Vector3d closest_boundary_pos = (1.0 - t) * a.position + t * b.position;
    Vector2d offset_uv = uv - closest_uv;

    out_position = closest_boundary_pos
        + offset_uv(0) * patch.tangent_u
        + offset_uv(1) * patch.tangent_v;

    return finite_vec3(out_position);
}

static void apply_boundary_transition(const RegionPatch& patch,
                                      const ReconstructionConfig& cfg,
                                      int sample_idx,
                                      const Vector2d& uv,
                                      Vector3d& q) {
    if (!cfg.enable_boundary_transition || patch.sample_spacing <= 1e-12) return;

    double band_width = cfg.boundary_transition_width_factor * patch.sample_spacing;
    if (band_width <= 1e-12) return;

    double dist = 0.0;
    Vector3d ribbon_pos;
    bool has_ribbon = false;
    if (sample_idx >= 0 && sample_idx < (int)patch.interior_sample_meta.size()) {
        has_ribbon = boundary_ribbon_point_from_meta(
            patch, uv, patch.interior_sample_meta[sample_idx], dist, ribbon_pos);
    }
    if (!has_ribbon && !boundary_ribbon_point(patch, uv, dist, ribbon_pos)) return;
    if (dist >= band_width) return;

    double w = smoothstep01(dist / band_width);
    q = (1.0 - w) * ribbon_pos + w * q;
}

static int append_fan_region_mesh(const RegionPatch& patch,
                                  const MatrixXd& V,
                                  vector<Vector3d>& vertices,
                                  vector<Vector3i>& faces,
                                  vector<int>& labels,
                                  map<int, int>& boundary_vertex_map,
                                  const Vector3d& interior_pos) {
    if (patch.ordered_boundary_vertex_ids.size() < 3) return 0;

    vector<int> boundary_out;
    boundary_out.reserve(patch.ordered_boundary_vertex_ids.size());
    for (int vid : patch.ordered_boundary_vertex_ids) {
        boundary_out.push_back(get_boundary_vertex(vid, V, vertices, boundary_vertex_map));
    }

    int center_id = append_vertex(vertices, interior_pos);
    int before = (int)faces.size();
    int n = (int)boundary_out.size();
    for (int i = 0; i < n; i++) {
        int a = boundary_out[i];
        int b = boundary_out[(i + 1) % n];
        if (a == b || b == center_id || center_id == a) continue;

        Vector3d pa = vertices[a];
        Vector3d pb = vertices[b];
        if (triangle_area3(pa, pb, interior_pos) < 1e-15) continue;

        faces.push_back(Vector3i(a, b, center_id));
        labels.push_back(patch.region_id);
    }

    return (int)faces.size() - before;
}

struct Tri2D {
    int a;
    int b;
    int c;
};

static double tri_area2(const Vector2d& a, const Vector2d& b, const Vector2d& c) {
    return 0.5 * cross2(b - a, c - a);
}

static bool circumcircle_contains(const vector<Vector2d>& pts,
                                  const Tri2D& tri,
                                  const Vector2d& p) {
    const Vector2d& a = pts[tri.a];
    const Vector2d& b = pts[tri.b];
    const Vector2d& c = pts[tri.c];

    double d = 2.0 * (a(0) * (b(1) - c(1)) +
                      b(0) * (c(1) - a(1)) +
                      c(0) * (a(1) - b(1)));
    if (abs(d) < 1e-14) return false;

    double aa = a.squaredNorm();
    double bb = b.squaredNorm();
    double cc = c.squaredNorm();

    Vector2d center;
    center(0) = (aa * (b(1) - c(1)) +
                 bb * (c(1) - a(1)) +
                 cc * (a(1) - b(1))) / d;
    center(1) = (aa * (c(0) - b(0)) +
                 bb * (a(0) - c(0)) +
                 cc * (b(0) - a(0))) / d;

    double r2 = (center - a).squaredNorm();
    double q2 = (center - p).squaredNorm();
    return q2 <= r2 * (1.0 + 1e-10) + 1e-12;
}

static void orient_ccw(const vector<Vector2d>& pts, Tri2D& tri) {
    if (tri_area2(pts[tri.a], pts[tri.b], pts[tri.c]) < 0.0) {
        swap(tri.b, tri.c);
    }
}

static bool bowyer_watson_triangulation(const vector<Vector2d>& input_pts,
                                        vector<Tri2D>& out_triangles) {
    out_triangles.clear();
    int n = (int)input_pts.size();
    if (n < 3) return false;

    Vector2d mn = input_pts[0];
    Vector2d mx = input_pts[0];
    for (const Vector2d& p : input_pts) {
        mn = mn.cwiseMin(p);
        mx = mx.cwiseMax(p);
    }
    Vector2d center = 0.5 * (mn + mx);
    double span = max((mx - mn).maxCoeff(), 1e-6);
    double r = 64.0 * span;

    vector<Vector2d> pts = input_pts;
    pts.push_back(center + Vector2d(-2.0 * r, -r));
    pts.push_back(center + Vector2d( 2.0 * r, -r));
    pts.push_back(center + Vector2d( 0.0,  2.0 * r));

    int s0 = n;
    int s1 = n + 1;
    int s2 = n + 2;

    vector<Tri2D> triangles;
    triangles.push_back({s0, s1, s2});
    orient_ccw(pts, triangles.back());

    for (int pi = 0; pi < n; pi++) {
        vector<Tri2D> kept;
        map<pair<int, int>, int> edge_count;

        for (const Tri2D& tri : triangles) {
            if (circumcircle_contains(pts, tri, pts[pi])) {
                int ids[3] = {tri.a, tri.b, tri.c};
                for (int e = 0; e < 3; e++) {
                    int u = ids[e];
                    int v = ids[(e + 1) % 3];
                    if (u > v) swap(u, v);
                    edge_count[make_pair(u, v)]++;
                }
            } else {
                kept.push_back(tri);
            }
        }

        triangles.swap(kept);
        for (const auto& kv : edge_count) {
            if (kv.second != 1) continue;
            Tri2D t{kv.first.first, kv.first.second, pi};
            if (abs(tri_area2(pts[t.a], pts[t.b], pts[t.c])) < 1e-14) continue;
            orient_ccw(pts, t);
            triangles.push_back(t);
        }
    }

    set<tuple<int, int, int>> seen;
    for (Tri2D tri : triangles) {
        if (tri.a >= n || tri.b >= n || tri.c >= n) continue;
        if (abs(tri_area2(input_pts[tri.a], input_pts[tri.b], input_pts[tri.c])) < 1e-14) continue;
        orient_ccw(input_pts, tri);

        vector<int> ids = {tri.a, tri.b, tri.c};
        sort(ids.begin(), ids.end());
        auto key = make_tuple(ids[0], ids[1], ids[2]);
        if (seen.count(key)) continue;
        seen.insert(key);
        out_triangles.push_back(tri);
    }

    return !out_triangles.empty();
}

static bool triangle_inside_polygon_2d(const vector<Vector2d>& poly,
                                       const Vector2d& a,
                                       const Vector2d& b,
                                       const Vector2d& c) {
    Vector2d centroid = (a + b + c) / 3.0;
    Vector2d ab = 0.5 * (a + b);
    Vector2d bc = 0.5 * (b + c);
    Vector2d ca = 0.5 * (c + a);
    return point_in_polygon(centroid, poly) &&
           point_in_polygon(ab, poly) &&
           point_in_polygon(bc, poly) &&
           point_in_polygon(ca, poly);
}

static bool points_close_2d(const Vector2d& a,
                            const Vector2d& b,
                            double eps = 1e-9) {
    return (a - b).squaredNorm() <= eps * eps;
}

static bool edge_crosses_polygon_boundary_2d(const Vector2d& a,
                                             const Vector2d& b,
                                             const vector<Vector2d>& poly) {
    int n = (int)poly.size();
    if (n < 3) return true;

    for (int i = 0; i < n; i++) {
        const Vector2d& c = poly[i];
        const Vector2d& d = poly[(i + 1) % n];
        if (!segments_intersect_2d(a, b, c, d)) continue;

        int o1 = orient2d_sign(a, b, c);
        int o2 = orient2d_sign(a, b, d);
        int o3 = orient2d_sign(c, d, a);
        int o4 = orient2d_sign(c, d, b);
        bool proper_cross = (o1 * o2 < 0) && (o3 * o4 < 0);
        if (proper_cross) return true;

        bool shared_endpoint =
            points_close_2d(a, c) || points_close_2d(a, d) ||
            points_close_2d(b, c) || points_close_2d(b, d);
        if (shared_endpoint) continue;

        bool a_on_boundary_edge = point_on_segment_2d(a, c, d);
        bool b_on_boundary_edge = point_on_segment_2d(b, c, d);
        if (a_on_boundary_edge && b_on_boundary_edge) continue;

        return true;
    }

    return false;
}

static ProjectionDropReason triangle_filter_reason(const vector<Vector2d>& uv_points,
                                                   const vector<Vector2d>& polygon,
                                                   const ReconstructionConfig& cfg,
                                                   double sample_spacing,
                                                   const Tri2D& tri) {
    if (tri.a < 0 || tri.b < 0 || tri.c < 0 ||
        tri.a >= (int)uv_points.size() ||
        tri.b >= (int)uv_points.size() ||
        tri.c >= (int)uv_points.size()) {
        return ProjectionDropReason::NaN;
    }
    const Vector2d& a = uv_points[tri.a];
    const Vector2d& b = uv_points[tri.b];
    const Vector2d& c = uv_points[tri.c];
    if (!finite_vec2(a) || !finite_vec2(b) || !finite_vec2(c)) {
        return ProjectionDropReason::NaN;
    }
    double area2 = abs(tri_area2(a, b, c));
    if (area2 < 1e-14) return ProjectionDropReason::DegenerateArea;
    if (!triangle_inside_polygon_2d(polygon, a, b, c)) {
        return ProjectionDropReason::OutsideDomain;
    }

    double ab = (a - b).norm();
    double bc = (b - c).norm();
    double ca = (c - a).norm();
    double min_edge = min(ab, min(bc, ca));
    double max_edge = max(ab, max(bc, ca));
    if (min_edge < 1e-12 || !finite_double(max_edge)) {
        return ProjectionDropReason::DegenerateArea;
    }

    double max_allowed_edge = cfg.max_delaunay_edge_factor * max(1e-12, sample_spacing);
    if (cfg.max_delaunay_edge_factor > 0.0 &&
        sample_spacing > 1e-12 &&
        max_edge > max_allowed_edge) {
        return ProjectionDropReason::OutsideDomain;
    }

    if (cfg.max_triangle_edge_ratio > 0.0 &&
        max_edge / min_edge > cfg.max_triangle_edge_ratio) {
        return ProjectionDropReason::DegenerateArea;
    }

    if (edge_crosses_polygon_boundary_2d(a, b, polygon) ||
        edge_crosses_polygon_boundary_2d(b, c, polygon) ||
        edge_crosses_polygon_boundary_2d(c, a, polygon)) {
        return ProjectionDropReason::OutsideDomain;
    }

    return ProjectionDropReason::None;
}

static ProjectionDropReason cdt_triangle_filter_reason(const vector<Vector2d>& uv_points,
                                                       const vector<Vector2d>& polygon,
                                                       const Tri2D& tri) {
    if (tri.a < 0 || tri.b < 0 || tri.c < 0 ||
        tri.a >= (int)uv_points.size() ||
        tri.b >= (int)uv_points.size() ||
        tri.c >= (int)uv_points.size()) {
        return ProjectionDropReason::NaN;
    }

    const Vector2d& a = uv_points[tri.a];
    const Vector2d& b = uv_points[tri.b];
    const Vector2d& c = uv_points[tri.c];
    if (!finite_vec2(a) || !finite_vec2(b) || !finite_vec2(c)) {
        return ProjectionDropReason::NaN;
    }
    if (abs(tri_area2(a, b, c)) < 1e-14) {
        return ProjectionDropReason::DegenerateArea;
    }
    if (!triangle_inside_polygon_2d(polygon, a, b, c)) {
        return ProjectionDropReason::OutsideDomain;
    }
    return ProjectionDropReason::None;
}

static bool triangle_passes_quality_filters(const vector<Vector2d>& uv_points,
                                            const vector<Vector2d>& polygon,
                                            const ReconstructionConfig& cfg,
                                            double sample_spacing,
                                            const Tri2D& tri) {
    return triangle_filter_reason(uv_points, polygon, cfg, sample_spacing, tri) ==
           ProjectionDropReason::None;
}

static bool add_unique_local_triangle(const vector<Vector2d>& uv_points,
                                      const vector<Vector2d>& polygon,
                                      int a,
                                      int b,
                                      int c,
                                      set<tuple<int, int, int>>& seen,
                                      vector<Tri2D>& out_tris) {
    if (a == b || b == c || c == a) return false;
    if (a < 0 || b < 0 || c < 0 ||
        a >= (int)uv_points.size() ||
        b >= (int)uv_points.size() ||
        c >= (int)uv_points.size()) {
        return false;
    }
    if (abs(tri_area2(uv_points[a], uv_points[b], uv_points[c])) < 1e-14) return false;
    if (!triangle_inside_polygon_2d(polygon, uv_points[a], uv_points[b], uv_points[c])) {
        return false;
    }

    vector<int> ids = {a, b, c};
    sort(ids.begin(), ids.end());
    auto key = make_tuple(ids[0], ids[1], ids[2]);
    if (!seen.insert(key).second) return false;

    Tri2D t{a, b, c};
    orient_ccw(uv_points, t);
    out_tris.push_back(t);
    return true;
}

static bool append_vertices_for_region(
    const RegionPatch& patch,
    const MatrixXd& V,
    const vector<Vector3d>& interior_positions,
    vector<Vector3d>& vertices,
    map<int, int>& boundary_vertex_map,
    map<BoundarySampleVertexKey, int>& boundary_sample_vertex_map,
    vector<int>& output_vertex_ids) {
    int nb = !patch.boundary_samples.empty()
           ? (int)patch.boundary_samples.size()
           : (int)patch.ordered_boundary_vertex_ids.size();

    output_vertex_ids.clear();
    output_vertex_ids.reserve(nb + interior_positions.size());
    for (int i = 0; i < nb; i++) {
        if (!patch.boundary_samples.empty()) {
            const auto& sample = patch.boundary_samples[i];
            if (sample.is_original_vertex()) {
                output_vertex_ids.push_back(
                    get_boundary_vertex(sample.original_vertex_id, V, vertices, boundary_vertex_map));
            } else {
                int id = get_boundary_sample_vertex(
                    sample, vertices, boundary_sample_vertex_map);
                if (id < 0) return false;
                output_vertex_ids.push_back(id);
            }
        } else {
            int original_vid = patch.ordered_boundary_vertex_ids[i];
            output_vertex_ids.push_back(
                get_boundary_vertex(original_vid, V, vertices, boundary_vertex_map));
        }
    }

    for (const Vector3d& p : interior_positions) {
        output_vertex_ids.push_back(append_vertex(vertices, p));
    }
    return true;
}

static int emit_region_triangles(
    const RegionPatch& patch,
    const vector<Tri2D>& local_tris,
    const vector<int>& output_vertex_ids,
    vector<Vector3d>& vertices,
    vector<Vector3i>& faces,
    vector<int>& labels) {
    int before = (int)faces.size();
    for (Tri2D tri : local_tris) {
        int a = output_vertex_ids[tri.a];
        int b = output_vertex_ids[tri.b];
        int c = output_vertex_ids[tri.c];
        push_oriented_region_face(patch, a, b, c, vertices, faces, labels);
    }
    return (int)faces.size() - before;
}

static int nearest_parent_face_id(const RegionPatch& patch,
                                  const MatrixXd& V,
                                  const MatrixXi& F,
                                  const Vector3d& p) {
    int best_face = -1;
    double best_d2 = numeric_limits<double>::infinity();
    for (int fi : patch.face_ids) {
        Vector3d c = (vertex3(V, F(fi, 0)) +
                      vertex3(V, F(fi, 1)) +
                      vertex3(V, F(fi, 2))) / 3.0;
        double d2 = (p - c).squaredNorm();
        if (d2 < best_d2) {
            best_d2 = d2;
            best_face = fi;
        }
    }
    return best_face;
}

static Vector3d debug_local_position(const RegionPatch& patch,
                                     const MatrixXd& V,
                                     const vector<Vector2d>& uv_points,
                                     const vector<Vector3d>& interior_positions,
                                     int nb,
                                     int local_id) {
    if (local_id < nb) {
        if (!patch.boundary_samples.empty() &&
            local_id < (int)patch.boundary_samples.size()) {
            return patch.boundary_samples[local_id].position;
        }
        if (local_id < (int)patch.ordered_boundary_vertex_ids.size()) {
            return vertex3(V, patch.ordered_boundary_vertex_ids[local_id]);
        }
    }

    int interior_id = local_id - nb;
    if (interior_id >= 0 && interior_id < (int)interior_positions.size()) {
        return interior_positions[interior_id];
    }
    if (local_id >= 0 && local_id < (int)uv_points.size()) {
        const Vector2d& uv = uv_points[local_id];
        return patch.center + uv(0) * patch.tangent_u + uv(1) * patch.tangent_v;
    }
    return patch.center;
}

static void add_projection_debug_triangle(
    const RegionPatch& patch,
    const MatrixXd& V,
    const MatrixXi& F,
    const vector<Vector2d>& uv_points,
    const vector<Vector3d>& interior_positions,
    int nb,
    const Tri2D& tri,
    ProjectionDropReason reason,
    bool kept,
    ProjectionFilterDebugMesh& debug_mesh) {
    Vector3d pa = debug_local_position(patch, V, uv_points, interior_positions, nb, tri.a);
    Vector3d pb = debug_local_position(patch, V, uv_points, interior_positions, nb, tri.b);
    Vector3d pc = debug_local_position(patch, V, uv_points, interior_positions, nb, tri.c);
    int base = (int)debug_mesh.vertices.size();
    debug_mesh.vertices.push_back(pa);
    debug_mesh.vertices.push_back(pb);
    debug_mesh.vertices.push_back(pc);
    debug_mesh.faces.push_back(Vector3i(base, base + 1, base + 2));

    ProjectionFilterDebugTriangle rec;
    rec.region_id = patch.region_id;
    rec.parent_region_id = patch.region_id;
    rec.reason = reason;
    rec.kept = kept;
    rec.parent_face_id = nearest_parent_face_id(
        patch, V, F, (pa + pb + pc) / 3.0);
    debug_mesh.triangles.push_back(rec);
}

static ProjectionDropReason proxy_space_triangle_filter_reason(
    const RegionPatch& patch,
    const ReconstructionConfig& cfg,
    const MatrixXd& V,
    const vector<Vector2d>& uv_points,
    const vector<Vector3d>& interior_positions,
    int nb,
    const Tri2D& tri) {
    if (!cfg.enable_proxy_triangle_quality_filter) return ProjectionDropReason::None;
    if (tri.a < 0 || tri.b < 0 || tri.c < 0 ||
        tri.a >= (int)uv_points.size() ||
        tri.b >= (int)uv_points.size() ||
        tri.c >= (int)uv_points.size()) {
        return ProjectionDropReason::NaN;
    }

    Vector3d pa = debug_local_position(patch, V, uv_points, interior_positions, nb, tri.a);
    Vector3d pb = debug_local_position(patch, V, uv_points, interior_positions, nb, tri.b);
    Vector3d pc = debug_local_position(patch, V, uv_points, interior_positions, nb, tri.c);
    if (!finite_vec3(pa) || !finite_vec3(pb) || !finite_vec3(pc)) {
        return ProjectionDropReason::NaN;
    }

    double ab = (pa - pb).norm();
    double bc = (pb - pc).norm();
    double ca = (pc - pa).norm();
    double min_edge = min(ab, min(bc, ca));
    double max_edge = max(ab, max(bc, ca));
    if (min_edge < 1e-12 || !finite_double(max_edge)) {
        return ProjectionDropReason::DegenerateArea;
    }

    double area = triangle_area3(pa, pb, pc);
    if (!finite_double(area) || area < 1e-15) {
        return ProjectionDropReason::DegenerateArea;
    }

    double ratio_limit = cfg.proxy_triangle_edge_ratio > 0.0
        ? cfg.proxy_triangle_edge_ratio
        : cfg.max_triangle_edge_ratio;
    if (ratio_limit > 0.0 && max_edge / min_edge > ratio_limit) {
        return ProjectionDropReason::DegenerateArea;
    }

    return ProjectionDropReason::None;
}

static void add_projection_failed_debug_marker(
    const RegionPatch& patch,
    const MatrixXd& V,
    const MatrixXi& F,
    const Vector2d& uv,
    ProjectionFilterDebugMesh& debug_mesh) {
    double s = max(1e-6, 0.2 * max(patch.sample_spacing, 1e-6));
    Vector3d p = patch.center + uv(0) * patch.tangent_u + uv(1) * patch.tangent_v;
    int base = (int)debug_mesh.vertices.size();
    debug_mesh.vertices.push_back(p + s * patch.tangent_u);
    debug_mesh.vertices.push_back(p - s * patch.tangent_u + s * patch.tangent_v);
    debug_mesh.vertices.push_back(p - s * patch.tangent_u - s * patch.tangent_v);
    debug_mesh.faces.push_back(Vector3i(base, base + 1, base + 2));

    ProjectionFilterDebugTriangle rec;
    rec.region_id = patch.region_id;
    rec.parent_region_id = patch.region_id;
    rec.parent_face_id = nearest_parent_face_id(patch, V, F, p);
    rec.reason = ProjectionDropReason::ProjectionFailed;
    rec.kept = false;
    debug_mesh.triangles.push_back(rec);
}

static int append_fast_grid_region_mesh(const RegionPatch& patch,
                                        const ReconstructionConfig& cfg,
                                        const MatrixXd& V,
                                        const vector<Vector2d>& interior_uvs,
                                        const vector<Vector3d>& interior_positions,
                                        const vector<int>& lifted_sample_indices,
                                        vector<Vector3d>& vertices,
                                        vector<Vector3i>& faces,
                                        vector<int>& labels,
                                        map<int, int>& boundary_vertex_map,
                                        map<BoundarySampleVertexKey, int>& boundary_sample_vertex_map,
                                        int& out_boundary_vertices,
                                        int& out_interior_vertices) {
    out_boundary_vertices = 0;
    out_interior_vertices = 0;

    int nb = !patch.boundary_samples.empty()
           ? (int)patch.boundary_samples.size()
           : (int)patch.ordered_boundary_vertex_ids.size();
    int ni = (int)interior_uvs.size();
    if (nb < 3 || ni != (int)interior_positions.size() ||
        ni != (int)lifted_sample_indices.size()) {
        return 0;
    }

    vector<Vector2d> uv_points;
    uv_points.reserve(nb + ni);
    for (int i = 0; i < nb; i++) uv_points.push_back(patch.boundary_uv[i]);
    for (const Vector2d& uv : interior_uvs) uv_points.push_back(uv);
    vector<double> boundary_distances(nb + ni, 0.0);

    set<tuple<int, int, int>> seen;
    vector<Tri2D> local_tris;

    map<pair<int, int>, int> grid_local;
    vector<bool> lifted_near_boundary(ni, true);
    for (int li = 0; li < ni; li++) {
        int sample_idx = lifted_sample_indices[li];
        if (sample_idx < 0 || sample_idx >= (int)patch.interior_sample_meta.size()) continue;
        const auto& meta = patch.interior_sample_meta[sample_idx];
        lifted_near_boundary[li] = meta.near_boundary || !meta.is_grid_sample;
        boundary_distances[nb + li] = meta.boundary_distance;
        if (meta.is_grid_sample && !meta.near_boundary && meta.ix >= 0 && meta.iy >= 0) {
            grid_local[make_pair(meta.ix, meta.iy)] = nb + li;
        }
    }

    for (int iy = 0; iy + 1 < patch.bbox_grid_resolution_y; iy++) {
        for (int ix = 0; ix + 1 < patch.bbox_grid_resolution_x; ix++) {
            auto it00 = grid_local.find(make_pair(ix, iy));
            auto it10 = grid_local.find(make_pair(ix + 1, iy));
            auto it01 = grid_local.find(make_pair(ix, iy + 1));
            auto it11 = grid_local.find(make_pair(ix + 1, iy + 1));
            if (it00 == grid_local.end() || it10 == grid_local.end() ||
                it01 == grid_local.end() || it11 == grid_local.end()) {
                continue;
            }

            int p00 = it00->second;
            int p10 = it10->second;
            int p01 = it01->second;
            int p11 = it11->second;
            add_unique_local_triangle(uv_points, patch.boundary_uv, p00, p10, p11, seen, local_tris);
            add_unique_local_triangle(uv_points, patch.boundary_uv, p00, p11, p01, seen, local_tris);
        }
    }

    vector<int> subset_to_local;
    subset_to_local.reserve(nb + ni);
    for (int i = 0; i < nb; i++) subset_to_local.push_back(i);
    for (int li = 0; li < ni; li++) {
        if (lifted_near_boundary[li]) subset_to_local.push_back(nb + li);
    }

    if (subset_to_local.size() >= 3) {
        vector<Vector2d> subset_uv;
        subset_uv.reserve(subset_to_local.size());
        for (int local_id : subset_to_local) subset_uv.push_back(uv_points[local_id]);

        vector<Tri2D> boundary_tris;
        if (bowyer_watson_triangulation(subset_uv, boundary_tris)) {
            double keep_band = (max(0.0, cfg.fast_triangulation_boundary_band_factor) + 0.75) *
                               max(patch.sample_spacing, 1e-12);
            for (Tri2D t : boundary_tris) {
                int a = subset_to_local[t.a];
                int b = subset_to_local[t.b];
                int c = subset_to_local[t.c];
                bool touches_boundary = (a < nb || b < nb || c < nb);
                double max_boundary_distance = max(boundary_distances[a],
                                                   max(boundary_distances[b],
                                                       boundary_distances[c]));
                if (!touches_boundary &&
                    finite_double(max_boundary_distance) &&
                    max_boundary_distance > keep_band) {
                    continue;
                }
                Tri2D local_tri{a, b, c};
                if (!triangle_passes_quality_filters(
                        uv_points, patch.boundary_uv, cfg, patch.sample_spacing, local_tri)) {
                    continue;
                }
                add_unique_local_triangle(uv_points, patch.boundary_uv, a, b, c, seen, local_tris);
            }
        }
    }

    if (local_tris.empty()) return 0;

    vector<int> output_vertex_ids;
    if (!append_vertices_for_region(patch, V, interior_positions, vertices,
                                    boundary_vertex_map, boundary_sample_vertex_map,
                                    output_vertex_ids)) {
        return 0;
    }

    out_boundary_vertices = nb;
    out_interior_vertices = ni;
    return emit_region_triangles(patch, local_tris, output_vertex_ids, vertices, faces, labels);
}

static int append_delaunay_region_mesh(const RegionPatch& patch,
                                       const ReconstructionConfig& cfg,
                                       const MatrixXd& V,
                                       const MatrixXi& F,
                                       const vector<Vector2d>& interior_uvs,
                                       const vector<Vector3d>& interior_positions,
                                       vector<Vector3d>& vertices,
                                       vector<Vector3i>& faces,
                                       vector<int>& labels,
                                       map<int, int>& boundary_vertex_map,
                                       map<BoundarySampleVertexKey, int>& boundary_sample_vertex_map,
                                       int& out_boundary_vertices,
                                       int& out_interior_vertices,
                                       RegionReconstructionDebugInfo* dbg,
                                       ProjectionFilterDebugMesh* debug_mesh) {
    out_boundary_vertices = 0;
    out_interior_vertices = 0;

    int nb = !patch.boundary_samples.empty()
           ? (int)patch.boundary_samples.size()
           : (int)patch.ordered_boundary_vertex_ids.size();
    if (nb < 3 || interior_uvs.size() != interior_positions.size()) return 0;

    vector<Vector2d> uv_points;
    uv_points.reserve(nb + interior_uvs.size());

    for (int i = 0; i < nb; i++) {
        uv_points.push_back(patch.boundary_uv[i]);
    }

    for (int i = 0; i < (int)interior_uvs.size(); i++) {
        uv_points.push_back(interior_uvs[i]);
    }

    vector<Vector2d> triangulation_uv_points = uv_points;
    Matrix2d metric_warp = Matrix2d::Identity();
    Vector2d metric_origin = 0.5 * (patch.bbox_uv_min + patch.bbox_uv_max);
    if (compute_metric_warp_transform(
            patch, cfg, interior_uvs, metric_warp, metric_origin)) {
        for (Vector2d& uv : triangulation_uv_points) {
            uv = metric_warp * (uv - metric_origin);
        }
    }

    vector<Tri2D> tri2;
    bool used_cgal_cdt = false;
#ifdef VSA_USE_CGAL_CDT
    if (cfg.use_cgal_cdt) {
        vector<Vector3i> cgal_tris;
        used_cgal_cdt = vsa_cgal_constrained_delaunay_2d(
            triangulation_uv_points, nb, cgal_tris);
        if (used_cgal_cdt) {
            tri2.reserve(cgal_tris.size());
            for (const Vector3i& t : cgal_tris) {
                tri2.push_back({t(0), t(1), t(2)});
            }
        }
    }
#endif
    if (!used_cgal_cdt &&
        !bowyer_watson_triangulation(triangulation_uv_points, tri2)) return 0;
    if (dbg) dbg->subdivided_triangles += (int)tri2.size();

    vector<Tri2D> kept_triangles;
    for (Tri2D tri : tri2) {
        ProjectionDropReason reason = used_cgal_cdt
            ? cdt_triangle_filter_reason(uv_points, patch.boundary_uv, tri)
            : triangle_filter_reason(
                uv_points, patch.boundary_uv, cfg, patch.sample_spacing, tri);
        if (reason != ProjectionDropReason::None) {
            if (dbg) count_drop_reason(*dbg, reason);
            if (debug_mesh) {
                add_projection_debug_triangle(
                    patch, V, F, uv_points, interior_positions, nb,
                    tri, reason, false, *debug_mesh);
            }
            continue;
        }

        ProjectionDropReason proxy_reason = proxy_space_triangle_filter_reason(
            patch, cfg, V, uv_points, interior_positions, nb, tri);
        if (proxy_reason != ProjectionDropReason::None) {
            if (dbg) count_drop_reason(*dbg, proxy_reason);
            if (debug_mesh) {
                add_projection_debug_triangle(
                    patch, V, F, uv_points, interior_positions, nb,
                    tri, proxy_reason, false, *debug_mesh);
            }
            continue;
        }

        if (dbg) dbg->kept_triangles++;
        if (debug_mesh) {
            add_projection_debug_triangle(
                patch, V, F, uv_points, interior_positions, nb,
                tri, ProjectionDropReason::None, true, *debug_mesh);
        }
        kept_triangles.push_back(tri);
    }

    if (kept_triangles.empty()) return 0;

    vector<int> output_vertex_ids;
    output_vertex_ids.reserve(uv_points.size());
    for (int i = 0; i < nb; i++) {
        if (!patch.boundary_samples.empty()) {
            const auto& sample = patch.boundary_samples[i];
            if (sample.is_original_vertex()) {
                output_vertex_ids.push_back(
                    get_boundary_vertex(sample.original_vertex_id, V, vertices, boundary_vertex_map));
            } else {
                int id = get_boundary_sample_vertex(
                    sample, vertices, boundary_sample_vertex_map);
                if (id < 0) return 0;
                output_vertex_ids.push_back(id);
            }
        } else {
            int original_vid = patch.ordered_boundary_vertex_ids[i];
            output_vertex_ids.push_back(
                get_boundary_vertex(original_vid, V, vertices, boundary_vertex_map));
        }
    }
    for (int i = 0; i < (int)interior_positions.size(); i++) {
        output_vertex_ids.push_back(append_vertex(vertices, interior_positions[i]));
    }

    int before = (int)faces.size();
    for (Tri2D tri : kept_triangles) {
        int a = output_vertex_ids[tri.a];
        int b = output_vertex_ids[tri.b];
        int c = output_vertex_ids[tri.c];
        push_oriented_region_face(patch, a, b, c, vertices, faces, labels);
    }

    out_boundary_vertices = nb;
    out_interior_vertices = (int)interior_uvs.size();
    return (int)faces.size() - before;
}

static void convert_output(const vector<Vector3d>& vertices,
                           const vector<Vector3i>& faces,
                           const vector<int>& labels,
                           ReconstructedMesh& out_mesh) {
    out_mesh.V.resize((int)vertices.size(), 3);
    for (int i = 0; i < (int)vertices.size(); i++) {
        out_mesh.V.row(i) = vertices[i].transpose();
    }

    out_mesh.F.resize((int)faces.size(), 3);
    for (int i = 0; i < (int)faces.size(); i++) {
        out_mesh.F.row(i) = faces[i].transpose();
    }

    out_mesh.R.resize((int)labels.size(), 1);
    for (int i = 0; i < (int)labels.size(); i++) {
        out_mesh.R(i, 0) = labels[i];
    }
}

static bool debug_region_selected(const ReconstructionDebugOptions& debug,
                                  int region_id,
                                  int exported_count) {
    if (!debug.export_region_2d_debug) return false;
    if (!debug.debug_region_ids.empty()) {
        return find(debug.debug_region_ids.begin(), debug.debug_region_ids.end(), region_id)
            != debug.debug_region_ids.end();
    }
    return exported_count < debug.max_debug_regions;
}

static bool export_region_2d_debug_file(const string& filename,
                                        const RegionPatch& patch) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Cannot write " << filename << endl;
        return false;
    }

    fout << "type,x,y" << endl;
    for (const Vector2d& uv : patch.boundary_uv)
        fout << "boundary," << uv(0) << "," << uv(1) << endl;
    fout << "bbox_min," << patch.bbox_uv_min(0) << "," << patch.bbox_uv_min(1) << endl;
    fout << "bbox_max," << patch.bbox_uv_max(0) << "," << patch.bbox_uv_max(1) << endl;
    for (const Vector2d& uv : patch.candidate_uv_samples)
        fout << "candidate," << uv(0) << "," << uv(1) << endl;
    for (const Vector2d& uv : patch.interior_uv_samples)
        fout << "inside," << uv(0) << "," << uv(1) << endl;
    for (const Vector2d& uv : patch.lift_success_uv_samples)
        fout << "lift_success," << uv(0) << "," << uv(1) << endl;
    for (const Vector2d& uv : patch.lift_failed_uv_samples)
        fout << "lift_fail," << uv(0) << "," << uv(1) << endl;

    fout.close();
    cout << "Exported 2D region debug: " << filename << endl;
    return true;
}

static void fill_debug_info_from_patch(const RegionPatch& patch,
                                       RegionReconstructionDebugInfo& dbg) {
    dbg.region_id = patch.region_id;
    dbg.face_count = (int)patch.face_ids.size();
    dbg.boundary_vertex_count = (int)patch.boundary_vertex_ids.size();
    dbg.boundary_edge_count = (int)patch.boundary_edges.size();
    dbg.boundary_loop_count = patch.boundary_loop_count;
    dbg.chart_valid = patch.valid_chart;
    dbg.polygon_valid = patch.valid_polygon;
    dbg.polygon_signed_area = patch.polygon_signed_area;
    dbg.polygon_abs_area = patch.polygon_abs_area;
    dbg.polygon_self_intersect = patch.polygon_self_intersect;
    dbg.bbox_grid_resolution_x = patch.bbox_grid_resolution_x;
    dbg.bbox_grid_resolution_y = patch.bbox_grid_resolution_y;
    dbg.candidate_sample_count = (int)patch.candidate_uv_samples.size();
    dbg.inside_sample_count = (int)patch.interior_uv_samples.size();
    dbg.inside_ratio = dbg.candidate_sample_count > 0
                     ? (double)dbg.inside_sample_count / (double)dbg.candidate_sample_count
                     : 0.0;
    dbg.elapsed_chart_ms = patch.elapsed_chart_ms;
    dbg.elapsed_sampling_ms = patch.elapsed_sampling_ms;
}

static void finalize_debug_ratios(RegionReconstructionDebugInfo& dbg) {
    dbg.lift_fail_count = dbg.lift_attempt_count - dbg.lift_success_count;
    if (dbg.lift_fail_count < 0) dbg.lift_fail_count = 0;
    dbg.lift_success_ratio = dbg.lift_attempt_count > 0
                           ? (double)dbg.lift_success_count / (double)dbg.lift_attempt_count
                           : 0.0;
}

static void print_region_debug_line(const RegionReconstructionDebugInfo& dbg) {
    cout << "[RegionRecon] id=" << dbg.region_id
         << " faces=" << dbg.face_count
         << " boundaryV=" << dbg.boundary_vertex_count
         << " loops=" << dbg.boundary_loop_count
         << " chart=" << (dbg.chart_valid ? "ok" : "bad")
         << " polygon=" << (dbg.polygon_valid ? "ok" : "bad")
         << " area=" << dbg.polygon_abs_area
         << " candidate=" << dbg.candidate_sample_count
         << " inside=" << dbg.inside_sample_count
         << " insideRatio=" << dbg.inside_ratio
         << " liftOk=" << dbg.lift_success_count
         << " liftFail=" << dbg.lift_fail_count
         << " finalInterior=" << dbg.final_interior_vertex_count
         << " tris=" << dbg.final_triangle_count
         << " fallback=" << regionFallbackTypeName(dbg.fallback_type)
         << " timeMs=" << dbg.elapsed_total_ms
         << endl;

    if (dbg.boundary_loop_count != 1) {
        cout << "[RegionReconWarning] region=" << dbg.region_id
             << " boundary vertices are not ordered as a valid single loop"
             << " loops=" << dbg.boundary_loop_count << endl;
    }
    if (dbg.polygon_valid && dbg.polygon_self_intersect) {
        cout << "[RegionReconWarning] region=" << dbg.region_id
             << " projected 2D polygon may self-intersect" << endl;
    }
    if (dbg.final_interior_vertex_count <= 1 &&
        dbg.final_triangle_count >= max(1, dbg.boundary_vertex_count / 2)) {
        cout << "[RegionReconWarning] region=" << dbg.region_id
             << " possible fan triangulation: finalInterior="
             << dbg.final_interior_vertex_count
             << " boundaryV=" << dbg.boundary_vertex_count
             << " tris=" << dbg.final_triangle_count
             << " fallback=" << regionFallbackTypeName(dbg.fallback_type)
             << endl;
    }
}

static void accumulate_global_debug(ReconstructionLog& log,
                                    const RegionReconstructionDebugInfo& dbg,
                                    bool original_mesh_fallback) {
    if (dbg.fallback_type == RegionFallbackType::None ||
        dbg.fallback_type == RegionFallbackType::GridTriangulation ||
        dbg.fallback_type == RegionFallbackType::ConstrainedTriangulation)
        log.normal_path_regions++;
    if (dbg.fallback_type == RegionFallbackType::FanTriangulation)
        log.fan_triangulation_regions++;
    if (original_mesh_fallback)
        log.original_mesh_fallback_regions++;
    if (dbg.fallback_type == RegionFallbackType::InvalidChart)
        log.invalid_chart_regions++;
    if (dbg.fallback_type == RegionFallbackType::InvalidPolygon)
        log.invalid_polygon_regions++;
    if (dbg.fallback_type == RegionFallbackType::TooFewInteriorSamples)
        log.too_few_sample_regions++;

    log.total_candidate_samples += dbg.candidate_sample_count;
    log.total_inside_samples += dbg.inside_sample_count;
    log.total_successful_lifts += dbg.lift_success_count;
    log.total_failed_lifts += dbg.lift_fail_count;
}

bool reconstructQuadricVSA(
    const MatrixXd& V,
    const MatrixXi& F,
    const MatrixXi& R,
    const vector<QuadricProxy>& QP,
    int num_regions,
    const ReconstructionConfig& cfg,
    DisplacementQuery* displacement_query,
    ReconstructedMesh& out_mesh,
    ReconstructionLog& log_out) {

    auto reconstruction_start = Clock::now();
    log_out = ReconstructionLog();
    log_out.num_regions = num_regions;

    vector<RegionPatch> patches = extractRegionPatches(V, F, R, QP, num_regions, cfg);
    vector<RegionRayProjector> ray_projectors;
    if (cfg.interior_projection_mode == InteriorProjectionMode::OriginalMeshRayCast ||
        cfg.interior_projection_mode == InteriorProjectionMode::RayCastThenQuadricFallback) {
        build_region_ray_projectors(V, F, patches, ray_projectors);
    }
    vector<Vector3d> out_vertices;
    vector<Vector3i> out_faces;
    vector<int> out_labels;
    map<int, int> boundary_vertex_map;
    map<BoundarySampleVertexKey, int> boundary_sample_vertex_map;
    map<int, int> fallback_vertex_map;
    int debug_exported_count = 0;

    for (RegionPatch& patch : patches) {
        auto region_start = Clock::now();
        RegionReconstructionStats st;
        RegionReconstructionDebugInfo dbg;
        st.region_id = patch.region_id;
        st.face_count = (int)patch.face_ids.size();
        st.boundary_vertex_count = (int)patch.boundary_vertex_ids.size();
        st.interior_sample_count = (int)patch.interior_uv_samples.size();
        st.chart_valid = patch.valid_chart;
        st.polygon_valid = patch.valid_polygon;
        fill_debug_info_from_patch(patch, dbg);
        dbg.input_faces = (int)patch.face_ids.size();

        bool original_mesh_fallback = false;
        auto finish_region = [&]() {
            auto region_end = Clock::now();
            dbg.elapsed_total_ms = patch.elapsed_chart_ms + patch.elapsed_sampling_ms +
                dbg.elapsed_lifting_ms + dbg.elapsed_triangulation_ms +
                elapsed_ms(region_start, region_end);
            finalize_debug_ratios(dbg);
            st.failed_lifts = dbg.lift_fail_count;
            st.fallback_type = dbg.fallback_type;
            log_out.region_stats.push_back(st);
            log_out.debug_infos.push_back(dbg);
            accumulate_global_debug(log_out, dbg, original_mesh_fallback);
            print_region_debug_line(dbg);

            if (debug_region_selected(cfg.debug, patch.region_id, debug_exported_count)) {
                string prefix = cfg.debug.output_prefix.empty()
                              ? string("reconstruction")
                              : cfg.debug.output_prefix;
                export_region_2d_debug_file(
                    prefix + "_debug_region_" + to_string(patch.region_id) + "_2d.csv",
                    patch);
                debug_exported_count++;
            }
        };

        auto fallback_or_skip = [&](const string& reason, RegionFallbackType type) {
            st.reason = reason;
            dbg.fallback_type = type;
            if (cfg.fallback_to_original) {
                st.fallback_used = true;
                original_mesh_fallback = true;
                auto tri_start = Clock::now();
                st.output_face_count = append_original_region_mesh(
                    patch, V, F, out_vertices, out_faces, out_labels,
                    boundary_vertex_map, fallback_vertex_map);
                auto tri_end = Clock::now();
                dbg.elapsed_triangulation_ms += elapsed_ms(tri_start, tri_end);
                dbg.final_boundary_vertex_count = (int)patch.boundary_vertex_ids.size();
                dbg.final_interior_vertex_count = 0;
                dbg.final_triangle_count = st.output_face_count;
                log_out.fallback_regions++;
                log_out.processed_regions++;
            } else {
                st.skipped = true;
                dbg.fallback_type = RegionFallbackType::SkipRegion;
                log_out.skipped_regions++;
            }
        };

        if (cfg.max_regions >= 0 && patch.region_id >= cfg.max_regions) {
            st.skipped = true;
            st.reason = "max_regions_limit";
            dbg.fallback_type = RegionFallbackType::SkipRegion;
            log_out.skipped_regions++;
            finish_region();
            continue;
        }

        if (patch.face_ids.empty()) {
            st.skipped = true;
            st.reason = "empty_region";
            dbg.fallback_type = RegionFallbackType::SkipRegion;
            log_out.skipped_regions++;
            finish_region();
            continue;
        }

        if (!patch.valid_chart) {
            fallback_or_skip("invalid_chart", RegionFallbackType::InvalidChart);
            finish_region();
            continue;
        }

        if (!patch.simple_boundary_loop || !patch.valid_polygon) {
            fallback_or_skip(!patch.simple_boundary_loop ? "complex_boundary" : "invalid_polygon",
                             RegionFallbackType::InvalidPolygon);
            finish_region();
            continue;
        }

        auto lift_start = Clock::now();
        patch.lift_success_uv_samples.clear();
        patch.lift_failed_uv_samples.clear();
        vector<Vector2d> lifted_uvs;
        vector<Vector3d> lifted_positions;
        vector<int> lifted_sample_indices;
        map<tuple<long long, long long, long long>, vector<int>> accepted_surface_cells;
        double min_surface_spacing = cfg.enable_surface_spacing_filter
            ? cfg.surface_min_sample_spacing_factor * max(0.0, patch.sample_spacing)
            : 0.0;
        double min_surface_spacing_sq = min_surface_spacing * min_surface_spacing;
        double surface_cell_size = max(1e-12, min_surface_spacing);
        auto surface_cell_key = [&](const Vector3d& p) {
            return make_tuple((long long)floor(p(0) / surface_cell_size),
                              (long long)floor(p(1) / surface_cell_size),
                              (long long)floor(p(2) / surface_cell_size));
        };
        auto accepts_surface_spacing = [&](const Vector3d& p) {
            if (min_surface_spacing <= 1e-12) return true;
            auto key = surface_cell_key(p);
            long long kx = get<0>(key);
            long long ky = get<1>(key);
            long long kz = get<2>(key);
            for (long long dz = -1; dz <= 1; dz++) {
                for (long long dy = -1; dy <= 1; dy++) {
                    for (long long dx = -1; dx <= 1; dx++) {
                        auto it = accepted_surface_cells.find(
                            make_tuple(kx + dx, ky + dy, kz + dz));
                        if (it == accepted_surface_cells.end()) continue;
                        for (int idx : it->second) {
                            if (idx >= 0 && idx < (int)lifted_positions.size() &&
                                (lifted_positions[idx] - p).squaredNorm() <
                                    min_surface_spacing_sq) {
                                return false;
                            }
                        }
                    }
                }
            }
            return true;
        };
        auto record_surface_spacing = [&](const Vector3d& p, int idx) {
            if (min_surface_spacing <= 1e-12) return;
            accepted_surface_cells[surface_cell_key(p)].push_back(idx);
        };

        auto project_and_append_uv = [&](const Vector2d& uv,
                                         int sample_idx,
                                         bool use_spacing_filter,
                                         bool emit_failure_marker) {
            Vector3d lifted;
            bool projected = false;
            bool projected_by_quadric = false;
            dbg.lift_attempt_count++;

            if (cfg.interior_projection_mode == InteriorProjectionMode::OriginalMeshRayCast ||
                cfg.interior_projection_mode == InteriorProjectionMode::RayCastThenQuadricFallback) {
                if (patch.region_id >= 0 &&
                    patch.region_id < (int)ray_projectors.size()) {
                    projected = projectUvToOriginalMeshRay(
                        patch, ray_projectors[patch.region_id], V, cfg, uv, lifted);
                }
            }

            if (!projected &&
                (cfg.interior_projection_mode == InteriorProjectionMode::QuadricLift ||
                 cfg.interior_projection_mode == InteriorProjectionMode::RayCastThenQuadricFallback)) {
                projected = liftToQuadric(patch, uv, lifted);
                projected_by_quadric = projected;
            }

            if (!projected) {
                dbg.projection_failed_vertices++;
                count_drop_reason(dbg, ProjectionDropReason::ProjectionFailed);
                if (emit_failure_marker) {
                    add_projection_failed_debug_marker(
                        patch, V, F, uv, log_out.projection_filter_debug_mesh);
                }
                patch.lift_failed_uv_samples.push_back(uv);
                return false;
            }

            if (projected_by_quadric) {
                apply_boundary_transition(patch, cfg, sample_idx, uv, lifted);
            }

            double abs_disp = 0.0;
            bool disp_failed = false;
            if (projected_by_quadric) {
                apply_optional_displacement(patch, cfg, displacement_query,
                                            lifted, abs_disp, disp_failed);
            }
            if (disp_failed) {
                st.failed_displacement_queries++;
                log_out.total_failed_displacement_queries++;
            } else if (cfg.enable_displacement) {
                st.average_displacement += abs_disp;
                st.max_displacement = max(st.max_displacement, abs_disp);
            }

            if (use_spacing_filter && !accepts_surface_spacing(lifted)) {
                patch.lift_failed_uv_samples.push_back(uv);
                return false;
            }

            dbg.lift_success_count++;
            dbg.projected_vertices++;
            patch.lift_success_uv_samples.push_back(uv);
            lifted_uvs.push_back(uv);
            lifted_positions.push_back(lifted);
            lifted_sample_indices.push_back(sample_idx);
            record_surface_spacing(lifted, (int)lifted_positions.size() - 1);
            return true;
        };

        for (int sample_idx = 0; sample_idx < (int)patch.interior_uv_samples.size(); sample_idx++) {
            const Vector2d& uv = patch.interior_uv_samples[sample_idx];
            project_and_append_uv(uv, sample_idx, true, true);
        }
        auto lift_end = Clock::now();
        dbg.elapsed_lifting_ms += elapsed_ms(lift_start, lift_end);
        if (cfg.enable_displacement && !lifted_positions.empty()) {
            st.average_displacement /= (double)lifted_positions.size();
        }

        auto tri_start = Clock::now();
        int final_boundary_vertices = 0;
        int final_interior_vertices = 0;
        int added = 0;
        bool used_full_patch_delaunay = false;
        if (cfg.prefer_full_patch_delaunay) {
            added = append_delaunay_region_mesh(
                patch, cfg, V, F, lifted_uvs, lifted_positions,
                out_vertices, out_faces, out_labels,
                boundary_vertex_map,
                boundary_sample_vertex_map,
                final_boundary_vertices, final_interior_vertices,
                &dbg, &log_out.projection_filter_debug_mesh);
            used_full_patch_delaunay = added > 0;
        }
        if (added <= 0 && cfg.use_simple_grid_triangulation) {
            added = append_fast_grid_region_mesh(
                patch, cfg, V, lifted_uvs, lifted_positions, lifted_sample_indices,
                out_vertices, out_faces, out_labels,
                boundary_vertex_map,
                boundary_sample_vertex_map,
                final_boundary_vertices, final_interior_vertices);
        }
        if (added <= 0) {
            added = append_delaunay_region_mesh(
                patch, cfg, V, F, lifted_uvs, lifted_positions,
                out_vertices, out_faces, out_labels,
                boundary_vertex_map,
                boundary_sample_vertex_map,
                final_boundary_vertices, final_interior_vertices,
                &dbg, &log_out.projection_filter_debug_mesh);
        }
        auto tri_end = Clock::now();
        dbg.elapsed_triangulation_ms += elapsed_ms(tri_start, tri_end);

        if (added > 0) {
            st.output_face_count = added;
            st.reason = used_full_patch_delaunay
                ? "full_patch_delaunay_triangulation"
                : "bbox_grid_triangulation";
            dbg.fallback_type = used_full_patch_delaunay
                ? RegionFallbackType::ConstrainedTriangulation
                : RegionFallbackType::GridTriangulation;
            dbg.final_boundary_vertex_count = final_boundary_vertices;
            dbg.final_interior_vertex_count = final_interior_vertices;
            dbg.final_triangle_count = added;
            log_out.processed_regions++;
        } else {
            if (lifted_positions.empty()) {
                fallback_or_skip(patch.interior_uv_samples.empty()
                    ? "no_interior_sample"
                    : "all_interior_lifts_failed",
                    patch.interior_uv_samples.empty()
                        ? RegionFallbackType::TooFewInteriorSamples
                        : RegionFallbackType::TooManyFailedLifts);
            } else {
                fallback_or_skip("delaunay_failed_original_region_mesh",
                                 RegionFallbackType::OriginalRegionMesh);
            }
        }

        finish_region();
    }

    convert_output(out_vertices, out_faces, out_labels, out_mesh);
    out_mesh.boundary_vertex_map = boundary_vertex_map;

    log_out.boundary_vertex_map = boundary_vertex_map;
    log_out.reconstructed_vertices = out_mesh.V.rows();
    log_out.reconstructed_faces = out_mesh.F.rows();
    log_out.total_reconstruction_time_ms = elapsed_ms(reconstruction_start, Clock::now());

    if (cfg.verbose) {
        cout << "[RegionReconSummary]" << endl;
        cout << "regions_total=" << log_out.num_regions << endl;
        cout << "regions_processed=" << log_out.processed_regions << endl;
        cout << "regions_skipped=" << log_out.skipped_regions << endl;
        cout << "normal_path=" << log_out.normal_path_regions << endl;
        cout << "fan_fallback=" << log_out.fan_triangulation_regions << endl;
        cout << "original_mesh_fallback=" << log_out.original_mesh_fallback_regions << endl;
        cout << "invalid_chart=" << log_out.invalid_chart_regions << endl;
        cout << "invalid_polygon=" << log_out.invalid_polygon_regions << endl;
        cout << "too_few_samples=" << log_out.too_few_sample_regions << endl;
        cout << "candidate_samples=" << log_out.total_candidate_samples << endl;
        cout << "inside_samples=" << log_out.total_inside_samples << endl;
        cout << "lift_success=" << log_out.total_successful_lifts << endl;
        cout << "lift_fail=" << log_out.total_failed_lifts << endl;
        cout << "output_vertices=" << log_out.reconstructed_vertices << endl;
        cout << "output_faces=" << log_out.reconstructed_faces << endl;
        cout << "time_total_ms=" << log_out.total_reconstruction_time_ms << endl;
    }

    return out_mesh.V.rows() > 0 && out_mesh.F.rows() > 0;
}

bool exportReconstructedMesh(
    const string& filename,
    const ReconstructedMesh& mesh) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Cannot write " << filename << endl;
        return false;
    }

    fout << "# Quadric VSA reconstructed mesh" << endl;
    fout << "# " << mesh.V.rows() << " vertices, " << mesh.F.rows() << " faces" << endl;
    for (int i = 0; i < mesh.V.rows(); i++) {
        fout << "v " << mesh.V(i, 0) << " " << mesh.V(i, 1) << " " << mesh.V(i, 2) << endl;
    }
    for (int i = 0; i < mesh.F.rows(); i++) {
        fout << "f " << (mesh.F(i, 0) + 1)
             << " " << (mesh.F(i, 1) + 1)
             << " " << (mesh.F(i, 2) + 1) << endl;
    }

    fout.close();
    cout << "Exported reconstructed OBJ: " << filename
         << " (" << mesh.V.rows() << " verts, " << mesh.F.rows() << " faces)" << endl;
    return true;
}

bool exportReconstructedMeshPLY(
    const string& filename,
    const ReconstructedMesh& mesh) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Cannot write " << filename << endl;
        return false;
    }

    fout << "ply" << endl;
    fout << "format ascii 1.0" << endl;
    fout << "comment Quadric VSA reconstructed mesh" << endl;
    fout << "element vertex " << mesh.V.rows() << endl;
    fout << "property float x" << endl;
    fout << "property float y" << endl;
    fout << "property float z" << endl;
    fout << "element face " << mesh.F.rows() << endl;
    fout << "property list uchar int vertex_indices" << endl;
    fout << "end_header" << endl;

    for (int i = 0; i < mesh.V.rows(); i++) {
        fout << mesh.V(i, 0) << " " << mesh.V(i, 1) << " " << mesh.V(i, 2) << endl;
    }
    for (int i = 0; i < mesh.F.rows(); i++) {
        fout << "3 " << mesh.F(i, 0) << " " << mesh.F(i, 1) << " " << mesh.F(i, 2) << endl;
    }

    fout.close();
    cout << "Exported reconstructed PLY: " << filename
         << " (" << mesh.V.rows() << " verts, " << mesh.F.rows() << " faces)" << endl;
    return true;
}

bool exportReconstructionLog(
    const string& filename,
    const ReconstructionLog& log) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Cannot write " << filename << endl;
        return false;
    }

    fout << "region_id,faces,boundary_vertices,interior_samples,chart_valid,"
         << "polygon_valid,output_faces,skipped,fallback,failed_lifts,"
         << "failed_displacement_queries,average_displacement,max_displacement,"
         << "fallback_type,reason" << endl;

    for (const RegionReconstructionStats& st : log.region_stats) {
        fout << st.region_id << ","
             << st.face_count << ","
             << st.boundary_vertex_count << ","
             << st.interior_sample_count << ","
             << (st.chart_valid ? 1 : 0) << ","
             << (st.polygon_valid ? 1 : 0) << ","
             << st.output_face_count << ","
             << (st.skipped ? 1 : 0) << ","
             << (st.fallback_used ? 1 : 0) << ","
             << st.failed_lifts << ","
             << st.failed_displacement_queries << ","
             << st.average_displacement << ","
             << st.max_displacement << ","
             << regionFallbackTypeName(st.fallback_type) << ","
             << "\"" << st.reason << "\"" << endl;
    }

    fout.close();
    cout << "Exported reconstruction log: " << filename << endl;
    return true;
}

bool exportReconstructionDebugReport(
    const string& filename,
    const ReconstructionLog& log) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Cannot write " << filename << endl;
        return false;
    }

    fout << "region_id,"
         << "face_count,"
         << "boundary_vertex_count,"
         << "boundary_edge_count,"
         << "boundary_loop_count,"
         << "chart_valid,"
         << "polygon_valid,"
         << "polygon_signed_area,"
         << "polygon_abs_area,"
         << "polygon_self_intersect,"
         << "bbox_grid_resolution_x,"
         << "bbox_grid_resolution_y,"
         << "candidate_sample_count,"
         << "inside_sample_count,"
         << "inside_ratio,"
         << "lift_attempt_count,"
         << "lift_success_count,"
         << "lift_fail_count,"
         << "lift_success_ratio,"
         << "final_boundary_vertex_count,"
         << "final_interior_vertex_count,"
         << "final_triangle_count,"
         << "inputFaces,"
         << "subdividedTriangles,"
         << "projectedVertices,"
         << "projectionFailedVertices,"
         << "keptTriangles,"
         << "droppedTriangles,"
         << "droppedProjectionFailed,"
         << "droppedNaN,"
         << "droppedDegenerateArea,"
         << "droppedFlippedNormal,"
         << "droppedOutsideDomain,"
         << "droppedSharpBoundaryInvalid,"
         << "droppedRegionMismatch,"
         << "fallback_type,"
         << "elapsed_chart_ms,"
         << "elapsed_sampling_ms,"
         << "elapsed_lifting_ms,"
         << "elapsed_triangulation_ms,"
         << "elapsed_total_ms" << endl;

    for (const RegionReconstructionDebugInfo& d : log.debug_infos) {
        fout << d.region_id << ","
             << d.face_count << ","
             << d.boundary_vertex_count << ","
             << d.boundary_edge_count << ","
             << d.boundary_loop_count << ","
             << (d.chart_valid ? 1 : 0) << ","
             << (d.polygon_valid ? 1 : 0) << ","
             << d.polygon_signed_area << ","
             << d.polygon_abs_area << ","
             << (d.polygon_self_intersect ? 1 : 0) << ","
             << d.bbox_grid_resolution_x << ","
             << d.bbox_grid_resolution_y << ","
             << d.candidate_sample_count << ","
             << d.inside_sample_count << ","
             << d.inside_ratio << ","
             << d.lift_attempt_count << ","
             << d.lift_success_count << ","
             << d.lift_fail_count << ","
             << d.lift_success_ratio << ","
             << d.final_boundary_vertex_count << ","
             << d.final_interior_vertex_count << ","
             << d.final_triangle_count << ","
             << d.input_faces << ","
             << d.subdivided_triangles << ","
             << d.projected_vertices << ","
             << d.projection_failed_vertices << ","
             << d.kept_triangles << ","
             << d.dropped_triangles << ","
             << d.dropped_projection_failed << ","
             << d.dropped_nan << ","
             << d.dropped_degenerate_area << ","
             << d.dropped_flipped_normal << ","
             << d.dropped_outside_domain << ","
             << d.dropped_sharp_boundary_invalid << ","
             << d.dropped_region_mismatch << ","
             << regionFallbackTypeName(d.fallback_type) << ","
             << d.elapsed_chart_ms << ","
             << d.elapsed_sampling_ms << ","
             << d.elapsed_lifting_ms << ","
             << d.elapsed_triangulation_ms << ","
             << d.elapsed_total_ms << endl;
    }

    fout.close();
    cout << "Exported reconstruction debug report: " << filename << endl;
    return true;
}

bool exportProjectionFilterDebugTrianglesCSV(
    const string& filename,
    const ReconstructionLog& log) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Cannot write " << filename << endl;
        return false;
    }

    fout << "childTriangleId,regionId,parentFaceId,parentRegionId,kept,reason\n";
    const auto& tris = log.projection_filter_debug_mesh.triangles;
    for (int i = 0; i < (int)tris.size(); i++) {
        fout << i << ","
             << tris[i].region_id << ","
             << tris[i].parent_face_id << ","
             << tris[i].parent_region_id << ","
             << (tris[i].kept ? 1 : 0) << ","
             << projectionDropReasonName(tris[i].reason) << "\n";
    }
    fout.close();
    cout << "Exported projection/filter triangle debug CSV: " << filename << endl;
    return true;
}

static RowVector3i projection_debug_color(ProjectionDropReason reason, bool kept) {
    if (kept && reason == ProjectionDropReason::None) return RowVector3i(80, 210, 110);
    switch (reason) {
        case ProjectionDropReason::ProjectionFailed: return RowVector3i(255, 60, 60);
        case ProjectionDropReason::NaN: return RowVector3i(255, 0, 255);
        case ProjectionDropReason::DegenerateArea: return RowVector3i(255, 170, 0);
        case ProjectionDropReason::FlippedNormal: return RowVector3i(40, 80, 255);
        case ProjectionDropReason::OutsideDomain: return RowVector3i(255, 255, 0);
        case ProjectionDropReason::SharpBoundaryInvalid: return RowVector3i(0, 220, 220);
        case ProjectionDropReason::RegionMismatch: return RowVector3i(180, 80, 255);
        case ProjectionDropReason::None: return RowVector3i(180, 180, 180);
    }
    return RowVector3i(180, 180, 180);
}

bool exportProjectionFilterDebugMeshPLY(
    const string& filename,
    const ReconstructionLog& log) {
    const auto& mesh = log.projection_filter_debug_mesh;
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Cannot write " << filename << endl;
        return false;
    }

    fout << "ply\nformat ascii 1.0\n";
    fout << "comment projection/filter debug colors\n";
    fout << "comment kept=None green\n";
    fout << "comment ProjectionFailed red\n";
    fout << "comment NaN magenta\n";
    fout << "comment DegenerateArea orange\n";
    fout << "comment FlippedNormal blue\n";
    fout << "comment OutsideDomain yellow\n";
    fout << "comment SharpBoundaryInvalid cyan\n";
    fout << "comment RegionMismatch purple\n";
    fout << "element vertex " << mesh.vertices.size() << "\n";
    fout << "property float x\nproperty float y\nproperty float z\n";
    fout << "element face " << mesh.faces.size() << "\n";
    fout << "property list uchar int vertex_indices\n";
    fout << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    fout << "end_header\n";

    for (const Vector3d& p : mesh.vertices) {
        fout << p(0) << " " << p(1) << " " << p(2) << "\n";
    }
    for (int i = 0; i < (int)mesh.faces.size(); i++) {
        ProjectionDropReason reason = ProjectionDropReason::None;
        bool kept = false;
        if (i < (int)mesh.triangles.size()) {
            reason = mesh.triangles[i].reason;
            kept = mesh.triangles[i].kept;
        }
        RowVector3i color = projection_debug_color(reason, kept);
        const Vector3i& f = mesh.faces[i];
        fout << "3 " << f(0) << " " << f(1) << " " << f(2) << " "
             << color(0) << " " << color(1) << " " << color(2) << "\n";
    }
    fout.close();
    cout << "Exported projection/filter debug mesh: " << filename << endl;
    return true;
}

bool reconstructAndExportQuadricVSA(
    const MatrixXd& V,
    const MatrixXi& F,
    const MatrixXi& R,
    const vector<QuadricProxy>& QP,
    int num_regions,
    const ReconstructionConfig& cfg,
    const string& base_filename) {

    bool ok = true;
    NullDisplacementQuery null_query;

    ReconstructionConfig quadric_only_cfg = cfg;
    quadric_only_cfg.enable_displacement = false;
    if (quadric_only_cfg.debug.output_prefix == "reconstruction")
        quadric_only_cfg.debug.output_prefix = base_filename;

    ReconstructedMesh quadric_mesh;
    ReconstructionLog quadric_log;
    ok = reconstructQuadricVSA(V, F, R, QP, num_regions,
                               quadric_only_cfg, &null_query,
                               quadric_mesh, quadric_log) && ok;
    ok = exportReconstructedMesh(base_filename + "_reconstructed_quadric_only.obj", quadric_mesh) && ok;
    if (cfg.export_ply) {
        ok = exportReconstructedMeshPLY(base_filename + "_reconstructed_quadric_only.ply", quadric_mesh) && ok;
    }
    ok = exportReconstructionLog(base_filename + "_reconstruction_quadric_only_log.csv", quadric_log) && ok;
    if (quadric_only_cfg.debug.enable_debug_report) {
        ok = exportReconstructionDebugReport(
            base_filename + "_reconstruction_debug_report.csv", quadric_log) && ok;
        ok = exportProjectionFilterDebugTrianglesCSV(
            base_filename + "_projection_filter_debug_triangles.csv", quadric_log) && ok;
        ok = exportProjectionFilterDebugMeshPLY(
            base_filename + "_projection_filter_debug_mesh.ply", quadric_log) && ok;
    }

    if (!cfg.enable_displacement) {
        ok = exportReconstructedMesh(base_filename + "_reconstructed.obj", quadric_mesh) && ok;
        if (cfg.export_ply) {
            ok = exportReconstructedMeshPLY(base_filename + "_reconstructed.ply", quadric_mesh) && ok;
        }
        return ok;
    }

    ReconstructionConfig displacement_cfg = cfg;
    if (displacement_cfg.debug.output_prefix == "reconstruction")
        displacement_cfg.debug.output_prefix = base_filename + "_displacement";

    OriginalMeshAABBDisplacementQuery aabb_query(V, F);
    ReconstructedMesh disp_mesh;
    ReconstructionLog disp_log;
    ok = reconstructQuadricVSA(V, F, R, QP, num_regions,
                               displacement_cfg, &aabb_query,
                               disp_mesh, disp_log) && ok;
    ok = exportReconstructedMesh(base_filename + "_reconstructed_quadric_displacement.obj", disp_mesh) && ok;
    if (cfg.export_ply) {
        ok = exportReconstructedMeshPLY(base_filename + "_reconstructed_quadric_displacement.ply", disp_mesh) && ok;
    }
    ok = exportReconstructionLog(base_filename + "_reconstruction_quadric_displacement_log.csv", disp_log) && ok;
    if (displacement_cfg.debug.enable_debug_report) {
        ok = exportReconstructionDebugReport(
            base_filename + "_reconstruction_displacement_debug_report.csv", disp_log) && ok;
        ok = exportProjectionFilterDebugTrianglesCSV(
            base_filename + "_projection_filter_displacement_debug_triangles.csv", disp_log) && ok;
        ok = exportProjectionFilterDebugMeshPLY(
            base_filename + "_projection_filter_displacement_debug_mesh.ply", disp_log) && ok;
    }

    ok = exportReconstructedMesh(base_filename + "_reconstructed.obj", disp_mesh) && ok;
    if (cfg.export_ply) {
        ok = exportReconstructedMeshPLY(base_filename + "_reconstructed.ply", disp_mesh) && ok;
    }

    return ok;
}
