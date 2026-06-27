#include "reusable_trimmed_bspline_surface.h"

#include "trimmed_mesh_validation.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <tuple>

#ifdef VSA_USE_CGAL_CDT
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#endif

#ifdef _WIN32
#include <direct.h>
#endif

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector3i;
using std::string;
using std::vector;

namespace {

static bool ensure_dir(const string& dir) {
    if (dir.empty()) return false;
#ifdef _WIN32
    if (_mkdir(dir.c_str()) == 0) return true;
#else
    if (mkdir(dir.c_str(), 0755) == 0) return true;
#endif
    struct stat st;
    return stat(dir.c_str(), &st) == 0;
}

static string path_join(const string& a, const string& b) {
    if (a.empty()) return b;
    char last = a.back();
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

static string json_escape(const string& s) {
    std::ostringstream out;
    for (char c : s) {
        if (c == '"' || c == '\\') out << "\\" << c;
        else if (c == '\n') out << "\\n";
        else out << c;
    }
    return out.str();
}

static bool finite_vec2(const Vector2d& p) {
    return std::isfinite(p.x()) && std::isfinite(p.y());
}

static bool point_on_segment(
    const Vector2d& p,
    const Vector2d& a,
    const Vector2d& b,
    double eps) {
    Vector2d ab = b - a;
    double len2 = ab.squaredNorm();
    if (len2 <= eps * eps) return (p - a).norm() <= eps;
    double t = (p - a).dot(ab) / len2;
    if (t < -eps || t > 1.0 + eps) return false;
    Vector2d q = a + std::max(0.0, std::min(1.0, t)) * ab;
    return (p - q).norm() <= eps;
}

static bool point_in_polygon_or_on_boundary(
    const Vector2d& p,
    const vector<Vector2d>& poly) {
    if (poly.size() < 3) return false;
    bool inside = false;
    const double eps = 1e-10;
    for (int i = 0, j = (int)poly.size() - 1; i < (int)poly.size(); j = i++) {
        const Vector2d& a = poly[j];
        const Vector2d& b = poly[i];
        if (point_on_segment(p, a, b, eps)) return true;
        bool crosses = ((a.y() > p.y()) != (b.y() > p.y()));
        if (crosses) {
            double dy = b.y() - a.y();
            if (std::abs(dy) <= 1e-30) continue;
            double x = (b.x() - a.x()) * (p.y() - a.y()) / dy + a.x();
            if (p.x() < x) inside = !inside;
        }
    }
    return inside;
}

static bool point_inside_asset_trim_region(
    const Vector2d& p,
    const vector<ReusableTrimLoop2D>& loops) {
    const ReusableTrimLoop2D* perimeter = nullptr;
    for (const ReusableTrimLoop2D& loop : loops) {
        if (loop.is_perimeter) {
            perimeter = &loop;
            break;
        }
    }
    if (!perimeter ||
        !point_in_polygon_or_on_boundary(p, perimeter->uv_polyline)) {
        return false;
    }
    for (const ReusableTrimLoop2D& loop : loops) {
        if (loop.is_perimeter) continue;
        if (point_in_polygon_or_on_boundary(p, loop.uv_polyline)) return false;
    }
    return true;
}

static double distance_to_polyline_segments(
    const Vector2d& p,
    const vector<Vector2d>& polyline,
    bool closed) {
    if (polyline.size() < 2) return std::numeric_limits<double>::infinity();
    double best = std::numeric_limits<double>::infinity();
    const int n = (int)polyline.size();
    const int edge_count = closed ? n : n - 1;
    for (int i = 0; i < edge_count; i++) {
        const Vector2d& a = polyline[i];
        const Vector2d& b = polyline[(i + 1) % n];
        Vector2d ab = b - a;
        double len2 = ab.squaredNorm();
        double t = len2 > 1e-30 ? (p - a).dot(ab) / len2 : 0.0;
        t = std::max(0.0, std::min(1.0, t));
        Vector2d q = a + t * ab;
        best = std::min(best, (p - q).norm());
    }
    return best;
}

static double distance_to_trim_loops(
    const Vector2d& p,
    const vector<ReusableTrimLoop2D>& loops) {
    double best = std::numeric_limits<double>::infinity();
    for (const ReusableTrimLoop2D& loop : loops) {
        best = std::min(
            best,
            distance_to_polyline_segments(p, loop.uv_polyline, true));
    }
    return best;
}

static double orient2d(const Vector2d& a, const Vector2d& b, const Vector2d& c) {
    return (b.x() - a.x()) * (c.y() - a.y()) -
           (b.y() - a.y()) * (c.x() - a.x());
}

static bool segments_intersect_proper(
    const Vector2d& a,
    const Vector2d& b,
    const Vector2d& c,
    const Vector2d& d,
    double eps) {
    if ((a - c).norm() <= eps || (a - d).norm() <= eps ||
        (b - c).norm() <= eps || (b - d).norm() <= eps) {
        return false;
    }
    double o1 = orient2d(a, b, c);
    double o2 = orient2d(a, b, d);
    double o3 = orient2d(c, d, a);
    double o4 = orient2d(c, d, b);
    if (((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps)) &&
        ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps))) {
        return true;
    }
    if (std::abs(o1) <= eps && point_on_segment(c, a, b, eps)) return true;
    if (std::abs(o2) <= eps && point_on_segment(d, a, b, eps)) return true;
    if (std::abs(o3) <= eps && point_on_segment(a, c, d, eps)) return true;
    if (std::abs(o4) <= eps && point_on_segment(b, c, d, eps)) return true;
    return false;
}

static std::pair<long long, long long> point_key(const Vector2d& p, double eps) {
    return {
        (long long)std::llround(p.x() / eps),
        (long long)std::llround(p.y() / eps)
    };
}

static int add_uv_point(
    vector<Vector2d>& points,
    std::map<std::pair<long long, long long>, int>& key_to_index,
    const Vector2d& p,
    double eps) {
    if (!finite_vec2(p)) return -1;
    auto key = point_key(p, eps);
    auto it = key_to_index.find(key);
    if (it != key_to_index.end()) return it->second;
    int idx = (int)points.size();
    key_to_index[key] = idx;
    points.push_back(p);
    return idx;
}

static int add_uv_point_if_separated(
    vector<Vector2d>& points,
    std::map<std::pair<long long, long long>, int>& key_to_index,
    const Vector2d& p,
    double eps,
    double min_distance) {
    if (!finite_vec2(p)) return -1;
    double min_dist2 = min_distance * min_distance;
    for (int i = 0; i < (int)points.size(); i++) {
        if ((points[i] - p).squaredNorm() < min_dist2) return i;
    }
    return add_uv_point(points, key_to_index, p, eps);
}

static vector<Vector2d> cleaned_loop(const vector<Vector2d>& input) {
    vector<Vector2d> out;
    for (const Vector2d& p : input) {
        if (!finite_vec2(p)) continue;
        Vector2d q = p.cwiseMax(Vector2d::Zero()).cwiseMin(Vector2d::Ones());
        if (!out.empty() && (q - out.back()).norm() < 1e-12) continue;
        out.push_back(q);
    }
    if (out.size() > 1 && (out.front() - out.back()).norm() < 1e-12) {
        out.pop_back();
    }
    return out;
}

static bool cdt_triangulate_uv(
    const vector<Vector2d>& points,
    const vector<std::pair<int, int>>& constraints,
    vector<Vector3i>& triangles) {
    triangles.clear();
#ifndef VSA_USE_CGAL_CDT
    (void)points;
    (void)constraints;
    return false;
#else
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef CGAL::Triangulation_vertex_base_with_info_2<int, Kernel> VertexBase;
    typedef CGAL::Constrained_triangulation_face_base_2<Kernel> FaceBase;
    typedef CGAL::Triangulation_data_structure_2<VertexBase, FaceBase> Tds;
    typedef CGAL::No_constraint_intersection_tag IntersectionTag;
    typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, Tds, IntersectionTag> CDT;
    typedef CDT::Point Point;
    typedef CDT::Vertex_handle VertexHandle;

    if (points.size() < 3) return false;
    try {
        CDT cdt;
        vector<VertexHandle> handles(points.size());
        for (int i = 0; i < (int)points.size(); i++) {
            VertexHandle vh = cdt.insert(Point(points[i].x(), points[i].y()));
            vh->info() = i;
            handles[i] = vh;
        }
        for (const auto& e : constraints) {
            if (e.first < 0 || e.second < 0 ||
                e.first >= (int)handles.size() ||
                e.second >= (int)handles.size() ||
                e.first == e.second) {
                continue;
            }
            cdt.insert_constraint(handles[e.first], handles[e.second]);
        }

        std::set<std::tuple<int, int, int>> seen;
        for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
            int a = fit->vertex(0)->info();
            int b = fit->vertex(1)->info();
            int c = fit->vertex(2)->info();
            if (a < 0 || b < 0 || c < 0 || a == b || b == c || c == a) continue;
            double area = orient2d(points[a], points[b], points[c]);
            if (std::abs(area) < 1e-14) continue;
            Vector3i tri(a, b, c);
            if (area < 0.0) std::swap(tri.y(), tri.z());
            vector<int> ids = {tri.x(), tri.y(), tri.z()};
            std::sort(ids.begin(), ids.end());
            auto key = std::make_tuple(ids[0], ids[1], ids[2]);
            if (!seen.insert(key).second) continue;
            triangles.push_back(tri);
        }
    } catch (...) {
        triangles.clear();
        return false;
    }
    return !triangles.empty();
#endif
}

static bool write_mesh_obj(const string& filename, const MatrixXd& V, const MatrixXi& F) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    for (int i = 0; i < V.rows(); i++) {
        out << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
    }
    for (int i = 0; i < F.rows(); i++) {
        out << "f " << F(i, 0) + 1
            << " " << F(i, 1) + 1
            << " " << F(i, 2) + 1 << "\n";
    }
    return true;
}

static bool write_uv_mesh_obj(const string& filename, const MatrixXd& UV, const MatrixXi& F) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    for (int i = 0; i < UV.rows(); i++) {
        out << "v " << UV(i, 0) << " " << UV(i, 1) << " 0\n";
    }
    for (int i = 0; i < F.rows(); i++) {
        out << "f " << F(i, 0) + 1
            << " " << F(i, 1) + 1
            << " " << F(i, 2) + 1 << "\n";
    }
    return true;
}

static bool write_mesh_subset_obj(
    const string& filename,
    const MatrixXd& V,
    const MatrixXi& F,
    const vector<int>& face_ids) {
    MatrixXi subset((int)face_ids.size(), 3);
    for (int i = 0; i < (int)face_ids.size(); i++) {
        int f = face_ids[i];
        if (f < 0 || f >= F.rows()) return false;
        subset.row(i) = F.row(f);
    }
    return write_mesh_obj(filename, V, subset);
}

static bool write_uv_mesh_subset_obj(
    const string& filename,
    const MatrixXd& UV,
    const MatrixXi& F,
    const vector<int>& face_ids) {
    MatrixXi subset((int)face_ids.size(), 3);
    for (int i = 0; i < (int)face_ids.size(); i++) {
        int f = face_ids[i];
        if (f < 0 || f >= F.rows()) return false;
        subset.row(i) = F.row(f);
    }
    return write_uv_mesh_obj(filename, UV, subset);
}

static bool write_trim_loops_obj(
    const string& filename,
    const vector<ReusableTrimLoop2D>& loops) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    int offset = 1;
    for (int li = 0; li < (int)loops.size(); li++) {
        out << "o trim_loop_" << li << "\n";
        for (const Vector2d& p : loops[li].uv_polyline) {
            out << "v " << p.x() << " " << p.y() << " 0\n";
        }
        int n = (int)loops[li].uv_polyline.size();
        for (int i = 0; i < n; i++) {
            out << "l " << (offset + i) << " " << (offset + ((i + 1) % n)) << "\n";
        }
        offset += n;
    }
    return true;
}

static bool write_surface_trim_loops_obj(
    const string& filename,
    const ReusableTrimmedBSplineSurface& asset) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    int offset = 1;
    for (int li = 0; li < (int)asset.trim_loops.size(); li++) {
        out << "o surface_trim_loop_" << li << "\n";
        for (const Vector2d& uv : asset.trim_loops[li].uv_polyline) {
            Vector3d p = asset.surface.evaluate(uv.x(), uv.y());
            out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
        }
        int n = (int)asset.trim_loops[li].uv_polyline.size();
        for (int i = 0; i < n; i++) {
            out << "l " << (offset + i) << " " << (offset + ((i + 1) % n)) << "\n";
        }
        offset += n;
    }
    return true;
}

static Vector2d safe_inward_normal(
    const vector<ReusableTrimLoop2D>& loops,
    const ReusableTrimLoop2D& loop,
    int edge_index) {
    int n = (int)loop.uv_polyline.size();
    if (n < 2) return Vector2d::Zero();
    const Vector2d& a = loop.uv_polyline[edge_index];
    const Vector2d& b = loop.uv_polyline[(edge_index + 1) % n];
    Vector2d e = b - a;
    if (e.norm() <= 1e-14) return Vector2d::Zero();
    e.normalize();
    Vector2d left(-e.y(), e.x());
    Vector2d mid = 0.5 * (a + b);
    double eps = 1e-4;
    bool left_inside = point_inside_asset_trim_region(mid + eps * left, loops);
    bool right_inside = point_inside_asset_trim_region(mid - eps * left, loops);
    if (left_inside && !right_inside) return left;
    if (right_inside && !left_inside) return -left;
    return loop.is_perimeter ? left : -left;
}

static Vector2d move_uv_inside(
    const vector<ReusableTrimLoop2D>& loops,
    const Vector2d& boundary_uv,
    const Vector2d& inward,
    double distance) {
    Vector2d uv = boundary_uv;
    double scale = 1.0;
    for (int tries = 0; tries < 12; tries++) {
        uv = (boundary_uv + scale * distance * inward)
                 .cwiseMax(Vector2d::Zero())
                 .cwiseMin(Vector2d::Ones());
        if (point_inside_asset_trim_region(uv, loops)) return uv;
        scale *= 0.5;
    }
    return boundary_uv;
}

static bool write_abc_boundary_ribbon_strips_obj(
    const string& filename,
    const ReusableTrimmedBSplineSurface& asset,
    int row_count,
    double ribbon_width) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    int rows = std::max(2, row_count);
    double width = ribbon_width > 0.0 ? ribbon_width : 0.06;
    int vertex_offset = 1;

    for (int li = 0; li < (int)asset.trim_loops.size(); li++) {
        const ReusableTrimLoop2D& loop = asset.trim_loops[li];
        int n = (int)loop.uv_polyline.size();
        if (n < 3 || (int)loop.spatial_polyline.size() != n) continue;

        vector<Vector2d> edge_normals(n, Vector2d::Zero());
        for (int i = 0; i < n; i++) {
            edge_normals[i] = safe_inward_normal(asset.trim_loops, loop, i);
        }

        vector<Vector2d> vertex_normals(n, Vector2d::Zero());
        for (int i = 0; i < n; i++) {
            Vector2d nn = edge_normals[(i - 1 + n) % n] + edge_normals[i];
            if (nn.norm() <= 1e-14) nn = edge_normals[i];
            if (nn.norm() > 1e-14) nn.normalize();
            vertex_normals[i] = nn;
        }

        out << "o abc_boundary_ribbon_loop_" << li << "\n";
        for (int r = 0; r < rows; r++) {
            double t = rows == 1 ? 0.0 : (double)r / (double)(rows - 1);
            double fade = (1.0 - t) * (1.0 - t) * (1.0 + 2.0 * t);
            for (int i = 0; i < n; i++) {
                Vector2d uv = move_uv_inside(
                    asset.trim_loops,
                    loop.uv_polyline[i],
                    vertex_normals[i],
                    t * width);
                Vector3d base = asset.surface.evaluate(uv.x(), uv.y());
                Vector3d boundary_base =
                    asset.surface.evaluate(loop.uv_polyline[i].x(), loop.uv_polyline[i].y());
                Vector3d residual = loop.spatial_polyline[i] - boundary_base;
                Vector3d p = (r == 0) ? loop.spatial_polyline[i] : base + fade * residual;
                out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
            }
        }
        for (int r = 0; r + 1 < rows; r++) {
            for (int i = 0; i < n; i++) {
                int a = vertex_offset + r * n + i;
                int b = vertex_offset + r * n + (i + 1) % n;
                int c = vertex_offset + (r + 1) * n + (i + 1) % n;
                int d = vertex_offset + (r + 1) * n + i;
                out << "f " << a << " " << b << " " << c << "\n";
                out << "f " << a << " " << c << " " << d << "\n";
            }
        }
        vertex_offset += rows * n;
    }
    return vertex_offset > 1;
}

static bool write_abc_boundary_ribbon_report_json(
    const string& filename,
    const ReusableTrimmedBSplineSurface& asset,
    int row_count,
    double ribbon_width) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    int loop_count = 0;
    int boundary_vertex_count = 0;
    double sum = 0.0;
    double sum2 = 0.0;
    double max_error = 0.0;
    for (const ReusableTrimLoop2D& loop : asset.trim_loops) {
        int n = (int)loop.uv_polyline.size();
        if (n < 3 || (int)loop.spatial_polyline.size() != n) continue;
        loop_count++;
        boundary_vertex_count += n;
        for (int i = 0; i < n; i++) {
            Vector3d base = asset.surface.evaluate(loop.uv_polyline[i].x(), loop.uv_polyline[i].y());
            double e = (base - loop.spatial_polyline[i]).norm();
            sum += e;
            sum2 += e * e;
            max_error = std::max(max_error, e);
        }
    }
    double mean = boundary_vertex_count > 0 ? sum / (double)boundary_vertex_count : 0.0;
    double rms = boundary_vertex_count > 0 ? std::sqrt(sum2 / (double)boundary_vertex_count) : 0.0;
    out << "{\n";
    out << "  \"format\": \"ABCBoundaryRibbonReport.v1\",\n";
    out << "  \"loop_count\": " << loop_count << ",\n";
    out << "  \"boundary_vertex_count\": " << boundary_vertex_count << ",\n";
    out << "  \"row_count\": " << std::max(2, row_count) << ",\n";
    out << "  \"ribbon_width_uv\": " << (ribbon_width > 0.0 ? ribbon_width : 0.06) << ",\n";
    out << "  \"base_boundary_mean_error\": " << mean << ",\n";
    out << "  \"base_boundary_rms_error\": " << rms << ",\n";
    out << "  \"base_boundary_max_error\": " << max_error << "\n";
    out << "}\n";
    return true;
}

static bool has_spatial_trim_data(const ReusableTrimmedBSplineSurface& asset) {
    for (const ReusableTrimLoop2D& loop : asset.trim_loops) {
        if (loop.spatial_polyline.size() == loop.uv_polyline.size() &&
            loop.spatial_polyline.size() >= 3) {
            return true;
        }
    }
    return false;
}

struct NearestBoundaryProjection {
    int loop_id = -1;
    int segment_id = -1;
    double segment_t = 0.0;
    double distance = std::numeric_limits<double>::infinity();
    Vector2d uv = Vector2d::Zero();
};

static NearestBoundaryProjection nearest_boundary_projection(
    const ReusableTrimmedBSplineSurface& asset,
    const Vector2d& uv) {
    NearestBoundaryProjection best;
    for (int li = 0; li < (int)asset.trim_loops.size(); li++) {
        const ReusableTrimLoop2D& loop = asset.trim_loops[li];
        int n = (int)loop.uv_polyline.size();
        if (n < 2 || (int)loop.spatial_polyline.size() != n) continue;
        for (int i = 0; i < n; i++) {
            const Vector2d& a = loop.uv_polyline[i];
            const Vector2d& b = loop.uv_polyline[(i + 1) % n];
            Vector2d ab = b - a;
            double len2 = ab.squaredNorm();
            double t = len2 > 1e-20 ? (uv - a).dot(ab) / len2 : 0.0;
            t = std::max(0.0, std::min(1.0, t));
            Vector2d q = a + t * ab;
            double d = (uv - q).norm();
            if (d >= best.distance) continue;
            best.loop_id = li;
            best.segment_id = i;
            best.segment_t = t;
            best.distance = d;
            best.uv = q;
        }
    }
    return best;
}

static const BoundaryRibbonSurface* find_ribbon_for_loop(
    const vector<BoundaryRibbonSurface>& ribbons,
    int loop_id) {
    for (const BoundaryRibbonSurface& ribbon : ribbons) {
        if (ribbon.valid && ribbon.loop_id == loop_id) return &ribbon;
    }
    return nullptr;
}

static Vector3d evaluate_abc_ribbon_preview(
    const ReusableTrimmedBSplineSurface& asset,
    const vector<BoundaryRibbonSurface>& ribbons,
    const Vector2d& uv,
    double ribbon_width) {
    Vector3d base = asset.surface.evaluate(uv.x(), uv.y());
    if (ribbon_width <= 1e-12 || ribbons.empty()) return base;

    NearestBoundaryProjection nearest = nearest_boundary_projection(asset, uv);
    if (nearest.loop_id < 0 ||
        !std::isfinite(nearest.distance) ||
        nearest.distance >= ribbon_width) {
        return base;
    }

    const BoundaryRibbonSurface* ribbon =
        find_ribbon_for_loop(ribbons, nearest.loop_id);
    if (!ribbon || ribbon->boundary_uv.empty()) return base;

    const int n = (int)ribbon->boundary_uv.size();
    const double ribbon_u =
        n > 0 ? ((double)nearest.segment_id + nearest.segment_t) / (double)n : 0.0;
    const double ribbon_v =
        std::max(0.0, std::min(1.0, nearest.distance / ribbon_width));
    const Vector3d ribbon_value = ribbon->surface.evaluate(ribbon_u, ribbon_v);

    const double smooth = ribbon_v * ribbon_v * (3.0 - 2.0 * ribbon_v);
    const double ribbon_weight = 1.0 - smooth;
    return (1.0 - ribbon_weight) * base + ribbon_weight * ribbon_value;
}

static bool build_cdt_trimmed_uv_mesh(
    const vector<ReusableTrimLoop2D>& loops,
    int sample_u,
    int sample_v,
    vector<Vector2d>& compact_uv,
    MatrixXi& F) {
    compact_uv.clear();
    F.resize(0, 3);

    vector<Vector2d> points;
    vector<std::pair<int, int>> constraints;
    std::map<std::pair<long long, long long>, int> key_to_index;
    std::set<std::pair<int, int>> constraint_edges;
    const double eps = 1e-10;

    for (const ReusableTrimLoop2D& loop : loops) {
        vector<int> ids;
        ids.reserve(loop.uv_polyline.size());
        for (const Vector2d& p : loop.uv_polyline) {
            int idx = add_uv_point(points, key_to_index, p, eps);
            if (idx < 0) continue;
            if (!ids.empty() && ids.back() == idx) continue;
            ids.push_back(idx);
        }
        if (ids.size() > 1 && ids.front() == ids.back()) ids.pop_back();
        if (ids.size() < 3) continue;
        for (int i = 0; i < (int)ids.size(); i++) {
            int a = ids[i];
            int b = ids[(i + 1) % ids.size()];
            if (a == b) continue;
            constraints.push_back({a, b});
            constraint_edges.insert(std::minmax(a, b));
        }
    }
    if (constraints.empty()) return false;

    int nu = std::max(8, sample_u);
    int nv = std::max(8, sample_v);
    const double grid_step = std::max(
        nu > 1 ? 1.0 / (double)(nu - 1) : 1.0,
        nv > 1 ? 1.0 / (double)(nv - 1) : 1.0);
    const double min_grid_boundary_distance = 0.60 * grid_step;
    const double min_aux_point_distance = 0.18 * grid_step;
    const double ring_distances[2] = {0.75 * grid_step, 1.50 * grid_step};
    for (const ReusableTrimLoop2D& loop : loops) {
        const int n = (int)loop.uv_polyline.size();
        if (n < 3) continue;

        vector<Vector2d> edge_normals(n, Vector2d::Zero());
        for (int i = 0; i < n; i++) {
            edge_normals[i] = safe_inward_normal(loops, loop, i);
        }
        vector<Vector2d> vertex_normals(n, Vector2d::Zero());
        for (int i = 0; i < n; i++) {
            Vector2d nn = edge_normals[(i - 1 + n) % n] + edge_normals[i];
            if (nn.norm() <= 1e-14) nn = edge_normals[i];
            if (nn.norm() > 1e-14) nn.normalize();
            vertex_normals[i] = nn;
        }

        for (int i = 0; i < n; i++) {
            const Vector2d& a = loop.uv_polyline[i];
            const Vector2d& b = loop.uv_polyline[(i + 1) % n];
            double len = (b - a).norm();
            if (len <= 1e-14) continue;
            int samples = std::max(1, (int)std::ceil(len / grid_step));
            for (int k = 0; k <= samples; k++) {
                double t = (double)k / (double)samples;
                Vector2d uv = (1.0 - t) * a + t * b;
                Vector2d normal =
                    (1.0 - t) * vertex_normals[i] +
                    t * vertex_normals[(i + 1) % n];
                if (normal.norm() <= 1e-14) normal = edge_normals[i];
                if (normal.norm() <= 1e-14) continue;
                normal.normalize();
                for (double distance : ring_distances) {
                    Vector2d ring_uv = move_uv_inside(loops, uv, normal, distance);
                    if (!point_inside_asset_trim_region(ring_uv, loops)) continue;
                    add_uv_point_if_separated(
                        points,
                        key_to_index,
                        ring_uv,
                        eps,
                        min_aux_point_distance);
                }
            }
        }
    }

    for (int j = 0; j < nv; j++) {
        for (int i = 0; i < nu; i++) {
            Vector2d uv((double)i / (double)(nu - 1),
                        (double)j / (double)(nv - 1));
            if (point_inside_asset_trim_region(uv, loops) &&
                distance_to_trim_loops(uv, loops) >= min_grid_boundary_distance) {
                add_uv_point_if_separated(
                    points,
                    key_to_index,
                    uv,
                    eps,
                    min_aux_point_distance);
            }
        }
    }

    vector<Vector3i> raw_faces;
    if (!cdt_triangulate_uv(points, constraints, raw_faces)) return false;

    auto edge_is_constraint = [&](int a, int b) {
        return constraint_edges.count(std::minmax(a, b)) > 0;
    };
    auto edge_crosses_trim_loop = [&](int a, int b) {
        const Vector2d& pa = points[a];
        const Vector2d& pb = points[b];
        for (const ReusableTrimLoop2D& loop : loops) {
            int n = (int)loop.uv_polyline.size();
            for (int i = 0; i < n; i++) {
                const Vector2d& c = loop.uv_polyline[i];
                const Vector2d& d = loop.uv_polyline[(i + 1) % n];
                if ((c - d).norm() <= eps) continue;
                if (segments_intersect_proper(pa, pb, c, d, eps)) return true;
            }
        }
        return false;
    };

    const double max_free_edge = 3.5 * grid_step;
    vector<Vector3i> kept;
    kept.reserve(raw_faces.size());
    for (const Vector3i& tri : raw_faces) {
        int ids[3] = {tri.x(), tri.y(), tri.z()};
        Vector2d centroid = (points[ids[0]] + points[ids[1]] + points[ids[2]]) / 3.0;
        if (!point_inside_asset_trim_region(centroid, loops)) continue;
        bool ok = true;
        for (int k = 0; k < 3 && ok; k++) {
            int a = ids[k];
            int b = ids[(k + 1) % 3];
            if (edge_is_constraint(a, b)) continue;
            if ((points[a] - points[b]).norm() > max_free_edge) ok = false;
            Vector2d mid = 0.5 * (points[a] + points[b]);
            if (!point_inside_asset_trim_region(mid, loops)) ok = false;
            if (edge_crosses_trim_loop(a, b)) ok = false;
        }
        if (ok) kept.push_back(tri);
    }
    if (kept.empty()) return false;

    vector<int> old_to_new(points.size(), -1);
    for (const Vector3i& tri : kept) {
        for (int k = 0; k < 3; k++) {
            int old = tri(k);
            if (old_to_new[old] >= 0) continue;
            old_to_new[old] = (int)compact_uv.size();
            compact_uv.push_back(points[old]);
        }
    }
    F.resize((int)kept.size(), 3);
    for (int i = 0; i < (int)kept.size(); i++) {
        F.row(i) << old_to_new[kept[i].x()],
                    old_to_new[kept[i].y()],
                    old_to_new[kept[i].z()];
    }
    return true;
}

static void write_vec2(std::ofstream& out, const Vector2d& p) {
    out << "[" << p.x() << "," << p.y() << "]";
}

static void write_vec3(std::ofstream& out, const Vector3d& p) {
    out << "[" << p.x() << "," << p.y() << "," << p.z() << "]";
}

static void write_bspline_surface_json(
    std::ofstream& out,
    const BSplineSurface3D& surface,
    const string& indent) {
    out << indent << "\"type\": \"tensor_product_bspline_surface_3d\",\n";
    out << indent << "\"degree_u\": " << surface.degree_u << ",\n";
    out << indent << "\"degree_v\": " << surface.degree_v << ",\n";
    out << indent << "\"knots_u\": [";
    for (int i = 0; i < (int)surface.knots_u.size(); i++) {
        if (i) out << ",";
        out << surface.knots_u[i];
    }
    out << "],\n";
    out << indent << "\"knots_v\": [";
    for (int i = 0; i < (int)surface.knots_v.size(); i++) {
        if (i) out << ",";
        out << surface.knots_v[i];
    }
    out << "],\n";
    out << indent << "\"control_count_u\": " << surface.control_grid.size() << ",\n";
    out << indent << "\"control_count_v\": "
        << (surface.control_grid.empty() ? 0 : surface.control_grid.front().size())
        << ",\n";
    out << indent << "\"control_grid\": [\n";
    for (int i = 0; i < (int)surface.control_grid.size(); i++) {
        out << indent << "  [";
        for (int j = 0; j < (int)surface.control_grid[i].size(); j++) {
            if (j) out << ",";
            write_vec3(out, surface.control_grid[i][j]);
        }
        out << "]" << (i + 1 < (int)surface.control_grid.size() ? "," : "") << "\n";
    }
    out << indent << "]";
}

} // namespace

ReusableTrimmedBSplineSurface make_reusable_trimmed_bspline_surface(
    const BSplineSurface3D& surface,
    const vector<TrimLoop2D>& normalized_trim_loops,
    const Vector2d& source_uv_min,
    const Vector2d& source_uv_max,
    int source_region_id,
    int trim_curve_control_count,
    double trim_curve_fairness_weight,
    const MatrixXd* source_vertices) {
    ReusableTrimmedBSplineSurface asset;
    asset.surface = surface;
    asset.source_uv_min = source_uv_min;
    asset.source_uv_max = source_uv_max;
    asset.source_region_id = source_region_id;
    asset.generator = "Vaitkus-Varady-style rectangular extension + tensor-product B-spline fit";

    bool has_perimeter = false;
    for (const TrimLoop2D& loop : normalized_trim_loops) {
        ReusableTrimLoop2D out_loop;
        out_loop.is_perimeter = loop.is_perimeter;
        out_loop.source_vertex_ids = loop.vertex_ids;
        out_loop.uv_polyline = cleaned_loop(loop.uv_polyline);
        if (source_vertices) {
            int count = std::min(
                (int)out_loop.uv_polyline.size(),
                (int)out_loop.source_vertex_ids.size());
            out_loop.spatial_polyline.reserve(count);
            for (int i = 0; i < count; i++) {
                int vid = out_loop.source_vertex_ids[i];
                if (vid >= 0 && vid < source_vertices->rows() &&
                    source_vertices->cols() >= 3) {
                    out_loop.spatial_polyline.push_back(
                        source_vertices->row(vid).transpose());
                }
            }
            if ((int)out_loop.spatial_polyline.size() !=
                (int)out_loop.uv_polyline.size()) {
                out_loop.spatial_polyline.clear();
            }
        }
        if (out_loop.uv_polyline.size() >= 4) {
            out_loop.fitted_curve = fit_trim_curve_2d_from_polyline(
                out_loop.uv_polyline,
                std::max(4, trim_curve_control_count),
                std::max(0.0, trim_curve_fairness_weight));
        }
        if (out_loop.is_perimeter) has_perimeter = true;
        if (!out_loop.uv_polyline.empty()) asset.trim_loops.push_back(out_loop);
    }

    if (surface.control_grid.empty() || asset.trim_loops.empty() || !has_perimeter) {
        asset.valid = false;
        asset.reason = "surface or perimeter trim loop is missing";
    } else {
        asset.valid = true;
        asset.reason = "ok";
    }
    return asset;
}

bool export_reusable_trimmed_bspline_surface_json(
    const string& filename,
    const ReusableTrimmedBSplineSurface& asset) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);

    out << "{\n";
    out << "  \"format\": \"ReusableTrimmedBSplineSurface.v1\",\n";
    out << "  \"valid\": " << (asset.valid ? "true" : "false") << ",\n";
    out << "  \"reason\": \"" << json_escape(asset.reason) << "\",\n";
    out << "  \"source_region_id\": " << asset.source_region_id << ",\n";
    out << "  \"generator\": \"" << json_escape(asset.generator) << "\",\n";
    out << "  \"uv_domain\": {\n";
    out << "    \"asset_min\": [0,0],\n";
    out << "    \"asset_max\": [1,1],\n";
    out << "    \"source_min\": ";
    write_vec2(out, asset.source_uv_min);
    out << ",\n    \"source_max\": ";
    write_vec2(out, asset.source_uv_max);
    out << "\n  },\n";

    out << "  \"surface\": {\n";
    write_bspline_surface_json(out, asset.surface, "    ");
    out << "\n";
    out << "  },\n";

    out << "  \"trim_loops\": [\n";
    for (int li = 0; li < (int)asset.trim_loops.size(); li++) {
        const ReusableTrimLoop2D& loop = asset.trim_loops[li];
        out << "    {\n";
        out << "      \"is_perimeter\": " << (loop.is_perimeter ? "true" : "false") << ",\n";
        out << "      \"source_vertex_ids\": [";
        for (int i = 0; i < (int)loop.source_vertex_ids.size(); i++) {
            if (i) out << ",";
            out << loop.source_vertex_ids[i];
        }
        out << "],\n";
        out << "      \"uv_polyline\": [";
        for (int i = 0; i < (int)loop.uv_polyline.size(); i++) {
            if (i) out << ",";
            write_vec2(out, loop.uv_polyline[i]);
        }
        out << "],\n";
        out << "      \"spatial_polyline\": [";
        for (int i = 0; i < (int)loop.spatial_polyline.size(); i++) {
            if (i) out << ",";
            write_vec3(out, loop.spatial_polyline[i]);
        }
        out << "],\n";
        out << "      \"fitted_trim_curve\": {\n";
        out << "        \"valid\": " << (loop.fitted_curve.valid ? "true" : "false") << ",\n";
        out << "        \"degree\": " << loop.fitted_curve.degree << ",\n";
        out << "        \"knots\": [";
        for (int i = 0; i < (int)loop.fitted_curve.knots.size(); i++) {
            if (i) out << ",";
            out << loop.fitted_curve.knots[i];
        }
        out << "],\n";
        out << "        \"control_points\": [";
        for (int i = 0; i < (int)loop.fitted_curve.control_points.size(); i++) {
            if (i) out << ",";
            write_vec2(out, loop.fitted_curve.control_points[i]);
        }
        out << "]\n";
        out << "      }\n";
        out << "    }" << (li + 1 < (int)asset.trim_loops.size() ? "," : "") << "\n";
    }
    out << "  ],\n";

    vector<BoundaryRibbonSurface> ribbons =
        build_g0_boundary_ribbon_surfaces(asset, 4, 0.06);
    out << "  \"boundary_ribbons\": [\n";
    int written = 0;
    for (int ri = 0; ri < (int)ribbons.size(); ri++) {
        const BoundaryRibbonSurface& ribbon = ribbons[ri];
        if (!ribbon.valid) continue;
        if (written > 0) out << ",\n";
        out << "    {\n";
        out << "      \"loop_id\": " << ribbon.loop_id << ",\n";
        out << "      \"ribbon_width_uv\": " << ribbon.ribbon_width_uv << ",\n";
        out << "      \"contact_order\": 0,\n";
        out << "      \"boundary_uv\": [";
        for (int i = 0; i < (int)ribbon.boundary_uv.size(); i++) {
            if (i) out << ",";
            write_vec2(out, ribbon.boundary_uv[i]);
        }
        out << "],\n";
        out << "      \"boundary_xyz\": [";
        for (int i = 0; i < (int)ribbon.boundary_xyz.size(); i++) {
            if (i) out << ",";
            write_vec3(out, ribbon.boundary_xyz[i]);
        }
        out << "],\n";
        out << "      \"surface\": {\n";
        write_bspline_surface_json(out, ribbon.surface, "        ");
        out << "\n";
        out << "      }\n";
        out << "    }";
        written++;
    }
    out << "\n  ]\n";
    out << "}\n";
    return true;
}

bool sample_reusable_trimmed_bspline_surface(
    const ReusableTrimmedBSplineSurface& asset,
    int sample_u,
    int sample_v,
    MatrixXd& V,
    MatrixXi& F,
    MatrixXd* UV) {
    V.resize(0, 3);
    F.resize(0, 3);
    if (UV) UV->resize(0, 2);
    if (!asset.valid || asset.trim_loops.empty()) return false;

    int nu = std::max(8, sample_u);
    int nv = std::max(8, sample_v);
    MatrixXd full_uv(nu * nv, 2);
    for (int j = 0; j < nv; j++) {
        for (int i = 0; i < nu; i++) {
            full_uv.row(j * nu + i) <<
                (double)i / (double)(nu - 1),
                (double)j / (double)(nv - 1);
        }
    }

    vector<Vector3i> full_faces;
    full_faces.reserve((nu - 1) * (nv - 1) * 2);
    for (int j = 0; j + 1 < nv; j++) {
        for (int i = 0; i + 1 < nu; i++) {
            int a = j * nu + i;
            int b = j * nu + i + 1;
            int c = (j + 1) * nu + i + 1;
            int d = (j + 1) * nu + i;
            Vector2d ca((i + 2.0 / 3.0) / (nu - 1),
                        (j + 1.0 / 3.0) / (nv - 1));
            Vector2d cb((i + 1.0 / 3.0) / (nu - 1),
                        (j + 2.0 / 3.0) / (nv - 1));
            if (point_inside_asset_trim_region(ca, asset.trim_loops)) {
                full_faces.push_back(Vector3i(a, b, c));
            }
            if (point_inside_asset_trim_region(cb, asset.trim_loops)) {
                full_faces.push_back(Vector3i(a, c, d));
            }
        }
    }
    if (full_faces.empty()) return false;

    vector<int> old_to_new(nu * nv, -1);
    vector<Vector2d> compact_uv;
    compact_uv.reserve(nu * nv);
    for (const Vector3i& tri : full_faces) {
        for (int k = 0; k < 3; k++) {
            int old = tri(k);
            if (old_to_new[old] >= 0) continue;
            old_to_new[old] = (int)compact_uv.size();
            compact_uv.push_back(full_uv.row(old).transpose());
        }
    }

    V.resize((int)compact_uv.size(), 3);
    MatrixXd uv_mat((int)compact_uv.size(), 2);
    for (int i = 0; i < (int)compact_uv.size(); i++) {
        uv_mat.row(i) = compact_uv[i].transpose();
        V.row(i) = asset.surface.evaluate(compact_uv[i].x(), compact_uv[i].y()).transpose();
    }
    F.resize((int)full_faces.size(), 3);
    for (int i = 0; i < (int)full_faces.size(); i++) {
        F.row(i) << old_to_new[full_faces[i].x()],
                    old_to_new[full_faces[i].y()],
                    old_to_new[full_faces[i].z()];
    }
    if (UV) *UV = uv_mat;
    return V.rows() > 0 && F.rows() > 0;
}

bool sample_abc_boundary_controlled_trimmed_surface(
    const ReusableTrimmedBSplineSurface& asset,
    int sample_u,
    int sample_v,
    double ribbon_width,
    MatrixXd& V,
    MatrixXi& F,
    MatrixXd* UV) {
    V.resize(0, 3);
    F.resize(0, 3);
    if (UV) UV->resize(0, 2);
    if (!asset.valid || !has_spatial_trim_data(asset)) return false;

    vector<Vector2d> compact_uv;
    if (!build_cdt_trimmed_uv_mesh(asset.trim_loops, sample_u, sample_v, compact_uv, F)) {
        return false;
    }

    MatrixXd uv_mat((int)compact_uv.size(), 2);
    V.resize((int)compact_uv.size(), 3);
    double width = ribbon_width > 0.0
                       ? ribbon_width
                       : 0.25 / (double)std::max(8, std::max(sample_u, sample_v));
    vector<BoundaryRibbonSurface> ribbons =
        build_g0_boundary_ribbon_surfaces(asset, 4, width);
    for (int i = 0; i < (int)compact_uv.size(); i++) {
        uv_mat.row(i) = compact_uv[i].transpose();
        V.row(i) = evaluate_abc_ribbon_preview(
            asset,
            ribbons,
            compact_uv[i],
            width).transpose();
    }
    if (UV) *UV = uv_mat;
    return V.rows() > 0 && F.rows() > 0;
}

vector<BoundaryRibbonSurface> build_g0_boundary_ribbon_surfaces(
    const ReusableTrimmedBSplineSurface& asset,
    int row_count,
    double ribbon_width) {
    vector<BoundaryRibbonSurface> ribbons;
    if (!asset.valid || asset.trim_loops.empty()) return ribbons;

    const int rows = std::max(2, row_count);
    const double width = ribbon_width > 0.0 ? ribbon_width : 0.06;
    const int degree_u = 1;
    const int degree_v = rows >= 4 ? 3 : 1;

    for (int li = 0; li < (int)asset.trim_loops.size(); li++) {
        const ReusableTrimLoop2D& loop = asset.trim_loops[li];
        const int n = (int)loop.uv_polyline.size();

        BoundaryRibbonSurface ribbon;
        ribbon.loop_id = li;
        ribbon.ribbon_width_uv = width;
        ribbon.boundary_uv = loop.uv_polyline;
        ribbon.boundary_xyz = loop.spatial_polyline;

        if (n < 3 || (int)loop.spatial_polyline.size() != n) {
            ribbon.valid = false;
            ribbon.reason = "trim loop has no matching authoritative 3D boundary";
            ribbons.push_back(ribbon);
            continue;
        }

        vector<Vector2d> edge_normals(n, Vector2d::Zero());
        for (int i = 0; i < n; i++) {
            edge_normals[i] = safe_inward_normal(asset.trim_loops, loop, i);
        }

        vector<Vector2d> vertex_normals(n, Vector2d::Zero());
        for (int i = 0; i < n; i++) {
            Vector2d nn = edge_normals[(i - 1 + n) % n] + edge_normals[i];
            if (nn.norm() <= 1e-14) nn = edge_normals[i];
            if (nn.norm() > 1e-14) nn.normalize();
            vertex_normals[i] = nn;
        }

        vector<vector<Vector3d>> control_grid(n + 1, vector<Vector3d>(rows));
        for (int i = 0; i <= n; i++) {
            const int src = i % n;
            const Vector2d& boundary_uv = loop.uv_polyline[src];
            const Vector3d& boundary_xyz = loop.spatial_polyline[src];

            for (int r = 0; r < rows; r++) {
                const double t = rows == 1 ? 0.0 : (double)r / (double)(rows - 1);
                Vector2d uv = move_uv_inside(
                    asset.trim_loops,
                    boundary_uv,
                    vertex_normals[src],
                    t * width);
                Vector3d base = asset.surface.evaluate(uv.x(), uv.y());
                control_grid[i][r] = (r == 0) ? boundary_xyz : base;
            }
        }

        try {
            ribbon.surface = BSplineSurface3D(
                degree_u,
                degree_v,
                make_open_uniform_knot_vector(n + 1, degree_u),
                make_open_uniform_knot_vector(rows, degree_v),
                control_grid);
            ribbon.valid = true;
            ribbon.reason = "ok";
        } catch (const std::exception& e) {
            ribbon.valid = false;
            ribbon.reason = e.what();
        }
        ribbons.push_back(ribbon);
    }

    return ribbons;
}

bool export_boundary_ribbon_surfaces_debug(
    const string& output_dir,
    const vector<BoundaryRibbonSurface>& ribbons,
    int sample_u_per_edge,
    int sample_v) {
    if (!ensure_dir(output_dir)) return false;

    MatrixXd aggregate_V(0, 3);
    MatrixXi aggregate_F(0, 3);
    int total_vertices = 0;
    int total_faces = 0;
    int valid_count = 0;
    double max_boundary_error = 0.0;
    bool ok = true;

    for (const BoundaryRibbonSurface& ribbon : ribbons) {
        if (!ribbon.valid) continue;
        valid_count++;
        const int edge_count = std::max(
            1,
            (int)ribbon.surface.control_grid.size() - 1);
        const int su = std::max(2, edge_count * std::max(1, sample_u_per_edge) + 1);
        const int sv = std::max(2, sample_v);

        MatrixXd V;
        MatrixXi F;
        sample_bspline_surface(ribbon.surface, su, sv, V, F);

        for (int i = 0; i < (int)ribbon.boundary_xyz.size(); i++) {
            double u = ribbon.boundary_xyz.size() <= 1
                           ? 0.0
                           : (double)i / (double)ribbon.boundary_xyz.size();
            Vector3d p = ribbon.surface.evaluate(u, 0.0);
            max_boundary_error = std::max(
                max_boundary_error,
                (p - ribbon.boundary_xyz[i]).norm());
        }

        std::ostringstream surface_name;
        surface_name << "abc_boundary_ribbon_surface_loop_" << ribbon.loop_id << ".obj";
        ok = write_mesh_obj(path_join(output_dir, surface_name.str()), V, F) && ok;

        std::ostringstream control_name;
        control_name << "abc_boundary_ribbon_surface_loop_" << ribbon.loop_id
                     << "_control_net.obj";
        ok = export_bspline_surface_control_net_obj(
                 path_join(output_dir, control_name.str()),
                 ribbon.surface) && ok;

        const int old_v = aggregate_V.rows();
        const int old_f = aggregate_F.rows();
        aggregate_V.conservativeResize(old_v + V.rows(), 3);
        aggregate_V.block(old_v, 0, V.rows(), 3) = V;
        aggregate_F.conservativeResize(old_f + F.rows(), 3);
        for (int f = 0; f < F.rows(); f++) {
            aggregate_F.row(old_f + f) =
                (F.row(f).array() + old_v).matrix();
        }
        total_vertices += V.rows();
        total_faces += F.rows();
    }

    if (aggregate_V.rows() > 0 && aggregate_F.rows() > 0) {
        ok = write_mesh_obj(
                 path_join(output_dir, "abc_boundary_ribbon_surfaces.obj"),
                 aggregate_V,
                 aggregate_F) && ok;
    } else {
        ok = false;
    }

    std::ofstream report(path_join(output_dir, "abc_boundary_ribbon_surfaces_report.json"));
    if (report.is_open()) {
        report.precision(17);
        report << "{\n";
        report << "  \"format\": \"ABCBoundaryRibbonSurfacesReport.v1\",\n";
        report << "  \"ribbon_count\": " << ribbons.size() << ",\n";
        report << "  \"valid_ribbon_count\": " << valid_count << ",\n";
        report << "  \"sampled_vertex_count\": " << total_vertices << ",\n";
        report << "  \"sampled_face_count\": " << total_faces << ",\n";
        report << "  \"max_boundary_control_error\": " << max_boundary_error << ",\n";
        report << "  \"ribbons\": [\n";
        for (int i = 0; i < (int)ribbons.size(); i++) {
            const BoundaryRibbonSurface& ribbon = ribbons[i];
            const int cu = ribbon.surface.control_grid.empty()
                               ? 0
                               : (int)ribbon.surface.control_grid.size();
            const int cv = ribbon.surface.control_grid.empty()
                               ? 0
                               : (int)ribbon.surface.control_grid.front().size();
            report << "    {\n";
            report << "      \"loop_id\": " << ribbon.loop_id << ",\n";
            report << "      \"valid\": " << (ribbon.valid ? "true" : "false") << ",\n";
            report << "      \"reason\": \"" << json_escape(ribbon.reason) << "\",\n";
            report << "      \"degree_u\": " << ribbon.surface.degree_u << ",\n";
            report << "      \"degree_v\": " << ribbon.surface.degree_v << ",\n";
            report << "      \"control_count_u\": " << cu << ",\n";
            report << "      \"control_count_v\": " << cv << ",\n";
            report << "      \"boundary_vertex_count\": " << ribbon.boundary_xyz.size() << "\n";
            report << "    }" << (i + 1 < (int)ribbons.size() ? "," : "") << "\n";
        }
        report << "  ]\n";
        report << "}\n";
    } else {
        ok = false;
    }

    return ok;
}

static bool write_abc_ribbon_preview_report_json(
    const string& filename,
    const ReusableTrimmedBSplineSurface& asset,
    int sample_u,
    int sample_v,
    double ribbon_width) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);

    const double width = ribbon_width > 0.0
                             ? ribbon_width
                             : 0.25 / (double)std::max(8, std::max(sample_u, sample_v));
    vector<BoundaryRibbonSurface> ribbons =
        build_g0_boundary_ribbon_surfaces(asset, 4, width);

    int valid_ribbons = 0;
    int boundary_count = 0;
    double boundary_sum = 0.0;
    double boundary_sum2 = 0.0;
    double boundary_max = 0.0;
    for (const BoundaryRibbonSurface& ribbon : ribbons) {
        if (!ribbon.valid) continue;
        valid_ribbons++;
        for (int i = 0; i < (int)ribbon.boundary_uv.size(); i++) {
            Vector3d p = evaluate_abc_ribbon_preview(
                asset,
                ribbons,
                ribbon.boundary_uv[i],
                width);
            double e = (p - ribbon.boundary_xyz[i]).norm();
            boundary_sum += e;
            boundary_sum2 += e * e;
            boundary_max = std::max(boundary_max, e);
            boundary_count++;
        }
    }
    double mean = boundary_count > 0 ? boundary_sum / (double)boundary_count : 0.0;
    double rms = boundary_count > 0 ? std::sqrt(boundary_sum2 / (double)boundary_count) : 0.0;

    out << "{\n";
    out << "  \"format\": \"ABCRibbonPreviewReport.v1\",\n";
    out << "  \"mode\": \"boundary-ribbon-surface-blend\",\n";
    out << "  \"sample_u\": " << sample_u << ",\n";
    out << "  \"sample_v\": " << sample_v << ",\n";
    out << "  \"ribbon_width_uv\": " << width << ",\n";
    out << "  \"ribbon_count\": " << ribbons.size() << ",\n";
    out << "  \"valid_ribbon_count\": " << valid_ribbons << ",\n";
    out << "  \"boundary_sample_count\": " << boundary_count << ",\n";
    out << "  \"boundary_mean_error\": " << mean << ",\n";
    out << "  \"boundary_rms_error\": " << rms << ",\n";
    out << "  \"boundary_max_error\": " << boundary_max << "\n";
    out << "}\n";
    return true;
}

struct ABCPreviewFaceDebug {
    int face_id = -1;
    double quality_3d = 0.0;
    double quality_uv = 0.0;
    double double_area_3d = 0.0;
    double double_area_uv = 0.0;
    double distance_to_trim = 0.0;
    double nearest_trim_vertex_distance = 0.0;
    int nearest_trim_loop = -1;
    int nearest_trim_segment = -1;
    int nearest_trim_vertex = -1;
    bool nearest_vertex_is_concave = false;
    bool spans_multiple_nearest_segments = false;
    double mean_base_delta = 0.0;
    double max_base_delta = 0.0;
    double base_delta_range = 0.0;
    Vector2d centroid_uv = Vector2d::Zero();
};

static double loop_signed_area(const vector<Vector2d>& polyline) {
    if (polyline.size() < 3) return 0.0;
    double area2 = 0.0;
    for (int i = 0; i < (int)polyline.size(); i++) {
        const Vector2d& a = polyline[i];
        const Vector2d& b = polyline[(i + 1) % (int)polyline.size()];
        area2 += a.x() * b.y() - a.y() * b.x();
    }
    return 0.5 * area2;
}

static double triangle_quality_from_edges(double a2, double b2, double c2, double double_area) {
    const double denom = a2 + b2 + c2;
    if (denom <= 1e-30) return 0.0;
    const double area = 0.5 * double_area;
    return 4.0 * std::sqrt(3.0) * area / denom;
}

static double triangle_quality_3d(
    const Vector3d& a,
    const Vector3d& b,
    const Vector3d& c,
    double* double_area_out) {
    const double double_area = ((b - a).cross(c - a)).norm();
    if (double_area_out) *double_area_out = double_area;
    return triangle_quality_from_edges(
        (b - a).squaredNorm(),
        (c - b).squaredNorm(),
        (a - c).squaredNorm(),
        double_area);
}

static double triangle_quality_2d(
    const Vector2d& a,
    const Vector2d& b,
    const Vector2d& c,
    double* double_area_out) {
    const double double_area = std::abs(orient2d(a, b, c));
    if (double_area_out) *double_area_out = double_area;
    return triangle_quality_from_edges(
        (b - a).squaredNorm(),
        (c - b).squaredNorm(),
        (a - c).squaredNorm(),
        double_area);
}

static double mean_uv_edge_length(const MatrixXd& UV, const MatrixXi& F) {
    if (UV.rows() == 0 || F.rows() == 0) return 0.02;
    double sum = 0.0;
    int count = 0;
    for (int f = 0; f < F.rows(); f++) {
        for (int k = 0; k < 3; k++) {
            int a = F(f, k);
            int b = F(f, (k + 1) % 3);
            if (a < 0 || b < 0 || a >= UV.rows() || b >= UV.rows()) continue;
            sum += (UV.row(a).transpose() - UV.row(b).transpose()).norm();
            count++;
        }
    }
    return count > 0 ? sum / (double)count : 0.02;
}

static bool trim_vertex_is_concave(
    const vector<ReusableTrimLoop2D>& loops,
    int loop_id,
    int vertex_id) {
    if (loop_id < 0 || loop_id >= (int)loops.size()) return false;
    const vector<Vector2d>& poly = loops[loop_id].uv_polyline;
    const int n = (int)poly.size();
    if (n < 3 || vertex_id < 0 || vertex_id >= n) return false;
    const Vector2d& prev = poly[(vertex_id - 1 + n) % n];
    const Vector2d& cur = poly[vertex_id];
    const Vector2d& next = poly[(vertex_id + 1) % n];
    const double turn = orient2d(prev, cur, next);
    const double area = loop_signed_area(poly);
    const double scale = std::max(1e-20, (prev - cur).norm() * (next - cur).norm());
    return (turn * area) < -1e-8 * scale * std::max(1e-12, std::abs(area));
}

static void nearest_trim_vertex(
    const vector<ReusableTrimLoop2D>& loops,
    const Vector2d& uv,
    int& loop_id,
    int& vertex_id,
    double& distance) {
    loop_id = -1;
    vertex_id = -1;
    distance = std::numeric_limits<double>::infinity();
    for (int li = 0; li < (int)loops.size(); li++) {
        const vector<Vector2d>& poly = loops[li].uv_polyline;
        for (int vi = 0; vi < (int)poly.size(); vi++) {
            double d = (uv - poly[vi]).norm();
            if (d >= distance) continue;
            loop_id = li;
            vertex_id = vi;
            distance = d;
        }
    }
}

static vector<int> sorted_unique_face_ids(vector<int> ids) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

static bool write_trim_corner_markers_uv_obj(
    const string& filename,
    const vector<ReusableTrimLoop2D>& loops,
    double marker_size) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    int offset = 1;
    for (int li = 0; li < (int)loops.size(); li++) {
        const vector<Vector2d>& poly = loops[li].uv_polyline;
        for (int vi = 0; vi < (int)poly.size(); vi++) {
            const Vector2d& p = poly[vi];
            const bool concave = trim_vertex_is_concave(loops, li, vi);
            const double s = concave ? 1.7 * marker_size : marker_size;
            out << "o trim_corner_loop_" << li << "_vertex_" << vi
                << (concave ? "_concave" : "") << "\n";
            out << "v " << p.x() - s << " " << p.y() << " 0\n";
            out << "v " << p.x() + s << " " << p.y() << " 0\n";
            out << "v " << p.x() << " " << p.y() - s << " 0\n";
            out << "v " << p.x() << " " << p.y() + s << " 0\n";
            out << "l " << offset << " " << offset + 1 << "\n";
            out << "l " << offset + 2 << " " << offset + 3 << "\n";
            offset += 4;
        }
    }
    return true;
}

static bool write_abc_preview_local_debug(
    const string& output_dir,
    const ReusableTrimmedBSplineSurface& asset,
    const MatrixXd& V,
    const MatrixXi& F,
    const MatrixXd& UV,
    int sample_u,
    int sample_v,
    double ribbon_width) {
    if (V.rows() == 0 || F.rows() == 0 || UV.rows() != V.rows()) return false;

    const double width = ribbon_width > 0.0
                             ? ribbon_width
                             : 0.25 / (double)std::max(8, std::max(sample_u, sample_v));
    const double avg_uv_edge = mean_uv_edge_length(UV, F);
    const double near_corner_radius = std::max(2.5 * avg_uv_edge, 1.25 * width);
    const double low_quality_threshold = 0.02;
    vector<ABCPreviewFaceDebug> records;
    records.reserve(F.rows());
    vector<int> low_quality_faces;
    vector<int> near_trim_corner_faces;
    vector<int> segment_transition_faces;

    for (int f = 0; f < F.rows(); f++) {
        const int ia = F(f, 0);
        const int ib = F(f, 1);
        const int ic = F(f, 2);
        if (ia < 0 || ib < 0 || ic < 0 ||
            ia >= V.rows() || ib >= V.rows() || ic >= V.rows()) {
            continue;
        }

        const Vector3d a = V.row(ia).transpose();
        const Vector3d b = V.row(ib).transpose();
        const Vector3d c = V.row(ic).transpose();
        const Vector2d auv = UV.row(ia).transpose();
        const Vector2d buv = UV.row(ib).transpose();
        const Vector2d cuv = UV.row(ic).transpose();
        ABCPreviewFaceDebug rec;
        rec.face_id = f;
        rec.quality_3d = triangle_quality_3d(a, b, c, &rec.double_area_3d);
        rec.quality_uv = triangle_quality_2d(auv, buv, cuv, &rec.double_area_uv);
        rec.centroid_uv = (auv + buv + cuv) / 3.0;
        NearestBoundaryProjection proj = nearest_boundary_projection(asset, rec.centroid_uv);
        rec.nearest_trim_loop = proj.loop_id;
        rec.nearest_trim_segment = proj.segment_id;
        rec.distance_to_trim = proj.distance;
        nearest_trim_vertex(
            asset.trim_loops,
            rec.centroid_uv,
            rec.nearest_trim_loop,
            rec.nearest_trim_vertex,
            rec.nearest_trim_vertex_distance);
        rec.nearest_vertex_is_concave = trim_vertex_is_concave(
            asset.trim_loops,
            rec.nearest_trim_loop,
            rec.nearest_trim_vertex);

        int vertex_segments[3] = {-1, -1, -1};
        double deltas[3] = {0.0, 0.0, 0.0};
        const int ids[3] = {ia, ib, ic};
        for (int k = 0; k < 3; k++) {
            const Vector2d uv = UV.row(ids[k]).transpose();
            NearestBoundaryProjection vp = nearest_boundary_projection(asset, uv);
            vertex_segments[k] = vp.segment_id;
            Vector3d base = asset.surface.evaluate(uv.x(), uv.y());
            deltas[k] = (V.row(ids[k]).transpose() - base).norm();
            rec.mean_base_delta += deltas[k] / 3.0;
            rec.max_base_delta = std::max(rec.max_base_delta, deltas[k]);
        }
        rec.base_delta_range =
            *std::max_element(deltas, deltas + 3) -
            *std::min_element(deltas, deltas + 3);
        rec.spans_multiple_nearest_segments =
            vertex_segments[0] != vertex_segments[1] ||
            vertex_segments[1] != vertex_segments[2];

        const bool low_quality =
            rec.quality_3d < low_quality_threshold ||
            rec.quality_uv < low_quality_threshold;
        const bool near_corner =
            rec.nearest_trim_vertex_distance <= near_corner_radius;
        const bool segment_transition =
            rec.spans_multiple_nearest_segments &&
            rec.distance_to_trim <= std::max(width, 2.0 * avg_uv_edge);

        if (low_quality) low_quality_faces.push_back(f);
        if (near_corner) near_trim_corner_faces.push_back(f);
        if (segment_transition) segment_transition_faces.push_back(f);
        if (low_quality || near_corner || segment_transition) records.push_back(rec);
    }

    std::sort(records.begin(), records.end(), [](const ABCPreviewFaceDebug& a, const ABCPreviewFaceDebug& b) {
        if (a.nearest_vertex_is_concave != b.nearest_vertex_is_concave) {
            return a.nearest_vertex_is_concave > b.nearest_vertex_is_concave;
        }
        if (std::abs(a.quality_3d - b.quality_3d) > 1e-14) {
            return a.quality_3d < b.quality_3d;
        }
        return a.nearest_trim_vertex_distance < b.nearest_trim_vertex_distance;
    });
    if ((int)records.size() > 256) records.resize(256);

    low_quality_faces = sorted_unique_face_ids(low_quality_faces);
    near_trim_corner_faces = sorted_unique_face_ids(near_trim_corner_faces);
    segment_transition_faces = sorted_unique_face_ids(segment_transition_faces);

    bool ok = true;
    ok = write_mesh_subset_obj(
             path_join(output_dir, "abc_preview_low_quality_faces.obj"),
             V,
             F,
             low_quality_faces) && ok;
    ok = write_uv_mesh_subset_obj(
             path_join(output_dir, "abc_preview_low_quality_faces_uv.obj"),
             UV,
             F,
             low_quality_faces) && ok;
    ok = write_mesh_subset_obj(
             path_join(output_dir, "abc_preview_near_trim_corner_faces.obj"),
             V,
             F,
             near_trim_corner_faces) && ok;
    ok = write_uv_mesh_subset_obj(
             path_join(output_dir, "abc_preview_near_trim_corner_faces_uv.obj"),
             UV,
             F,
             near_trim_corner_faces) && ok;
    ok = write_mesh_subset_obj(
             path_join(output_dir, "abc_preview_segment_transition_faces.obj"),
             V,
             F,
             segment_transition_faces) && ok;
    ok = write_uv_mesh_subset_obj(
             path_join(output_dir, "abc_preview_segment_transition_faces_uv.obj"),
             UV,
             F,
             segment_transition_faces) && ok;
    ok = write_trim_corner_markers_uv_obj(
             path_join(output_dir, "abc_preview_trim_corner_markers_uv.obj"),
             asset.trim_loops,
             std::max(0.25 * avg_uv_edge, 0.002)) && ok;

    std::ofstream out(path_join(output_dir, "abc_preview_local_debug.json"));
    if (!out.is_open()) return false;
    out.precision(17);
    out << "{\n";
    out << "  \"format\": \"ABCPreviewLocalDebug.v1\",\n";
    out << "  \"sample_u\": " << sample_u << ",\n";
    out << "  \"sample_v\": " << sample_v << ",\n";
    out << "  \"ribbon_width_uv\": " << width << ",\n";
    out << "  \"mean_uv_edge_length\": " << avg_uv_edge << ",\n";
    out << "  \"near_corner_radius\": " << near_corner_radius << ",\n";
    out << "  \"low_quality_face_count\": " << low_quality_faces.size() << ",\n";
    out << "  \"near_trim_corner_face_count\": " << near_trim_corner_faces.size() << ",\n";
    out << "  \"segment_transition_face_count\": " << segment_transition_faces.size() << ",\n";
    out << "  \"records\": [\n";
    for (int i = 0; i < (int)records.size(); i++) {
        const ABCPreviewFaceDebug& r = records[i];
        out << "    {\n";
        out << "      \"face_id\": " << r.face_id << ",\n";
        out << "      \"quality_3d\": " << r.quality_3d << ",\n";
        out << "      \"quality_uv\": " << r.quality_uv << ",\n";
        out << "      \"double_area_3d\": " << r.double_area_3d << ",\n";
        out << "      \"double_area_uv\": " << r.double_area_uv << ",\n";
        out << "      \"centroid_uv\": [" << r.centroid_uv.x() << ", " << r.centroid_uv.y() << "],\n";
        out << "      \"distance_to_trim\": " << r.distance_to_trim << ",\n";
        out << "      \"nearest_trim_loop\": " << r.nearest_trim_loop << ",\n";
        out << "      \"nearest_trim_segment\": " << r.nearest_trim_segment << ",\n";
        out << "      \"nearest_trim_vertex\": " << r.nearest_trim_vertex << ",\n";
        out << "      \"nearest_trim_vertex_distance\": " << r.nearest_trim_vertex_distance << ",\n";
        out << "      \"nearest_vertex_is_concave\": "
            << (r.nearest_vertex_is_concave ? "true" : "false") << ",\n";
        out << "      \"spans_multiple_nearest_segments\": "
            << (r.spans_multiple_nearest_segments ? "true" : "false") << ",\n";
        out << "      \"mean_base_delta\": " << r.mean_base_delta << ",\n";
        out << "      \"max_base_delta\": " << r.max_base_delta << ",\n";
        out << "      \"base_delta_range\": " << r.base_delta_range << "\n";
        out << "    }" << (i + 1 < (int)records.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return ok;
}

bool export_reusable_trimmed_bspline_surface_debug(
    const string& output_dir,
    const ReusableTrimmedBSplineSurface& asset,
    int sample_u,
    int sample_v) {
    if (!ensure_dir(output_dir)) return false;
    bool ok = export_reusable_trimmed_bspline_surface_json(
        path_join(output_dir, "trimmed_bspline_asset.json"), asset);
    ok = write_trim_loops_obj(
        path_join(output_dir, "trimmed_bspline_asset_trim_loops.obj"),
        asset.trim_loops) && ok;
    ok = write_surface_trim_loops_obj(
        path_join(output_dir, "trimmed_bspline_asset_surface_trim_loops.obj"),
        asset) && ok;
    ok = write_abc_boundary_ribbon_strips_obj(
        path_join(output_dir, "abc_boundary_ribbon_strips.obj"),
        asset, 4, 0.0) && ok;
    ok = write_abc_boundary_ribbon_report_json(
        path_join(output_dir, "abc_boundary_ribbon_report.json"),
        asset, 4, 0.0) && ok;
    vector<BoundaryRibbonSurface> ribbons =
        build_g0_boundary_ribbon_surfaces(asset, 4, 0.06);
    ok = export_boundary_ribbon_surfaces_debug(
        output_dir,
        ribbons,
        4,
        8) && ok;
    ok = export_bspline_surface_control_net_obj(
        path_join(output_dir, "trimmed_bspline_asset_control_net.obj"),
        asset.surface) && ok;

    MatrixXd V;
    MatrixXi F;
    MatrixXd UV;
    if (sample_reusable_trimmed_bspline_surface(asset, sample_u, sample_v, V, F, &UV)) {
        MeshValidationReport report = validate_trimmed_mesh(V, F, &UV);
        ok = write_mesh_obj(
            path_join(output_dir, "bspline_trimmed_surface_asset.obj"), V, F) && ok;
        ok = write_uv_mesh_obj(
            path_join(output_dir, "bspline_trimmed_surface_asset_uv.obj"), UV, F) && ok;
        ok = write_mesh_validation_json(
            path_join(output_dir, "trimmed_bspline_asset_mesh_validation.json"),
            report, &UV) && ok;
    } else {
        ok = false;
    }

    MatrixXd abcV;
    MatrixXi abcF;
    MatrixXd abcUV;
    if (sample_abc_boundary_controlled_trimmed_surface(
            asset, sample_u, sample_v, 0.0, abcV, abcF, &abcUV)) {
        MeshValidationReport abc_report = validate_trimmed_mesh(abcV, abcF, &abcUV);
        ok = write_mesh_obj(
            path_join(output_dir, "bspline_trimmed_surface_abc_preview.obj"),
            abcV, abcF) && ok;
        ok = write_uv_mesh_obj(
            path_join(output_dir, "bspline_trimmed_surface_abc_preview_uv.obj"),
            abcUV, abcF) && ok;
        ok = write_mesh_validation_json(
            path_join(output_dir, "trimmed_bspline_abc_preview_validation.json"),
            abc_report, &abcUV) && ok;
        ok = write_abc_ribbon_preview_report_json(
            path_join(output_dir, "abc_ribbon_preview_report.json"),
            asset,
            sample_u,
            sample_v,
            0.0) && ok;
        ok = write_abc_preview_local_debug(
            output_dir,
            asset,
            abcV,
            abcF,
            abcUV,
            sample_u,
            sample_v,
            0.0) && ok;
    }
    return ok;
}
