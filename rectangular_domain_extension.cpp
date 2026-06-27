#include "rectangular_domain_extension.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <tuple>

#ifdef VSA_USE_CGAL_CDT
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#endif

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector2d;
using Eigen::Vector3i;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::tuple;
using std::vector;

namespace {

static bool finite_vec2(const Vector2d& p) {
    return std::isfinite(p.x()) && std::isfinite(p.y());
}

static double cross2(const Vector2d& a, const Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

static double signed_area2(const vector<Vector2d>& poly) {
    if (poly.size() < 3) return 0.0;
    double a = 0.0;
    for (int i = 0; i < (int)poly.size(); i++) {
        a += cross2(poly[i], poly[(i + 1) % poly.size()]);
    }
    return 0.5 * a;
}

static double tri_area(const Vector2d& a, const Vector2d& b, const Vector2d& c) {
    return 0.5 * cross2(b - a, c - a);
}

static bool point_on_segment(const Vector2d& p, const Vector2d& a, const Vector2d& b, double eps) {
    Vector2d ab = b - a;
    double len2 = ab.squaredNorm();
    if (len2 <= eps * eps) return (p - a).norm() <= eps;
    double t = (p - a).dot(ab) / len2;
    if (t < -eps || t > 1.0 + eps) return false;
    Vector2d q = a + std::max(0.0, std::min(1.0, t)) * ab;
    return (p - q).norm() <= eps;
}

static bool point_in_polygon_or_boundary(const Vector2d& p, const vector<Vector2d>& poly, double eps) {
    if (poly.size() < 3) return false;
    for (int i = 0; i < (int)poly.size(); i++) {
        if (point_on_segment(p, poly[i], poly[(i + 1) % poly.size()], eps)) return true;
    }
    bool inside = false;
    for (int i = 0, j = (int)poly.size() - 1; i < (int)poly.size(); j = i++) {
        const Vector2d& a = poly[j];
        const Vector2d& b = poly[i];
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

static vector<Vector2d> clean_loop(const vector<Vector2d>& input, double eps) {
    vector<Vector2d> out;
    for (const Vector2d& p : input) {
        if (!finite_vec2(p)) continue;
        if (!out.empty() && (p - out.back()).norm() <= eps) continue;
        out.push_back(p);
    }
    if (out.size() > 1 && (out.front() - out.back()).norm() <= eps) out.pop_back();
    vector<Vector2d> filtered;
    for (int i = 0; i < (int)out.size(); i++) {
        const Vector2d& prev = out[(i + (int)out.size() - 1) % out.size()];
        const Vector2d& cur = out[i];
        const Vector2d& next = out[(i + 1) % out.size()];
        if ((cur - prev).norm() <= eps || (next - cur).norm() <= eps) continue;
        filtered.push_back(cur);
    }
    return filtered.size() >= 3 ? filtered : out;
}

static pair<long long, long long> point_key(const Vector2d& p, double eps) {
    return std::make_pair((long long)std::llround(p.x() / eps),
                          (long long)std::llround(p.y() / eps));
}

static int add_unique_point(
    vector<Vector2d>& points,
    vector<bool>& original_mask,
    map<pair<long long, long long>, int>& key_to_index,
    const Vector2d& p,
    bool original,
    double eps) {
    auto key = point_key(p, eps);
    auto it = key_to_index.find(key);
    if (it != key_to_index.end()) {
        if (original) original_mask[it->second] = true;
        return it->second;
    }
    int idx = (int)points.size();
    key_to_index[key] = idx;
    points.push_back(p);
    original_mask.push_back(original);
    return idx;
}

static bool point_in_original_region(
    const Vector2d& p,
    const vector<TrimLoop2D>& loops,
    double eps) {
    const TrimLoop2D* outer = nullptr;
    for (const TrimLoop2D& loop : loops) {
        if (loop.is_perimeter) {
            outer = &loop;
            break;
        }
    }
    if (!outer) return false;
    if (!point_in_polygon_or_boundary(p, outer->uv_polyline, eps)) return false;
    for (const TrimLoop2D& loop : loops) {
        if (loop.is_perimeter) continue;
        if (point_in_polygon_or_boundary(p, loop.uv_polyline, eps)) return false;
    }
    return true;
}

static bool point_in_rectangle(const Vector2d& p, const Vector2d& mn, const Vector2d& mx, double eps) {
    return p.x() >= mn.x() - eps && p.x() <= mx.x() + eps &&
           p.y() >= mn.y() - eps && p.y() <= mx.y() + eps;
}

static bool cdt_triangulate(
    const vector<Vector2d>& points,
    const vector<pair<int, int>>& constraints,
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
            handles[i] = cdt.insert(Point(points[i].x(), points[i].y()));
            handles[i]->info() = i;
        }
        for (const auto& e : constraints) {
            if (e.first < 0 || e.second < 0 ||
                e.first >= (int)points.size() || e.second >= (int)points.size() ||
                e.first == e.second) {
                continue;
            }
            cdt.insert_constraint(handles[e.first], handles[e.second]);
        }
        set<tuple<int, int, int>> seen;
        for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
            int a = fit->vertex(0)->info();
            int b = fit->vertex(1)->info();
            int c = fit->vertex(2)->info();
            if (a == b || b == c || c == a) continue;
            if (std::abs(tri_area(points[a], points[b], points[c])) < 1e-14) continue;
            Vector3i tri(a, b, c);
            if (tri_area(points[tri.x()], points[tri.y()], points[tri.z()]) < 0.0) {
                std::swap(tri.y(), tri.z());
            }
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

static bool same_face_vertices(const Vector3i& a, const Vector3i& b) {
    vector<int> av = {a.x(), a.y(), a.z()};
    vector<int> bv = {b.x(), b.y(), b.z()};
    std::sort(av.begin(), av.end());
    std::sort(bv.begin(), bv.end());
    return av == bv;
}

static double polygon_abs_area(const vector<Vector2d>& poly) {
    return std::abs(signed_area2(poly));
}

static bool write_uv_obj(const string& filename, const MatrixXd& V, const MatrixXi& F) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    for (int i = 0; i < V.rows(); i++) {
        out << "v " << V(i, 0) << " " << V(i, 1) << " 0\n";
    }
    for (int i = 0; i < F.rows(); i++) {
        out << "f " << F(i, 0) + 1 << " " << F(i, 1) + 1 << " " << F(i, 2) + 1 << "\n";
    }
    return true;
}

} // namespace

RectangularDomainExtensionInput make_rectangular_extension_input(
    const BoundarySegmentationResult& boundary,
    const KktGlobalStitchingResult& parameterization) {
    RectangularDomainExtensionInput input;
    input.UV = parameterization.UV;
    input.region_vertex_ids = parameterization.region_vertex_ids;
    input.local_faces = parameterization.local_faces;
    input.local_face_to_global_face = parameterization.local_face_to_global_face;
    for (const AuthoritativeBoundaryLoop& loop : boundary.loops) {
        TrimLoop2D trim;
        trim.is_perimeter = loop.is_perimeter;
        trim.vertex_ids = loop.vertex_ids;
        for (int vid : loop.vertex_ids) {
            if (vid >= 0 && vid < input.UV.rows()) {
                trim.uv_polyline.push_back(input.UV.row(vid).transpose());
            }
        }
        input.trim_loops.push_back(trim);
    }
    return input;
}

RectangularDomainExtensionResult build_rectangular_domain_extension(
    const RectangularDomainExtensionInput& input,
    const RectangularDomainExtensionConfig& config) {
    RectangularDomainExtensionResult result;
    if (input.UV.cols() != 2) {
        result.reason = "UV must have two columns";
        return result;
    }
    if (input.local_faces.cols() != 3) {
        result.reason = "local_faces must have three columns";
        return result;
    }
    if (input.region_vertex_ids.empty() || input.local_faces.rows() == 0) {
        result.reason = "input region mesh is empty";
        return result;
    }

    result.authoritative_trim_loops = input.trim_loops;
    for (TrimLoop2D& loop : result.authoritative_trim_loops) {
        loop.uv_polyline = clean_loop(loop.uv_polyline, config.min_constraint_edge_length);
        if (loop.uv_polyline.size() < 3) {
            result.reason = "trim loop has fewer than three usable vertices";
            return result;
        }
    }

    const TrimLoop2D* perimeter = nullptr;
    for (const TrimLoop2D& loop : result.authoritative_trim_loops) {
        if (loop.is_perimeter) {
            perimeter = &loop;
            break;
        }
    }
    if (!perimeter) {
        result.reason = "missing perimeter trim loop";
        return result;
    }

    Vector2d mn(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
    Vector2d mx(-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity());
    for (const Vector2d& p : perimeter->uv_polyline) {
        mn.x() = std::min(mn.x(), p.x());
        mn.y() = std::min(mn.y(), p.y());
        mx.x() = std::max(mx.x(), p.x());
        mx.y() = std::max(mx.y(), p.y());
    }
    Vector2d extent = mx - mn;
    if (extent.x() <= 0.0 || extent.y() <= 0.0) {
        result.reason = "perimeter trim loop has degenerate rectangle bounds";
        return result;
    }
    double margin = std::max(0.0, config.margin);
    mn -= margin * extent;
    mx += margin * extent;
    result.rectangle_min = mn;
    result.rectangle_max = mx;
    extent = mx - mn;

    double original_area = polygon_abs_area(perimeter->uv_polyline);
    for (const TrimLoop2D& loop : result.authoritative_trim_loops) {
        if (!loop.is_perimeter) original_area -= polygon_abs_area(loop.uv_polyline);
    }
    original_area = std::max(original_area, 1e-12);
    double rect_area = extent.x() * extent.y();
    double base = std::sqrt((double)input.region_vertex_ids.size() * rect_area / original_area);
    int auto_u = (int)std::ceil(base * std::sqrt(std::max(1e-12, extent.x() / extent.y())));
    int auto_v = (int)std::ceil(base * std::sqrt(std::max(1e-12, extent.y() / extent.x())));
    result.grid_sample_u = config.sample_count_u > 0 ? config.sample_count_u : auto_u;
    result.grid_sample_v = config.sample_count_v > 0 ? config.sample_count_v : auto_v;
    result.grid_sample_u = std::max(config.min_sample_count, std::min(config.max_sample_count, result.grid_sample_u));
    result.grid_sample_v = std::max(config.min_sample_count, std::min(config.max_sample_count, result.grid_sample_v));

    vector<Vector2d> points;
    vector<bool> original_vertex_mask;
    map<pair<long long, long long>, int> key_to_index;
    map<int, int> global_to_output;
    double eps = std::max(1e-12, config.dedup_epsilon);
    for (int gid : input.region_vertex_ids) {
        if (gid < 0 || gid >= input.UV.rows()) {
            result.reason = "region vertex id outside UV rows";
            return result;
        }
        int idx = add_unique_point(points, original_vertex_mask, key_to_index,
                                   input.UV.row(gid).transpose(), true, eps);
        global_to_output[gid] = idx;
    }

    vector<pair<int, int>> constraints;
    auto add_constraint_loop = [&](const vector<Vector2d>& loop) {
        vector<int> ids;
        ids.reserve(loop.size());
        for (const Vector2d& p : loop) {
            ids.push_back(add_unique_point(points, original_vertex_mask, key_to_index, p, false, eps));
        }
        for (int i = 0; i < (int)ids.size(); i++) {
            int a = ids[i];
            int b = ids[(i + 1) % ids.size()];
            if (a == b) continue;
            if ((points[a] - points[b]).norm() < config.min_constraint_edge_length) continue;
            constraints.push_back({a, b});
        }
    };

    vector<Vector2d> rectangle = {
        Vector2d(mn.x(), mn.y()),
        Vector2d(mx.x(), mn.y()),
        Vector2d(mx.x(), mx.y()),
        Vector2d(mn.x(), mx.y())};
    add_constraint_loop(rectangle);
    for (const TrimLoop2D& loop : result.authoritative_trim_loops) {
        add_constraint_loop(loop.uv_polyline);
    }

    for (int iu = 0; iu < result.grid_sample_u; iu++) {
        double u = result.grid_sample_u == 1 ? 0.5 : (double)iu / (double)(result.grid_sample_u - 1);
        for (int iv = 0; iv < result.grid_sample_v; iv++) {
            double v = result.grid_sample_v == 1 ? 0.5 : (double)iv / (double)(result.grid_sample_v - 1);
            Vector2d p(mn.x() + u * extent.x(), mn.y() + v * extent.y());
            add_unique_point(points, original_vertex_mask, key_to_index, p, false, eps);
        }
    }

    vector<Vector3i> cdt_faces;
    if (!cdt_triangulate(points, constraints, cdt_faces)) {
        result.reason = "CGAL constrained triangulation failed";
        return result;
    }

    vector<Vector3i> full_faces;
    vector<bool> full_face_original_mask;
    for (const Vector3i& tri : cdt_faces) {
        Vector2d centroid = (points[tri.x()] + points[tri.y()] + points[tri.z()]) / 3.0;
        if (!point_in_rectangle(centroid, mn, mx, eps)) continue;
        full_faces.push_back(tri);
        full_face_original_mask.push_back(
            point_in_original_region(centroid, result.authoritative_trim_loops, eps));
    }

    if (full_faces.empty()) {
        result.reason = "CDT generated no rectangle triangles";
        return result;
    }

    vector<int> old_to_new(points.size(), -1);
    vector<Vector2d> compact_points;
    vector<bool> compact_original_mask;
    for (const Vector3i& tri : full_faces) {
        for (int k = 0; k < 3; k++) {
            int old = tri(k);
            if (old_to_new[old] >= 0) continue;
            old_to_new[old] = (int)compact_points.size();
            compact_points.push_back(points[old]);
            compact_original_mask.push_back(original_vertex_mask[old]);
        }
    }

    result.full_uv_vertices.resize((int)compact_points.size(), 2);
    for (int i = 0; i < (int)compact_points.size(); i++) {
        result.full_uv_vertices.row(i) = compact_points[i].transpose();
    }
    result.full_faces.resize((int)full_faces.size(), 3);
    result.original_face_mask.clear();
    result.original_face_count = 0;
    result.artificial_face_count = 0;
    for (int i = 0; i < (int)full_faces.size(); i++) {
        Vector3i tri = full_faces[i];
        result.full_faces.row(i) << old_to_new[tri.x()], old_to_new[tri.y()], old_to_new[tri.z()];
        bool is_original = full_face_original_mask[i];
        result.original_face_mask.push_back(is_original);
        if (is_original) result.original_face_count++;
        else result.artificial_face_count++;
    }
    result.original_vertex_mask = compact_original_mask;
    result.original_vertex_count = 0;
    for (bool b : result.original_vertex_mask) {
        if (b) result.original_vertex_count++;
    }
    result.artificial_vertex_count = (int)result.original_vertex_mask.size() - result.original_vertex_count;
    if (result.artificial_face_count == 0) {
        result.reason = "no artificial extension triangles were generated";
        return result;
    }
    result.valid = true;
    result.reason = "ok";
    if (config.export_debug) export_rectangular_domain_extension_debug(config.debug_prefix, result);
    return result;
}

bool export_rectangular_domain_extension_debug(
    const string& prefix,
    const RectangularDomainExtensionResult& result) {
    bool ok = true;
    ok = write_uv_obj(prefix + "_full_uv.obj", result.full_uv_vertices, result.full_faces) && ok;
    {
        std::ofstream out(prefix + "_vertex_mask.csv");
        ok = out.is_open() && ok;
        out << "vertex_id,is_original\n";
        for (int i = 0; i < (int)result.original_vertex_mask.size(); i++) {
            out << i << "," << (result.original_vertex_mask[i] ? 1 : 0) << "\n";
        }
    }
    {
        std::ofstream out(prefix + "_face_mask.csv");
        ok = out.is_open() && ok;
        out << "face_id,is_original\n";
        for (int i = 0; i < (int)result.original_face_mask.size(); i++) {
            out << i << "," << (result.original_face_mask[i] ? 1 : 0) << "\n";
        }
    }
    {
        std::ofstream out(prefix + "_trim_loops.obj");
        ok = out.is_open() && ok;
        int cursor = 1;
        for (int li = 0; li < (int)result.authoritative_trim_loops.size(); li++) {
            const TrimLoop2D& loop = result.authoritative_trim_loops[li];
            out << "g trim_loop_" << li << (loop.is_perimeter ? "_perimeter" : "_inner") << "\n";
            int start = cursor;
            for (const Vector2d& p : loop.uv_polyline) {
                out << "v " << p.x() << " " << p.y() << " 0\n";
                cursor++;
            }
            for (int i = 0; i < (int)loop.uv_polyline.size(); i++) {
                out << "l " << (start + i) << " " << (start + ((i + 1) % loop.uv_polyline.size())) << "\n";
            }
        }
    }
    {
        std::ofstream out(prefix + "_summary.csv");
        ok = out.is_open() && ok;
        out << "valid,reason,rect_min_u,rect_min_v,rect_max_u,rect_max_v,grid_u,grid_v,original_vertices,artificial_vertices,original_faces,artificial_faces\n";
        out << (result.valid ? 1 : 0) << "," << result.reason << ","
            << result.rectangle_min.x() << "," << result.rectangle_min.y() << ","
            << result.rectangle_max.x() << "," << result.rectangle_max.y() << ","
            << result.grid_sample_u << "," << result.grid_sample_v << ","
            << result.original_vertex_count << "," << result.artificial_vertex_count << ","
            << result.original_face_count << "," << result.artificial_face_count << "\n";
    }
    return ok;
}
