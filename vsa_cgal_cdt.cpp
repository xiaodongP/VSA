#include "vsa_cgal_cdt.h"

#ifdef VSA_USE_CGAL_CDT
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>

using Eigen::Vector2d;
using Eigen::Vector3i;
using std::map;
using std::pair;
using std::set;
using std::tuple;
using std::vector;

namespace {

static bool finite_vec2(const Vector2d& p) {
    return std::isfinite(p.x()) && std::isfinite(p.y());
}

static double tri_area2(const Vector2d& a, const Vector2d& b, const Vector2d& c) {
    return 0.5 * ((b.x() - a.x()) * (c.y() - a.y()) -
                  (b.y() - a.y()) * (c.x() - a.x()));
}

static void orient_ccw(const vector<Vector2d>& pts, Vector3i& tri) {
    if (tri_area2(pts[tri.x()], pts[tri.y()], pts[tri.z()]) < 0.0) {
        std::swap(tri.y(), tri.z());
    }
}

} // namespace
#endif

bool vsa_cgal_constrained_delaunay_2d(
    const vector<Vector2d>& input_points,
    int boundary_count,
    vector<Vector3i>& out_triangles) {
    out_triangles.clear();

#ifndef VSA_USE_CGAL_CDT
    (void)input_points;
    (void)boundary_count;
    return false;
#else
    if ((int)input_points.size() < 3 || boundary_count < 3) return false;

    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef CGAL::Triangulation_vertex_base_with_info_2<int, Kernel> VertexBase;
    typedef CGAL::Constrained_triangulation_face_base_2<Kernel> FaceBase;
    typedef CGAL::Triangulation_data_structure_2<VertexBase, FaceBase> Tds;
    typedef CGAL::No_constraint_intersection_tag IntersectionTag;
    typedef CGAL::Constrained_Delaunay_triangulation_2<
        Kernel, Tds, IntersectionTag> CDT;
    typedef CDT::Point Point;
    typedef CDT::Vertex_handle VertexHandle;

    vector<Vector2d> unique_points;
    vector<int> unique_to_local;
    vector<int> local_to_unique(input_points.size(), -1);

    double eps = 1e-10;
    for (const Vector2d& p : input_points) {
        if (!finite_vec2(p)) continue;
        eps = std::max(eps, 1e-10 * std::max(1.0, std::max(std::abs(p.x()), std::abs(p.y()))));
    }

    map<pair<long long, long long>, int> key_to_unique;
    auto key_for = [&](const Vector2d& p) {
        return std::make_pair((long long)std::llround(p.x() / eps),
                              (long long)std::llround(p.y() / eps));
    };

    for (int i = 0; i < (int)input_points.size(); i++) {
        const Vector2d& p = input_points[i];
        if (!finite_vec2(p)) continue;

        auto key = key_for(p);
        auto it = key_to_unique.find(key);
        if (it == key_to_unique.end()) {
            int uid = (int)unique_points.size();
            key_to_unique[key] = uid;
            unique_points.push_back(p);
            unique_to_local.push_back(i);
            local_to_unique[i] = uid;
        } else {
            local_to_unique[i] = it->second;
        }
    }

    if ((int)unique_points.size() < 3) return false;

    try {
        CDT cdt;
        vector<VertexHandle> handles(unique_points.size());
        for (int uid = 0; uid < (int)unique_points.size(); uid++) {
            const Vector2d& p = unique_points[uid];
            VertexHandle vh = cdt.insert(Point(p.x(), p.y()));
            vh->info() = uid;
            handles[uid] = vh;
        }

        for (int i = 0; i < boundary_count; i++) {
            int j = (i + 1) % boundary_count;
            int ui = local_to_unique[i];
            int uj = local_to_unique[j];
            if (ui < 0 || uj < 0 || ui == uj) continue;
            cdt.insert_constraint(handles[ui], handles[uj]);
        }

        set<tuple<int, int, int>> seen;
        for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
            int u0 = fit->vertex(0)->info();
            int u1 = fit->vertex(1)->info();
            int u2 = fit->vertex(2)->info();
            if (u0 < 0 || u1 < 0 || u2 < 0) continue;

            int a = unique_to_local[u0];
            int b = unique_to_local[u1];
            int c = unique_to_local[u2];
            if (a == b || b == c || c == a) continue;
            if (std::abs(tri_area2(input_points[a], input_points[b], input_points[c])) < 1e-14) {
                continue;
            }

            Vector3i tri(a, b, c);
            orient_ccw(input_points, tri);

            vector<int> ids = {tri.x(), tri.y(), tri.z()};
            std::sort(ids.begin(), ids.end());
            auto key = std::make_tuple(ids[0], ids[1], ids[2]);
            if (!seen.insert(key).second) continue;

            out_triangles.push_back(tri);
        }
    } catch (const std::exception&) {
        out_triangles.clear();
        return false;
    } catch (...) {
        out_triangles.clear();
        return false;
    }


    return !out_triangles.empty();
#endif
}
