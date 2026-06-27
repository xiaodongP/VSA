#include "trimmed_bspline_pipeline.h"

#include "bezier_guiding_frame.h"
#include "constrained_arap_parameterization.h"
#include "polyharmonic_3d_extension.h"
#include "quad_like_boundary.h"
#include "rectangular_domain_extension.h"
#include "rotation_angle_parameterization.h"
#include "reusable_trimmed_bspline_surface.h"
#include "trimmed_labeling.h"
#include "trimmed_mesh_validation.h"
#include "trimmed_region_input.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
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
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector3i;
using std::string;
using std::vector;

namespace {

static string path_join(const string& a, const string& b) {
    if (a.empty()) return b;
    char last = a.back();
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

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

static string json_escape(const string& s) {
    std::ostringstream out;
    for (char c : s) {
        if (c == '"' || c == '\\') out << "\\" << c;
        else if (c == '\n') out << "\\n";
        else out << c;
    }
    return out.str();
}

static bool write_mesh_obj(const string& filename, const MatrixXd& V, const MatrixXi& F) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    for (int i = 0; i < V.rows(); i++) {
        out << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
    }
    for (int i = 0; i < F.rows(); i++) {
        out << "f " << F(i, 0) + 1 << " " << F(i, 1) + 1 << " " << F(i, 2) + 1 << "\n";
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
        out << "f " << F(i, 0) + 1 << " " << F(i, 1) + 1 << " " << F(i, 2) + 1 << "\n";
    }
    return true;
}

static bool write_guiding_frame_obj(const string& filename, const BezierGuidingFrameResult& frame) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    int base = 1;
    for (int ci = 0; ci < (int)frame.curves.size(); ci++) {
        out << "g frame_curve_" << ci << "\n";
        vector<Vector3d> samples = frame.curves[ci].sample(80);
        for (const Vector3d& p : samples) out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
        for (int i = 1; i < (int)samples.size(); i++) out << "l " << base + i - 1 << " " << base + i << "\n";
        base += (int)samples.size();
    }
    return true;
}

static bool write_trim_loops_obj(const string& filename, const vector<TrimLoop2D>& loops) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    int base = 1;
    for (int li = 0; li < (int)loops.size(); li++) {
        out << "g trim_loop_" << li << (loops[li].is_perimeter ? "_perimeter" : "_inner") << "\n";
        for (const Vector2d& p : loops[li].uv_polyline) out << "v " << p.x() << " " << p.y() << " 0\n";
        int n = (int)loops[li].uv_polyline.size();
        for (int i = 0; i < n; i++) out << "l " << base + i << " " << base + ((i + 1) % n) << "\n";
        base += n;
    }
    return true;
}

static bool point_on_segment(const Vector2d& p, const Vector2d& a, const Vector2d& b, double eps) {
    Vector2d ab = b - a;
    double len2 = ab.squaredNorm();
    if (len2 <= eps * eps) return (p - a).norm() <= eps;
    double t = (p - a).dot(ab) / len2;
    if (t < -eps || t > 1.0 + eps) return false;
    return (p - (a + std::max(0.0, std::min(1.0, t)) * ab)).norm() <= eps;
}

static bool point_in_polygon_or_on_boundary(const Vector2d& p, const vector<Vector2d>& poly) {
    if (poly.size() < 3) return false;
    bool inside = false;
    double eps = 1e-10;
    for (int i = 0, j = (int)poly.size() - 1; i < (int)poly.size(); j = i++) {
        const Vector2d& a = poly[j];
        const Vector2d& b = poly[i];
        if (point_on_segment(p, a, b, eps)) return true;
        bool crosses = ((a.y() > p.y()) != (b.y() > p.y()));
        if (crosses) {
            double x = (b.x() - a.x()) * (p.y() - a.y()) / (b.y() - a.y()) + a.x();
            if (p.x() < x) inside = !inside;
        }
    }
    return inside;
}

static vector<TrimLoop2D> normalized_trim_loops(
    const vector<TrimLoop2D>& loops,
    const Vector2d& mn,
    const Vector2d& mx) {
    vector<TrimLoop2D> out = loops;
    Vector2d d = mx - mn;
    for (TrimLoop2D& loop : out) {
        for (Vector2d& p : loop.uv_polyline) {
            p.x() = d.x() > 1e-14 ? (p.x() - mn.x()) / d.x() : 0.0;
            p.y() = d.y() > 1e-14 ? (p.y() - mn.y()) / d.y() : 0.0;
            p = p.cwiseMax(Vector2d::Zero()).cwiseMin(Vector2d::Ones());
        }
    }
    return out;
}

static bool point_inside_trim_region(const Vector2d& p, const vector<TrimLoop2D>& loops) {
    const TrimLoop2D* perimeter = nullptr;
    for (const TrimLoop2D& loop : loops) {
        if (loop.is_perimeter) {
            perimeter = &loop;
            break;
        }
    }
    if (!perimeter || !point_in_polygon_or_on_boundary(p, perimeter->uv_polyline)) return false;
    for (const TrimLoop2D& loop : loops) {
        if (loop.is_perimeter) continue;
        if (point_in_polygon_or_on_boundary(p, loop.uv_polyline)) return false;
    }
    return true;
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

static bool sample_trimmed_surface_regular_grid(
    const BSplineSurface3D& surface,
    const vector<TrimLoop2D>& normalized_loops,
    int su,
    int sv,
    MatrixXd& V,
    MatrixXi& F) {
    int nu = std::max(2, su);
    int nv = std::max(2, sv);
    MatrixXd gridV(nu * nv, 3);
    for (int j = 0; j < nv; j++) {
        for (int i = 0; i < nu; i++) {
            double u = (double)i / (double)(nu - 1);
            double v = (double)j / (double)(nv - 1);
            gridV.row(j * nu + i) = surface.evaluate(u, v).transpose();
        }
    }
    vector<Eigen::Vector3i> faces;
    for (int j = 0; j + 1 < nv; j++) {
        for (int i = 0; i + 1 < nu; i++) {
            int a = j * nu + i;
            int b = j * nu + i + 1;
            int c = (j + 1) * nu + i + 1;
            int d = (j + 1) * nu + i;
            Vector2d ca((i + 2.0 / 3.0) / (nu - 1), (j + 1.0 / 3.0) / (nv - 1));
            Vector2d cb((i + 1.0 / 3.0) / (nu - 1), (j + 2.0 / 3.0) / (nv - 1));
            if (point_inside_trim_region(ca, normalized_loops)) faces.push_back(Eigen::Vector3i(a, b, c));
            if (point_inside_trim_region(cb, normalized_loops)) faces.push_back(Eigen::Vector3i(a, c, d));
        }
    }
    V = gridV;
    F.resize((int)faces.size(), 3);
    for (int i = 0; i < (int)faces.size(); i++) F.row(i) = faces[i].transpose();
    return !faces.empty();
}

struct LockedTrimPoint {
    Vector2d uv = Vector2d::Zero();
    Vector3d position = Vector3d::Zero();
    bool locked = false;
};

static bool finite_vec2(const Vector2d& p) {
    return std::isfinite(p.x()) && std::isfinite(p.y());
}

static std::pair<long long, long long> trim_point_key(const Vector2d& p, double eps) {
    return std::make_pair(
        (long long)std::llround(p.x() / eps),
        (long long)std::llround(p.y() / eps));
}

static int add_locked_trim_point(
    vector<LockedTrimPoint>& points,
    std::map<std::pair<long long, long long>, int>& key_to_index,
    const Vector2d& uv,
    const Vector3d& position,
    bool locked,
    double eps) {
    if (!finite_vec2(uv)) return -1;
    auto key = trim_point_key(uv, eps);
    auto it = key_to_index.find(key);
    if (it != key_to_index.end()) {
        int idx = it->second;
        if (locked) {
            points[idx].position = position;
            points[idx].locked = true;
        }
        return idx;
    }
    int idx = (int)points.size();
    key_to_index[key] = idx;
    LockedTrimPoint p;
    p.uv = uv;
    p.position = position;
    p.locked = locked;
    points.push_back(p);
    return idx;
}

static bool cdt_triangulate_locked_trim_points(
    const vector<LockedTrimPoint>& points,
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
            VertexHandle vh = cdt.insert(Point(points[i].uv.x(), points[i].uv.y()));
            vh->info() = i;
            handles[i] = vh;
        }
        for (const auto& e : constraints) {
            if (e.first < 0 || e.second < 0 ||
                e.first >= (int)handles.size() || e.second >= (int)handles.size() ||
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
            double area = (points[b].uv.x() - points[a].uv.x()) *
                          (points[c].uv.y() - points[a].uv.y()) -
                          (points[b].uv.y() - points[a].uv.y()) *
                          (points[c].uv.x() - points[a].uv.x());
            if (std::abs(area) < 1e-14) continue;
            Vector3i tri(a, b, c);
            if (area < 0.0) std::swap(tri.y(), tri.z());
            vector<int> ids = {tri.x(), tri.y(), tri.z()};
            std::sort(ids.begin(), ids.end());
            auto key = std::make_tuple(ids[0], ids[1], ids[2]);
            if (!seen.insert(key).second) continue;
            triangles.push_back(tri);
        }
    } catch (const std::exception&) {
        triangles.clear();
        return false;
    } catch (...) {
        triangles.clear();
        return false;
    }
    return !triangles.empty();
#endif
}

static bool sample_trimmed_surface_with_authoritative_boundary(
    const BSplineSurface3D& surface,
    const vector<TrimLoop2D>& normalized_loops,
    const MatrixXd& original_V,
    int su,
    int sv,
    MatrixXd& V,
    MatrixXi& F,
    MatrixXd* UV = nullptr,
    bool apply_harmonic_boundary_correction = true) {
    V.resize(0, 3);
    F.resize(0, 3);
    if (UV) UV->resize(0, 2);

    vector<LockedTrimPoint> points;
    std::map<std::pair<long long, long long>, int> key_to_index;
    vector<std::pair<int, int>> constraints;
    std::set<std::pair<int, int>> constraint_edges;
    double eps = 1e-10;
    double boundary_length_sum = 0.0;
    int boundary_edge_count = 0;

    for (const TrimLoop2D& loop : normalized_loops) {
        vector<int> loop_indices;
        loop_indices.reserve(loop.uv_polyline.size());
        for (int i = 0; i < (int)loop.uv_polyline.size(); i++) {
            Vector2d uv = loop.uv_polyline[i];
            if (!finite_vec2(uv)) continue;
            Vector3d position = surface.evaluate(uv.x(), uv.y());
            bool locked = false;
            if (i < (int)loop.vertex_ids.size()) {
                int vid = loop.vertex_ids[i];
                if (vid >= 0 && vid < original_V.rows()) {
                    position = original_V.row(vid).transpose();
                    locked = true;
                }
            }
            if (!loop_indices.empty() &&
                (uv - points[loop_indices.back()].uv).norm() <= eps) {
                if (locked) {
                    points[loop_indices.back()].position = position;
                    points[loop_indices.back()].locked = true;
                }
                continue;
            }
            int idx = add_locked_trim_point(points, key_to_index, uv, position, locked, eps);
            if (idx >= 0) loop_indices.push_back(idx);
        }
        if (loop_indices.size() > 1 &&
            (points[loop_indices.front()].uv - points[loop_indices.back()].uv).norm() <= eps) {
            loop_indices.pop_back();
        }
        if (loop_indices.size() < 3) continue;
        for (int i = 0; i < (int)loop_indices.size(); i++) {
            int a = loop_indices[i];
            int b = loop_indices[(i + 1) % loop_indices.size()];
            if (a == b) continue;
            if ((points[a].uv - points[b].uv).norm() <= eps) continue;
            constraints.push_back({a, b});
            constraint_edges.insert(std::minmax(a, b));
            boundary_length_sum += (points[a].uv - points[b].uv).norm();
            boundary_edge_count++;
        }
    }

    if (constraints.empty()) return false;

    int nu = std::max(2, su);
    int nv = std::max(2, sv);
    for (int j = 0; j < nv; j++) {
        for (int i = 0; i < nu; i++) {
            double u = (double)i / (double)(nu - 1);
            double v = (double)j / (double)(nv - 1);
            Vector2d uv(u, v);
            if (!point_inside_trim_region(uv, normalized_loops)) continue;
            add_locked_trim_point(
                points, key_to_index, uv, surface.evaluate(u, v), false, eps);
        }
    }

    vector<Vector3i> raw_faces;
    if (!cdt_triangulate_locked_trim_points(points, constraints, raw_faces)) {
        return false;
    }

    const double grid_step = std::max(
        nu > 1 ? 1.0 / (double)(nu - 1) : 1.0,
        nv > 1 ? 1.0 / (double)(nv - 1) : 1.0);
    const double mean_boundary_step =
        boundary_edge_count > 0 ? boundary_length_sum / (double)boundary_edge_count : grid_step;
    const double max_free_edge_length =
        std::max(3.0 * grid_step, 3.0 * mean_boundary_step);

    auto edge_is_constraint = [&](int a, int b) {
        return constraint_edges.count(std::minmax(a, b)) > 0;
    };
    auto edge_crosses_trim_loop = [&](int a, int b) {
        const Vector2d& pa = points[a].uv;
        const Vector2d& pb = points[b].uv;
        for (const TrimLoop2D& loop : normalized_loops) {
            int n = (int)loop.uv_polyline.size();
            if (n < 2) continue;
            for (int i = 0; i < n; i++) {
                const Vector2d& c = loop.uv_polyline[i];
                const Vector2d& d = loop.uv_polyline[(i + 1) % n];
                if ((c - d).norm() <= eps) continue;
                if (segments_intersect_proper(pa, pb, c, d, 1e-10)) return true;
            }
        }
        return false;
    };
    auto triangle_is_safe = [&](const Vector3i& tri) {
        int ids[3] = {tri.x(), tri.y(), tri.z()};
        for (int k = 0; k < 3; k++) {
            if (!point_inside_trim_region(points[ids[k]].uv, normalized_loops)) return false;
        }
        Vector2d centroid =
            (points[tri.x()].uv + points[tri.y()].uv + points[tri.z()].uv) / 3.0;
        if (!point_inside_trim_region(centroid, normalized_loops)) return false;
        for (int k = 0; k < 3; k++) {
            int a = ids[k];
            int b = ids[(k + 1) % 3];
            if (!edge_is_constraint(a, b)) {
                if ((points[a].uv - points[b].uv).norm() > max_free_edge_length) return false;
                Vector2d mid = 0.5 * (points[a].uv + points[b].uv);
                if (!point_inside_trim_region(mid, normalized_loops)) return false;
                if (edge_crosses_trim_loop(a, b)) return false;
            }
        }
        return true;
    };

    vector<Vector3i> kept;
    kept.reserve(raw_faces.size());
    for (const Vector3i& tri : raw_faces) {
        if (!triangle_is_safe(tri)) continue;
        kept.push_back(tri);
    }
    if (kept.empty()) return false;

    vector<int> old_to_new(points.size(), -1);
    vector<LockedTrimPoint> compact_points;
    for (const Vector3i& tri : kept) {
        for (int k = 0; k < 3; k++) {
            int old = tri(k);
            if (old_to_new[old] >= 0) continue;
            old_to_new[old] = (int)compact_points.size();
            compact_points.push_back(points[old]);
        }
    }

    V.resize((int)compact_points.size(), 3);
    MatrixXd baseV((int)compact_points.size(), 3);
    vector<char> is_locked(compact_points.size(), 0);
    for (int i = 0; i < (int)compact_points.size(); i++) {
        baseV.row(i) = surface.evaluate(
            compact_points[i].uv.x(),
            compact_points[i].uv.y()).transpose();
        V.row(i) = compact_points[i].position.transpose();
        is_locked[i] = compact_points[i].locked ? 1 : 0;
    }
    F.resize((int)kept.size(), 3);
    for (int i = 0; i < (int)kept.size(); i++) {
        F.row(i) << old_to_new[kept[i].x()],
                    old_to_new[kept[i].y()],
                    old_to_new[kept[i].z()];
    }
    if (apply_harmonic_boundary_correction) {
        vector<set<int>> neighbors(V.rows());
        for (int fi = 0; fi < F.rows(); fi++) {
            int a = F(fi, 0);
            int b = F(fi, 1);
            int c = F(fi, 2);
            if (a < 0 || b < 0 || c < 0 ||
                a >= V.rows() || b >= V.rows() || c >= V.rows()) {
                continue;
            }
            neighbors[a].insert(b);
            neighbors[a].insert(c);
            neighbors[b].insert(a);
            neighbors[b].insert(c);
            neighbors[c].insert(a);
            neighbors[c].insert(b);
        }

        MatrixXd boundary_residual = MatrixXd::Zero(V.rows(), 3);
        for (int i = 0; i < V.rows(); i++) {
            if (is_locked[i]) boundary_residual.row(i) = V.row(i) - baseV.row(i);
        }

        vector<int> unknown_id(V.rows(), -1);
        int unknown_count = 0;
        for (int i = 0; i < V.rows(); i++) {
            if (!is_locked[i]) unknown_id[i] = unknown_count++;
        }
        if (unknown_count > 0) {
            vector<Triplet<double>> trips;
            MatrixXd rhs = MatrixXd::Zero(unknown_count, 3);
            trips.reserve(unknown_count * 7);
            for (int vi = 0; vi < V.rows(); vi++) {
                int row = unknown_id[vi];
                if (row < 0) continue;
                double degree = 0.0;
                for (int nb : neighbors[vi]) {
                    degree += 1.0;
                    if (is_locked[nb]) {
                        rhs.row(row) += boundary_residual.row(nb);
                    } else if (unknown_id[nb] >= 0) {
                        trips.emplace_back(row, unknown_id[nb], -1.0);
                    }
                }
                trips.emplace_back(row, row, degree > 0.0 ? degree : 1.0);
            }
            SparseMatrix<double> A(unknown_count, unknown_count);
            A.setFromTriplets(trips.begin(), trips.end());
            Eigen::SimplicialLDLT<SparseMatrix<double>> solver;
            solver.compute(A);
            if (solver.info() == Eigen::Success) {
                MatrixXd correction = solver.solve(rhs);
                if (solver.info() == Eigen::Success) {
                    for (int vi = 0; vi < V.rows(); vi++) {
                        if (is_locked[vi]) {
                            V.row(vi) = baseV.row(vi) + boundary_residual.row(vi);
                        } else if (unknown_id[vi] >= 0) {
                            V.row(vi) = baseV.row(vi) + correction.row(unknown_id[vi]);
                        }
                    }
                }
            }
        }
    }
    if (UV) {
        UV->resize((int)compact_points.size(), 2);
        for (int i = 0; i < (int)compact_points.size(); i++) {
            UV->row(i) = compact_points[i].uv.transpose();
        }
    }
    return V.rows() > 0 && F.rows() > 0;
}

static bool sample_bspline_on_region_topology_with_authoritative_boundary(
    const BSplineSurface3D& surface,
    const Vector2d& uv_min,
    const Vector2d& uv_max,
    const MatrixXd& original_V,
    const BoundarySegmentationResult& boundary,
    const KktGlobalStitchingResult& param,
    MatrixXd& V,
    MatrixXi& F,
    MatrixXd* UV = nullptr,
    bool snap_boundary_to_authoritative = false,
    bool apply_harmonic_boundary_correction = false) {
    V.resize(0, 3);
    F.resize(0, 3);
    if (UV) UV->resize(0, 2);
    if (param.region_vertex_ids.empty() || param.local_faces.rows() == 0) return false;

    std::set<int> boundary_vertices;
    for (const AuthoritativeBoundaryLoop& loop : boundary.loops) {
        for (int vid : loop.vertex_ids) boundary_vertices.insert(vid);
    }

    Vector2d scale = uv_max - uv_min;
    V.resize((int)param.region_vertex_ids.size(), 3);
    MatrixXd baseV((int)param.region_vertex_ids.size(), 3);
    MatrixXd local_uv((int)param.region_vertex_ids.size(), 2);
    for (int li = 0; li < (int)param.region_vertex_ids.size(); li++) {
        int gid = param.region_vertex_ids[li];
        if (gid < 0 || gid >= original_V.rows() || gid >= param.UV.rows()) return false;
        Vector2d uv = param.UV.row(gid).transpose();
        double u = scale.x() > 1e-14 ? (uv.x() - uv_min.x()) / scale.x() : 0.0;
        double v = scale.y() > 1e-14 ? (uv.y() - uv_min.y()) / scale.y() : 0.0;
        u = std::max(0.0, std::min(1.0, u));
        v = std::max(0.0, std::min(1.0, v));
        local_uv.row(li) << u, v;
        baseV.row(li) = surface.evaluate(u, v).transpose();
        if (snap_boundary_to_authoritative && boundary_vertices.count(gid)) {
            V.row(li) = original_V.row(gid);
        } else {
            V.row(li) = baseV.row(li);
        }
    }

    vector<Vector3i> faces;
    faces.reserve(param.local_faces.rows());
    for (int fi = 0; fi < param.local_faces.rows(); fi++) {
        int a = param.local_faces(fi, 0);
        int b = param.local_faces(fi, 1);
        int c = param.local_faces(fi, 2);
        if (a < 0 || b < 0 || c < 0 ||
            a >= V.rows() || b >= V.rows() || c >= V.rows() ||
            a == b || b == c || c == a) {
            continue;
        }
        faces.push_back(Vector3i(a, b, c));
    }
    if (faces.empty()) {
        V.resize(0, 3);
        return false;
    }
    F.resize((int)faces.size(), 3);
    for (int i = 0; i < (int)faces.size(); i++) F.row(i) = faces[i].transpose();

    if (apply_harmonic_boundary_correction && !boundary_vertices.empty()) {
        vector<char> is_boundary(param.region_vertex_ids.size(), 0);
        MatrixXd boundary_residual = MatrixXd::Zero(V.rows(), 3);
        for (int li = 0; li < (int)param.region_vertex_ids.size(); li++) {
            int gid = param.region_vertex_ids[li];
            if (!boundary_vertices.count(gid)) continue;
            is_boundary[li] = 1;
            boundary_residual.row(li) = original_V.row(gid) - baseV.row(li);
        }

        vector<set<int>> neighbors(V.rows());
        for (int fi = 0; fi < F.rows(); fi++) {
            int a = F(fi, 0);
            int b = F(fi, 1);
            int c = F(fi, 2);
            neighbors[a].insert(b);
            neighbors[a].insert(c);
            neighbors[b].insert(a);
            neighbors[b].insert(c);
            neighbors[c].insert(a);
            neighbors[c].insert(b);
        }

        vector<int> unknown_id(V.rows(), -1);
        int unknown_count = 0;
        for (int i = 0; i < V.rows(); i++) {
            if (!is_boundary[i]) unknown_id[i] = unknown_count++;
        }
        if (unknown_count > 0) {
            vector<Triplet<double>> trips;
            MatrixXd rhs = MatrixXd::Zero(unknown_count, 3);
            for (int vi = 0; vi < V.rows(); vi++) {
                int row = unknown_id[vi];
                if (row < 0) continue;
                double degree = 0.0;
                for (int nb : neighbors[vi]) {
                    degree += 1.0;
                    if (is_boundary[nb]) {
                        rhs.row(row) += boundary_residual.row(nb);
                    } else if (unknown_id[nb] >= 0) {
                        trips.emplace_back(row, unknown_id[nb], -1.0);
                    }
                }
                if (degree <= 0.0) {
                    trips.emplace_back(row, row, 1.0);
                } else {
                    trips.emplace_back(row, row, degree);
                }
            }
            SparseMatrix<double> A(unknown_count, unknown_count);
            A.setFromTriplets(trips.begin(), trips.end());
            Eigen::SimplicialLDLT<SparseMatrix<double>> solver;
            solver.compute(A);
            if (solver.info() == Eigen::Success) {
                MatrixXd correction = solver.solve(rhs);
                if (solver.info() == Eigen::Success) {
                    for (int vi = 0; vi < V.rows(); vi++) {
                        if (is_boundary[vi]) {
                            V.row(vi) = baseV.row(vi) + boundary_residual.row(vi);
                        } else if (unknown_id[vi] >= 0) {
                            V.row(vi) = baseV.row(vi) + correction.row(unknown_id[vi]);
                        }
                    }
                }
            }
        } else {
            for (int vi = 0; vi < V.rows(); vi++) {
                if (is_boundary[vi]) V.row(vi) = baseV.row(vi) + boundary_residual.row(vi);
            }
        }
    }

    if (UV) *UV = local_uv;
    return true;
}

static void write_labels_json(
    const string& filename,
    const BoundarySegmentationResult& boundary,
    const AutomaticLabelingResult& labeling) {
    std::ofstream out(filename);
    out << "{\n";
    out << "  \"valid\": " << (labeling.valid ? "true" : "false") << ",\n";
    out << "  \"reason\": \"" << json_escape(labeling.reason) << "\",\n";
    out << "  \"ambiguous\": " << (labeling.ambiguous ? "true" : "false") << ",\n";
    out << "  \"perimeter_segments\": " << boundary.perimeter_segments.size() << ",\n";
    out << "  \"final_label_count\": " << labeling.final_label_count << ",\n";
    out << "  \"final_corner_count\": " << labeling.final_corner_count << ",\n";
    out << "  \"orientation\": {\n";
    out << "    \"valid\": " << (labeling.orientation.valid ? "true" : "false") << ",\n";
    out << "    \"confidence\": " << labeling.orientation.confidence << "\n";
    out << "  },\n";
    out << "  \"sides\": [\n";
    for (int i = 0; i < (int)labeling.abstract_sides.size(); i++) {
        const AbstractSide& side = labeling.abstract_sides[i];
        out << "    {\"side_index\": " << side.side_index << ", \"segments\": [";
        for (int j = 0; j < (int)side.segment_ids.size(); j++) {
            if (j) out << ", ";
            out << side.segment_ids[j];
        }
        out << "]}";
        out << (i + 1 == (int)labeling.abstract_sides.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

static void write_metrics_json(const string& filename, const TrimmedBSplinePipelineMetrics& m) {
    std::ofstream out(filename);
    out.precision(17);
    out << "{\n";
    out << "  \"valid\": " << (m.valid ? "true" : "false") << ",\n";
    out << "  \"reason\": \"" << json_escape(m.reason) << "\",\n";
    out << "  \"labeling_configuration\": \"" << json_escape(m.labeling_configuration) << "\",\n";
    out << "  \"ambiguity_flag\": " << (m.ambiguous ? "true" : "false") << ",\n";
    out << "  \"parameterization_flipped_triangle_count\": " << m.flipped_triangle_count << ",\n";
    out << "  \"mean_arap_distortion\": " << m.mean_arap_distortion << ",\n";
    out << "  \"max_arap_distortion\": " << m.max_arap_distortion << ",\n";
    out << "  \"label_coordinate_constraint_error\": " << m.label_coordinate_constraint_error << ",\n";
    out << "  \"guiding_frame_length_ratio_error\": " << m.guiding_frame_length_ratio_error << ",\n";
    out << "  \"original_region_fitting_rms_error\": " << m.original_region_rms_error << ",\n";
    out << "  \"original_region_fitting_max_error\": " << m.original_region_max_error << ",\n";
    out << "  \"boundary_fitting_rms_error\": " << m.boundary_rms_error << ",\n";
    out << "  \"boundary_fitting_max_error\": " << m.boundary_max_error << ",\n";
    out << "  \"artificial_extension_curvature_mean\": " << m.artificial_extension_curvature_mean << ",\n";
    out << "  \"artificial_extension_curvature_max\": " << m.artificial_extension_curvature_max << ",\n";
    out << "  \"surface_area_growth_ratio\": " << m.surface_area_growth_ratio << ",\n";
    out << "  \"bounding_box_growth_ratio\": " << m.bounding_box_growth_ratio << ",\n";
    out << "  \"weak_control_point_count\": " << m.weak_control_point_count << ",\n";
    out << "  \"linear_system_condition_estimate\": " << m.linear_system_condition_estimate << "\n";
    out << "}\n";
}

static void write_trimmed_mesh_flicker_diagnosis(
    const string& filename,
    const string& render_mode,
    const MeshValidationReport& before,
    const MeshValidationReport& after) {
    std::ofstream out(filename);
    if (!out.is_open()) return;
    out << "# Trimmed Mesh Flicker Diagnosis\n\n";
    out << "## 1. Actual Root Cause\n\n";
    out << "`bspline_trimmed_surface.obj` is generated in mode `" << render_mode << "`. "
        << "Boundary flicker is treated as a geometry defect first: duplicate faces, degenerate faces, "
        << "bad winding, nonmanifold edges, and adjacent-face normal jumps are validated before export. "
        << "For authoritative-boundary CDT output, local folded sliver faces are removed iteratively "
        << "instead of being hidden with render-state changes.\n\n";
    out << "## 2. Viewer Mesh Overlap\n\n";
    out << "Viewer-side lifecycle now provides `TrimRenderDebugMode`. `TrimmedOnly` clears extra "
        << "viewer data slots and displays only the current trimmed mesh. `TrimmedAndOriginal` is the "
        << "explicit comparison mode and may intentionally reveal z-fighting if surfaces coincide.\n\n";
    out << "## 3. Mesh Validation\n\n";
    out << "| metric | before fix | after fix |\n";
    out << "| --- | ---: | ---: |\n";
    out << "| vertices | " << before.vertex_count << " | " << after.vertex_count << " |\n";
    out << "| faces | " << before.face_count << " | " << after.face_count << " |\n";
    out << "| exact duplicate faces | " << before.exact_duplicate_faces << " | " << after.exact_duplicate_faces << " |\n";
    out << "| geometric duplicate faces | " << before.geometric_duplicate_faces << " | " << after.geometric_duplicate_faces << " |\n";
    out << "| degenerate faces | " << before.degenerate_faces << " | " << after.degenerate_faces << " |\n";
    out << "| near-degenerate faces | " << before.near_degenerate_faces << " | " << after.near_degenerate_faces << " |\n";
    out << "| inconsistent winding edges | " << before.inconsistent_winding_edges << " | " << after.inconsistent_winding_edges << " |\n";
    out << "| normal-jump edges | " << before.normal_jump_edges << " | " << after.normal_jump_edges << " |\n";
    out << "| nonmanifold edges | " << before.nonmanifold_edges << " | " << after.nonmanifold_edges << " |\n";
    out << "| isolated vertices | " << before.isolated_vertices << " | " << after.isolated_vertices << " |\n";
    out << "| min double area | " << before.min_double_area << " | " << after.min_double_area << " |\n";
    out << "| min quality | " << before.min_quality << " | " << after.min_quality << " |\n\n";
    out << "## 4. Problem Code Path\n\n";
    out << "- Parameter-domain legacy sampling: `sample_trimmed_surface_regular_grid`.\n";
    out << "- CDT boundary-locked reference: `sample_trimmed_surface_with_authoritative_boundary`.\n";
    out << "- Default stable output: `sample_bspline_on_region_topology_with_authoritative_boundary` with harmonic boundary residual correction.\n";
    out << "- Mesh cleanup: `remove_degenerate_faces`, `orient_mesh_faces_consistently`.\n";
    out << "- Viewer lifecycle: `TrimRenderDebugMode` in `main.cpp`.\n\n";
    out << "## 5. Debug Artifacts\n\n";
    out << "- `bspline_trimmed_surface.obj`: fixed default viewer mesh.\n";
    out << "- `bspline_trimmed_surface_authoritative_boundary.obj`: hard-snapped boundary comparison mesh.\n";
    out << "- `bspline_trimmed_surface_unlocked.obj`: regular-grid clipped reference.\n";
    out << "- `bspline_trimmed_surface_cdt_boundary_locked.obj`: CDT boundary-locked reference.\n";
    out << "- `trimmed_mesh_validation_before_fix.json`, `trimmed_mesh_validation.json`.\n";
    out << "- `duplicate_faces.obj`, `degenerate_faces.obj`, `bad_winding_faces.obj`, `normal_jump_faces.obj`, `boundary_intersection_faces.obj`.\n\n";
    out << "## 6. Current Output Mode\n\n";
    out << "`" << render_mode << "`\n\n";
    out << "## 7. Remaining Known Boundary Exceptions\n\n";
    if (after.exact_duplicate_faces == 0 &&
        after.geometric_duplicate_faces == 0 &&
        after.degenerate_faces == 0 &&
        after.normal_jump_edges == 0 &&
        after.inconsistent_winding_edges == 0 &&
        after.nonmanifold_edges == 0) {
        out << "No structural mesh defect remains in the exported default trimmed mesh according to the validator. "
            << "If flicker persists in `TrimmedOnly`, compare face-based and vertex-based shading and inspect GPU-side overlays.\n";
    } else {
        out << "The exported default trimmed mesh still has validation warnings; inspect the issue OBJ files.\n";
    }
}

static MatrixXd region_positions_for_parameterization(
    const MatrixXd& V,
    const vector<int>& region_vertex_ids) {
    MatrixXd P((int)region_vertex_ids.size(), 3);
    for (int i = 0; i < (int)region_vertex_ids.size(); i++) P.row(i) = V.row(region_vertex_ids[i]);
    return P;
}

template <typename ParamResult>
static TensorProductBSplineFitInput make_trimmed_fit_input(
    const MatrixXd& V,
    const ParamResult& param,
    TensorProductFitBaseline baseline) {
    TensorProductBSplineFitInput input;
    input.baseline = baseline;
    input.UV.resize((int)param.region_vertex_ids.size(), 2);
    for (int i = 0; i < (int)param.region_vertex_ids.size(); i++) {
        input.UV.row(i) = param.UV.row(param.region_vertex_ids[i]);
    }
    input.F = param.local_faces;
    input.positions = region_positions_for_parameterization(V, param.region_vertex_ids);
    input.original_vertex_mask.assign(input.UV.rows(), true);
    input.original_face_mask.assign(input.F.rows(), true);
    input.sample_weights.assign(input.UV.rows(), 1.0);
    return input;
}

static MatrixXd positions_for_extended_domain(
    const MatrixXd& V,
    const KktGlobalStitchingResult& param,
    const RectangularDomainExtensionResult& domain) {
    MatrixXd P = MatrixXd::Zero(domain.full_uv_vertices.rows(), 3);
    for (int i = 0; i < domain.full_uv_vertices.rows(); i++) {
        double best = std::numeric_limits<double>::infinity();
        int best_gid = -1;
        for (int gid : param.region_vertex_ids) {
            double d = (domain.full_uv_vertices.row(i) - param.UV.row(gid)).squaredNorm();
            if (d < best) {
                best = d;
                best_gid = gid;
            }
        }
        if (best_gid >= 0) P.row(i) = V.row(best_gid);
    }
    return P;
}

static TensorProductBSplineFitInput make_extended_fit_input(
    const RectangularDomainExtensionResult& domain,
    const Polyharmonic3DExtensionResult& ext,
    TensorProductFitBaseline baseline) {
    TensorProductBSplineFitInput input;
    input.baseline = baseline;
    input.UV = domain.full_uv_vertices;
    input.F = domain.full_faces;
    input.positions = ext.extended_vertices;
    input.original_vertex_mask = domain.original_vertex_mask;
    input.original_face_mask = domain.original_face_mask;
    input.sample_weights.assign(input.UV.rows(), 1.0);
    return input;
}

struct BoundaryFitSample {
    Vector2d uv = Vector2d::Zero();
    Vector3d p = Vector3d::Zero();
};

static double boundary_bbox_diagonal(const BoundarySegmentationResult& boundary) {
    bool any = false;
    Vector3d mn = Vector3d::Zero();
    Vector3d mx = Vector3d::Zero();
    for (const AuthoritativeBoundaryLoop& loop : boundary.loops) {
        for (const Vector3d& p : loop.positions) {
            if (!any) {
                mn = p;
                mx = p;
                any = true;
            } else {
                mn = mn.cwiseMin(p);
                mx = mx.cwiseMax(p);
            }
        }
    }
    return any ? (mx - mn).norm() : 1.0;
}

static vector<BoundaryFitSample> collect_authoritative_boundary_fit_samples(
    const BoundarySegmentationResult& boundary,
    const KktGlobalStitchingResult& param,
    const MatrixXd& V) {
    vector<BoundaryFitSample> samples;
    for (const AuthoritativeBoundaryLoop& loop : boundary.loops) {
        int n = (int)loop.vertex_ids.size();
        if (n < 2) continue;
        double total_length = 0.0;
        int edge_count = loop.closed ? n : n - 1;
        for (int i = 0; i < edge_count; i++) {
            int a = loop.vertex_ids[i];
            int b = loop.vertex_ids[(i + 1) % n];
            if (a < 0 || b < 0 || a >= V.rows() || b >= V.rows()) continue;
            total_length += (V.row(b) - V.row(a)).norm();
        }
        int target_samples = std::max(64, 4 * n);
        double target_step = total_length > 1e-14
                                 ? total_length / (double)target_samples
                                 : 0.0;
        for (int i = 0; i < edge_count; i++) {
            int a = loop.vertex_ids[i];
            int b = loop.vertex_ids[(i + 1) % n];
            if (a < 0 || b < 0 ||
                a >= param.UV.rows() || b >= param.UV.rows() ||
                a >= V.rows() || b >= V.rows()) {
                continue;
            }
            double len = (V.row(b) - V.row(a)).norm();
            int subdivisions = target_step > 1e-14
                                   ? (int)std::ceil(len / target_step)
                                   : 1;
            subdivisions = std::max(1, std::min(12, subdivisions));
            for (int k = 0; k < subdivisions; k++) {
                double t = (double)k / (double)subdivisions;
                BoundaryFitSample s;
                s.uv = (1.0 - t) * param.UV.row(a).transpose() +
                       t * param.UV.row(b).transpose();
                s.p = (1.0 - t) * V.row(a).transpose() +
                      t * V.row(b).transpose();
                samples.push_back(s);
            }
        }
        if (!loop.closed) {
            int b = loop.vertex_ids.back();
            if (b >= 0 && b < param.UV.rows() && b < V.rows()) {
                BoundaryFitSample s;
                s.uv = param.UV.row(b).transpose();
                s.p = V.row(b).transpose();
                samples.push_back(s);
            }
        }
    }
    return samples;
}

static void append_authoritative_boundary_fit_samples(
    TensorProductBSplineFitInput& input,
    const BoundarySegmentationResult& boundary,
    const KktGlobalStitchingResult& param,
    const MatrixXd& V,
    double weight) {
    if (weight <= 0.0) return;
    int old_rows = input.UV.rows();
    vector<BoundaryFitSample> samples =
        collect_authoritative_boundary_fit_samples(boundary, param, V);
    int add_count = (int)samples.size();
    if (samples.empty()) return;
    MatrixXd new_uv(old_rows + add_count, 2);
    MatrixXd new_pos(old_rows + add_count, 3);
    if (old_rows > 0) {
        new_uv.topRows(old_rows) = input.UV;
        new_pos.topRows(old_rows) = input.positions;
    }
    if ((int)input.original_vertex_mask.size() < old_rows) {
        input.original_vertex_mask.resize(old_rows, false);
    }
    if ((int)input.sample_weights.size() < old_rows) {
        input.sample_weights.resize(old_rows, 1.0);
    }

    int row = old_rows;
    for (const BoundaryFitSample& s : samples) {
        new_uv.row(row) = s.uv.transpose();
        new_pos.row(row) = s.p.transpose();
        input.original_vertex_mask.push_back(true);
        input.sample_weights.push_back(weight);
        row++;
    }
    input.UV = new_uv;
    input.positions = new_pos;
}

static TensorProductBSplineFitConfig fit_config(
    const TrimmedBSplinePipelineConfig& cfg,
    const string& prefix,
    bool original_only) {
    TensorProductBSplineFitConfig fcfg;
    fcfg.control_count_u = cfg.control_count_u;
    fcfg.control_count_v = cfg.control_count_v;
    fcfg.degree_u = cfg.spline_degree;
    fcfg.degree_v = cfg.spline_degree;
    fcfg.control_net_fairness_weight = cfg.fitting_regularization;
    fcfg.control_net_initial_weight = cfg.fitting_initial_weight;
    fcfg.fit_original_vertices_only = original_only;
    fcfg.export_debug = cfg.export_debug_artifacts;
    fcfg.estimate_condition_number = cfg.estimate_condition_number;
    fcfg.debug_prefix = prefix;
    fcfg.sample_u = cfg.surface_sample_u;
    fcfg.sample_v = cfg.surface_sample_v;
    return fcfg;
}

static TensorProductBSplineFitStats boundary_error_stats(
    const BoundarySegmentationResult& boundary,
    const KktGlobalStitchingResult& param,
    const MatrixXd& V,
    const TensorProductBSplineFitResult& fit) {
    TensorProductBSplineFitStats stats;
    double sum = 0.0;
    double sum2 = 0.0;
    int count = 0;
    Vector2d d = fit.uv_max - fit.uv_min;
    vector<BoundaryFitSample> samples =
        collect_authoritative_boundary_fit_samples(boundary, param, V);
    for (const BoundaryFitSample& s : samples) {
        Vector2d uv = s.uv;
        uv.x() = d.x() > 1e-14 ? (uv.x() - fit.uv_min.x()) / d.x() : 0.0;
        uv.y() = d.y() > 1e-14 ? (uv.y() - fit.uv_min.y()) / d.y() : 0.0;
        uv = uv.cwiseMax(Vector2d::Zero()).cwiseMin(Vector2d::Ones());
        double e = (fit.surface.evaluate(uv.x(), uv.y()) - s.p).norm();
        sum += e;
        sum2 += e * e;
        stats.max_error = std::max(stats.max_error, e);
        count++;
    }
    if (count > 0) {
        stats.mean_error = sum / (double)count;
        stats.rms_error = std::sqrt(sum2 / (double)count);
    }
    return stats;
}

static void write_baseline_report(
    const string& filename,
    const vector<TensorProductBSplineFitResult>& baselines) {
    std::ofstream out(filename);
    out << "baseline,valid,original_rms,original_max,weak_controls,condition,extension_smoothness\n";
    for (const TensorProductBSplineFitResult& r : baselines) {
        out << to_string(r.baseline) << ","
            << (r.valid ? 1 : 0) << ","
            << r.original_region_error.rms_error << ","
            << r.original_region_error.max_error << ","
            << r.weak_support.weak_control_point_count << ","
            << r.condition_estimate << ","
            << r.extension_region_smoothness << "\n";
    }
}

static int count_sampled_surface_fold_edges(
    const BSplineSurface3D& surface,
    int sample_u,
    int sample_v,
    double normal_dot_threshold = -0.5) {
    MatrixXd V;
    MatrixXi F;
    sample_bspline_surface(
        surface,
        std::max(8, sample_u),
        std::max(8, sample_v),
        V,
        F);
    if (V.rows() == 0 || F.rows() == 0) return 0;

    vector<Vector3d> normals(F.rows(), Vector3d::Zero());
    for (int fi = 0; fi < F.rows(); fi++) {
        int a = F(fi, 0);
        int b = F(fi, 1);
        int c = F(fi, 2);
        if (a < 0 || b < 0 || c < 0 ||
            a >= V.rows() || b >= V.rows() || c >= V.rows()) {
            continue;
        }
        Vector3d pa = V.row(a).transpose();
        Vector3d pb = V.row(b).transpose();
        Vector3d pc = V.row(c).transpose();
        Vector3d n = (pb - pa).cross(pc - pa);
        if (n.norm() > 1e-14) normals[fi] = n.normalized();
    }

    std::map<std::pair<int, int>, vector<int>> edge_faces;
    for (int fi = 0; fi < F.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            edge_faces[std::minmax(F(fi, k), F(fi, (k + 1) % 3))].push_back(fi);
        }
    }

    int folded_edges = 0;
    for (const auto& kv : edge_faces) {
        const vector<int>& faces = kv.second;
        if (faces.size() != 2) continue;
        const Vector3d& n0 = normals[faces[0]];
        const Vector3d& n1 = normals[faces[1]];
        if (n0.norm() <= 0.0 || n1.norm() <= 0.0) continue;
        if (n0.dot(n1) < normal_dot_threshold) folded_edges++;
    }
    return folded_edges;
}

static int count_sampled_trimmed_surface_fold_edges(
    const BSplineSurface3D& surface,
    const vector<TrimLoop2D>& normalized_loops,
    int sample_u,
    int sample_v,
    double normal_dot_threshold = -0.5) {
    MatrixXd V;
    MatrixXi F;
    if (!sample_trimmed_surface_regular_grid(
            surface,
            normalized_loops,
            std::max(8, sample_u),
            std::max(8, sample_v),
            V,
            F)) {
        return 0;
    }
    if (V.rows() == 0 || F.rows() == 0) return 0;

    vector<Vector3d> normals(F.rows(), Vector3d::Zero());
    for (int fi = 0; fi < F.rows(); fi++) {
        int a = F(fi, 0);
        int b = F(fi, 1);
        int c = F(fi, 2);
        if (a < 0 || b < 0 || c < 0 ||
            a >= V.rows() || b >= V.rows() || c >= V.rows()) {
            continue;
        }
        Vector3d pa = V.row(a).transpose();
        Vector3d pb = V.row(b).transpose();
        Vector3d pc = V.row(c).transpose();
        Vector3d n = (pb - pa).cross(pc - pa);
        if (n.norm() > 1e-14) normals[fi] = n.normalized();
    }

    std::map<std::pair<int, int>, vector<int>> edge_faces;
    for (int fi = 0; fi < F.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            edge_faces[std::minmax(F(fi, k), F(fi, (k + 1) % 3))].push_back(fi);
        }
    }

    int folded_edges = 0;
    for (const auto& kv : edge_faces) {
        const vector<int>& faces = kv.second;
        if (faces.size() != 2) continue;
        const Vector3d& n0 = normals[faces[0]];
        const Vector3d& n1 = normals[faces[1]];
        if (n0.norm() <= 0.0 || n1.norm() <= 0.0) continue;
        if (n0.dot(n1) < normal_dot_threshold) folded_edges++;
    }
    return folded_edges;
}

static int count_mesh_normal_jump_edges(
    const MatrixXd& V,
    const MatrixXi& F,
    double normal_dot_threshold = 0.5) {
    if (V.rows() == 0 || F.rows() == 0) return 0;
    vector<Vector3d> normals(F.rows(), Vector3d::Zero());
    for (int fi = 0; fi < F.rows(); fi++) {
        int a = F(fi, 0);
        int b = F(fi, 1);
        int c = F(fi, 2);
        if (a < 0 || b < 0 || c < 0 ||
            a >= V.rows() || b >= V.rows() || c >= V.rows()) {
            continue;
        }
        Vector3d pa = V.row(a).transpose();
        Vector3d pb = V.row(b).transpose();
        Vector3d pc = V.row(c).transpose();
        Vector3d n = (pb - pa).cross(pc - pa);
        if (n.norm() > 1e-14) normals[fi] = n.normalized();
    }

    std::map<std::pair<int, int>, vector<int>> edge_faces;
    for (int fi = 0; fi < F.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            edge_faces[std::minmax(F(fi, k), F(fi, (k + 1) % 3))].push_back(fi);
        }
    }
    int bad_edges = 0;
    for (const auto& kv : edge_faces) {
        const vector<int>& faces = kv.second;
        if (faces.size() != 2) continue;
        const Vector3d& n0 = normals[faces[0]];
        const Vector3d& n1 = normals[faces[1]];
        if (n0.norm() <= 0.0 || n1.norm() <= 0.0) continue;
        if (n0.dot(n1) < normal_dot_threshold) bad_edges++;
    }
    return bad_edges;
}

static int remove_normal_jump_faces(
    MatrixXd& V,
    MatrixXi& F,
    MatrixXd* UV,
    double normal_dot_threshold = 0.5,
    double low_quality_threshold = 0.02) {
    if (V.rows() == 0 || F.rows() == 0) return 0;

    auto face_normal_area_quality = [&](int fi, Vector3d& n, double& area2, double& quality) {
        n = Vector3d::Zero();
        area2 = 0.0;
        quality = 0.0;
        int a = F(fi, 0);
        int b = F(fi, 1);
        int c = F(fi, 2);
        if (a < 0 || b < 0 || c < 0 ||
            a >= V.rows() || b >= V.rows() || c >= V.rows()) {
            return;
        }
        Vector3d pa = V.row(a).transpose();
        Vector3d pb = V.row(b).transpose();
        Vector3d pc = V.row(c).transpose();
        Vector3d raw = (pb - pa).cross(pc - pa);
        area2 = raw.norm();
        if (area2 > 1e-14) n = raw / area2;
        double e0 = (pb - pa).squaredNorm();
        double e1 = (pc - pb).squaredNorm();
        double e2 = (pa - pc).squaredNorm();
        double denom = e0 + e1 + e2;
        if (denom > 1e-30) quality = 2.0 * std::sqrt(3.0) * area2 / denom;
    };

    vector<Vector3d> normals(F.rows(), Vector3d::Zero());
    vector<double> area2(F.rows(), 0.0);
    vector<double> quality(F.rows(), 0.0);
    for (int fi = 0; fi < F.rows(); fi++) {
        face_normal_area_quality(fi, normals[fi], area2[fi], quality[fi]);
    }

    std::map<std::pair<int, int>, vector<int>> edge_faces;
    for (int fi = 0; fi < F.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            edge_faces[std::minmax(F(fi, k), F(fi, (k + 1) % 3))].push_back(fi);
        }
    }

    vector<char> remove(F.rows(), 0);
    for (int fi = 0; fi < F.rows(); fi++) {
        if (quality[fi] > 0.0 && quality[fi] < low_quality_threshold) remove[fi] = 1;
    }
    for (const auto& kv : edge_faces) {
        const vector<int>& faces = kv.second;
        if (faces.size() != 2) continue;
        int f0 = faces[0];
        int f1 = faces[1];
        if (normals[f0].norm() <= 0.0 || normals[f1].norm() <= 0.0) continue;
        if (normals[f0].dot(normals[f1]) >= normal_dot_threshold) continue;
        int kill = f0;
        if (quality[f1] < quality[f0] ||
            (std::abs(quality[f1] - quality[f0]) <= 1e-12 && area2[f1] < area2[f0])) {
            kill = f1;
        }
        remove[kill] = 1;
    }

    int removed = 0;
    for (char r : remove) {
        if (r) removed++;
    }
    if (removed == 0) return 0;

    vector<int> old_to_new(V.rows(), -1);
    vector<Vector3d> new_vertices;
    vector<Vector2d> new_uvs;
    vector<Vector3i> new_faces;
    new_faces.reserve(F.rows() - removed);
    for (int fi = 0; fi < F.rows(); fi++) {
        if (remove[fi]) continue;
        Vector3i tri;
        bool ok = true;
        for (int k = 0; k < 3; k++) {
            int old = F(fi, k);
            if (old < 0 || old >= V.rows()) {
                ok = false;
                break;
            }
            if (old_to_new[old] < 0) {
                old_to_new[old] = (int)new_vertices.size();
                new_vertices.push_back(V.row(old).transpose());
                if (UV && old < UV->rows()) {
                    new_uvs.push_back(UV->row(old).transpose());
                }
            }
            tri(k) = old_to_new[old];
        }
        if (ok) new_faces.push_back(tri);
    }

    V.resize((int)new_vertices.size(), 3);
    for (int i = 0; i < (int)new_vertices.size(); i++) V.row(i) = new_vertices[i].transpose();
    F.resize((int)new_faces.size(), 3);
    for (int i = 0; i < (int)new_faces.size(); i++) F.row(i) = new_faces[i].transpose();
    if (UV) {
        UV->resize((int)new_uvs.size(), 2);
        for (int i = 0; i < (int)new_uvs.size(); i++) UV->row(i) = new_uvs[i].transpose();
    }
    return removed;
}

static int remove_normal_jump_faces_until_stable(
    MatrixXd& V,
    MatrixXi& F,
    MatrixXd* UV,
    int max_iterations = 8) {
    int total_removed = 0;
    for (int iter = 0; iter < max_iterations; iter++) {
        int jumps_before = count_mesh_normal_jump_edges(V, F);
        int removed = remove_normal_jump_faces(V, F, UV);
        total_removed += removed;
        if (removed == 0 || jumps_before == 0) break;
    }
    return total_removed;
}

static int fallback_priority(TensorProductFitBaseline baseline) {
    switch (baseline) {
    case TensorProductFitBaseline::ArapOnly: return 0;
    case TensorProductFitBaseline::TrimmedOnly: return 1;
    case TensorProductFitBaseline::ExtensionOnly: return 2;
    case TensorProductFitBaseline::LabelingExtension: return 3;
    }
    return 10;
}

static string role_cardinal_name(ParameterSideRole role) {
    switch (role) {
    case ParameterSideRole::South: return "South";
    case ParameterSideRole::East: return "East";
    case ParameterSideRole::North: return "North";
    case ParameterSideRole::West: return "West";
    case ParameterSideRole::Unassigned: return "Unassigned";
    }
    return "Unassigned";
}

static vector<std::array<ParameterSideRole, 4>> role_mapping_candidates(
    const AutomaticLabelingResult& labeling) {
    std::array<ParameterSideRole, 4> ccw = {
        ParameterSideRole::South,
        ParameterSideRole::East,
        ParameterSideRole::North,
        ParameterSideRole::West};
    std::array<ParameterSideRole, 4> cw = {
        ParameterSideRole::South,
        ParameterSideRole::West,
        ParameterSideRole::North,
        ParameterSideRole::East};
    vector<std::array<ParameterSideRole, 4>> candidates;

    if (labeling.orientation.valid) {
        std::array<ParameterSideRole, 4> oriented = {
            ParameterSideRole::Unassigned,
            ParameterSideRole::Unassigned,
            ParameterSideRole::Unassigned,
            ParameterSideRole::Unassigned};
        bool complete = labeling.abstract_sides.size() >= 4;
        for (int i = 0; i < 4 && i < (int)labeling.orientation.side_to_cardinal.size(); i++) {
            const string& name = labeling.orientation.side_to_cardinal[i];
            if (name == "South") oriented[i] = ParameterSideRole::South;
            else if (name == "East") oriented[i] = ParameterSideRole::East;
            else if (name == "North") oriented[i] = ParameterSideRole::North;
            else if (name == "West") oriented[i] = ParameterSideRole::West;
            else complete = false;
        }
        if (complete) candidates.push_back(oriented);
    }

    auto add_rotations = [&](const std::array<ParameterSideRole, 4>& base) {
        for (int r = 0; r < 4; r++) {
            std::array<ParameterSideRole, 4> roles;
            for (int i = 0; i < 4; i++) roles[i] = base[(i + r) % 4];
            bool duplicate = false;
            for (const auto& existing : candidates) {
                if (existing == roles) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) candidates.push_back(roles);
        }
    };
    add_rotations(ccw);
    add_rotations(cw);
    return candidates;
}

static AutomaticLabelingResult labeling_with_roles(
    const AutomaticLabelingResult& labeling,
    const std::array<ParameterSideRole, 4>& roles) {
    AutomaticLabelingResult out = labeling;
    out.orientation.valid = true;
    out.orientation.used_vsa_axes = false;
    out.orientation.confidence = 0.0;
    out.orientation.mean_direction_error = 0.0;
    for (int i = 0; i < 4; i++) {
        out.orientation.side_to_cardinal[i] = role_cardinal_name(roles[i]);
    }
    return out;
}

static string roles_to_string(const std::array<ParameterSideRole, 4>& roles) {
    std::ostringstream out;
    for (int i = 0; i < 4; i++) {
        if (i) out << ",";
        out << "Side" << i << "=" << role_cardinal_name(roles[i]);
    }
    return out.str();
}

static KktGlobalStitchingResult kkt_result_from_constrained_arap(
    const ConstrainedArapResult& arap,
    const string& reason) {
    KktGlobalStitchingResult out;
    out.valid = arap.valid && arap.flipped_triangle_count == 0;
    out.reason = reason;
    out.UV = arap.UV;
    out.region_vertex_ids = arap.region_vertex_ids;
    out.local_faces = arap.local_faces;
    out.local_face_to_global_face = arap.local_face_to_global_face;
    out.before_triangle_stats = arap.triangle_stats;
    out.after_triangle_stats = arap.triangle_stats;
    out.flipped_triangle_count = arap.flipped_triangle_count;
    out.mean_arap_residual = arap.mean_arap_residual;
    out.max_arap_residual = arap.max_arap_residual;
    out.constraint_rank = 0;
    out.constraint_count = (int)arap.constrained_vertex_ids.size() * 2;
    out.redundant_constraints = false;
    out.max_label_coordinate_error = 0.0;
    for (int i = 0; i < (int)arap.constrained_vertex_ids.size(); i++) {
        int gid = arap.constrained_vertex_ids[i];
        if (gid < 0 || gid >= arap.UV.rows()) continue;
        out.max_label_coordinate_error =
            std::max(out.max_label_coordinate_error,
                     (arap.UV.row(gid).transpose() - arap.constrained_uvs[i]).norm());
    }
    out.max_length_ratio_error = 0.0;
    out.adjacent_residual_variation = 0.0;
    return out;
}

static void write_region_topology_output_report(
    const string& filename,
    const MatrixXd& original_V,
    const KktGlobalStitchingResult& param,
    const MatrixXd& output_V,
    const MatrixXi& output_F,
    const string& output_mode,
    bool topology_output) {
    std::ofstream out(filename);
    if (!out.is_open()) return;
    out.precision(17);

    Vector3d src_min = Vector3d::Zero();
    Vector3d src_max = Vector3d::Zero();
    bool have_src = false;
    for (int gid : param.region_vertex_ids) {
        if (gid < 0 || gid >= original_V.rows()) continue;
        Vector3d p = original_V.row(gid).transpose();
        if (!have_src) {
            src_min = p;
            src_max = p;
            have_src = true;
        } else {
            src_min = src_min.cwiseMin(p);
            src_max = src_max.cwiseMax(p);
        }
    }

    Vector3d out_min = Vector3d::Zero();
    Vector3d out_max = Vector3d::Zero();
    bool have_out = output_V.rows() > 0;
    if (have_out) {
        out_min = output_V.colwise().minCoeff().transpose();
        out_max = output_V.colwise().maxCoeff().transpose();
    }

    const bool same_vertex_count =
        output_V.rows() == (int)param.region_vertex_ids.size();
    const bool same_face_count = output_F.rows() == param.local_faces.rows();
    double bbox_delta = 0.0;
    if (have_src && have_out) {
        bbox_delta = std::max((src_min - out_min).norm(), (src_max - out_max).norm());
    }

    out << "{\n";
    out << "  \"format\": \"RegionTopologyOutputReport.v1\",\n";
    out << "  \"output_mode\": \"" << json_escape(output_mode) << "\",\n";
    out << "  \"uses_original_region_topology\": "
        << (topology_output ? "true" : "false") << ",\n";
    out << "  \"region_vertex_count\": " << param.region_vertex_ids.size() << ",\n";
    out << "  \"region_face_count\": " << param.local_faces.rows() << ",\n";
    out << "  \"output_vertex_count\": " << output_V.rows() << ",\n";
    out << "  \"output_face_count\": " << output_F.rows() << ",\n";
    out << "  \"same_vertex_count_as_region\": "
        << (same_vertex_count ? "true" : "false") << ",\n";
    out << "  \"same_face_count_as_region\": "
        << (same_face_count ? "true" : "false") << ",\n";
    out << "  \"bbox_delta\": " << bbox_delta << ",\n";
    out << "  \"source_region_bbox_min\": ["
        << src_min.x() << ", " << src_min.y() << ", " << src_min.z() << "],\n";
    out << "  \"source_region_bbox_max\": ["
        << src_max.x() << ", " << src_max.y() << ", " << src_max.z() << "],\n";
    out << "  \"output_bbox_min\": ["
        << out_min.x() << ", " << out_min.y() << ", " << out_min.z() << "],\n";
    out << "  \"output_bbox_max\": ["
        << out_max.x() << ", " << out_max.y() << ", " << out_max.z() << "],\n";
    out << "  \"interpretation\": \"When uses_original_region_topology is true and counts match, visible extra lobes are inherited from the selected VSA region face set rather than generated by the tensor-product B-spline sampler.\"\n";
    out << "}\n";
}

static int active_labeled_side_count(const AutomaticLabelingResult& labeling) {
    int count = 0;
    for (const AbstractSide& side : labeling.abstract_sides) {
        if (!side.segment_ids.empty()) count++;
    }
    return count;
}

static vector<int> loop_indices_between(int a, int b, int n) {
    vector<int> out;
    int cur = a;
    for (int guard = 0; guard <= n; guard++) {
        out.push_back(cur);
        if (cur == b) break;
        cur = (cur + 1) % n;
    }
    return out;
}

static double segment_length_3d(const vector<Vector3d>& pts) {
    double length = 0.0;
    for (int i = 1; i < (int)pts.size(); i++) length += (pts[i] - pts[i - 1]).norm();
    return length;
}

static Vector3d average_segment_tangent(const vector<Vector3d>& pts) {
    if (pts.size() < 2) return Vector3d::Zero();
    Vector3d sum = Vector3d::Zero();
    for (int i = 1; i < (int)pts.size(); i++) sum += pts[i] - pts[i - 1];
    return sum.norm() > 1e-12 ? sum.normalized() : Vector3d::Zero();
}

static bool promote_three_label_sides_to_four(
    const BoundarySegmentationResult& boundary,
    AutomaticLabelingResult& labeling,
    string& reason) {
    vector<AbstractSide> active_sides;
    for (const AbstractSide& side : labeling.abstract_sides) {
        if (!side.segment_ids.empty()) active_sides.push_back(side);
    }
    if (active_sides.size() != 3) {
        reason = "labeling does not have exactly three active sides";
        return false;
    }

    auto segment_by_id = [&](int id) -> const BoundarySegment* {
        for (const BoundarySegment& segment : boundary.perimeter_segments) {
            if (segment.id == id) return &segment;
        }
        return nullptr;
    };
    auto segment_ids_length = [&](const vector<int>& ids) {
        double length = 0.0;
        for (int id : ids) {
            const BoundarySegment* segment = segment_by_id(id);
            if (segment) length += std::max(segment->length, 0.0);
        }
        return length;
    };

    int split_side = -1;
    double split_length = -1.0;
    for (int i = 0; i < (int)active_sides.size(); i++) {
        if (active_sides[i].segment_ids.size() < 2) continue;
        double length = segment_ids_length(active_sides[i].segment_ids);
        if (length > split_length) {
            split_side = i;
            split_length = length;
        }
    }
    if (split_side < 0) {
        reason = "three-label promotion needs one multi-segment side to split";
        return false;
    }

    vector<vector<int>> promoted_segments;
    for (int i = 0; i < (int)active_sides.size(); i++) {
        const vector<int>& ids = active_sides[i].segment_ids;
        if (i != split_side) {
            promoted_segments.push_back(ids);
            continue;
        }

        double total = segment_ids_length(ids);
        double prefix = 0.0;
        int split_after = 1;
        double best_error = std::numeric_limits<double>::infinity();
        for (int j = 1; j < (int)ids.size(); j++) {
            const BoundarySegment* segment = segment_by_id(ids[j - 1]);
            if (segment) prefix += std::max(segment->length, 0.0);
            double error = std::abs(prefix - 0.5 * total);
            if (error < best_error) {
                best_error = error;
                split_after = j;
            }
        }
        split_after = std::max(1, std::min((int)ids.size() - 1, split_after));
        promoted_segments.emplace_back(ids.begin(), ids.begin() + split_after);
        promoted_segments.emplace_back(ids.begin() + split_after, ids.end());
    }
    if (promoted_segments.size() != 4) {
        reason = "three-label promotion did not produce four sides";
        return false;
    }

    labeling.final_label_groups.clear();
    labeling.abstract_sides.clear();
    labeling.final_corners.clear();
    for (int i = 0; i < 4; i++) {
        vector<Vector3d> guide;
        for (int segment_id : promoted_segments[i]) {
            const BoundarySegment* segment = segment_by_id(segment_id);
            if (!segment) continue;
            const vector<Vector3d>& src =
                segment->guide_positions.empty()
                    ? segment->authoritative_positions
                    : segment->guide_positions;
            for (int k = 0; k < (int)src.size(); k++) {
                if (!guide.empty() && k == 0 &&
                    (guide.back() - src[k]).norm() <= 1e-10) {
                    continue;
                }
                guide.push_back(src[k]);
            }
        }
        if (guide.size() < 2) {
            reason = "promoted side has too few guide samples";
            return false;
        }

        LabelGroup group;
        group.id = i;
        group.active = true;
        group.segment_ids = promoted_segments[i];
        group.guide_polyline = guide;
        labeling.final_label_groups.push_back(group);

        AbstractSide side;
        side.side_index = i;
        side.label_group_ids.push_back(i);
        side.segment_ids = promoted_segments[i];
        side.average_tangent = average_segment_tangent(guide);
        labeling.abstract_sides.push_back(side);
    }

    labeling.valid = true;
    labeling.ambiguous = true;
    labeling.reason += "; promoted 3 active sides to 4 by splitting a multi-segment label";
    labeling.final_label_count = 4;
    labeling.final_corner_count = 4;
    labeling.orientation.valid = false;
    return true;
}

static Vector3d average_region_normal(
    const MatrixXd& V,
    const KktGlobalStitchingResult& param) {
    Vector3d n = Vector3d::Zero();
    for (int fi = 0; fi < param.local_faces.rows(); fi++) {
        int ia = param.local_faces(fi, 0);
        int ib = param.local_faces(fi, 1);
        int ic = param.local_faces(fi, 2);
        if (ia < 0 || ib < 0 || ic < 0 ||
            ia >= (int)param.region_vertex_ids.size() ||
            ib >= (int)param.region_vertex_ids.size() ||
            ic >= (int)param.region_vertex_ids.size()) {
            continue;
        }
        int a = param.region_vertex_ids[ia];
        int b = param.region_vertex_ids[ib];
        int c = param.region_vertex_ids[ic];
        if (a < 0 || b < 0 || c < 0 || a >= V.rows() || b >= V.rows() || c >= V.rows()) continue;
        Vector3d pa = V.row(a).transpose();
        Vector3d pb = V.row(b).transpose();
        Vector3d pc = V.row(c).transpose();
        n += (pb - pa).cross(pc - pa);
    }
    return n.norm() > 1e-12 ? n.normalized() : Vector3d(0.0, 0.0, 1.0);
}

static bool install_quadlike_labeling_fallback(
    BoundarySegmentationResult& boundary,
    AutomaticLabelingResult& labeling,
    const string& output_dir,
    bool export_debug,
    string& reason) {
    if (boundary.perimeter_loop_index < 0 ||
        boundary.perimeter_loop_index >= (int)boundary.loops.size()) {
        reason = "quad-like fallback needs a perimeter loop";
        return false;
    }
    const AuthoritativeBoundaryLoop& loop =
        boundary.loops[boundary.perimeter_loop_index];
    if (loop.vertex_ids.size() < 4 || loop.positions.size() != loop.vertex_ids.size()) {
        reason = "quad-like fallback perimeter loop is too small";
        return false;
    }

    RegionBoundaryLoop rloop;
    rloop.vertex_ids = loop.vertex_ids;
    rloop.positions = loop.positions;
    rloop.closed = loop.closed;

    auto install_from_segments = [&](const vector<BoundarySegment>& segments,
                                     const string& label_reason) -> bool {
        if (segments.size() != 4) return false;
        boundary.perimeter_segments = segments;
        labeling = AutomaticLabelingResult();
        labeling.valid = true;
        labeling.ambiguous = true;
        labeling.reason = label_reason;
        labeling.final_label_count = 4;
        labeling.final_corner_count = 4;
        labeling.orientation.valid = false;

        for (int s = 0; s < 4; s++) {
            const BoundarySegment& seg = boundary.perimeter_segments[s];
            if (seg.authoritative_vertex_ids.size() < 2) return false;
            LabelGroup group;
            group.id = s;
            group.active = true;
            group.segment_ids.push_back(seg.id);
            group.guide_polyline = seg.guide_positions;
            labeling.final_label_groups.push_back(group);

            AbstractSide side;
            side.side_index = s;
            side.label_group_ids.push_back(s);
            side.segment_ids.push_back(seg.id);
            side.average_tangent = average_segment_tangent(seg.guide_positions);
            labeling.abstract_sides.push_back(side);

            VirtualCorner corner;
            corner.id = s;
            corner.prev_group_id = (s + 3) % 4;
            corner.next_group_id = s;
            corner.position = seg.authoritative_positions.front();
            corner.closest_prev = corner.position;
            corner.closest_next = corner.position;
            corner.type = VirtualCornerType::StrongConvex;
            labeling.final_corners.push_back(corner);
        }
        return true;
    };

    if (boundary.perimeter_segments.size() == 3) {
        int split_index = -1;
        double best_len = -1.0;
        for (int i = 0; i < 3; i++) {
            const BoundarySegment& s = boundary.perimeter_segments[i];
            if (s.authoritative_vertex_ids.size() >= 3 && s.length > best_len) {
                split_index = i;
                best_len = s.length;
            }
        }
        if (split_index >= 0) {
            vector<BoundarySegment> split_segments;
            for (int i = 0; i < 3; i++) {
                const BoundarySegment& src = boundary.perimeter_segments[i];
                if (i != split_index) {
                    BoundarySegment dst = src;
                    dst.id = (int)split_segments.size();
                    split_segments.push_back(dst);
                    continue;
                }
                int m = (int)src.authoritative_vertex_ids.size();
                int mid = m / 2;
                if (src.authoritative_positions.size() == src.authoritative_vertex_ids.size()) {
                    vector<double> cumulative(m, 0.0);
                    for (int k = 1; k < m; k++) {
                        cumulative[k] = cumulative[k - 1] +
                            (src.authoritative_positions[k] -
                             src.authoritative_positions[k - 1]).norm();
                    }
                    const double half_length = 0.5 * cumulative.back();
                    mid = 1;
                    while (mid < m - 2 && cumulative[mid] < half_length) mid++;
                }
                mid = std::max(1, std::min(m - 2, mid));
                for (int part = 0; part < 2; part++) {
                    int begin = part == 0 ? 0 : mid;
                    int end = part == 0 ? mid : m - 1;
                    BoundarySegment dst;
                    dst.id = (int)split_segments.size();
                    dst.loop_id = src.loop_id;
                    dst.adjacent_region_id = src.adjacent_region_id;
                    dst.touches_feature_barrier = src.touches_feature_barrier;
                    dst.touches_mesh_boundary = src.touches_mesh_boundary;
                    dst.touches_user_marker = src.touches_user_marker;
                    for (int k = begin; k <= end; k++) {
                        dst.authoritative_vertex_ids.push_back(src.authoritative_vertex_ids[k]);
                        dst.authoritative_positions.push_back(src.authoritative_positions[k]);
                        dst.guide_positions.push_back(
                            k < (int)src.guide_positions.size()
                                ? src.guide_positions[k]
                                : src.authoritative_positions[k]);
                    }
                    for (int k = 1; k < (int)dst.authoritative_vertex_ids.size(); k++) {
                        dst.edge_keys.push_back(EdgeKey(
                            dst.authoritative_vertex_ids[k - 1],
                            dst.authoritative_vertex_ids[k]));
                    }
                    dst.length = segment_length_3d(dst.authoritative_positions);
                    if (dst.authoritative_positions.size() >= 2) {
                        Vector3d tb = dst.authoritative_positions[1] -
                                      dst.authoritative_positions[0];
                        int dm = (int)dst.authoritative_positions.size();
                        Vector3d te = dst.authoritative_positions[dm - 1] -
                                      dst.authoritative_positions[dm - 2];
                        dst.tangent_begin = tb.norm() > 1e-12 ? tb.normalized() : Vector3d::Zero();
                        dst.tangent_end = te.norm() > 1e-12 ? te.normalized() : Vector3d::Zero();
                    }
                    split_segments.push_back(dst);
                }
            }
            if (install_from_segments(
                    split_segments,
                    "segment-preserving four-side fallback after automatic labeling degeneracy")) {
                return true;
            }
        }
    }

    QuadLikeBoundaryConfig qcfg;
    QuadLikeBoundaryResult quad;
    std::array<int, 4> manual;
    string manual_reason;
    if (load_quad_like_boundary_manual_config("quad_like_manual_corners.txt", manual, manual_reason)) {
        bool manual_in_range = true;
        for (int idx : manual) {
            if (idx < 0 || idx >= (int)rloop.vertex_ids.size()) manual_in_range = false;
        }
        if (manual_in_range) {
            qcfg.use_manual_corners = true;
            qcfg.manual_corner_loop_indices = manual;
            quad = split_quad_like_boundary(rloop, qcfg);
        }
    }
    if (!quad.success || !quad.boundary.valid) {
        qcfg = QuadLikeBoundaryConfig();
        quad = split_quad_like_boundary(rloop, qcfg);
    }
    if (export_debug) {
        export_quad_like_boundary_pca_debug_obj(
            path_join(output_dir, "labeling_quadlike_fallback_pca.obj"), quad.debug);
    }
    if (!quad.success || !quad.boundary.valid) {
        reason = "quad-like fallback failed: " + quad.reason;
        return false;
    }

    boundary.perimeter_segments.clear();
    labeling = AutomaticLabelingResult();
    labeling.valid = true;
    labeling.ambiguous = true;
    labeling.reason = "quad-like fallback after automatic labeling degeneracy";
    labeling.final_label_count = 4;
    labeling.final_corner_count = 4;
    labeling.orientation.valid = false;

    int n = (int)loop.vertex_ids.size();
    for (int s = 0; s < 4; s++) {
        int a = quad.boundary.corner_loop_indices[s];
        int b = quad.boundary.corner_loop_indices[(s + 1) % 4];
        if (a < 0 || a >= n || b < 0 || b >= n) {
            reason = "quad-like fallback produced invalid corner index";
            return false;
        }
        vector<int> loop_ids = loop_indices_between(a, b, n);
        if (loop_ids.size() < 2) {
            reason = "quad-like fallback side has too few vertices";
            return false;
        }

        BoundarySegment seg;
        seg.id = s;
        seg.loop_id = loop.id;
        seg.adjacent_region_id = -1;
        for (int idx : loop_ids) {
            seg.authoritative_vertex_ids.push_back(loop.vertex_ids[idx]);
            seg.authoritative_positions.push_back(loop.positions[idx]);
            seg.guide_positions.push_back(loop.positions[idx]);
        }
        for (int i = 1; i < (int)seg.authoritative_vertex_ids.size(); i++) {
            int va = seg.authoritative_vertex_ids[i - 1];
            int vb = seg.authoritative_vertex_ids[i];
            seg.edge_keys.push_back(EdgeKey(va, vb));
            for (const DirectedBoundaryEdge& e : loop.directed_edges) {
                if ((e.from == va && e.to == vb) || (e.from == vb && e.to == va)) {
                    seg.touches_feature_barrier = seg.touches_feature_barrier || e.is_feature_barrier;
                    seg.touches_mesh_boundary = seg.touches_mesh_boundary || e.is_mesh_boundary;
                    seg.touches_user_marker = seg.touches_user_marker || e.is_user_marker;
                    if (seg.adjacent_region_id < 0) seg.adjacent_region_id = e.adjacent_region_id;
                    break;
                }
            }
        }
        seg.length = segment_length_3d(seg.authoritative_positions);
        if (seg.authoritative_positions.size() >= 2) {
            Vector3d tb = seg.authoritative_positions[1] - seg.authoritative_positions[0];
            int m = (int)seg.authoritative_positions.size();
            Vector3d te = seg.authoritative_positions[m - 1] - seg.authoritative_positions[m - 2];
            seg.tangent_begin = tb.norm() > 1e-12 ? tb.normalized() : Vector3d::Zero();
            seg.tangent_end = te.norm() > 1e-12 ? te.normalized() : Vector3d::Zero();
        }
        boundary.perimeter_segments.push_back(seg);

        LabelGroup group;
        group.id = s;
        group.active = true;
        group.segment_ids.push_back(s);
        group.guide_polyline = seg.guide_positions;
        labeling.final_label_groups.push_back(group);

        AbstractSide side;
        side.side_index = s;
        side.label_group_ids.push_back(s);
        side.segment_ids.push_back(s);
        side.average_tangent = average_segment_tangent(seg.guide_positions);
        labeling.abstract_sides.push_back(side);

        VirtualCorner corner;
        corner.id = s;
        corner.prev_group_id = (s + 3) % 4;
        corner.next_group_id = s;
        corner.position = loop.positions[a];
        corner.closest_prev = corner.position;
        corner.closest_next = corner.position;
        corner.type = VirtualCornerType::StrongConvex;
        labeling.final_corners.push_back(corner);
    }
    return true;
}

} // namespace

TrimmedBSplinePipelineResult run_single_region_trimmed_bspline_pipeline(
    const MatrixXd& V,
    const MatrixXi& F,
    const vector<int>& face_region_ids,
    int target_region_id,
    const TrimmedBSplinePipelineConfig& config) {
    TrimmedBSplinePipelineResult result;
    ensure_dir(config.output_dir);
    std::ofstream log(path_join(config.output_dir, "pipeline.log"));
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed_seconds = [&]() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - t0).count();
    };
    auto progress = [&](const string& message) {
        if (log.is_open()) {
            log << "[" << elapsed_seconds() << "s] " << message << "\n";
            log.flush();
        }
        if (config.print_progress_to_console) {
            std::cout << "[trimmed pipeline] [" << elapsed_seconds()
                      << "s] " << message << std::endl;
        }
    };
    auto fail = [&](const string& reason) {
        result.valid = false;
        result.reason = reason;
        result.metrics.valid = false;
        result.metrics.reason = reason;
        write_metrics_json(path_join(config.output_dir, "metrics.json"), result.metrics);
        progress("FAIL: " + reason);
        return result;
    };
    if (V.cols() != 3 || F.cols() != 3 || (int)face_region_ids.size() != F.rows()) {
        return fail("invalid mesh or face region labels");
    }

    progress("1/9 boundary segmentation: start");
    BoundarySegmentationConfig bcfg;
    bcfg.guide_smoothing_iterations = 1;
    BoundarySegmentationResult boundary =
        build_trimmed_region_input(V, F, face_region_ids, target_region_id, bcfg);
    if (config.export_debug_artifacts) {
        export_trimmed_region_input_debug_obj(
            path_join(config.output_dir, "boundary_segments.obj"), V, F, boundary);
    }
    if (!boundary.valid) return fail("boundary segmentation failed: " + boundary.reason);
    progress("1/9 boundary segmentation: done, loops=" +
             std::to_string(boundary.loops.size()) +
             ", perimeter_segments=" +
             std::to_string(boundary.perimeter_segments.size()));

    progress("2/9 automatic labeling: start");
    AutomaticLabelingConfig lcfg;
    lcfg.debug_output_dir = path_join(config.output_dir, "labeling_debug");
    lcfg.export_debug = config.export_debug_artifacts;
    lcfg.min_supported_label_count = 4;
    AutomaticLabelingResult labeling = run_automatic_labeling(V, F, boundary, lcfg);
    if (!labeling.valid) {
        progress("2/9 automatic labeling failed (" + labeling.reason +
                 "); using four-side fallback");
        string fallback_reason;
        if (!install_quadlike_labeling_fallback(
                boundary, labeling, config.output_dir,
                config.export_debug_artifacts, fallback_reason)) {
            return fail("automatic labeling failed and fallback failed: " + fallback_reason);
        }
    }
    if (labeling.valid &&
        active_labeled_side_count(labeling) == 4 &&
        labeling.final_corner_count < 4) {
        progress("2/9 automatic labeling has weak 4-side corners; trying 3-label reduction promotion");
        AutomaticLabelingConfig reduced_lcfg = lcfg;
        reduced_lcfg.min_supported_label_count = 3;
        AutomaticLabelingResult reduced_labeling =
            run_automatic_labeling(V, F, boundary, reduced_lcfg);
        string promotion_reason;
        if (reduced_labeling.valid &&
            active_labeled_side_count(reduced_labeling) == 3 &&
            promote_three_label_sides_to_four(
                boundary, reduced_labeling, promotion_reason)) {
            labeling = reduced_labeling;
            progress("2/9 weak 4-side labeling replaced by promoted 3-label split");
        } else {
            progress("2/9 weak 4-side labeling could not be promoted (" +
                     (promotion_reason.empty() ? reduced_labeling.reason
                                               : promotion_reason) +
                     "); keeping automatic labeling for primary parameterization");
        }
    }
    int active_sides = active_labeled_side_count(labeling);
    if (active_sides == 3) {
        string promotion_reason;
        if (promote_three_label_sides_to_four(boundary, labeling, promotion_reason)) {
            progress("2/9 automatic labeling promoted 3 active sides to 4 by splitting a multi-segment label");
            active_sides = active_labeled_side_count(labeling);
        } else {
            progress("2/9 automatic labeling kept 3 active sides; promotion skipped: " +
                     promotion_reason);
        }
    }
    if (active_sides < 2 || active_sides > 4) {
        progress("2/9 automatic labeling produced " +
                 std::to_string(active_sides) +
                 " active sides; using quad-like fallback");
        string fallback_reason;
        if (!install_quadlike_labeling_fallback(
                boundary, labeling, config.output_dir,
                config.export_debug_artifacts, fallback_reason)) {
            return fail("automatic labeling degenerates and fallback failed: " + fallback_reason);
        }
    }
    BezierGuidingFrameConfig fcfg;
    fcfg.debug_output_dir = path_join(config.output_dir, "guiding_frame_debug");
    fcfg.max_degree = 4;
    fcfg.export_debug = config.export_debug_artifacts;
    BezierGuidingFrameResult frame;
    ConstrainedArapResult arap;
    RotationAngleInitializationResult rot;
    KktGlobalStitchingResult kkt;
    bool parameterization_ok = false;
    string parameterization_fail_log;
    bool parameterization_fallback_used = false;

    while (true) {
    write_labels_json(path_join(config.output_dir, "labels.json"), boundary, labeling);
    progress("2/9 automatic labeling: done, labels=" +
             std::to_string(labeling.final_label_count) +
             ", corners=" + std::to_string(labeling.final_corner_count) +
             ", ambiguous=" + string(labeling.ambiguous ? "true" : "false"));

    progress("3/9 Bezier guiding frame: start");
    frame = build_bezier_guiding_frame(V, F, boundary, labeling, fcfg);
    write_guiding_frame_obj(path_join(config.output_dir, "guiding_frame.obj"), frame);
    if (!frame.valid) return fail("guiding frame failed: " + frame.reason);
    progress("3/9 Bezier guiding frame: done, curves=" +
             std::to_string(frame.curves.size()) +
             ", rms=" + std::to_string(frame.rms_error));

    progress("4/9 constrained ARAP/KKT role search: start");
    vector<std::array<ParameterSideRole, 4>> role_candidates =
        role_mapping_candidates(labeling);
    parameterization_ok = false;
    parameterization_fail_log.clear();

    for (int ri = 0; ri < (int)role_candidates.size(); ri++) {
        const auto& roles = role_candidates[ri];
        AutomaticLabelingResult trial_labeling = labeling_with_roles(labeling, roles);
        string role_desc = roles_to_string(roles);
        progress("4/9 role candidate " + std::to_string(ri + 1) + "/" +
                 std::to_string((int)role_candidates.size()) + ": " + role_desc);

        ConstrainedArapConfig acfg;
        acfg.debug_output_prefix =
            path_join(config.output_dir, "arap_mvp_role_" + std::to_string(ri));
        acfg.fail_on_flips = true;
        acfg.export_debug = config.export_debug_artifacts;
        acfg.use_labeling_orientation = true;
        for (int i = 0; i < 4; i++) acfg.fallback_side_roles[i] = roles[i];
        ConstrainedArapResult trial_arap =
            parameterize_constrained_arap_mvp(V, F, boundary, trial_labeling, frame, acfg);
        if (!trial_arap.valid) {
            parameterization_fail_log +=
                "roles[" + role_desc + "] ARAP failed: " + trial_arap.reason + "; ";
            progress("4/9 role rejected by ARAP: " + trial_arap.reason);
            continue;
        }

        RotationAngleInitializationConfig rcfg;
        rcfg.debug_prefix =
            path_join(config.output_dir, "rotation_angle_role_" + std::to_string(ri));
        rcfg.export_debug = config.export_debug_artifacts;
        rcfg.print_progress_to_console = config.print_progress_to_console;
        rcfg.use_sparse_linear_solvers = true;
        RotationAngleInitializationResult trial_rot =
            initialize_rotation_angles_section421(V, F, boundary, trial_labeling, rcfg);
        if (!trial_rot.valid) {
            parameterization_fail_log +=
                "roles[" + role_desc + "] rotation failed: " + trial_rot.reason + "; ";
            progress("5/9 role rejected by rotation init: " + trial_rot.reason);
            continue;
        }

        KktGlobalStitchingConfig kcfg;
        kcfg.debug_prefix =
            path_join(config.output_dir, "parameterized_region_role_" + std::to_string(ri));
        kcfg.enable_smoothed_arap = config.enable_smoothed_arap;
        kcfg.lambda_smooth = config.arap_smoothing_weight;
        kcfg.export_debug = config.export_debug_artifacts;
        kcfg.enable_dense_diagnostics = false;
        kcfg.print_progress_to_console = config.print_progress_to_console;
        KktGlobalStitchingResult trial_kkt =
            parameterize_kkt_global_stitching(
                V, F, boundary, trial_labeling, frame, trial_rot, kcfg);
        if (!trial_kkt.valid) {
            if (trial_arap.valid &&
                trial_arap.flipped_triangle_count == 0 &&
                trial_kkt.reason.find("flipped or degenerate") != string::npos) {
                labeling = trial_labeling;
                arap = trial_arap;
                rot = trial_rot;
                kkt = kkt_result_from_constrained_arap(
                    trial_arap,
                    "constrained ARAP parameterization kept because KKT stitching lost injectivity");
                parameterization_ok = true;
                progress("6/9 KKT lost injectivity; using validated constrained ARAP UV for role: " +
                         role_desc +
                         ", arap_flips=" + std::to_string(arap.flipped_triangle_count) +
                         ", kkt_reason=" + trial_kkt.reason);
                break;
            }
            parameterization_fail_log +=
                "roles[" + role_desc + "] KKT failed: " + trial_kkt.reason + "; ";
            progress("6/9 role rejected by KKT: " + trial_kkt.reason);
            continue;
        }

        labeling = trial_labeling;
        arap = trial_arap;
        rot = trial_rot;
        kkt = trial_kkt;
        parameterization_ok = true;
        progress("4-6/9 selected role mapping: " + role_desc +
                 ", arap_flips=" + std::to_string(arap.flipped_triangle_count) +
                 ", kkt_flips=" + std::to_string(kkt.flipped_triangle_count));
        break;
    }
    if (!parameterization_ok) {
        if (!parameterization_fallback_used) {
            progress("4-6/9 primary labeling parameterization failed; retrying with four-side fallback");
            string fallback_reason;
            if (!install_quadlike_labeling_fallback(
                    boundary, labeling, config.output_dir,
                    config.export_debug_artifacts, fallback_reason)) {
                return fail("parameterization failed and fallback failed: " +
                            fallback_reason + "; primary failures: " +
                            parameterization_fail_log);
            }
            parameterization_fallback_used = true;
            continue;
        }
        return fail("parameterization failed for all side-role mappings: " +
                    parameterization_fail_log);
    }
    break;
    }
    write_labels_json(path_join(config.output_dir, "labels.json"), boundary, labeling);
    if (config.export_debug_artifacts) {
        std::remove(path_join(config.output_dir, "parameterized_region_uv.obj").c_str());
        export_kkt_global_stitching_debug(path_join(config.output_dir, "parameterized_region"), kkt);
        std::rename(path_join(config.output_dir, "parameterized_region_uv.obj").c_str(),
                    path_join(config.output_dir, "parameterized_region.obj").c_str());
    } else {
        MatrixXd local_uv(kkt.region_vertex_ids.size(), 2);
        for (int i = 0; i < (int)kkt.region_vertex_ids.size(); i++) {
            local_uv.row(i) = kkt.UV.row(kkt.region_vertex_ids[i]);
        }
        write_uv_mesh_obj(path_join(config.output_dir, "parameterized_region.obj"),
                          local_uv, kkt.local_faces);
    }
    progress("6/9 KKT stitching: done, flips=" +
             std::to_string(kkt.flipped_triangle_count) +
             ", mean_arap=" + std::to_string(kkt.mean_arap_residual));

    progress("7/9 rectangular UV extension: start");
    RectangularDomainExtensionInput rect_input =
        make_rectangular_extension_input(boundary, kkt);
    RectangularDomainExtensionConfig xcfg;
    xcfg.debug_prefix = path_join(config.output_dir, "rectangular_extension");
    xcfg.sample_count_u = config.extension_sample_u;
    xcfg.sample_count_v = config.extension_sample_v;
    xcfg.margin = config.extension_margin;
    xcfg.export_debug = config.export_debug_artifacts;
    RectangularDomainExtensionResult domain =
        build_rectangular_domain_extension(rect_input, xcfg);
    if (!domain.valid) return fail("rectangular domain extension failed: " + domain.reason);
    write_uv_mesh_obj(path_join(config.output_dir, "extended_uv_mesh.obj"),
                      domain.full_uv_vertices, domain.full_faces);
    write_trim_loops_obj(path_join(config.output_dir, "uv_trim_loops.obj"),
                         domain.authoritative_trim_loops);
    progress("7/9 rectangular UV extension: done, uv_vertices=" +
             std::to_string((int)domain.full_uv_vertices.rows()) +
             ", faces=" + std::to_string((int)domain.full_faces.rows()) +
             ", artificial_vertices=" +
             std::to_string(domain.artificial_vertex_count));

    progress("8/9 3D extension: start");
    MatrixXd domain_positions = positions_for_extended_domain(V, kkt, domain);
    Polyharmonic3DExtensionResult ext_g1;
    if (config.run_ablation_baselines) {
        progress("8/9 3D extension: G1 start");
        Polyharmonic3DExtensionConfig g1cfg;
        g1cfg.mode = PolyharmonicContinuityMode::G1;
        g1cfg.debug_prefix = path_join(config.output_dir, "extended_3d_g1");
        g1cfg.export_debug = config.export_debug_artifacts;
        ext_g1 = extend_polyharmonic_3d(domain, domain_positions, g1cfg);
        if (!ext_g1.valid) return fail("G1 polyharmonic extension failed: " + ext_g1.reason);
        progress("8/9 3D extension: G1 done");
    }
    progress("8/9 3D extension: G2 start");
    Polyharmonic3DExtensionConfig g2cfg;
    g2cfg.mode = PolyharmonicContinuityMode::G2;
    g2cfg.debug_prefix = path_join(config.output_dir, "extended_3d_g2");
    g2cfg.export_debug = config.export_debug_artifacts;
    g2cfg.fairness_weight = config.enable_extension_fairness ? config.extension_fairness_weight : 0.0;
    g2cfg.isocurve_fairness_weight = config.extension_isocurve_weight;
    Polyharmonic3DExtensionResult ext_g2 =
        extend_polyharmonic_3d(domain, domain_positions, g2cfg);
    if (!ext_g2.valid) return fail("G2 polyharmonic extension failed: " + ext_g2.reason);
    auto extension_overgrown = [&](const Polyharmonic3DExtensionResult& ext) {
        return ext.bbox_growth > config.max_extension_bbox_growth ||
               ext.surface_area_growth > config.max_extension_area_growth;
    };
    if (extension_overgrown(ext_g2)) {
        progress("8/9 3D extension: G2 overgrown; retrying stabilized anchors");
        Polyharmonic3DExtensionResult best_ext = ext_g2;
        auto extension_score = [&](const Polyharmonic3DExtensionResult& ext) {
            double bbox_excess = std::max(0.0, ext.bbox_growth - 1.0);
            double area_excess = std::max(0.0, ext.surface_area_growth - 1.0);
            return 2.0 * bbox_excess + area_excess + 0.25 * ext.mean_unknown_displacement;
        };
        vector<double> anchor_weights = {0.05, 0.1, 0.25, 0.5, 1.0, 10.0, 100.0, 1000.0};
        for (double anchor_weight : anchor_weights) {
            Polyharmonic3DExtensionConfig retry_cfg = g2cfg;
            retry_cfg.initial_position_weight = anchor_weight;
            retry_cfg.export_debug = false;
            Polyharmonic3DExtensionResult retry =
                extend_polyharmonic_3d(domain, domain_positions, retry_cfg);
            if (!retry.valid) {
                progress("8/9 3D extension stabilized retry failed: anchor=" +
                         std::to_string(anchor_weight) + ", reason=" + retry.reason);
                continue;
            }
            progress("8/9 3D extension stabilized retry: anchor=" +
                     std::to_string(anchor_weight) +
                     ", area_growth=" + std::to_string(retry.surface_area_growth) +
                     ", bbox_growth=" + std::to_string(retry.bbox_growth) +
                     ", mean_displacement=" +
                     std::to_string(retry.mean_unknown_displacement));
            if (extension_score(retry) < extension_score(best_ext)) {
                best_ext = retry;
            }
            if (!extension_overgrown(best_ext)) break;
        }
        if (extension_score(best_ext) < extension_score(ext_g2) - 1e-10) {
            progress("8/9 3D extension stabilized: bbox_growth=" +
                     std::to_string(ext_g2.bbox_growth) + " -> " +
                     std::to_string(best_ext.bbox_growth) +
                     ", area_growth=" +
                     std::to_string(ext_g2.surface_area_growth) + " -> " +
                     std::to_string(best_ext.surface_area_growth));
            ext_g2 = best_ext;
            if (config.export_debug_artifacts) {
                export_polyharmonic_3d_extension_debug(
                    path_join(config.output_dir, "extended_3d_g2_stabilized"), ext_g2);
            }
        } else {
            progress("8/9 3D extension stabilized retry did not improve over original G2");
        }
    }
    write_mesh_obj(path_join(config.output_dir, "extended_3d_mesh.obj"),
                   ext_g2.extended_vertices, ext_g2.faces);
    progress("8/9 3D extension: G2 done, area_growth=" +
             std::to_string(ext_g2.surface_area_growth) +
             ", bbox_growth=" + std::to_string(ext_g2.bbox_growth));

    progress("9/9 tensor-product B-spline fitting: start");
    vector<TensorProductBSplineFitResult> baselines;
    if (config.run_ablation_baselines) {
        progress("9/9 fitting baseline A: start");
        TensorProductBSplineFitInput in_a =
            make_trimmed_fit_input(V, arap, TensorProductFitBaseline::TrimmedOnly);
        TensorProductBSplineFitResult fit_a = fit_tensor_product_cubic_bspline_surface(
            in_a, fit_config(config, path_join(config.output_dir, "baseline_A_existing_parameterization"), true));
        baselines.push_back(fit_a);
        progress("9/9 fitting baseline A: done, valid=" +
                 string(fit_a.valid ? "true" : "false"));
        progress("9/9 fitting baseline B: start");
        TensorProductBSplineFitInput in_b =
            make_trimmed_fit_input(V, kkt, TensorProductFitBaseline::ArapOnly);
        TensorProductBSplineFitResult fit_b = fit_tensor_product_cubic_bspline_surface(
            in_b, fit_config(config, path_join(config.output_dir, "baseline_B_arap"), true));
        baselines.push_back(fit_b);
        progress("9/9 fitting baseline B: done, valid=" +
                 string(fit_b.valid ? "true" : "false"));
        progress("9/9 fitting baseline C: start");
        TensorProductBSplineFitInput in_c =
            make_extended_fit_input(domain, ext_g1, TensorProductFitBaseline::ExtensionOnly);
        TensorProductBSplineFitResult fit_c = fit_tensor_product_cubic_bspline_surface(
            in_c, fit_config(config, path_join(config.output_dir, "baseline_C_extension"), false));
        baselines.push_back(fit_c);
        progress("9/9 fitting baseline C: done, valid=" +
                 string(fit_c.valid ? "true" : "false"));
    }
    progress("9/9 fitting final D: start");
    TensorProductBSplineFitInput in_d =
        make_extended_fit_input(domain, ext_g2, TensorProductFitBaseline::LabelingExtension);
    append_authoritative_boundary_fit_samples(
        in_d, boundary, kkt, V, config.boundary_fit_weight);
    TensorProductBSplineFitConfig d_cfg = fit_config(
        config, path_join(config.output_dir, "baseline_D_labeling_extension"), false);
    d_cfg.control_net_initial_weight =
        std::max(d_cfg.control_net_initial_weight, 1e-3);
    TensorProductBSplineFitResult fit_d = fit_tensor_product_cubic_bspline_surface(
        in_d, d_cfg);
    baselines.push_back(fit_d);
    progress("9/9 fitting final D: done, valid=" +
             string(fit_d.valid ? "true" : "false") +
             ", original_rms=" +
             std::to_string(fit_d.original_region_error.rms_error));
    result.baseline_results = baselines;
    write_baseline_report(path_join(config.output_dir, "baseline_ablation.csv"), baselines);
    if (!fit_d.valid) return fail("final tensor-product B-spline fit failed: " + fit_d.reason);

    auto visible_fold_edges = [&](const TensorProductBSplineFitResult& fit) {
        vector<TrimLoop2D> loops = normalized_trim_loops(
            domain.authoritative_trim_loops, fit.uv_min, fit.uv_max);
        return count_sampled_trimmed_surface_fold_edges(
            fit.surface,
            loops,
            config.surface_sample_u,
            config.surface_sample_v);
    };

    int final_fold_edges = visible_fold_edges(fit_d);
    int final_full_fold_edges = count_sampled_surface_fold_edges(
        fit_d.surface, config.surface_sample_u, config.surface_sample_v);
    progress("9/9 final visible fold check: baseline=" +
             string(to_string(fit_d.baseline)) +
             ", visible_folded_edges=" + std::to_string(final_fold_edges) +
             ", full_rect_folded_edges=" + std::to_string(final_full_fold_edges));
    TensorProductBSplineFitStats final_boundary_stats =
        boundary_error_stats(boundary, kkt, V, fit_d);
    double boundary_limit =
        std::max(config.max_boundary_error_absolute,
                 config.max_boundary_error_ratio * boundary_bbox_diagonal(boundary));
    progress("9/9 final boundary fit check: rms=" +
             std::to_string(final_boundary_stats.rms_error) +
             ", max=" + std::to_string(final_boundary_stats.max_error) +
             ", limit=" + std::to_string(boundary_limit));
    if (final_fold_edges > 0) {
        vector<double> boundary_weights;
        auto add_boundary_weight = [&](double w) {
            w = std::max(config.boundary_fit_weight, w);
            for (double existing : boundary_weights) {
                if (std::abs(existing - w) <= std::max(1e-12, std::abs(w) * 1e-9)) return;
            }
            boundary_weights.push_back(w);
        };
        add_boundary_weight(config.boundary_fit_weight);

        vector<int> control_counts;
        auto add_control_count = [&](int c) {
            c = std::min(14, std::max(4, c));
            if (std::find(control_counts.begin(), control_counts.end(), c) == control_counts.end()) {
                control_counts.push_back(c);
            }
        };
        int base_control_count = std::max(config.control_count_u, config.control_count_v);
        add_control_count(base_control_count);
        add_control_count(base_control_count + 2);
        add_control_count(base_control_count + 4);
        add_control_count(12);
        add_control_count(14);

        vector<double> initial_weights;
        auto add_initial_weight = [&](double w) {
            w = std::max(0.0, w);
            for (double existing : initial_weights) {
                if (std::abs(existing - w) <= std::max(1e-12, std::abs(w) * 1e-9)) return;
            }
            initial_weights.push_back(w);
        };
        add_initial_weight(std::max(config.fitting_initial_weight, 1e-3));
        add_initial_weight(1e-1);
        add_initial_weight(1.0);

        vector<double> fairness_weights;
        auto add_fairness_weight = [&](double w) {
            w = std::max(0.0, w);
            for (double existing : fairness_weights) {
                if (std::abs(existing - w) <= std::max(1e-14, std::abs(w) * 1e-9)) return;
            }
            fairness_weights.push_back(w);
        };
        add_fairness_weight(std::max(config.fitting_regularization, 1e-7));
        add_fairness_weight(1e-5);
        add_fairness_weight(1e-4);
        add_fairness_weight(1e-3);
        add_fairness_weight(1e-2);

        TensorProductBSplineFitResult best_d = fit_d;
        int best_folds = final_fold_edges;
        TensorProductBSplineFitStats best_boundary_stats =
            final_boundary_stats;
        auto boundary_ok = [&](const TensorProductBSplineFitStats& stats) {
            return stats.max_error <= boundary_limit;
        };
        auto candidate_better = [&](int candidate_folds,
                                    const TensorProductBSplineFitResult& candidate,
                                    const TensorProductBSplineFitStats& candidate_boundary) {
            if (candidate_folds > best_folds) return false;
            bool cand_ok = boundary_ok(candidate_boundary);
            bool best_ok = boundary_ok(best_boundary_stats);
            if (!cand_ok && best_ok) return false;
            if (cand_ok && !best_ok) return true;
            if (!cand_ok && !best_ok) {
                if (candidate_folds < best_folds &&
                    candidate_boundary.max_error <= 1.10 * best_boundary_stats.max_error) {
                    return true;
                }
                if (candidate_folds == best_folds &&
                    candidate_boundary.max_error < best_boundary_stats.max_error) {
                    return true;
                }
                return candidate_folds < best_folds;
            }
            if (candidate_folds != best_folds) return candidate_folds < best_folds;
            double candidate_score =
                candidate.original_region_error.rms_error +
                2.0 * candidate_boundary.rms_error +
                5.0 * candidate_boundary.max_error;
            double best_score =
                best_d.original_region_error.rms_error +
                2.0 * best_boundary_stats.rms_error +
                5.0 * best_boundary_stats.max_error;
            return candidate_score < best_score;
        };
        for (double boundary_weight : boundary_weights) {
            for (int control_count : control_counts) {
                for (double fairness_weight : fairness_weights) {
                    for (double initial_weight : initial_weights) {
                        if (boundary_weight == config.boundary_fit_weight &&
                            control_count == base_control_count &&
                            initial_weight <= d_cfg.control_net_initial_weight + 1e-15 &&
                            fairness_weight <= d_cfg.control_net_fairness_weight + 1e-15) {
                            continue;
                        }
                        TensorProductBSplineFitInput repaired_input =
                            make_extended_fit_input(domain, ext_g2, TensorProductFitBaseline::LabelingExtension);
                        append_authoritative_boundary_fit_samples(
                            repaired_input, boundary, kkt, V, boundary_weight);
                        string tag = "baseline_D_labeling_extension_repair_w" +
                                     std::to_string((int)std::round(boundary_weight)) +
                                     "_c" + std::to_string(control_count) +
                                     "_f" + std::to_string((int)std::round(-std::log10(std::max(fairness_weight, 1e-12)))) +
                                     "_i" + std::to_string((int)std::round(-std::log10(std::max(initial_weight, 1e-12))));
                        TensorProductBSplineFitConfig repaired_cfg = fit_config(
                            config,
                            path_join(config.output_dir, tag),
                            false);
                        repaired_cfg.control_count_u = std::max(config.control_count_u, control_count);
                        repaired_cfg.control_count_v = std::max(config.control_count_v, control_count);
                        repaired_cfg.control_net_fairness_weight = fairness_weight;
                        repaired_cfg.control_net_initial_weight = initial_weight;
                        repaired_cfg.export_debug = false;
                        progress("9/9 repairing LabelingExtension: boundary_weight=" +
                                 std::to_string(boundary_weight) +
                                 ", fairness_weight=" + std::to_string(fairness_weight) +
                                 ", initial_weight=" + std::to_string(initial_weight) +
                                 ", controls=" +
                                 std::to_string(repaired_cfg.control_count_u) + "x" +
                                 std::to_string(repaired_cfg.control_count_v));
                        TensorProductBSplineFitResult repaired =
                            fit_tensor_product_cubic_bspline_surface(repaired_input, repaired_cfg);
                        if (!repaired.valid) {
                            progress("9/9 repairing LabelingExtension failed: " + repaired.reason);
                            continue;
                        }
                        int repaired_folds = visible_fold_edges(repaired);
                        int repaired_full_folds = count_sampled_surface_fold_edges(
                            repaired.surface,
                            config.surface_sample_u,
                            config.surface_sample_v);
                        TensorProductBSplineFitStats repaired_boundary_stats =
                            boundary_error_stats(boundary, kkt, V, repaired);
                        progress("9/9 repaired LabelingExtension visible fold check: boundary_weight=" +
                                 std::to_string(boundary_weight) +
                                 ", fairness_weight=" + std::to_string(fairness_weight) +
                                 ", initial_weight=" + std::to_string(initial_weight) +
                                 ", controls=" +
                                 std::to_string(repaired_cfg.control_count_u) + "x" +
                                 std::to_string(repaired_cfg.control_count_v) +
                                 ", visible_folded_edges=" + std::to_string(repaired_folds) +
                                 ", full_rect_folded_edges=" + std::to_string(repaired_full_folds) +
                                 ", original_rms=" +
                                 std::to_string(repaired.original_region_error.rms_error) +
                                 ", boundary_rms=" +
                                 std::to_string(repaired_boundary_stats.rms_error) +
                                 ", boundary_max=" +
                                 std::to_string(repaired_boundary_stats.max_error) +
                                 ", boundary_ok=" +
                                 string(boundary_ok(repaired_boundary_stats) ? "true" : "false"));

                        if (candidate_better(repaired_folds, repaired, repaired_boundary_stats)) {
                            best_d = repaired;
                            best_folds = repaired_folds;
                            best_boundary_stats = repaired_boundary_stats;
                        }
                        if (best_folds == 0 && boundary_ok(best_boundary_stats)) break;
                    }
                    if (best_folds == 0 && boundary_ok(best_boundary_stats)) break;
                }
                if (best_folds == 0 && boundary_ok(best_boundary_stats)) break;
            }
            if (best_folds == 0 && boundary_ok(best_boundary_stats)) break;
        }
        if (best_folds < final_fold_edges ||
            best_boundary_stats.max_error < final_boundary_stats.max_error) {
            progress("9/9 LabelingExtension repaired: folded_edges=" +
                     std::to_string(final_fold_edges) + " -> " +
                     std::to_string(best_folds) +
                     ", boundary_max=" +
                     std::to_string(final_boundary_stats.max_error) + " -> " +
                     std::to_string(best_boundary_stats.max_error));
            fit_d = best_d;
            final_fold_edges = best_folds;
            final_boundary_stats = best_boundary_stats;
        }
    }
    if (final_fold_edges > 0 && config.allow_non_labeling_surface_fallback) {
        int best_index = -1;
        int best_priority = 100;
        int best_fold_edges = final_fold_edges;
        for (int i = 0; i < (int)baselines.size(); i++) {
            const TensorProductBSplineFitResult& candidate = baselines[i];
            if (!candidate.valid ||
                candidate.baseline == TensorProductFitBaseline::LabelingExtension) {
                continue;
            }
            int folds = visible_fold_edges(candidate);
            progress("9/9 fallback fold check: baseline=" +
                     string(to_string(candidate.baseline)) +
                     ", visible_folded_edges=" + std::to_string(folds));
            int priority = fallback_priority(candidate.baseline);
            if (folds == 0 && priority < best_priority) {
                best_index = i;
                best_priority = priority;
                best_fold_edges = folds;
            }
        }
        if (best_index >= 0) {
            progress("9/9 final surface fallback: " +
                     string(to_string(fit_d.baseline)) +
                     " folded_edges=" + std::to_string(final_fold_edges) +
                     ", using " +
                     string(to_string(baselines[best_index].baseline)));
            fit_d = baselines[best_index];
            final_fold_edges = best_fold_edges;
        } else {
            progress("9/9 final surface fold warning: no non-folded fallback surface available");
        }
    } else if (final_fold_edges > 0) {
        progress("9/9 final surface fold warning: LabelingExtension still has visible folded edges; "
                 "non-labeling baseline fallback is disabled");
    }

    export_bspline_surface_control_net_obj(
        path_join(config.output_dir, "bspline_control_net.obj"), fit_d.surface);
    export_bspline_surface_mesh_obj(
        path_join(config.output_dir, "bspline_full_surface.obj"),
        fit_d.surface, config.surface_sample_u, config.surface_sample_v);

    vector<TrimLoop2D> norm_loops = normalized_trim_loops(
        domain.authoritative_trim_loops, fit_d.uv_min, fit_d.uv_max);
    progress("9/9 reusable trimmed B-spline asset: start");
    ReusableTrimmedBSplineSurface reusable_asset =
        make_reusable_trimmed_bspline_surface(
            fit_d.surface,
            norm_loops,
            fit_d.uv_min,
            fit_d.uv_max,
            target_region_id,
            config.trim_curve_control_count,
            config.trim_curve_fairness_weight,
            &V);
    if (!reusable_asset.valid) {
        return fail("reusable trimmed B-spline asset failed: " + reusable_asset.reason);
    }
    if (!export_reusable_trimmed_bspline_surface_debug(
            config.output_dir,
            reusable_asset,
            config.surface_sample_u,
            config.surface_sample_v)) {
        return fail("reusable trimmed B-spline asset export failed");
    }
    progress("9/9 reusable trimmed B-spline asset: done, trim_loops=" +
             std::to_string((int)reusable_asset.trim_loops.size()));

    MatrixXd trimV;
    MatrixXi trimF;
    if (!sample_trimmed_surface_regular_grid(
            fit_d.surface, norm_loops, config.surface_sample_u, config.surface_sample_v,
            trimV, trimF)) {
        return fail("trimmed B-spline surface sampling failed");
    }
    write_mesh_obj(path_join(config.output_dir, "bspline_trimmed_surface_unlocked.obj"), trimV, trimF);
    MatrixXd lockedTrimV;
    MatrixXi lockedTrimF;
    MatrixXd lockedTrimUV;
    bool locked_output_uses_harmonic = config.apply_harmonic_boundary_correction;
    bool locked_boundary_output = sample_trimmed_surface_with_authoritative_boundary(
        fit_d.surface, norm_loops, V,
        config.surface_sample_u, config.surface_sample_v,
        lockedTrimV, lockedTrimF, &lockedTrimUV,
        locked_output_uses_harmonic);
    if (locked_boundary_output) {
        int locked_normal_jumps = count_mesh_normal_jump_edges(lockedTrimV, lockedTrimF);
        progress("9/9 boundary-locked trimmed mesh check: harmonic=" +
                 string(locked_output_uses_harmonic ? "true" : "false") +
                 ", normal_jump_edges=" + std::to_string(locked_normal_jumps));
        if (locked_output_uses_harmonic && locked_normal_jumps > 0) {
            MatrixXd noHarmV;
            MatrixXi noHarmF;
            MatrixXd noHarmUV;
            bool no_harm_ok = sample_trimmed_surface_with_authoritative_boundary(
                fit_d.surface, norm_loops, V,
                config.surface_sample_u, config.surface_sample_v,
                noHarmV, noHarmF, &noHarmUV, false);
            if (no_harm_ok) {
                int no_harm_jumps = count_mesh_normal_jump_edges(noHarmV, noHarmF);
                progress("9/9 boundary-locked retry without harmonic correction: normal_jump_edges=" +
                         std::to_string(no_harm_jumps));
                write_mesh_obj(
                    path_join(config.output_dir, "bspline_trimmed_surface_cdt_boundary_locked_no_harmonic.obj"),
                    noHarmV,
                    noHarmF);
                write_uv_mesh_obj(
                    path_join(config.output_dir, "bspline_trimmed_surface_cdt_boundary_locked_no_harmonic_uv.obj"),
                    noHarmUV,
                    noHarmF);
                if (no_harm_jumps < locked_normal_jumps) {
                    lockedTrimV = noHarmV;
                    lockedTrimF = noHarmF;
                    lockedTrimUV = noHarmUV;
                    locked_output_uses_harmonic = false;
                    progress("9/9 boundary-locked mesh switched to no-harmonic correction");
                }
            }
        }
        int removed_unstable_faces =
            remove_normal_jump_faces_until_stable(lockedTrimV, lockedTrimF, &lockedTrimUV);
        if (removed_unstable_faces > 0) {
            int remaining_jumps = count_mesh_normal_jump_edges(lockedTrimV, lockedTrimF);
            progress("9/9 boundary-locked mesh removed unstable faces: removed=" +
                     std::to_string(removed_unstable_faces) +
                     ", remaining_normal_jump_edges=" +
                     std::to_string(remaining_jumps));
        }
        write_mesh_obj(path_join(config.output_dir, "bspline_trimmed_surface_cdt_boundary_locked.obj"),
                       lockedTrimV, lockedTrimF);
        write_uv_mesh_obj(path_join(config.output_dir, "bspline_trimmed_surface_cdt_boundary_locked_uv.obj"),
                          lockedTrimUV, lockedTrimF);
    }

    MatrixXd topologyTrimV;
    MatrixXi topologyTrimF;
    MatrixXd topologyTrimUV;
    bool topology_output = sample_bspline_on_region_topology_with_authoritative_boundary(
        fit_d.surface, fit_d.uv_min, fit_d.uv_max, V, boundary, kkt,
        topologyTrimV, topologyTrimF, &topologyTrimUV,
        config.snap_output_boundary_to_authoritative,
        config.apply_harmonic_boundary_correction);
    MatrixXd snappedTopologyV;
    MatrixXi snappedTopologyF;
    MatrixXd snappedTopologyUV;
    bool snapped_topology_output = sample_bspline_on_region_topology_with_authoritative_boundary(
        fit_d.surface, fit_d.uv_min, fit_d.uv_max, V, boundary, kkt,
        snappedTopologyV, snappedTopologyF, &snappedTopologyUV, true, false);
    if (snapped_topology_output) {
        write_mesh_obj(
            path_join(config.output_dir, "bspline_trimmed_surface_authoritative_boundary.obj"),
            snappedTopologyV, snappedTopologyF);
    }
    if (topology_output) {
        write_mesh_obj(
            path_join(config.output_dir, "bspline_trimmed_surface_region_topology.obj"),
            topologyTrimV,
            topologyTrimF);
    }
    bool final_prefers_topology = false;
    if (topology_output) {
        MeshValidationReport topology_validation =
            validate_trimmed_mesh(topologyTrimV, topologyTrimF, &topologyTrimUV);
        int topology_jumps = topology_validation.normal_jump_edges;
        progress("9/9 region-topology output check: faces=" +
                 std::to_string((int)topologyTrimF.rows()) +
                 ", normal_jump_edges=" + std::to_string(topology_jumps) +
                 ", nonmanifold=" +
                 std::to_string(topology_validation.nonmanifold_edges));
        final_prefers_topology =
            topology_validation.exact_duplicate_faces == 0 &&
            topology_validation.geometric_duplicate_faces == 0 &&
            topology_validation.degenerate_faces == 0 &&
            topology_validation.inconsistent_winding_edges == 0 &&
            topology_validation.normal_jump_edges == 0 &&
            topology_validation.nonmanifold_edges == 0;
    }

    if (final_prefers_topology) {
        trimV = topologyTrimV;
        trimF = topologyTrimF;
        progress("9/9 final render mesh uses region topology to avoid CDT boundary redundancy");
    } else if (locked_boundary_output) {
        trimV = lockedTrimV;
        trimF = lockedTrimF;
    }

    const bool final_uses_topology = final_prefers_topology;
    const MatrixXd* validation_uv =
        final_uses_topology ? &topologyTrimUV :
        (locked_boundary_output ? &lockedTrimUV : nullptr);
    MeshValidationReport validation_before =
        validate_trimmed_mesh(trimV, trimF, validation_uv);
    write_mesh_validation_json(
        path_join(config.output_dir, "trimmed_mesh_validation_before_fix.json"),
        validation_before, validation_uv);
    Vector3d target_normal = average_region_normal(V, kkt);
    remove_degenerate_faces(trimV, trimF);
    orient_mesh_faces_consistently(trimV, trimF, target_normal);
    MeshValidationReport validation_after =
        validate_trimmed_mesh(trimV, trimF, validation_uv);
    write_mesh_validation_json(
        path_join(config.output_dir, "trimmed_mesh_validation.json"),
        validation_after, validation_uv);
    export_mesh_validation_issue_objs(config.output_dir, trimV, trimF, validation_after);
    string final_output_mode =
        final_uses_topology ? (config.apply_harmonic_boundary_correction ?
            "region-topology-harmonic-boundary-corrected" :
            (config.snap_output_boundary_to_authoritative ?
            "region-topology-authoritative-boundary" :
            "region-topology-surface-consistent-boundary-weighted-fit")) :
        (locked_boundary_output ?
            (locked_output_uses_harmonic ?
                "cdt-authoritative-boundary-harmonic-cleaned" :
                "cdt-authoritative-boundary-no-harmonic-cleaned") :
            "regular-grid");
    write_trimmed_mesh_flicker_diagnosis(
        path_join(config.output_dir, "TRIMMED_MESH_FLICKER_DIAGNOSIS.md"),
        final_output_mode,
        validation_before,
        validation_after);
    write_region_topology_output_report(
        path_join(config.output_dir, "region_topology_output_report.json"),
        V,
        kkt,
        trimV,
        trimF,
        final_output_mode,
        final_uses_topology);

    write_mesh_obj(path_join(config.output_dir, "bspline_trimmed_surface.obj"), trimV, trimF);
    progress("9/9 trimmed surface sampling: done, mode=" +
             final_output_mode +
             ", vertices=" +
             std::to_string((int)trimV.rows()) +
             ", faces=" + std::to_string((int)trimF.rows()) +
             ", duplicates=" + std::to_string(validation_after.exact_duplicate_faces) +
             ", degenerate=" + std::to_string(validation_after.degenerate_faces) +
             ", bad_winding=" + std::to_string(validation_after.inconsistent_winding_edges));

    TensorProductBSplineFitStats boundary_stats =
        boundary_error_stats(boundary, kkt, V, fit_d);
    result.metrics.valid = true;
    result.metrics.reason = "ok";
    result.metrics.labeling_configuration =
        "automatic labeling, " + std::to_string(labeling.final_label_count) +
        " labels, " + std::to_string(labeling.final_corner_count) + " corners";
    result.metrics.ambiguous = labeling.ambiguous;
    result.metrics.flipped_triangle_count = kkt.flipped_triangle_count;
    result.metrics.mean_arap_distortion = kkt.mean_arap_residual;
    result.metrics.max_arap_distortion = kkt.max_arap_residual;
    result.metrics.label_coordinate_constraint_error = kkt.max_label_coordinate_error;
    result.metrics.guiding_frame_length_ratio_error = kkt.max_length_ratio_error;
    result.metrics.original_region_rms_error = fit_d.original_region_error.rms_error;
    result.metrics.original_region_max_error = fit_d.original_region_error.max_error;
    result.metrics.boundary_rms_error = boundary_stats.rms_error;
    result.metrics.boundary_max_error = boundary_stats.max_error;
    result.metrics.artificial_extension_curvature_mean = ext_g2.mean_boundary_curvature;
    result.metrics.artificial_extension_curvature_max = ext_g2.max_boundary_curvature;
    result.metrics.surface_area_growth_ratio = ext_g2.surface_area_growth;
    result.metrics.bounding_box_growth_ratio = ext_g2.bbox_growth;
    result.metrics.weak_control_point_count = fit_d.weak_support.weak_control_point_count;
    result.metrics.linear_system_condition_estimate = fit_d.condition_estimate;
    write_metrics_json(path_join(config.output_dir, "metrics.json"), result.metrics);

    progress("DONE");
    result.valid = true;
    result.reason = "ok";
    return result;
}

bool write_trimmed_bspline_reproduction_doc(const string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << "# Trimmed B-Spline Reproduction\n\n";
    out << "This project implements a single-region baseline pipeline for reproducing the main engineering stages around Vaitkus and Varady 2018/2019.\n\n";
    out << "## 2018 Modules\n\n";
    out << "- Rectangular parameter-domain extension: `rectangular_domain_extension.*`.\n";
    out << "- Polyharmonic 3D extension with optional isocurve and mesh fairness: `polyharmonic_3d_extension.*`.\n";
    out << "- Tensor-product cubic B-spline fitting on the completed rectangle: `tensor_product_bspline_fitter.*`.\n";
    out << "- Reusable trimmed-surface asset export: `reusable_trimmed_bspline_surface.*` stores the untrimmed tensor-product B-spline surface plus 2D trim loops.\n";
    out << "- Trim-loop based sampled surface clipping in `trimmed_bspline_pipeline.*` and asset-derived tessellation in `reusable_trimmed_bspline_surface.*`.\n\n";
    out << "## 2019 Modules\n\n";
    out << "- Boundary segmentation from Quadric VSA face sets: `trimmed_region_input.*`.\n";
    out << "- Automatic labeling with candidate rejection, virtual corners, concatenation and reduction: `trimmed_labeling.*`.\n";
    out << "- Bezier guiding frame construction: `bezier_guiding_frame.*`.\n";
    out << "- Constrained ARAP/KKT parameterization and residual smoothing: `constrained_arap_parameterization.*` and `rotation_angle_parameterization.*`.\n\n";
    out << "## Unpublished Parameters\n\n";
    out << "- Candidate rejection thresholds, weak/parallel/obtuse corner angles, labeling tie epsilon.\n";
    out << "- Guiding-frame degree, projection sample count, frame fairness weight.\n";
    out << "- Rectangular grid density and margin.\n";
    out << "- Polyharmonic regularization, fairness blend, isocurve/mesh fairness blend.\n";
    out << "- B-spline control-grid size and control-net regularization.\n\n";
    out << "## Engineering Approximations\n\n";
    out << "- Multiply connected rotation-angle initialization is explicitly rejected until a non-contractible loop basis is added.\n";
    out << "- Trimmed surface output is produced by regular-grid clipping against UV trim loops; the rectangular extension itself uses CGAL CDT.\n";
    out << "- `trimmed_bspline_asset.json` is the reusable surface representation. Viewer meshes with authoritative-boundary snapping or harmonic boundary correction are preview/debug derivatives, not the canonical trimmed B-spline asset.\n";
    out << "- Baseline A-D share spline degree, knot type, control-grid size and regularization; their parameter domains and fitting sample sets differ by design.\n";
    out << "- Boundary fitting is measured by evaluating the final surface at trim-loop UV vertices and comparing to authoritative 3D boundary vertices.\n\n";
    out << "## Quadric VSA Integration\n\n";
    out << "- Input is a Quadric VSA region face set represented by `face_region_ids` and `target_region_id`.\n";
    out << "- Shared authoritative boundaries are extracted by `trimmed_region_input` and are not smoothed for watertight reconstruction.\n";
    out << "- Guide positions may be smoothed for labeling/frame estimation only.\n";
    out << "- Feature barrier and user marker hooks are carried by `BoundarySegmentationConfig`.\n\n";
    out << "## Defaults\n\n";
    out << "- Cubic tensor-product surface, open-uniform knots, 8x8 control grid.\n";
    out << "- KKT ARAP residual smoothing enabled with small weight.\n";
    out << "- G1 and G2 polyharmonic extension are both run; final result uses G2 with optional fairness.\n";
    out << "- Output directory: `trimmed_bspline_output/`.\n\n";
    out << "## Known Failures\n\n";
    out << "- Regions with holes are segmented, but the current rotation-angle initialization marks multiply connected regions unsupported.\n";
    out << "- Highly non-quad-like regions may fail admissible labeling reduction or produce weak guiding-frame confidence.\n";
    out << "- Severe UV flips fail rather than being clamped or repaired.\n";
    out << "- Thin sliver triangles can make the B-spline normal equations ill-conditioned.\n\n";
    out << "## Ablation Outputs\n\n";
    out << "- A: existing parameterization + B-spline fitting.\n";
    out << "- B: ordinary ARAP/KKT + B-spline fitting.\n";
    out << "- C: ARAP + 2D/3D extension + B-spline fitting.\n";
    out << "- D: automatic labeling + constrained ARAP + extension + B-spline fitting.\n";
    out << "- Compare `baseline_ablation.csv` and `metrics.json` for RMS/max fitting error, weak control points and condition estimate.\n";
    return true;
}
