#include "trimmed_bspline_surface.h"

#include "vsa_cgal_cdt.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector3i;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::vector;

namespace {

static bool finite_vec2(const Vector2d& p) {
    return std::isfinite(p.x()) && std::isfinite(p.y());
}

static double cross2(const Vector2d& a, const Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

static double signed_area2(const vector<Vector2d>& polyline) {
    if (polyline.size() < 3) return 0.0;
    double a = 0.0;
    int n = (int)polyline.size();
    for (int i = 0; i < n; i++) {
        const Vector2d& p = polyline[i];
        const Vector2d& q = polyline[(i + 1) % n];
        a += cross2(p, q);
    }
    return a;
}

static bool point_on_segment(
    const Vector2d& p,
    const Vector2d& a,
    const Vector2d& b,
    double eps) {
    Vector2d ab = b - a;
    Vector2d ap = p - a;
    double len2 = ab.squaredNorm();
    if (len2 <= eps * eps) return (p - a).norm() <= eps;
    double t = ap.dot(ab) / len2;
    if (t < -eps || t > 1.0 + eps) return false;
    Vector2d closest = a + std::max(0.0, std::min(1.0, t)) * ab;
    return (p - closest).norm() <= eps;
}

static bool point_in_polygon_or_on_boundary(
    const Vector2d& p,
    const vector<Vector2d>& polyline) {
    if (polyline.size() < 3) return false;

    double eps = 1e-10;
    for (const Vector2d& q : polyline) {
        eps = std::max(eps, 1e-10 * std::max(1.0, std::max(std::abs(q.x()), std::abs(q.y()))));
    }

    bool inside = false;
    int n = (int)polyline.size();
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const Vector2d& a = polyline[j];
        const Vector2d& b = polyline[i];
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

static void add_unique_point(
    vector<Vector2d>& points,
    map<pair<long long, long long>, int>& key_to_index,
    const Vector2d& p,
    double eps) {
    if (!finite_vec2(p)) return;
    pair<long long, long long> key(
        (long long)std::llround(p.x() / eps),
        (long long)std::llround(p.y() / eps));
    if (key_to_index.count(key)) return;
    key_to_index[key] = (int)points.size();
    points.push_back(p);
}

static vector<Vector2d> cleaned_trim_polyline(const vector<Vector2d>& input) {
    vector<Vector2d> out;
    for (const Vector2d& p : input) {
        if (!finite_vec2(p)) continue;
        if (!out.empty() && (p - out.back()).norm() < 1e-12) continue;
        out.push_back(p);
    }
    if (out.size() > 1 && (out.front() - out.back()).norm() < 1e-12) {
        out.pop_back();
    }
    return out;
}

static bool write_mesh_obj(
    const string& filename,
    const MatrixXd& V,
    const MatrixXi& F) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout.precision(17);
    fout << "# Trimmed B-spline sampled mesh\n";
    for (int i = 0; i < V.rows(); i++) {
        fout << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
    }
    for (int i = 0; i < F.rows(); i++) {
        fout << "f " << (F(i, 0) + 1)
             << " " << (F(i, 1) + 1)
             << " " << (F(i, 2) + 1) << "\n";
    }
    return true;
}

static bool write_uv_mesh_obj(
    const string& filename,
    const MatrixXd& UV,
    const MatrixXi& F) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout.precision(17);
    fout << "# Trimmed B-spline UV triangulation\n";
    for (int i = 0; i < UV.rows(); i++) {
        fout << "v " << UV(i, 0) << " " << UV(i, 1) << " 0\n";
    }
    for (int i = 0; i < F.rows(); i++) {
        fout << "f " << (F(i, 0) + 1)
             << " " << (F(i, 1) + 1)
             << " " << (F(i, 2) + 1) << "\n";
    }
    return true;
}

} // namespace

TrimCurve2D::TrimCurve2D()
    : degree(3), valid(false) {}

TrimmedBSplineSurfacePatch::TrimmedBSplineSurfacePatch()
    : valid(false) {}

TrimCurve2D fit_trim_curve_2d_from_polyline(
    const vector<Vector2d>& polyline,
    int control_count,
    double fairness_weight) {
    TrimCurve2D result;
    result.polyline = cleaned_trim_polyline(polyline);
    if (result.polyline.size() < 4) {
        result.reason = "trim polyline has too few points";
        return result;
    }

    vector<Vector3d> lifted;
    lifted.reserve(result.polyline.size());
    for (const Vector2d& p : result.polyline) {
        lifted.push_back(Vector3d(p.x(), p.y(), 0.0));
    }

    BSplineCurveFitResult fit =
        fit_cubic_bspline_curve_least_squares(
            lifted,
            std::max(4, control_count),
            std::max(0.0, fairness_weight));
    if (!fit.success) {
        result.reason = fit.message;
        return result;
    }

    result.degree = fit.curve.degree;
    result.knots = fit.curve.knots;
    result.control_points.clear();
    result.control_points.reserve(fit.curve.control_points.size());
    for (const Vector3d& p : fit.curve.control_points) {
        result.control_points.push_back(Vector2d(p.x(), p.y()));
    }
    result.valid = true;
    result.reason = "ok";
    return result;
}

bool sample_trimmed_bspline_surface(
    const TrimmedBSplineSurfacePatch& patch,
    int grid_resolution_u,
    int grid_resolution_v,
    MatrixXd& V,
    MatrixXi& F,
    MatrixXd* UV) {
    V.resize(0, 3);
    F.resize(0, 3);
    if (UV) UV->resize(0, 2);
    if (!patch.valid || patch.outer_trim_polyline.size() < 3) return false;

    vector<Vector2d> trim = cleaned_trim_polyline(patch.outer_trim_polyline);
    if (trim.size() < 3 || std::abs(signed_area2(trim)) < 1e-14) return false;

    vector<Vector2d> points;
    points.reserve(trim.size() + grid_resolution_u * grid_resolution_v);
    double eps = 1e-10;
    map<pair<long long, long long>, int> key_to_index;
    for (const Vector2d& p : trim) {
        add_unique_point(points, key_to_index, p, eps);
    }
    int boundary_count = (int)points.size();
    if (boundary_count < 3) return false;

    int gu = std::max(2, grid_resolution_u);
    int gv = std::max(2, grid_resolution_v);
    for (int i = 0; i < gu; i++) {
        double u = (double)i / (double)(gu - 1);
        for (int j = 0; j < gv; j++) {
            double v = (double)j / (double)(gv - 1);
            Vector2d p(u, v);
            if (point_in_polygon_or_on_boundary(p, trim)) {
                add_unique_point(points, key_to_index, p, eps);
            }
        }
    }

    vector<Vector3i> raw_triangles;
    if (!vsa_cgal_constrained_delaunay_2d(points, boundary_count, raw_triangles)) {
        return false;
    }

    vector<Vector3i> kept;
    kept.reserve(raw_triangles.size());
    for (const Vector3i& tri : raw_triangles) {
        Vector2d centroid =
            (points[tri.x()] + points[tri.y()] + points[tri.z()]) / 3.0;
        if (!point_in_polygon_or_on_boundary(centroid, trim)) continue;
        kept.push_back(tri);
    }
    if (kept.empty()) return false;

    MatrixXd uv_mat(points.size(), 2);
    V.resize(points.size(), 3);
    for (int i = 0; i < (int)points.size(); i++) {
        uv_mat.row(i) = points[i].transpose();
        Vector3d p = patch.surface.evaluate(points[i].x(), points[i].y());
        V.row(i) = p.transpose();
    }
    F.resize(kept.size(), 3);
    for (int i = 0; i < (int)kept.size(); i++) {
        F.row(i) = kept[i].transpose();
    }
    if (UV) *UV = uv_mat;
    return true;
}

bool export_uv_trim_loop_obj(
    const string& filename,
    const vector<Vector2d>& trim_polyline) {
    vector<Vector2d> trim = cleaned_trim_polyline(trim_polyline);
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout.precision(17);
    fout << "# UV trim loop\n";
    for (const Vector2d& p : trim) {
        fout << "v " << p.x() << " " << p.y() << " 0\n";
    }
    int n = (int)trim.size();
    for (int i = 0; i < n; i++) {
        fout << "l " << (i + 1) << " " << ((i + 1) % n + 1) << "\n";
    }
    return true;
}

bool export_trim_curve_control_polygon_obj(
    const string& filename,
    const TrimCurve2D& curve) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;
    fout.precision(17);
    fout << "# 2D trim B-spline control polygon\n";
    for (const Vector2d& p : curve.control_points) {
        fout << "v " << p.x() << " " << p.y() << " 0\n";
    }
    for (int i = 0; i + 1 < (int)curve.control_points.size(); i++) {
        fout << "l " << (i + 1) << " " << (i + 2) << "\n";
    }
    return true;
}

bool export_trimmed_bspline_surface_debug(
    const string& prefix,
    const TrimmedBSplineSurfacePatch& patch,
    int grid_resolution_u,
    int grid_resolution_v) {
    MatrixXd V;
    MatrixXi F;
    MatrixXd UV;
    bool ok = sample_trimmed_bspline_surface(
        patch, grid_resolution_u, grid_resolution_v, V, F, &UV);
    if (ok) {
        ok = write_mesh_obj(prefix + "_sampled_surface.obj", V, F) && ok;
        ok = write_uv_mesh_obj(prefix + "_uv_triangulation.obj", UV, F) && ok;
    }
    ok = export_uv_trim_loop_obj(prefix + "_uv_trim_loop.obj",
                                 patch.outer_trim_polyline) && ok;
    if (patch.outer_trim_curve.valid) {
        ok = export_trim_curve_control_polygon_obj(
            prefix + "_uv_trim_control_polygon.obj",
            patch.outer_trim_curve) && ok;
    }
    return ok;
}
