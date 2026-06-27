#include "trimmed_mesh_validation.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <tuple>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector3d;
using Eigen::Vector3i;
using std::map;
using std::set;
using std::string;
using std::tuple;
using std::vector;

namespace {

static double bbox_diagonal(const MatrixXd& V) {
    if (V.rows() == 0) return 1.0;
    Vector3d mn = V.colwise().minCoeff();
    Vector3d mx = V.colwise().maxCoeff();
    double d = (mx - mn).norm();
    return std::max(d, 1.0);
}

static tuple<int, int, int> sorted_face_key(const Vector3i& f) {
    int ids[3] = {f.x(), f.y(), f.z()};
    std::sort(ids, ids + 3);
    return std::make_tuple(ids[0], ids[1], ids[2]);
}

static tuple<long long, long long, long long> quantized_point_key(
    const Vector3d& p,
    double eps) {
    return std::make_tuple(
        (long long)std::llround(p.x() / eps),
        (long long)std::llround(p.y() / eps),
        (long long)std::llround(p.z() / eps));
}

static tuple<
    tuple<long long, long long, long long>,
    tuple<long long, long long, long long>,
    tuple<long long, long long, long long>>
geometric_face_key(const MatrixXd& V, const Vector3i& f, double eps) {
    vector<tuple<long long, long long, long long>> keys = {
        quantized_point_key(V.row(f.x()).transpose(), eps),
        quantized_point_key(V.row(f.y()).transpose(), eps),
        quantized_point_key(V.row(f.z()).transpose(), eps)};
    std::sort(keys.begin(), keys.end());
    return std::make_tuple(keys[0], keys[1], keys[2]);
}

static double triangle_double_area(const MatrixXd& V, const Vector3i& f) {
    Vector3d a = V.row(f.x()).transpose();
    Vector3d b = V.row(f.y()).transpose();
    Vector3d c = V.row(f.z()).transpose();
    return (b - a).cross(c - a).norm();
}

static double triangle_quality(const MatrixXd& V, const Vector3i& f) {
    Vector3d a = V.row(f.x()).transpose();
    Vector3d b = V.row(f.y()).transpose();
    Vector3d c = V.row(f.z()).transpose();
    double l0 = (b - a).squaredNorm();
    double l1 = (c - b).squaredNorm();
    double l2 = (a - c).squaredNorm();
    double denom = l0 + l1 + l2;
    if (denom <= 1e-30) return 0.0;
    double double_area = (b - a).cross(c - a).norm();
    return 2.0 * std::sqrt(3.0) * double_area / denom;
}

static Vector3d triangle_unit_normal(const MatrixXd& V, const Vector3i& f) {
    Vector3d a = V.row(f.x()).transpose();
    Vector3d b = V.row(f.y()).transpose();
    Vector3d c = V.row(f.z()).transpose();
    Vector3d n = (b - a).cross(c - a);
    double len = n.norm();
    if (len <= 1e-14) return Vector3d::Zero();
    return n / len;
}

static int directed_edge_sign(const Vector3i& f, int a, int b) {
    for (int k = 0; k < 3; k++) {
        int x = f(k);
        int y = f((k + 1) % 3);
        if (x == a && y == b) return 1;
        if (x == b && y == a) return -1;
    }
    return 0;
}

static bool write_issue_obj(
    const string& filename,
    const MatrixXd& V,
    const MatrixXi& F,
    const vector<int>& face_ids) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    set<int> unique_faces(face_ids.begin(), face_ids.end());
    for (int i = 0; i < V.rows(); i++) {
        out << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
    }
    for (int fid : unique_faces) {
        if (fid < 0 || fid >= F.rows()) continue;
        out << "f " << F(fid, 0) + 1 << " " << F(fid, 1) + 1
            << " " << F(fid, 2) + 1 << "\n";
    }
    return true;
}

static void append_unique(vector<int>& dst, int value) {
    if (std::find(dst.begin(), dst.end(), value) == dst.end()) dst.push_back(value);
}

} // namespace

MeshValidationReport validate_trimmed_mesh(
    const MatrixXd& V,
    const MatrixXi& F,
    const MatrixXd* UV) {
    (void)UV;
    MeshValidationReport r;
    r.vertex_count = (int)V.rows();
    r.face_count = (int)F.rows();
    if (V.rows() == 0 || F.rows() == 0) return r;

    double diag = bbox_diagonal(V);
    double position_eps = diag * 1e-10;
    double degenerate_threshold = diag * diag * 1e-14;
    double near_degenerate_threshold = diag * diag * 1e-10;
    double low_quality_threshold = 0.02;
    double normal_jump_dot_threshold = 0.5;

    r.min_double_area = std::numeric_limits<double>::infinity();
    r.min_quality = std::numeric_limits<double>::infinity();

    map<tuple<int, int, int>, vector<int>> exact_faces;
    map<tuple<
            tuple<long long, long long, long long>,
            tuple<long long, long long, long long>,
            tuple<long long, long long, long long>>,
        vector<int>>
        geometric_faces;
    vector<int> vertex_use(V.rows(), 0);
    vector<Vector3d> face_normals(F.rows(), Vector3d::Zero());

    for (int fi = 0; fi < F.rows(); fi++) {
        Vector3i f = F.row(fi).transpose();
        bool valid = true;
        for (int k = 0; k < 3; k++) {
            if (f(k) < 0 || f(k) >= V.rows()) valid = false;
        }
        if (!valid) {
            append_unique(r.degenerate_face_ids, fi);
            continue;
        }
        for (int k = 0; k < 3; k++) vertex_use[f(k)]++;
        exact_faces[sorted_face_key(f)].push_back(fi);
        geometric_faces[geometric_face_key(V, f, std::max(position_eps, 1e-30))].push_back(fi);

        double da = triangle_double_area(V, f);
        double q = triangle_quality(V, f);
        face_normals[fi] = triangle_unit_normal(V, f);
        r.min_double_area = std::min(r.min_double_area, da);
        r.min_quality = std::min(r.min_quality, q);
        if (da <= degenerate_threshold) append_unique(r.degenerate_face_ids, fi);
        else if (da <= near_degenerate_threshold) append_unique(r.near_degenerate_face_ids, fi);
        if (q <= low_quality_threshold) append_unique(r.low_quality_face_ids, fi);
    }

    for (const auto& kv : exact_faces) {
        if (kv.second.size() <= 1) continue;
        r.exact_duplicate_faces += (int)kv.second.size() - 1;
        for (int fid : kv.second) append_unique(r.exact_duplicate_face_ids, fid);
    }
    for (const auto& kv : geometric_faces) {
        if (kv.second.size() <= 1) continue;
        r.geometric_duplicate_faces += (int)kv.second.size() - 1;
        for (int fid : kv.second) append_unique(r.geometric_duplicate_face_ids, fid);
    }

    struct EdgeUse {
        int face = -1;
        int sign = 0;
    };
    map<std::pair<int, int>, vector<EdgeUse>> edge_faces;
    for (int fi = 0; fi < F.rows(); fi++) {
        Vector3i f = F.row(fi).transpose();
        if (f.x() < 0 || f.y() < 0 || f.z() < 0 ||
            f.x() >= V.rows() || f.y() >= V.rows() || f.z() >= V.rows()) {
            continue;
        }
        for (int k = 0; k < 3; k++) {
            int a = f(k);
            int b = f((k + 1) % 3);
            auto key = std::minmax(a, b);
            edge_faces[key].push_back({fi, directed_edge_sign(f, key.first, key.second)});
        }
    }
    for (const auto& kv : edge_faces) {
        const vector<EdgeUse>& uses = kv.second;
        if (uses.size() > 2) {
            r.nonmanifold_edges++;
            for (const EdgeUse& u : uses) append_unique(r.nonmanifold_face_ids, u.face);
        } else if (uses.size() == 2 && uses[0].sign == uses[1].sign) {
            r.inconsistent_winding_edges++;
            append_unique(r.bad_winding_face_ids, uses[0].face);
            append_unique(r.bad_winding_face_ids, uses[1].face);
        }
        if (uses.size() == 2) {
            int f0 = uses[0].face;
            int f1 = uses[1].face;
            if (f0 >= 0 && f0 < (int)face_normals.size() &&
                f1 >= 0 && f1 < (int)face_normals.size() &&
                face_normals[f0].norm() > 0.0 &&
                face_normals[f1].norm() > 0.0 &&
                face_normals[f0].dot(face_normals[f1]) < normal_jump_dot_threshold) {
                r.normal_jump_edges++;
                append_unique(r.normal_jump_face_ids, f0);
                append_unique(r.normal_jump_face_ids, f1);
            }
        }
    }

    for (int vi = 0; vi < (int)vertex_use.size(); vi++) {
        if (vertex_use[vi] == 0) {
            r.isolated_vertices++;
            r.isolated_vertex_ids.push_back(vi);
        }
    }
    if (!std::isfinite(r.min_double_area)) r.min_double_area = 0.0;
    if (!std::isfinite(r.min_quality)) r.min_quality = 0.0;
    r.degenerate_faces = (int)r.degenerate_face_ids.size();
    r.near_degenerate_faces = (int)r.near_degenerate_face_ids.size();
    return r;
}

bool orient_mesh_faces_consistently(
    const MatrixXd& V,
    MatrixXi& F,
    const Vector3d& target_normal) {
    if (F.rows() == 0) return true;
    map<std::pair<int, int>, vector<int>> edge_to_faces;
    for (int fi = 0; fi < F.rows(); fi++) {
        Vector3i f = F.row(fi).transpose();
        for (int k = 0; k < 3; k++) {
            edge_to_faces[std::minmax(f(k), f((k + 1) % 3))].push_back(fi);
        }
    }

    vector<vector<int>> adjacency(F.rows());
    for (const auto& kv : edge_to_faces) {
        if (kv.second.size() != 2) continue;
        int a = kv.second[0];
        int b = kv.second[1];
        adjacency[a].push_back(b);
        adjacency[b].push_back(a);
    }

    vector<char> seen(F.rows(), 0);
    for (int seed = 0; seed < F.rows(); seed++) {
        if (seen[seed]) continue;
        vector<int> stack = {seed};
        vector<int> component;
        seen[seed] = 1;
        while (!stack.empty()) {
            int f0 = stack.back();
            stack.pop_back();
            component.push_back(f0);
            Vector3i tri0 = F.row(f0).transpose();
            for (int f1 : adjacency[f0]) {
                if (seen[f1]) continue;
                Vector3i tri1 = F.row(f1).transpose();
                bool should_flip = false;
                bool shares_edge = false;
                for (int k = 0; k < 3; k++) {
                    int a = tri0(k);
                    int b = tri0((k + 1) % 3);
                    int s0 = directed_edge_sign(tri0, a, b);
                    int s1 = directed_edge_sign(tri1, a, b);
                    if (s1 != 0) {
                        shares_edge = true;
                        should_flip = (s0 == s1);
                        break;
                    }
                }
                if (!shares_edge) continue;
                if (should_flip) std::swap(F(f1, 1), F(f1, 2));
                seen[f1] = 1;
                stack.push_back(f1);
            }
        }

        Vector3d n = Vector3d::Zero();
        for (int fid : component) {
            Vector3d a = V.row(F(fid, 0)).transpose();
            Vector3d b = V.row(F(fid, 1)).transpose();
            Vector3d c = V.row(F(fid, 2)).transpose();
            n += (b - a).cross(c - a);
        }
        if (target_normal.norm() > 1e-12 && n.dot(target_normal) < 0.0) {
            for (int fid : component) std::swap(F(fid, 1), F(fid, 2));
        }
    }
    return true;
}

bool remove_degenerate_faces(const MatrixXd& V, MatrixXi& F) {
    if (F.rows() == 0) return true;
    double diag = bbox_diagonal(V);
    double threshold = diag * diag * 1e-10;
    vector<Vector3i> kept;
    kept.reserve(F.rows());
    set<tuple<int, int, int>> seen;
    for (int fi = 0; fi < F.rows(); fi++) {
        Vector3i f = F.row(fi).transpose();
        if (f.x() < 0 || f.y() < 0 || f.z() < 0 ||
            f.x() >= V.rows() || f.y() >= V.rows() || f.z() >= V.rows() ||
            f.x() == f.y() || f.y() == f.z() || f.z() == f.x()) {
            continue;
        }
        if (triangle_double_area(V, f) <= threshold) continue;
        auto key = sorted_face_key(f);
        if (!seen.insert(key).second) continue;
        kept.push_back(f);
    }
    F.resize((int)kept.size(), 3);
    for (int i = 0; i < (int)kept.size(); i++) F.row(i) = kept[i].transpose();
    return true;
}

bool write_mesh_validation_json(
    const string& filename,
    const MeshValidationReport& r,
    const MatrixXd* UV) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    auto write_ids = [&](const char* name, const vector<int>& ids) {
        out << "  \"" << name << "\": [";
        for (int i = 0; i < (int)ids.size(); i++) {
            if (i) out << ", ";
            out << ids[i];
        }
        out << "]";
    };
    out.precision(17);
    out << "{\n";
    out << "  \"vertex_count\": " << r.vertex_count << ",\n";
    out << "  \"face_count\": " << r.face_count << ",\n";
    out << "  \"exact_duplicate_faces\": " << r.exact_duplicate_faces << ",\n";
    out << "  \"geometric_duplicate_faces\": " << r.geometric_duplicate_faces << ",\n";
    out << "  \"degenerate_faces\": " << r.degenerate_faces << ",\n";
    out << "  \"near_degenerate_faces\": " << r.near_degenerate_faces << ",\n";
    out << "  \"inconsistent_winding_edges\": " << r.inconsistent_winding_edges << ",\n";
    out << "  \"normal_jump_edges\": " << r.normal_jump_edges << ",\n";
    out << "  \"nonmanifold_edges\": " << r.nonmanifold_edges << ",\n";
    out << "  \"isolated_vertices\": " << r.isolated_vertices << ",\n";
    out << "  \"min_double_area\": " << r.min_double_area << ",\n";
    out << "  \"min_quality\": " << r.min_quality << ",\n";
    write_ids("exact_duplicate_face_ids", r.exact_duplicate_face_ids); out << ",\n";
    write_ids("geometric_duplicate_face_ids", r.geometric_duplicate_face_ids); out << ",\n";
    write_ids("degenerate_face_ids", r.degenerate_face_ids); out << ",\n";
    write_ids("near_degenerate_face_ids", r.near_degenerate_face_ids); out << ",\n";
    write_ids("low_quality_face_ids", r.low_quality_face_ids); out << ",\n";
    write_ids("bad_winding_face_ids", r.bad_winding_face_ids); out << ",\n";
    write_ids("normal_jump_face_ids", r.normal_jump_face_ids); out << ",\n";
    write_ids("nonmanifold_face_ids", r.nonmanifold_face_ids); out << ",\n";
    write_ids("isolated_vertex_ids", r.isolated_vertex_ids); out << ",\n";
    out << "  \"uv_available\": " << (UV ? "true" : "false") << "\n";
    out << "}\n";
    return true;
}

bool export_mesh_validation_issue_objs(
    const string& output_dir,
    const MatrixXd& V,
    const MatrixXi& F,
    const MeshValidationReport& r) {
    bool ok = true;
    vector<int> duplicate = r.exact_duplicate_face_ids;
    for (int fid : r.geometric_duplicate_face_ids) append_unique(duplicate, fid);
    vector<int> degenerate = r.degenerate_face_ids;
    for (int fid : r.near_degenerate_face_ids) append_unique(degenerate, fid);
    for (int fid : r.low_quality_face_ids) append_unique(degenerate, fid);
    ok = write_issue_obj(output_dir + "/duplicate_faces.obj", V, F, duplicate) && ok;
    ok = write_issue_obj(output_dir + "/degenerate_faces.obj", V, F, degenerate) && ok;
    ok = write_issue_obj(output_dir + "/bad_winding_faces.obj", V, F, r.bad_winding_face_ids) && ok;
    ok = write_issue_obj(output_dir + "/normal_jump_faces.obj", V, F, r.normal_jump_face_ids) && ok;
    ok = write_issue_obj(output_dir + "/boundary_intersection_faces.obj", V, F, r.nonmanifold_face_ids) && ok;
    return ok;
}
