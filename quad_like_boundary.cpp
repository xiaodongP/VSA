#include "quad_like_boundary.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::SelfAdjointEigenSolver;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::array;
using std::set;
using std::string;
using std::vector;

namespace {

static const double kPi = 3.141592653589793238462643383279502884;

static double clamp01(double x) {
    return std::max(0.0, std::min(1.0, x));
}

static bool finite_vec3(const Vector3d& p) {
    return std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z());
}

static double cross2(const Vector2d& a, const Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

static double polygon_signed_area(const vector<Vector2d>& pts) {
    double area = 0.0;
    int n = (int)pts.size();
    for (int i = 0; i < n; i++) {
        const Vector2d& a = pts[i];
        const Vector2d& b = pts[(i + 1) % n];
        area += cross2(a, b);
    }
    return 0.5 * area;
}

static double polyline_length_2d(
    const vector<Vector2d>& pts,
    int a,
    int b) {
    int n = (int)pts.size();
    double len = 0.0;
    int cur = a;
    while (cur != b) {
        int next = (cur + 1) % n;
        len += (pts[next] - pts[cur]).norm();
        cur = next;
    }
    return len;
}

static Vector2d side_direction_2d(
    const vector<Vector2d>& pts,
    int a,
    int b) {
    Vector2d d = pts[b] - pts[a];
    if (d.norm() > 1e-12) return d.normalized();

    int n = (int)pts.size();
    Vector2d sum = Vector2d::Zero();
    int cur = a;
    while (cur != b) {
        int next = (cur + 1) % n;
        sum += pts[next] - pts[cur];
        cur = next;
    }
    return sum.norm() > 1e-12 ? sum.normalized() : Vector2d::UnitX();
}

static double side_mean_y(
    const vector<Vector2d>& pts,
    int a,
    int b) {
    int n = (int)pts.size();
    double sum = 0.0;
    int count = 0;
    int cur = a;
    for (;;) {
        sum += pts[cur].y();
        count++;
        if (cur == b) break;
        cur = (cur + 1) % n;
    }
    return count > 0 ? sum / (double)count : 0.0;
}

static double side_bending(
    const vector<double>& abs_turns,
    int a,
    int b) {
    int n = (int)abs_turns.size();
    double bend = 0.0;
    int cur = (a + 1) % n;
    while (cur != b) {
        bend += abs_turns[cur];
        cur = (cur + 1) % n;
    }
    return bend;
}

static bool compute_pca_projection(
    const RegionBoundaryLoop& loop,
    vector<Vector2d>& pca_positions,
    bool& reversed) {
    reversed = false;
    pca_positions.clear();
    const int n = (int)loop.positions.size();
    if (n < 4) return false;

    Vector3d center = Vector3d::Zero();
    for (const Vector3d& p : loop.positions) {
        if (!finite_vec3(p)) return false;
        center += p;
    }
    center /= (double)n;

    Matrix3d cov = Matrix3d::Zero();
    for (const Vector3d& p : loop.positions) {
        Vector3d d = p - center;
        cov += d * d.transpose();
    }

    SelfAdjointEigenSolver<Matrix3d> eig(cov);
    if (eig.info() != Eigen::Success) return false;

    Vector3d axis0 = eig.eigenvectors().col(2);
    Vector3d axis1 = eig.eigenvectors().col(1);
    if (axis0.norm() < 1e-12 || axis1.norm() < 1e-12) return false;
    axis0.normalize();
    axis1.normalize();

    pca_positions.reserve(n);
    for (const Vector3d& p : loop.positions) {
        Vector3d d = p - center;
        pca_positions.push_back(Vector2d(d.dot(axis0), d.dot(axis1)));
    }

    if (polygon_signed_area(pca_positions) < 0.0) {
        for (Vector2d& p : pca_positions) p.y() = -p.y();
        reversed = true;
    }
    return true;
}

static vector<double> compute_turn_angles(const vector<Vector2d>& pts) {
    int n = (int)pts.size();
    vector<double> turns(n, 0.0);
    for (int i = 0; i < n; i++) {
        Vector2d prev = pts[i] - pts[(i + n - 1) % n];
        Vector2d next = pts[(i + 1) % n] - pts[i];
        if (prev.norm() <= 1e-12 || next.norm() <= 1e-12) continue;
        prev.normalize();
        next.normalize();
        turns[i] = std::atan2(cross2(prev, next), prev.dot(next));
    }
    return turns;
}

static vector<QuadLikeBoundaryCandidate> build_candidates(
    const vector<Vector2d>& pts,
    const vector<double>& turns,
    const QuadLikeBoundaryConfig& cfg) {
    vector<QuadLikeBoundaryCandidate> candidates;
    candidates.reserve(pts.size());
    for (int i = 0; i < (int)pts.size(); i++) {
        double angle = std::abs(turns[i]);
        double score = clamp01(angle / (0.5 * kPi));
        QuadLikeBoundaryCandidate c;
        c.loop_index = i;
        c.turn_angle = turns[i];
        c.score = score;
        c.pca_position = pts[i];
        candidates.push_back(c);
    }

    std::sort(candidates.begin(), candidates.end(),
        [](const QuadLikeBoundaryCandidate& a, const QuadLikeBoundaryCandidate& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.loop_index < b.loop_index;
        });

    int keep = std::max(4, cfg.max_corner_candidates);
    if ((int)candidates.size() > keep) candidates.resize(keep);
    return candidates;
}

static int circular_distance(int a, int b, int n) {
    int d = std::abs(a - b);
    return std::min(d, n - d);
}

static bool valid_corner_spacing(
    const array<int, 4>& corners,
    int n,
    int min_sep) {
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (circular_distance(corners[i], corners[j], n) < min_sep) {
                return false;
            }
        }
    }
    return true;
}

static array<int, 4> sorted_corners(array<int, 4> corners) {
    std::sort(corners.begin(), corners.end());
    return corners;
}

static double evaluate_corner_set(
    const vector<Vector2d>& pts,
    const vector<double>& turns,
    const array<int, 4>& sorted,
    const QuadLikeBoundaryConfig& cfg) {
    array<double, 4> lengths;
    array<Vector2d, 4> dirs;
    array<double, 4> bends;
    vector<double> abs_turns(turns.size(), 0.0);
    for (int i = 0; i < (int)turns.size(); i++) abs_turns[i] = std::abs(turns[i]);

    double length_sum = 0.0;
    for (int s = 0; s < 4; s++) {
        int a = sorted[s];
        int b = sorted[(s + 1) % 4];
        lengths[s] = polyline_length_2d(pts, a, b);
        dirs[s] = side_direction_2d(pts, a, b);
        bends[s] = side_bending(abs_turns, a, b);
        length_sum += lengths[s];
    }
    if (length_sum <= 1e-12) return -std::numeric_limits<double>::infinity();

    double corner_score = 0.0;
    for (int idx : sorted) {
        corner_score += clamp01(std::abs(turns[idx]) / (0.5 * kPi));
    }
    corner_score /= 4.0;

    double mean_len = length_sum / 4.0;
    double len_dev = 0.0;
    for (double len : lengths) {
        len_dev += std::abs(len - mean_len) / std::max(1e-12, mean_len);
    }
    double length_score = clamp01(1.0 - len_dev / 4.0);

    double parallel_score =
        0.5 * (std::abs(dirs[0].dot(dirs[2])) + std::abs(dirs[1].dot(dirs[3])));

    double orth_sum = 0.0;
    for (int s = 0; s < 4; s++) {
        orth_sum += std::abs(dirs[s].dot(dirs[(s + 1) % 4]));
    }
    double orth_score = clamp01(1.0 - orth_sum / 4.0);

    double bend_norm = 0.0;
    for (double b : bends) bend_norm += b;
    bend_norm /= std::max(1e-12, 2.0 * kPi);
    double bend_score = clamp01(1.0 - bend_norm);

    double weight_sum = cfg.corner_angle_weight +
                        cfg.length_balance_weight +
                        cfg.opposite_parallel_weight +
                        cfg.adjacent_orthogonal_weight +
                        cfg.side_bending_weight;
    if (weight_sum <= 1e-12) weight_sum = 1.0;

    return (
        cfg.corner_angle_weight * corner_score +
        cfg.length_balance_weight * length_score +
        cfg.opposite_parallel_weight * parallel_score +
        cfg.adjacent_orthogonal_weight * orth_score +
        cfg.side_bending_weight * bend_score) / weight_sum;
}

static bool choose_corners(
    const vector<Vector2d>& pts,
    const vector<double>& turns,
    const vector<QuadLikeBoundaryCandidate>& candidates,
    const QuadLikeBoundaryConfig& cfg,
    array<int, 4>& out_corners,
    double& out_score) {
    int n = (int)pts.size();
    int min_sep = std::max(1, (int)std::floor(cfg.min_corner_separation_fraction * n));
    out_score = -std::numeric_limits<double>::infinity();
    bool found = false;
    set<array<int, 4>> seen;

    for (int a = 0; a < (int)candidates.size(); a++) {
        for (int b = a + 1; b < (int)candidates.size(); b++) {
            for (int c = b + 1; c < (int)candidates.size(); c++) {
                for (int d = c + 1; d < (int)candidates.size(); d++) {
                    array<int, 4> corners = {
                        candidates[a].loop_index,
                        candidates[b].loop_index,
                        candidates[c].loop_index,
                        candidates[d].loop_index
                    };
                    corners = sorted_corners(corners);
                    if (!seen.insert(corners).second) continue;
                    if (!valid_corner_spacing(corners, n, min_sep)) continue;
                    double score = evaluate_corner_set(pts, turns, corners, cfg);
                    if (score > out_score) {
                        out_score = score;
                        out_corners = corners;
                        found = true;
                    }
                }
            }
        }
    }
    return found;
}

static array<int, 4> orient_bottom_right_top_left(
    const vector<Vector2d>& ccw_pts,
    const array<int, 4>& sorted_corners) {
    int bottom_side = 0;
    double best_y = std::numeric_limits<double>::infinity();
    for (int s = 0; s < 4; s++) {
        double y = side_mean_y(
            ccw_pts, sorted_corners[s], sorted_corners[(s + 1) % 4]);
        if (y < best_y) {
            best_y = y;
            bottom_side = s;
        }
    }

    array<int, 4> oriented;
    for (int s = 0; s < 4; s++) {
        oriented[s] = sorted_corners[(bottom_side + s) % 4];
    }
    return oriented;
}

static vector<Vector3d> side_polyline_from_loop(
    const RegionBoundaryLoop& loop,
    int a,
    int b) {
    int n = (int)loop.positions.size();
    vector<Vector3d> side;
    int cur = a;
    for (;;) {
        side.push_back(loop.positions[cur]);
        if (cur == b) break;
        cur = (cur + 1) % n;
    }
    return side;
}

static bool build_boundary_from_corners(
    const RegionBoundaryLoop& working_loop,
    const vector<Vector2d>& ccw_pts,
    array<int, 4> sorted_ccw_corners,
    double quality,
    QuadLikeBoundary& out) {
    if ((int)working_loop.positions.size() < 4) return false;
    array<int, 4> oriented = orient_bottom_right_top_left(ccw_pts, sorted_ccw_corners);
    out.corner_loop_indices = oriented;
    for (int s = 0; s < 4; s++) {
        out.side_polylines[s] = side_polyline_from_loop(
            working_loop, oriented[s], oriented[(s + 1) % 4]);
    }
    out.quality_score = quality;
    out.valid = true;
    return true;
}

} // namespace

QuadLikeBoundary::QuadLikeBoundary()
    : corner_loop_indices({{-1, -1, -1, -1}}),
      quality_score(0.0),
      valid(false) {}

QuadLikeBoundaryConfig::QuadLikeBoundaryConfig()
    : max_corner_candidates(16),
      min_corner_separation_fraction(0.08),
      corner_angle_weight(1.0),
      length_balance_weight(0.65),
      opposite_parallel_weight(0.75),
      adjacent_orthogonal_weight(0.75),
      side_bending_weight(0.45),
      min_quality_score(0.35),
      use_manual_corners(false),
      manual_corner_loop_indices({{-1, -1, -1, -1}}) {}

QuadLikeBoundaryResult::QuadLikeBoundaryResult()
    : used_manual_fallback(false),
      success(false) {}

bool load_quad_like_boundary_manual_config(
    const string& filename,
    array<int, 4>& corner_loop_indices,
    string& reason) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        reason = "cannot open manual corner config";
        return false;
    }

    string line;
    while (std::getline(fin, line)) {
        size_t hash = line.find('#');
        if (hash != string::npos) line = line.substr(0, hash);
        std::stringstream ss(line);
        string key;
        if (!(ss >> key)) continue;
        if (key == "corner_loop_indices" || key == "corners") {
            for (int i = 0; i < 4; i++) {
                if (!(ss >> corner_loop_indices[i])) {
                    reason = "manual corner config needs four indices";
                    return false;
                }
            }
            reason = "ok";
            return true;
        }
    }

    reason = "manual corner config did not contain corner_loop_indices";
    return false;
}

QuadLikeBoundaryResult split_quad_like_boundary(
    const RegionBoundaryLoop& loop,
    const QuadLikeBoundaryConfig& cfg) {
    QuadLikeBoundaryResult result;
    if (!loop.closed) {
        result.reason = "boundary loop is not closed";
        return result;
    }
    if (loop.positions.size() < 4 || loop.vertex_ids.size() != loop.positions.size()) {
        result.reason = "boundary loop needs at least four matching vertices and positions";
        return result;
    }

    vector<Vector2d> pca_positions;
    bool reversed = false;
    if (!compute_pca_projection(loop, pca_positions, reversed)) {
        result.reason = "PCA projection failed";
        return result;
    }
    (void)reversed;
    result.debug.pca_positions = pca_positions;

    const RegionBoundaryLoop& working_loop = loop;
    vector<double> turns = compute_turn_angles(pca_positions);
    result.debug.candidates = build_candidates(pca_positions, turns, cfg);

    array<int, 4> sorted_ccw_corners = {{-1, -1, -1, -1}};
    double quality = 0.0;
    bool auto_success = false;

    auto_success = choose_corners(
        pca_positions, turns, result.debug.candidates,
        cfg, sorted_ccw_corners, quality);
    if (auto_success && quality < cfg.min_quality_score) {
        result.reason = "automatic corner quality below threshold";
        auto_success = false;
    }

    if (!auto_success && cfg.use_manual_corners) {
        sorted_ccw_corners = sorted_corners(cfg.manual_corner_loop_indices);
        for (int i = 0; i < 4; i++) {
            if (sorted_ccw_corners[i] < 0 ||
                sorted_ccw_corners[i] >= (int)loop.positions.size()) {
                result.reason = "manual corner index out of range";
                return result;
            }
            if (i > 0 && sorted_ccw_corners[i] == sorted_ccw_corners[i - 1]) {
                result.reason = "manual corner indices must be unique";
                return result;
            }
        }
        vector<double> manual_turns = turns;
        quality = evaluate_corner_set(pca_positions, manual_turns, sorted_ccw_corners, cfg);
        result.used_manual_fallback = true;
    } else if (!auto_success) {
        if (result.reason.empty()) result.reason = "automatic corner selection failed";
        return result;
    }

    if (!valid_corner_spacing(
            sorted_ccw_corners,
            (int)working_loop.positions.size(),
            std::max(1, (int)std::floor(cfg.min_corner_separation_fraction *
                                        (int)working_loop.positions.size())))) {
        result.reason = "selected corners are too close together";
        return result;
    }

    if (!build_boundary_from_corners(
            working_loop, pca_positions, sorted_ccw_corners,
            quality, result.boundary)) {
        result.reason = "failed to build side polylines";
        return result;
    }

    result.debug.selected_corner_loop_indices = result.boundary.corner_loop_indices;
    result.debug.reason = "bottom,right,top,left sides follow counter-clockwise boundary order";
    result.success = true;
    result.reason = result.used_manual_fallback ? "ok_manual_fallback" : "ok";
    return result;
}

bool export_quad_like_boundary_pca_debug_obj(
    const string& filename,
    const QuadLikeBoundaryDebugInfo& debug) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;

    fout << "# Quad-like boundary PCA debug OBJ\n";
    fout << "# " << debug.reason << "\n";
    fout << "g pca_boundary\n";
    for (const Vector2d& p : debug.pca_positions) {
        fout << "v " << p.x() << " " << p.y() << " 0\n";
    }
    for (int i = 0; i < (int)debug.pca_positions.size(); i++) {
        fout << "l " << (i + 1) << " "
             << (((i + 1) % (int)debug.pca_positions.size()) + 1) << "\n";
    }

    int base = (int)debug.pca_positions.size();
    fout << "g corner_candidates\n";
    for (const QuadLikeBoundaryCandidate& c : debug.candidates) {
        const Vector2d& p = c.pca_position;
        fout << "v " << p.x() << " " << p.y() << " 0.03\n";
    }
    for (int i = 0; i < (int)debug.candidates.size(); i++) {
        fout << "p " << (base + i + 1) << "\n";
    }

    base += (int)debug.candidates.size();
    fout << "g final_corners\n";
    for (int idx : debug.selected_corner_loop_indices) {
        if (idx < 0 || idx >= (int)debug.pca_positions.size()) continue;
        const Vector2d& p = debug.pca_positions[idx];
        fout << "v " << p.x() << " " << p.y() << " 0.08\n";
    }
    int written = 0;
    for (int idx : debug.selected_corner_loop_indices) {
        if (idx < 0 || idx >= (int)debug.pca_positions.size()) continue;
        fout << "p " << (base + written + 1) << "\n";
        written++;
    }

    return true;
}
