#include "trimmed_labeling.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#endif

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SelfAdjointEigenSolver;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::string;
using std::vector;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 1e-12;

struct ProjectionFrame {
    Vector3d origin = Vector3d::Zero();
    Vector3d u = Vector3d::UnitX();
    Vector3d v = Vector3d::UnitY();
    Vector3d n = Vector3d::UnitZ();
    double loop_area_sign = 1.0;
};

static bool make_directory(const string& directory) {
    if (directory.empty()) return false;
#ifdef _WIN32
    int rc = _mkdir(directory.c_str());
#else
    int rc = mkdir(directory.c_str(), 0755);
#endif
    return rc == 0 || errno == EEXIST;
}

static string path_join(const string& dir, const string& file) {
    if (dir.empty()) return file;
    char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\') return dir + file;
    return dir + "/" + file;
}

static string json_escape(const string& s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << c; break;
        }
    }
    return out.str();
}

static Vector3d normalized_or_zero(const Vector3d& v) {
    double n = v.norm();
    if (n <= kEps) return Vector3d::Zero();
    return v / n;
}

static double clamp_dot(double x) {
    return std::max(-1.0, std::min(1.0, x));
}

static double angle_between(const Vector3d& a, const Vector3d& b) {
    Vector3d na = normalized_or_zero(a);
    Vector3d nb = normalized_or_zero(b);
    if (na.norm() <= kEps || nb.norm() <= kEps) return 0.0;
    return std::acos(clamp_dot(na.dot(nb)));
}

static Vector3d first_tangent(const vector<Vector3d>& polyline) {
    for (int i = 1; i < (int)polyline.size(); i++) {
        Vector3d t = normalized_or_zero(polyline[i] - polyline[i - 1]);
        if (t.norm() > kEps) return t;
    }
    return Vector3d::Zero();
}

static Vector3d last_tangent(const vector<Vector3d>& polyline) {
    for (int i = (int)polyline.size() - 1; i > 0; i--) {
        Vector3d t = normalized_or_zero(polyline[i] - polyline[i - 1]);
        if (t.norm() > kEps) return t;
    }
    return Vector3d::Zero();
}

static Vector3d average_tangent(const vector<Vector3d>& polyline) {
    Vector3d sum = Vector3d::Zero();
    for (int i = 1; i < (int)polyline.size(); i++) {
        sum += polyline[i] - polyline[i - 1];
    }
    return normalized_or_zero(sum);
}

static ProjectionFrame compute_frame(const vector<Vector3d>& points) {
    ProjectionFrame frame;
    if (points.empty()) return frame;

    for (const Vector3d& p : points) frame.origin += p;
    frame.origin /= (double)points.size();

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const Vector3d& p : points) {
        Vector3d d = p - frame.origin;
        cov += d * d.transpose();
    }
    SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if (solver.info() == Eigen::Success) {
        frame.u = normalized_or_zero(solver.eigenvectors().col(2));
        frame.v = normalized_or_zero(solver.eigenvectors().col(1));
        frame.n = normalized_or_zero(frame.u.cross(frame.v));
    }
    if (frame.n.norm() <= kEps) frame.n = Vector3d::UnitZ();

    double area2 = 0.0;
    for (int i = 0; i < (int)points.size(); i++) {
        Vector3d da = points[i] - frame.origin;
        Vector3d db = points[(i + 1) % points.size()] - frame.origin;
        Vector2d a(da.dot(frame.u), da.dot(frame.v));
        Vector2d b(db.dot(frame.u), db.dot(frame.v));
        area2 += a.x() * b.y() - a.y() * b.x();
    }
    frame.loop_area_sign = area2 >= 0.0 ? 1.0 : -1.0;
    return frame;
}

static Vector2d project_vec(const ProjectionFrame& frame, const Vector3d& v) {
    return Vector2d(v.dot(frame.u), v.dot(frame.v));
}

static double signed_turn_in_frame(
    const ProjectionFrame& frame,
    const Vector3d& t0,
    const Vector3d& t1) {
    Vector2d a = project_vec(frame, t0);
    Vector2d b = project_vec(frame, t1);
    if (a.norm() <= kEps || b.norm() <= kEps) return 0.0;
    a.normalize();
    b.normalize();
    double turn = std::atan2(a.x() * b.y() - a.y() * b.x(), a.dot(b));
    return frame.loop_area_sign * turn;
}

static double triangle_area_3d(
    const Vector3d& a,
    const Vector3d& b,
    const Vector3d& c) {
    return 0.5 * (b - a).cross(c - a).norm();
}

static double region_area(
    const MatrixXd& V,
    const MatrixXi& F,
    const RegionFaceSet& region) {
    double area = 0.0;
    for (int fi : region.face_ids) {
        Vector3d a = V.row(F(fi, 0));
        Vector3d b = V.row(F(fi, 1));
        Vector3d c = V.row(F(fi, 2));
        area += triangle_area_3d(a, b, c);
    }
    return std::max(area, kEps);
}

static vector<Vector3d> concat_group_polyline(const vector<BoundarySegment>& segments) {
    vector<Vector3d> polyline;
    for (const BoundarySegment& segment : segments) {
        const vector<Vector3d>& src =
            segment.guide_positions.empty()
                ? segment.authoritative_positions
                : segment.guide_positions;
        for (int i = 0; i < (int)src.size(); i++) {
            if (!polyline.empty() && i == 0 &&
                (polyline.back() - src[i]).norm() <= 1e-10) {
                continue;
            }
            polyline.push_back(src[i]);
        }
    }
    return polyline;
}

static vector<LabelGroup> make_initial_groups(const BoundarySegmentationResult& input) {
    vector<LabelGroup> groups;
    for (const BoundarySegment& segment : input.perimeter_segments) {
        LabelGroup group;
        group.id = segment.id;
        group.segment_ids.push_back(segment.id);
        group.guide_polyline = segment.guide_positions.empty()
                                   ? segment.authoritative_positions
                                   : segment.guide_positions;
        groups.push_back(group);
    }
    return groups;
}

static vector<BoundarySegment> segments_for_group(
    const BoundarySegmentationResult& input,
    const LabelGroup& group) {
    vector<BoundarySegment> segments;
    for (int segment_id : group.segment_ids) {
        for (const BoundarySegment& segment : input.perimeter_segments) {
            if (segment.id == segment_id) {
                segments.push_back(segment);
                break;
            }
        }
    }
    return segments;
}

static LabelCandidateMetrics candidate_metrics_for_group(
    const ProjectionFrame& frame,
    const vector<LabelGroup>& groups,
    int group_index) {
    const int n = (int)groups.size();
    const LabelGroup& group = groups[group_index];
    const LabelGroup& prev = groups[(group_index + n - 1) % n];
    const LabelGroup& next = groups[(group_index + 1) % n];

    LabelCandidateMetrics metrics;
    metrics.group_id = group.id;
    metrics.segment_ids = group.segment_ids;

    const vector<Vector3d>& p = group.guide_polyline;
    for (int i = 1; i + 1 < (int)p.size(); i++) {
        Vector3d t0 = p[i] - p[i - 1];
        Vector3d t1 = p[i + 1] - p[i];
        double phi = signed_turn_in_frame(frame, t0, t1);
        metrics.raw_energy += phi * phi;
    }

    Vector3d prev_t = last_tangent(prev.guide_polyline);
    Vector3d cur_begin = first_tangent(group.guide_polyline);
    Vector3d cur_end = last_tangent(group.guide_polyline);
    Vector3d next_t = first_tangent(next.guide_polyline);
    metrics.phi_begin = angle_between(prev_t, cur_begin);
    metrics.phi_end = angle_between(cur_end, next_t);
    metrics.begin_concave =
        signed_turn_in_frame(frame, prev_t, cur_begin) < 0.0;
    metrics.end_concave =
        signed_turn_in_frame(frame, cur_end, next_t) < 0.0;
    metrics.label_energy =
        metrics.raw_energy +
        std::pow(kPi * 0.5 - metrics.phi_begin, 2.0) +
        std::pow(kPi * 0.5 - metrics.phi_end, 2.0);
    return metrics;
}

static vector<LabelCandidateMetrics> compute_candidate_metrics(
    const ProjectionFrame& frame,
    const vector<LabelGroup>& groups,
    const AutomaticLabelingConfig& config) {
    vector<LabelCandidateMetrics> metrics;
    for (int i = 0; i < (int)groups.size(); i++) {
        LabelCandidateMetrics m = candidate_metrics_for_group(frame, groups, i);
        if (config.enable_candidate_rejection) {
            if (m.begin_concave || m.end_concave) {
                m.rejected = true;
                m.rejection_reason = LabelRejectionReason::Concave;
            } else if (m.label_energy > config.label_energy_threshold) {
                m.rejected = true;
                m.rejection_reason = LabelRejectionReason::HighEnergy;
            }
        }
        metrics.push_back(m);
    }
    return metrics;
}

static vector<Vector3d> hermite_bridge(
    const Vector3d& p0,
    const Vector3d& t0,
    const Vector3d& p1,
    const Vector3d& t1) {
    vector<Vector3d> bridge;
    const double d = (p1 - p0).norm();
    if (d <= 1e-10) return bridge;
    Vector3d m0 = normalized_or_zero(t0) * d;
    Vector3d m1 = normalized_or_zero(t1) * d;
    for (int i = 1; i <= 5; i++) {
        double u = (double)i / 6.0;
        double h00 = 2.0 * u * u * u - 3.0 * u * u + 1.0;
        double h10 = u * u * u - 2.0 * u * u + u;
        double h01 = -2.0 * u * u * u + 3.0 * u * u;
        double h11 = u * u * u - u * u;
        bridge.push_back(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1);
    }
    return bridge;
}

static void append_polyline_with_optional_bridge(
    LabelGroup& target,
    const LabelGroup& source,
    bool insert_bridge) {
    if (source.guide_polyline.empty()) return;
    if (target.guide_polyline.empty()) {
        target.guide_polyline = source.guide_polyline;
    } else {
        if (insert_bridge) {
            vector<Vector3d> bridge = hermite_bridge(
                target.guide_polyline.back(),
                last_tangent(target.guide_polyline),
                source.guide_polyline.front(),
                first_tangent(source.guide_polyline));
            target.guide_polyline.insert(
                target.guide_polyline.end(), bridge.begin(), bridge.end());
        }
        int start = (target.guide_polyline.back() -
                     source.guide_polyline.front()).norm() <= 1e-10
                        ? 1
                        : 0;
        for (int i = start; i < (int)source.guide_polyline.size(); i++) {
            target.guide_polyline.push_back(source.guide_polyline[i]);
        }
    }
    target.segment_ids.insert(
        target.segment_ids.end(),
        source.segment_ids.begin(),
        source.segment_ids.end());
}

static void merge_corner(vector<LabelGroup>& groups, int corner_index, bool insert_bridge) {
    if (groups.size() <= 1) return;
    int n = (int)groups.size();
    int next = (corner_index + 1) % n;
    append_polyline_with_optional_bridge(groups[corner_index], groups[next], insert_bridge);
    if (next == 0) {
        groups.erase(groups.begin());
    } else {
        groups.erase(groups.begin() + next);
    }
}

static void remove_label_group(vector<LabelGroup>& groups, int remove_index) {
    if (groups.size() <= 1) return;
    int n = (int)groups.size();
    int prev = (remove_index + n - 1) % n;
    if (prev == remove_index) return;
    append_polyline_with_optional_bridge(groups[prev], groups[remove_index], true);
    groups.erase(groups.begin() + remove_index);
}

static void closest_points_between_lines(
    const Vector3d& p0,
    const Vector3d& d0,
    const Vector3d& p1,
    const Vector3d& d1,
    Vector3d& c0,
    Vector3d& c1) {
    Vector3d u = normalized_or_zero(d0);
    Vector3d v = normalized_or_zero(d1);
    Vector3d w0 = p0 - p1;
    double a = u.dot(u);
    double b = u.dot(v);
    double c = v.dot(v);
    double d = u.dot(w0);
    double e = v.dot(w0);
    double denom = a * c - b * b;
    double s = 0.0;
    double t = 0.0;
    if (std::abs(denom) > 1e-12) {
        s = (b * e - c * d) / denom;
        t = (a * e - b * d) / denom;
    }
    c0 = p0 + s * u;
    c1 = p1 + t * v;
}

static vector<VirtualCorner> compute_virtual_corners(
    const ProjectionFrame& frame,
    const vector<LabelGroup>& groups,
    const AutomaticLabelingConfig& config,
    double region_area_value) {
    vector<VirtualCorner> corners;
    int n = (int)groups.size();
    if (n < 2) return corners;

    for (int i = 0; i < n; i++) {
        const LabelGroup& prev = groups[i];
        const LabelGroup& next = groups[(i + 1) % n];
        Vector3d p0 = prev.guide_polyline.empty()
                          ? Vector3d::Zero()
                          : prev.guide_polyline.back();
        Vector3d p1 = next.guide_polyline.empty()
                          ? Vector3d::Zero()
                          : next.guide_polyline.front();
        Vector3d t0 = last_tangent(prev.guide_polyline);
        Vector3d t1 = first_tangent(next.guide_polyline);

        VirtualCorner corner;
        corner.id = i;
        corner.prev_group_id = prev.id;
        corner.next_group_id = next.id;
        closest_points_between_lines(p0, t0, p1, t1,
                                     corner.closest_prev,
                                     corner.closest_next);
        corner.position = 0.5 * (corner.closest_prev + corner.closest_next);
        corner.theta = angle_between(t0, t1);
        corner.signed_turn = signed_turn_in_frame(frame, t0, t1);

        if (corner.signed_turn < -config.concave_turn_threshold) {
            corner.type = VirtualCornerType::Concave;
        } else if (corner.theta < config.parallel_angle_threshold ||
                   std::abs(kPi - corner.theta) < config.parallel_angle_threshold) {
            corner.type = VirtualCornerType::Parallel;
        } else if (corner.theta > config.weak_convex_angle_threshold) {
            corner.type = VirtualCornerType::WeakConvex;
        } else {
            corner.type = VirtualCornerType::StrongConvex;
        }

        corner.e_ang = std::abs(kPi * 0.5 - corner.theta) / (kPi * 0.5);
        corner.e_dist = triangle_area_3d(p0, p1, corner.position) / region_area_value;
        double solid_angle = std::abs(corner.signed_turn);
        corner.e_tri = solid_angle / (2.0 * kPi);
        corner.e_cvx =
            (1.0 - config.w_dist) * corner.e_ang +
            config.w_dist * corner.e_dist;
        corner.e_cor =
            (1.0 - config.w_tri) * corner.e_cvx +
            config.w_tri * corner.e_tri;
        corners.push_back(corner);
    }
    return corners;
}

static int strong_corner_count(const vector<VirtualCorner>& corners) {
    int count = 0;
    for (const VirtualCorner& corner : corners) {
        if (corner.type == VirtualCornerType::StrongConvex) count++;
    }
    return count;
}

static bool admissible_pair(int labels, int corners) {
    return (labels == 0 && corners == 0) ||
           (labels == 1 && corners == 0) ||
           (labels == 2 && corners == 0) ||
           (labels == 2 && corners == 1) ||
           (labels == 3 && corners == 2) ||
           (labels == 4 && corners == 4);
}

static int distance_to_admissible(int labels, int corners) {
    const int allowed[][2] = {{0, 0}, {1, 0}, {2, 0}, {2, 1}, {3, 2}, {4, 4}};
    int best = 1000;
    for (const auto& p : allowed) {
        best = std::min(best, std::abs(labels - p[0]) + std::abs(corners - p[1]));
    }
    return best;
}

static double total_corner_energy(const vector<VirtualCorner>& corners) {
    double e = 0.0;
    for (const VirtualCorner& corner : corners) e += corner.e_cor;
    return e;
}

static void write_vec3_json(std::ofstream& out, const Vector3d& p) {
    out << "[" << p.x() << "," << p.y() << "," << p.z() << "]";
}

static bool write_candidate_metrics_json(
    const string& filename,
    const vector<LabelCandidateMetrics>& metrics) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << std::setprecision(17);
    out << "{\n  \"candidates\": [\n";
    for (int i = 0; i < (int)metrics.size(); i++) {
        const LabelCandidateMetrics& m = metrics[i];
        out << "    {\n";
        out << "      \"group_id\": " << m.group_id << ",\n";
        out << "      \"segment_ids\": [";
        for (int j = 0; j < (int)m.segment_ids.size(); j++) {
            if (j) out << ",";
            out << m.segment_ids[j];
        }
        out << "],\n";
        out << "      \"rawEnergy\": " << m.raw_energy << ",\n";
        out << "      \"phiBeg\": " << m.phi_begin << ",\n";
        out << "      \"phiEnd\": " << m.phi_end << ",\n";
        out << "      \"labelEnergy\": " << m.label_energy << ",\n";
        out << "      \"beginConcave\": " << (m.begin_concave ? "true" : "false") << ",\n";
        out << "      \"endConcave\": " << (m.end_concave ? "true" : "false") << ",\n";
        out << "      \"rejected\": " << (m.rejected ? "true" : "false") << ",\n";
        out << "      \"rejectionReason\": \"" << to_string(m.rejection_reason) << "\"\n";
        out << "    }" << (i + 1 == (int)metrics.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    return true;
}

static bool write_corners_json(
    const string& filename,
    const vector<VirtualCorner>& corners) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << std::setprecision(17);
    out << "{\n  \"corners\": [\n";
    for (int i = 0; i < (int)corners.size(); i++) {
        const VirtualCorner& c = corners[i];
        out << "    {\n";
        out << "      \"id\": " << c.id << ",\n";
        out << "      \"prev_group_id\": " << c.prev_group_id << ",\n";
        out << "      \"next_group_id\": " << c.next_group_id << ",\n";
        out << "      \"type\": \"" << to_string(c.type) << "\",\n";
        out << "      \"position\": ";
        write_vec3_json(out, c.position);
        out << ",\n";
        out << "      \"closest_prev\": ";
        write_vec3_json(out, c.closest_prev);
        out << ",\n";
        out << "      \"closest_next\": ";
        write_vec3_json(out, c.closest_next);
        out << ",\n";
        out << "      \"theta\": " << c.theta << ",\n";
        out << "      \"signed_turn\": " << c.signed_turn << ",\n";
        out << "      \"EAng\": " << c.e_ang << ",\n";
        out << "      \"EDist\": " << c.e_dist << ",\n";
        out << "      \"ETri\": " << c.e_tri << ",\n";
        out << "      \"ECvx\": " << c.e_cvx << ",\n";
        out << "      \"ECor\": " << c.e_cor << "\n";
        out << "    }" << (i + 1 == (int)corners.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    return true;
}

static AbstractSide make_side(int side_index, const LabelGroup* group) {
    AbstractSide side;
    side.side_index = side_index;
    if (!group) return side;
    side.label_group_ids.push_back(group->id);
    side.segment_ids = group->segment_ids;
    side.average_tangent = average_tangent(group->guide_polyline);
    return side;
}

static vector<AbstractSide> make_abstract_sides(const vector<LabelGroup>& groups) {
    vector<AbstractSide> sides;
    sides.reserve(4);
    for (int i = 0; i < 4; i++) {
        const LabelGroup* group = i < (int)groups.size() ? &groups[i] : nullptr;
        sides.push_back(make_side(i, group));
    }
    return sides;
}

static OrientationResolution resolve_orientation(
    const ProjectionFrame& frame,
    const vector<AbstractSide>& sides,
    const AutomaticLabelingConfig& config) {
    OrientationResolution best;
    best.side_to_cardinal = {"Unassigned", "Unassigned", "Unassigned", "Unassigned"};
    if (!config.enable_vsa_orientation || sides.empty()) return best;

    Vector3d u = config.use_supplied_orientation_axes
                     ? normalized_or_zero(config.orientation_axis_u)
                     : frame.u;
    Vector3d v = config.use_supplied_orientation_axes
                     ? normalized_or_zero(config.orientation_axis_v)
                     : frame.v;
    if (u.norm() <= kEps || v.norm() <= kEps) return best;
    v = normalized_or_zero(v - v.dot(u) * u);
    if (v.norm() <= kEps) return best;

    std::array<Vector3d, 4> axes = {u, v, -u, -v};
    std::array<string, 4> names = {"East", "North", "West", "South"};

    double best_error = std::numeric_limits<double>::infinity();
    int best_rotation = 0;
    bool best_reflected = false;
    std::array<string, 4> best_names = best.side_to_cardinal;

    for (int reflected = 0; reflected <= 1; reflected++) {
        for (int rotation = 0; rotation < 4; rotation++) {
            double error = 0.0;
            int used = 0;
            std::array<string, 4> side_names = {"Unassigned", "Unassigned", "Unassigned", "Unassigned"};
            for (int si = 0; si < (int)sides.size() && si < 4; si++) {
                if (sides[si].average_tangent.norm() <= kEps) continue;
                int axis_index = reflected
                                     ? (rotation - si + 8) % 4
                                     : (rotation + si) % 4;
                error += angle_between(sides[si].average_tangent, axes[axis_index]);
                side_names[si] = names[axis_index];
                used++;
            }
            if (used == 0) continue;
            error /= (double)used;
            if (error + 1e-12 < best_error) {
                best_error = error;
                best_rotation = rotation;
                best_reflected = reflected != 0;
                best_names = side_names;
            }
        }
    }

    if (!std::isfinite(best_error)) return best;
    best.valid = true;
    best.used_vsa_axes = config.use_supplied_orientation_axes;
    best.rotation = best_rotation;
    best.reflected = best_reflected;
    best.mean_direction_error = best_error;
    best.confidence = std::max(0.0, 1.0 - best_error / (0.5 * kPi));
    best.side_to_cardinal = best_names;
    return best;
}

static bool export_segments_obj(
    const string& filename,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& result) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << "# Labeling segments and virtual corners\n";
    int vertex_base = 1;
    auto write_polyline = [&](const string& group_name,
                              const vector<Vector3d>& points) {
        if (points.empty()) return;
        out << "g " << group_name << "\n";
        int start = vertex_base;
        for (const Vector3d& p : points) {
            out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
            vertex_base++;
        }
        for (int i = 1; i < (int)points.size(); i++) {
            out << "l " << (start + i - 1) << " " << (start + i) << "\n";
        }
    };
    auto find_segment = [&](int segment_id) -> const BoundarySegment* {
        for (const BoundarySegment& segment : input.perimeter_segments) {
            if (segment.id == segment_id) return &segment;
        }
        return nullptr;
    };

    out << "g raw_segments\n";
    for (const BoundarySegment& segment : input.perimeter_segments) {
        write_polyline("segment_" + std::to_string(segment.id),
                       segment.authoritative_positions);
    }

    for (const LabelCandidateMetrics& m : result.initial_candidates) {
        if (!m.rejected) continue;
        string group = m.rejection_reason == LabelRejectionReason::Concave
                           ? "rejected_concave"
                           : "rejected_high_energy";
        for (int segment_id : m.segment_ids) {
            const BoundarySegment* segment = find_segment(segment_id);
            if (segment) {
                write_polyline(group + "_segment_" + std::to_string(segment_id),
                               segment->authoritative_positions);
            }
        }
    }

    out << "g tangent_rays\n";
    double avg_len = 0.0;
    int len_count = 0;
    for (const BoundarySegment& segment : input.perimeter_segments) {
        avg_len += segment.length;
        len_count++;
    }
    avg_len = len_count > 0 ? avg_len / (double)len_count : 1.0;
    const double ray_len = std::max(avg_len * 0.18, 1e-3);
    for (const BoundarySegment& segment : input.perimeter_segments) {
        if (segment.authoritative_positions.empty()) continue;
        Vector3d b0 = segment.authoritative_positions.front();
        Vector3d b1 = segment.authoritative_positions.back();
        write_polyline("tangent_begin_segment_" + std::to_string(segment.id),
                       {b0, b0 + ray_len * segment.tangent_begin});
        write_polyline("tangent_end_segment_" + std::to_string(segment.id),
                       {b1, b1 + ray_len * segment.tangent_end});
    }

    out << "g virtual_corners_by_type\n";
    for (const VirtualCorner& corner : result.final_corners) {
        out << "g virtual_corner_" << corner.id << "_" << to_string(corner.type) << "\n";
        out << "v " << corner.position.x() << " "
            << corner.position.y() << " "
            << corner.position.z() << "\n";
        out << "p " << vertex_base << "\n";
        vertex_base++;
    }

    for (const LabelGroup& group : result.final_label_groups) {
        if (group.segment_ids.size() <= 1) continue;
        for (int segment_id : group.segment_ids) {
            const BoundarySegment* segment = find_segment(segment_id);
            if (segment) {
                write_polyline("multi_segment_label_" + std::to_string(group.id) +
                                   "_segment_" + std::to_string(segment_id),
                               segment->authoritative_positions);
            }
        }
    }

    for (const AbstractSide& side : result.abstract_sides) {
        for (int segment_id : side.segment_ids) {
            const BoundarySegment* segment = find_segment(segment_id);
            if (segment) {
                write_polyline("final_Side" + std::to_string(side.side_index) +
                                   "_segment_" + std::to_string(segment_id),
                               segment->authoritative_positions);
            }
        }
    }
    return true;
}

static bool write_operations_log(
    const string& filename,
    const vector<string>& log) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    for (const string& line : log) out << line << "\n";
    return true;
}

static bool write_final_labels_json(
    const string& filename,
    const AutomaticLabelingResult& result) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"valid\": " << (result.valid ? "true" : "false") << ",\n";
    out << "  \"ambiguous\": " << (result.ambiguous ? "true" : "false") << ",\n";
    out << "  \"reason\": \"" << json_escape(result.reason) << "\",\n";
    out << "  \"final_label_count\": " << result.final_label_count << ",\n";
    out << "  \"final_corner_count\": " << result.final_corner_count << ",\n";
    out << "  \"abstract_sides\": [\n";
    for (int i = 0; i < (int)result.abstract_sides.size(); i++) {
        const AbstractSide& side = result.abstract_sides[i];
        out << "    {\n";
        out << "      \"side\": \"Side" << side.side_index << "\",\n";
        out << "      \"label_group_ids\": [";
        for (int j = 0; j < (int)side.label_group_ids.size(); j++) {
            if (j) out << ",";
            out << side.label_group_ids[j];
        }
        out << "],\n";
        out << "      \"segment_ids\": [";
        for (int j = 0; j < (int)side.segment_ids.size(); j++) {
            if (j) out << ",";
            out << side.segment_ids[j];
        }
        out << "],\n";
        out << "      \"average_tangent\": ";
        write_vec3_json(out, side.average_tangent);
        out << ",\n";
        out << "      \"orientation\": \"" << result.orientation.side_to_cardinal[i] << "\"\n";
        out << "    }" << (i + 1 == (int)result.abstract_sides.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"orientation\": {\n";
    out << "    \"valid\": " << (result.orientation.valid ? "true" : "false") << ",\n";
    out << "    \"used_vsa_axes\": " << (result.orientation.used_vsa_axes ? "true" : "false") << ",\n";
    out << "    \"rotation\": " << result.orientation.rotation << ",\n";
    out << "    \"reflected\": " << (result.orientation.reflected ? "true" : "false") << ",\n";
    out << "    \"mean_direction_error\": " << result.orientation.mean_direction_error << ",\n";
    out << "    \"confidence\": " << result.orientation.confidence << "\n";
    out << "  }\n";
    out << "}\n";
    return true;
}

} // namespace

const char* to_string(LabelRejectionReason reason) {
    switch (reason) {
    case LabelRejectionReason::None: return "None";
    case LabelRejectionReason::Concave: return "Concave";
    case LabelRejectionReason::HighEnergy: return "HighEnergy";
    }
    return "Unknown";
}

const char* to_string(VirtualCornerType type) {
    switch (type) {
    case VirtualCornerType::Concave: return "Concave";
    case VirtualCornerType::Parallel: return "Parallel";
    case VirtualCornerType::WeakConvex: return "WeakConvex";
    case VirtualCornerType::StrongConvex: return "StrongConvex";
    }
    return "Unknown";
}

AutomaticLabelingResult run_automatic_labeling(
    const MatrixXd& V,
    const MatrixXi& F,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingConfig& config) {
    AutomaticLabelingResult result;
    if (!input.valid) {
        result.reason = "boundary segmentation input is invalid: " + input.reason;
        return result;
    }
    if (input.perimeter_loop_index < 0 ||
        input.perimeter_loop_index >= (int)input.loops.size()) {
        result.reason = "input has no valid perimeter loop";
        return result;
    }
    if (input.perimeter_segments.empty()) {
        result.reason = "input perimeter has no boundary segments";
        return result;
    }

    if (config.export_debug) make_directory(config.debug_output_dir);

    const AuthoritativeBoundaryLoop& perimeter =
        input.loops[input.perimeter_loop_index];
    ProjectionFrame frame = compute_frame(perimeter.positions);
    double area = region_area(V, F, input.region);
    vector<LabelGroup> groups = make_initial_groups(input);
    result.initial_candidates = compute_candidate_metrics(frame, groups, config);
    int min_supported_labels = std::max(0, config.min_supported_label_count);

    if (config.enable_candidate_rejection) {
        bool changed = true;
        while (changed && (int)groups.size() > min_supported_labels) {
            changed = false;
            vector<LabelCandidateMetrics> metrics =
                compute_candidate_metrics(frame, groups, config);
            for (int i = 0; i < (int)metrics.size(); i++) {
                if (!metrics[i].rejected) continue;
                if ((int)groups.size() - 1 < min_supported_labels) continue;
                std::ostringstream log;
                log << "reject label group " << groups[i].id
                    << " reason=" << to_string(metrics[i].rejection_reason)
                    << " labelEnergy=" << metrics[i].label_energy;
                result.operation_log.push_back(log.str());
                remove_label_group(groups, i);
                changed = true;
                break;
            }
        }
    }

    int round = 0;
    bool changed = true;
    while (config.enable_candidate_concatenation &&
           changed &&
           (int)groups.size() > min_supported_labels) {
        changed = false;
        vector<VirtualCorner> corners =
            compute_virtual_corners(frame, groups, config, area);
        result.corner_rounds.push_back(corners);

        for (int i = 0; i < (int)corners.size(); i++) {
            const VirtualCorner& corner = corners[i];
            bool merge = corner.type == VirtualCornerType::Concave ||
                         corner.type == VirtualCornerType::Parallel ||
                         (corner.type == VirtualCornerType::WeakConvex &&
                          corner.theta > config.obtuse_angle_threshold);
            if (!merge) continue;
            if ((int)groups.size() - 1 < min_supported_labels) continue;

            bool bridge = corner.type == VirtualCornerType::WeakConvex;
            std::ostringstream log;
            log << "round " << round
                << " merge corner " << i
                << " prev=" << corner.prev_group_id
                << " next=" << corner.next_group_id
                << " type=" << to_string(corner.type)
                << " theta=" << corner.theta;
            result.operation_log.push_back(log.str());
            merge_corner(groups, i, bridge);
            changed = true;
            break;
        }
        round++;
    }

    for (int iter = 0; config.enable_reduction && iter < config.max_reduction_rounds; iter++) {
        vector<VirtualCorner> corners =
            compute_virtual_corners(frame, groups, config, area);
        result.corner_rounds.push_back(corners);
        int L = (int)groups.size();
        int C = strong_corner_count(corners);
        if (admissible_pair(L, C)) break;
        if ((int)groups.size() <= min_supported_labels) break;

        struct Move {
            vector<int> remove_indices;
            double score = std::numeric_limits<double>::infinity();
            bool admissible = false;
        };
        vector<Move> moves;

        auto evaluate_move = [&](const vector<int>& remove_indices) {
            vector<LabelGroup> trial = groups;
            vector<int> sorted = remove_indices;
            std::sort(sorted.begin(), sorted.end(), std::greater<int>());
            for (int idx : sorted) {
                if (idx >= 0 && idx < (int)trial.size()) remove_label_group(trial, idx);
            }
            if ((int)trial.size() < min_supported_labels) return;
            vector<VirtualCorner> trial_corners =
                compute_virtual_corners(frame, trial, config, area);
            int trial_L = (int)trial.size();
            int trial_C = strong_corner_count(trial_corners);
            Move move;
            move.remove_indices = remove_indices;
            move.admissible = admissible_pair(trial_L, trial_C);
            move.score =
                (move.admissible ? -1000.0 : 0.0) +
                10.0 * distance_to_admissible(trial_L, trial_C) +
                total_corner_energy(trial_corners);
            for (int idx : remove_indices) {
                if (idx >= 0 && idx < (int)groups.size()) {
                    LabelCandidateMetrics m =
                        candidate_metrics_for_group(frame, groups, idx);
                    move.score += config.high_energy_merge_bias * m.label_energy;
                }
            }
            moves.push_back(move);
        };

        for (int i = 0; i < (int)groups.size(); i++) evaluate_move({i});
        if ((L == 5 && C == 5) || (L == 4 && C == 3)) {
            for (int i = 0; i < (int)groups.size(); i++) {
                for (int j = i + 1; j < (int)groups.size(); j++) {
                    evaluate_move({i, j});
                }
            }
        }

        std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
            if (std::abs(a.score - b.score) > config.tie_epsilon)
                return a.score < b.score;
            return a.remove_indices < b.remove_indices;
        });
        if (moves.empty() || !std::isfinite(moves.front().score)) break;
        if (moves.size() > 1 &&
            std::abs(moves[0].score - moves[1].score) <= config.tie_epsilon) {
            result.ambiguous = true;
        }

        std::ostringstream log;
        log << "reduction round " << iter << " L=" << L << " C=" << C
            << " remove";
        vector<int> sorted = moves.front().remove_indices;
        std::sort(sorted.begin(), sorted.end(), std::greater<int>());
        for (int idx : sorted) {
            if (idx >= 0 && idx < (int)groups.size()) {
                log << " group=" << groups[idx].id;
            }
        }
        log << " score=" << moves.front().score
            << " admissible=" << (moves.front().admissible ? 1 : 0);
        result.operation_log.push_back(log.str());
        for (int idx : sorted) {
            if (idx >= 0 && idx < (int)groups.size()) remove_label_group(groups, idx);
        }
    }

    result.final_label_groups = groups;
    result.final_corners = compute_virtual_corners(frame, groups, config, area);
    result.final_label_count = (int)groups.size();
    result.final_corner_count = strong_corner_count(result.final_corners);
    result.abstract_sides = make_abstract_sides(groups);
    result.orientation = resolve_orientation(frame, result.abstract_sides, config);
    bool admissible = admissible_pair(result.final_label_count, result.final_corner_count);
    if (!admissible && result.final_label_count == 4) {
        result.ambiguous = true;
        admissible = true;
    }
    result.valid =
        result.final_label_count >= min_supported_labels &&
        admissible;
    result.reason = result.valid ? "ok" : "final label/corner count is not admissible";

    if (config.export_debug) {
        export_labeling_debug_artifacts(config.debug_output_dir, V, F, input, result);
    }
    return result;
}

bool export_labeling_debug_artifacts(
    const string& directory,
    const MatrixXd&,
    const MatrixXi&,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& result) {
    if (!make_directory(directory)) return false;
    bool ok = true;
    ok = write_candidate_metrics_json(
             path_join(directory, "candidate_metrics.json"),
             result.initial_candidates) && ok;
    for (int i = 0; i < (int)result.corner_rounds.size(); i++) {
        std::ostringstream name;
        name << "corner_round_" << i << ".json";
        ok = write_corners_json(
                 path_join(directory, name.str()),
                 result.corner_rounds[i]) && ok;
    }
    ok = write_corners_json(
             path_join(directory, "final_corners.json"),
             result.final_corners) && ok;
    ok = write_operations_log(
             path_join(directory, "operations.log"),
             result.operation_log) && ok;
    ok = write_final_labels_json(
             path_join(directory, "final_labels.json"),
             result) && ok;
    ok = export_segments_obj(
             path_join(directory, "segments.obj"),
             input,
             result) && ok;
    return ok;
}
