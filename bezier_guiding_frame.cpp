#include "bezier_guiding_frame.h"

#include <Eigen/Sparse>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#endif

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::Vector3d;
using std::string;
using std::vector;

namespace {

constexpr double kEps = 1e-12;

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

static Vector3d normalized_or_zero(const Vector3d& v) {
    double n = v.norm();
    if (n <= kEps) return Vector3d::Zero();
    return v / n;
}

static double clamp01(double t) {
    return std::max(0.0, std::min(1.0, t));
}

static double binomial(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    k = std::min(k, n - k);
    double c = 1.0;
    for (int i = 1; i <= k; i++) {
        c *= (double)(n - k + i);
        c /= (double)i;
    }
    return c;
}

static double bernstein(int degree, int i, double t) {
    if (degree < 0 || i < 0 || i > degree) return 0.0;
    t = clamp01(t);
    if (degree == 0) return 1.0;
    return binomial(degree, i) *
           std::pow(t, i) *
           std::pow(1.0 - t, degree - i);
}

static vector<double> chord_parameters(const vector<Vector3d>& samples) {
    vector<double> t(samples.size(), 0.0);
    if (samples.size() < 2) return t;
    double total = 0.0;
    for (int i = 1; i < (int)samples.size(); i++) {
        total += (samples[i] - samples[i - 1]).norm();
        t[i] = total;
    }
    if (total <= kEps) return t;
    for (double& x : t) x /= total;
    return t;
}

static double curve_point_distance2(const BezierCurve3D& curve, double t, const Vector3d& p) {
    return (curve.evaluate(t) - p).squaredNorm();
}

static double project_point_to_curve(
    const BezierCurve3D& curve,
    const Vector3d& p,
    int sample_count) {
    sample_count = std::max(sample_count, 8);
    double best_t = 0.0;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (int i = 0; i < sample_count; i++) {
        double t = (double)i / (double)(sample_count - 1);
        double d2 = curve_point_distance2(curve, t, p);
        if (d2 < best_d2) {
            best_d2 = d2;
            best_t = t;
        }
    }

    double t = best_t;
    for (int iter = 0; iter < 8; iter++) {
        Vector3d c = curve.evaluate(t);
        Vector3d d = curve.derivative(t);
        double denom = d.squaredNorm();
        if (denom <= kEps) break;
        double step = (c - p).dot(d) / denom;
        t = clamp01(t - step);
        if (std::abs(step) < 1e-10) break;
    }
    return t;
}

static vector<double> project_samples(
    const BezierCurve3D& curve,
    const vector<Vector3d>& samples,
    int projection_samples) {
    vector<double> t(samples.size(), 0.0);
    for (int i = 0; i < (int)samples.size(); i++) {
        t[i] = project_point_to_curve(curve, samples[i], projection_samples);
    }
    if (!t.empty()) {
        t.front() = 0.0;
        t.back() = 1.0;
    }
    return t;
}

static void compute_error(
    const BezierCurve3D& curve,
    const vector<Vector3d>& samples,
    const vector<double>& params,
    double& mean_error,
    double& rms_error,
    double& max_error) {
    mean_error = 0.0;
    rms_error = 0.0;
    max_error = 0.0;
    if (samples.empty()) return;
    for (int i = 0; i < (int)samples.size(); i++) {
        double e = (curve.evaluate(params[i]) - samples[i]).norm();
        mean_error += e;
        rms_error += e * e;
        max_error = std::max(max_error, e);
    }
    mean_error /= (double)samples.size();
    rms_error = std::sqrt(rms_error / (double)samples.size());
}

static BezierCurve3D make_line_curve(
    const Vector3d& p0,
    const Vector3d& p1) {
    BezierCurve3D curve;
    curve.degree = 1;
    curve.control_points = {p0, p1};
    return curve;
}

static BezierCurve3D make_cubic_hermite_curve(
    const Vector3d& p0,
    const Vector3d& p1,
    const Vector3d& t0,
    const Vector3d& t1) {
    double length = std::max((p1 - p0).norm(), 1e-6);
    BezierCurve3D curve;
    curve.degree = 3;
    curve.control_points.resize(4);
    curve.control_points[0] = p0;
    curve.control_points[1] = p0 + normalized_or_zero(t0) * length / 3.0;
    curve.control_points[2] = p1 - normalized_or_zero(t1) * length / 3.0;
    curve.control_points[3] = p1;
    return curve;
}

static BezierCurve3D elevate_degree_once(const BezierCurve3D& curve) {
    BezierCurve3D elevated;
    elevated.degree = curve.degree + 1;
    elevated.control_points.resize(elevated.degree + 1);
    elevated.control_points.front() = curve.control_points.front();
    elevated.control_points.back() = curve.control_points.back();
    for (int i = 1; i < elevated.degree; i++) {
        double alpha = (double)i / (double)elevated.degree;
        elevated.control_points[i] =
            alpha * curve.control_points[i - 1] +
            (1.0 - alpha) * curve.control_points[i];
    }
    return elevated;
}

static double estimate_condition(const SparseMatrix<double>& normal) {
    MatrixXd dense(normal);
    if (dense.rows() == 0 || dense.cols() == 0) return 1.0;
    Eigen::JacobiSVD<MatrixXd> svd(dense);
    if (svd.singularValues().size() == 0) return 1.0;
    double smax = svd.singularValues()(0);
    double smin = svd.singularValues()(svd.singularValues().size() - 1);
    if (smin <= 1e-14) return std::numeric_limits<double>::infinity();
    return smax / smin;
}

static bool solve_curve_control_points(
    BezierCurve3D& curve,
    const vector<Vector3d>& samples,
    const vector<double>& params,
    const BezierGuidingFrameConfig& config,
    double& condition_estimate,
    string& reason) {
    const int degree = curve.degree;
    const int interior_count = degree - 1;
    if (interior_count <= 0) {
        condition_estimate = 1.0;
        return true;
    }
    if ((int)samples.size() != (int)params.size()) {
        reason = "sample/parameter size mismatch";
        return false;
    }

    struct RowTerm {
        int col;
        double value;
    };
    vector<vector<RowTerm>> rows;
    vector<Vector3d> rhs;

    for (int si = 0; si < (int)samples.size(); si++) {
        double t = clamp01(params[si]);
        vector<RowTerm> row;
        Vector3d b = samples[si];
        for (int j = 0; j <= degree; j++) {
            double coeff = bernstein(degree, j, t);
            if (j == 0 || j == degree) {
                b -= coeff * curve.control_points[j];
            } else {
                row.push_back({j - 1, coeff});
            }
        }
        rows.push_back(row);
        rhs.push_back(b);
    }

    if (degree >= 2 && config.lambda_frame > 0.0) {
        int qn = std::max(config.fairness_quadrature_samples, degree * 4);
        double weight = std::sqrt(config.lambda_frame / (double)qn);
        double scale = (double)(degree * (degree - 1));
        for (int qi = 0; qi < qn; qi++) {
            double t = qn == 1 ? 0.5 : (double)qi / (double)(qn - 1);
            vector<double> cp_coeff(degree + 1, 0.0);
            for (int k = 0; k <= degree - 2; k++) {
                double b2 = scale * bernstein(degree - 2, k, t) * weight;
                cp_coeff[k] += b2;
                cp_coeff[k + 1] -= 2.0 * b2;
                cp_coeff[k + 2] += b2;
            }
            Vector3d fair_rhs = Vector3d::Zero();
            vector<RowTerm> row;
            for (int j = 0; j <= degree; j++) {
                if (std::abs(cp_coeff[j]) <= 1e-14) continue;
                if (j == 0 || j == degree) {
                    fair_rhs -= cp_coeff[j] * curve.control_points[j];
                } else {
                    row.push_back({j - 1, cp_coeff[j]});
                }
            }
            rows.push_back(row);
            rhs.push_back(fair_rhs);
        }
    }

    if (rows.empty()) {
        reason = "no rows in Bézier fitting system";
        return false;
    }

    vector<Triplet<double>> triplets;
    triplets.reserve(rows.size() * std::max(1, interior_count));
    for (int r = 0; r < (int)rows.size(); r++) {
        for (const RowTerm& term : rows[r]) {
            triplets.emplace_back(r, term.col, term.value);
        }
    }
    SparseMatrix<double> A((int)rows.size(), interior_count);
    A.setFromTriplets(triplets.begin(), triplets.end());
    SparseMatrix<double> normal = A.transpose() * A;
    condition_estimate = estimate_condition(normal);

    Eigen::SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(normal);
    if (solver.info() != Eigen::Success) {
        reason = "failed to factor Bézier sparse normal matrix";
        return false;
    }

    MatrixXd B(rows.size(), 3);
    for (int r = 0; r < (int)rhs.size(); r++) {
        B.row(r) = rhs[r].transpose();
    }
    MatrixXd AtB = A.transpose() * B;
    MatrixXd X = solver.solve(AtB);
    if (solver.info() != Eigen::Success) {
        reason = "failed to solve Bézier sparse normal system";
        return false;
    }

    for (int j = 1; j < degree; j++) {
        curve.control_points[j] = X.row(j - 1).transpose();
    }
    return true;
}

static const BoundarySegment* find_segment(
    const BoundarySegmentationResult& input,
    int segment_id) {
    for (const BoundarySegment& segment : input.perimeter_segments) {
        if (segment.id == segment_id) return &segment;
    }
    return nullptr;
}

static vector<Vector3d> samples_for_side(
    const BoundarySegmentationResult& input,
    const AbstractSide& side) {
    vector<Vector3d> samples;
    for (int segment_id : side.segment_ids) {
        const BoundarySegment* segment = find_segment(input, segment_id);
        if (!segment) continue;
        const vector<Vector3d>& src =
            segment->guide_positions.empty()
                ? segment->authoritative_positions
                : segment->guide_positions;
        for (int i = 0; i < (int)src.size(); i++) {
            if (!samples.empty() && i == 0 &&
                (samples.back() - src[i]).norm() <= 1e-10) {
                continue;
            }
            samples.push_back(src[i]);
        }
    }
    return samples;
}

static vector<int> active_side_indices(const AutomaticLabelingResult& labeling) {
    vector<int> indices;
    for (int i = 0; i < (int)labeling.abstract_sides.size(); i++) {
        if (!labeling.abstract_sides[i].segment_ids.empty()) indices.push_back(i);
    }
    return indices;
}

static vector<Vector3d> compute_shared_corners_from_samples(
    const vector<vector<Vector3d>>& samples,
    const vector<Vector3d>& optional_corners) {
    const int n = (int)samples.size();
    if ((int)optional_corners.size() == n) return optional_corners;

    vector<Vector3d> corners(n, Vector3d::Zero());
    for (int i = 0; i < n; i++) {
        Vector3d a = samples[i].empty() ? Vector3d::Zero() : samples[i].front();
        Vector3d b = samples[(i + n - 1) % n].empty()
                         ? a
                         : samples[(i + n - 1) % n].back();
        corners[i] = 0.5 * (a + b);
    }
    return corners;
}

static Vector3d side_begin_tangent(const vector<Vector3d>& samples, const Vector3d& fallback) {
    for (int i = 1; i < (int)samples.size(); i++) {
        Vector3d t = normalized_or_zero(samples[i] - samples[i - 1]);
        if (t.norm() > kEps) return t;
    }
    return normalized_or_zero(fallback);
}

static Vector3d side_end_tangent(const vector<Vector3d>& samples, const Vector3d& fallback) {
    for (int i = (int)samples.size() - 1; i > 0; i--) {
        Vector3d t = normalized_or_zero(samples[i] - samples[i - 1]);
        if (t.norm() > kEps) return t;
    }
    return normalized_or_zero(fallback);
}

static vector<BezierCurve3D> initial_curves(
    const vector<vector<Vector3d>>& samples,
    const vector<Vector3d>& corners,
    int degree) {
    vector<BezierCurve3D> curves;
    const int n = (int)samples.size();
    for (int i = 0; i < n; i++) {
        Vector3d p0 = corners[i];
        Vector3d p1 = corners[(i + 1) % n];
        BezierCurve3D curve = make_line_curve(p0, p1);
        if (samples[i].size() < 2 && degree >= 3) {
            Vector3d fallback = p1 - p0;
            curve = make_cubic_hermite_curve(
                p0, p1,
                side_begin_tangent(samples[i], fallback),
                side_end_tangent(samples[i], fallback));
        }
        while (curve.degree < degree) curve = elevate_degree_once(curve);
        curve.control_points.front() = p0;
        curve.control_points.back() = p1;
        curves.push_back(curve);
    }
    return curves;
}

static void append_report_error(
    BezierGuidingFrameResult& result,
    const vector<BezierCurve3D>& curves,
    const vector<vector<Vector3d>>& samples,
    const vector<vector<double>>& params) {
    double total = 0.0;
    double sum = 0.0;
    double sum2 = 0.0;
    double maxe = 0.0;
    for (int ci = 0; ci < (int)curves.size(); ci++) {
        for (int si = 0; si < (int)samples[ci].size(); si++) {
            double e = (curves[ci].evaluate(params[ci][si]) - samples[ci][si]).norm();
            sum += e;
            sum2 += e * e;
            maxe = std::max(maxe, e);
            total += 1.0;
        }
    }
    result.mean_error = total > 0.0 ? sum / total : 0.0;
    result.rms_error = total > 0.0 ? std::sqrt(sum2 / total) : 0.0;
    result.max_error = maxe;
}

static bool write_reports_csv(
    const string& filename,
    const BezierGuidingFrameResult& result) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << "curve_index,side_index,degree,iteration,mean_error,rms_error,max_error,condition_estimate\n";
    out << std::setprecision(17);
    for (const BezierFrameIterationReport& r : result.reports) {
        out << r.curve_index << ","
            << r.side_index << ","
            << r.degree << ","
            << r.iteration << ","
            << r.mean_error << ","
            << r.rms_error << ","
            << r.max_error << ","
            << r.condition_estimate << "\n";
    }
    return true;
}

static bool write_curves_obj(
    const string& filename,
    const BezierGuidingFrameResult& result) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    int cursor = 1;
    for (int ci = 0; ci < (int)result.curves.size(); ci++) {
        out << "g frame_curve_" << ci << "_side_" << result.side_indices[ci] << "\n";
        vector<Vector3d> points = result.curves[ci].sample(64);
        int start = cursor;
        for (const Vector3d& p : points) {
            out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
            cursor++;
        }
        for (int i = 1; i < (int)points.size(); i++) {
            out << "l " << (start + i - 1) << " " << (start + i) << "\n";
        }
    }
    out << "g shared_corners\n";
    for (const Vector3d& p : result.shared_corners) {
        out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
        out << "p " << cursor << "\n";
        cursor++;
    }
    return true;
}

static bool write_control_net_obj(
    const string& filename,
    const BezierGuidingFrameResult& result) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    int cursor = 1;
    for (int ci = 0; ci < (int)result.curves.size(); ci++) {
        out << "g control_polygon_" << ci << "\n";
        int start = cursor;
        for (const Vector3d& p : result.curves[ci].control_points) {
            out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
            cursor++;
        }
        for (int i = 1; i < (int)result.curves[ci].control_points.size(); i++) {
            out << "l " << (start + i - 1) << " " << (start + i) << "\n";
        }
    }
    return true;
}

static BezierGuidingFrameResult fit_frame_from_labeled_samples(
    const vector<int>& side_indices,
    const vector<vector<Vector3d>>& labeled_samples,
    const vector<Vector3d>& optional_corners,
    const BezierGuidingFrameConfig& config) {
    BezierGuidingFrameResult result;
    result.side_indices = side_indices;
    result.labeled_samples = labeled_samples;

    if ((int)labeled_samples.size() < 2 || (int)labeled_samples.size() > 4) {
        result.reason = "guiding frame requires 2, 3, or 4 labeled sample sets";
        return result;
    }
    if (!optional_corners.empty() &&
        (int)optional_corners.size() != (int)labeled_samples.size()) {
        result.reason = "optional virtual corner count must match label count";
        return result;
    }

    bool has_missing_samples = false;
    for (const vector<Vector3d>& samples : labeled_samples) {
        if (samples.empty()) has_missing_samples = true;
    }
    if (has_missing_samples && optional_corners.empty()) {
        result.reason = "missing label samples require optional virtual corners";
        return result;
    }

    result.shared_corners =
        compute_shared_corners_from_samples(labeled_samples, optional_corners);

    int start_degree = std::max(1, config.initial_degree);
    int max_degree = std::max(start_degree, config.max_degree);
    if (has_missing_samples) max_degree = std::max(max_degree, 3);
    vector<BezierCurve3D> curves =
        initial_curves(result.labeled_samples, result.shared_corners, start_degree);

    string reason;
    bool tolerance_met = false;
    for (int degree = start_degree; degree <= max_degree; degree++) {
        if (degree > start_degree) {
            for (BezierCurve3D& curve : curves) {
                curve = elevate_degree_once(curve);
            }
        }
        if (has_missing_samples && degree == 3) {
            for (int ci = 0; ci < (int)curves.size(); ci++) {
                if (!result.labeled_samples[ci].empty()) continue;
                Vector3d p0 = result.shared_corners[ci];
                Vector3d p1 =
                    result.shared_corners[(ci + 1) % result.shared_corners.size()];
                curves[ci] = make_cubic_hermite_curve(p0, p1, p1 - p0, p1 - p0);
            }
        }
        for (int ci = 0; ci < (int)curves.size(); ci++) {
            curves[ci].degree = degree;
            curves[ci].control_points.front() = result.shared_corners[ci];
            curves[ci].control_points.back() =
                result.shared_corners[(ci + 1) % result.shared_corners.size()];
        }

        double previous_rms = std::numeric_limits<double>::infinity();
        vector<vector<double>> params(curves.size());
        for (int ci = 0; ci < (int)curves.size(); ci++) {
            params[ci] = chord_parameters(result.labeled_samples[ci]);
        }

        for (int iter = 0; iter < config.max_iterations_per_degree; iter++) {
            for (int ci = 0; ci < (int)curves.size(); ci++) {
                params[ci] = project_samples(
                    curves[ci],
                    result.labeled_samples[ci],
                    config.projection_samples);
                double condition = 1.0;
                if (!solve_curve_control_points(
                        curves[ci],
                        result.labeled_samples[ci],
                        params[ci],
                        config,
                        condition,
                        reason)) {
                    result.reason = reason;
                    return result;
                }

                BezierFrameIterationReport report;
                report.curve_index = ci;
                report.side_index = result.side_indices.empty() ? ci : result.side_indices[ci];
                report.degree = degree;
                report.iteration = iter;
                report.condition_estimate = condition;
                compute_error(
                    curves[ci],
                    result.labeled_samples[ci],
                    params[ci],
                    report.mean_error,
                    report.rms_error,
                    report.max_error);
                result.reports.push_back(report);
            }

            append_report_error(result, curves, result.labeled_samples, params);
            bool can_stop_on_tolerance = !has_missing_samples || degree >= 3;
            if (can_stop_on_tolerance &&
                result.rms_error <= config.fitting_tolerance) {
                tolerance_met = true;
                break;
            }
            if (std::abs(previous_rms - result.rms_error) <=
                config.convergence_tolerance) {
                break;
            }
            previous_rms = result.rms_error;
        }

        if (tolerance_met || degree == max_degree) {
            result.curves = curves;
            result.final_degree = degree;
            break;
        }
    }

    for (int ci = 0; ci < (int)result.curves.size(); ci++) {
        result.curves[ci].control_points.front() = result.shared_corners[ci];
        result.curves[ci].control_points.back() =
            result.shared_corners[(ci + 1) % result.shared_corners.size()];
    }

    result.valid = !result.curves.empty();
    result.reason = result.valid ? "ok" : "failed to build guiding frame";
    if (config.export_debug) {
        export_bezier_guiding_frame_debug(config.debug_output_dir, result);
    }
    return result;
}

} // namespace

Vector3d BezierCurve3D::evaluate(double t) const {
    if (control_points.empty()) return Vector3d::Zero();
    t = clamp01(t);
    Vector3d p = Vector3d::Zero();
    for (int i = 0; i <= degree && i < (int)control_points.size(); i++) {
        p += bernstein(degree, i, t) * control_points[i];
    }
    return p;
}

Vector3d BezierCurve3D::derivative(double t) const {
    if (degree <= 0 || (int)control_points.size() < degree + 1) {
        return Vector3d::Zero();
    }
    t = clamp01(t);
    Vector3d d = Vector3d::Zero();
    for (int i = 0; i < degree; i++) {
        d += bernstein(degree - 1, i, t) *
             (double)degree *
             (control_points[i + 1] - control_points[i]);
    }
    return d;
}

vector<Vector3d> BezierCurve3D::sample(int count) const {
    count = std::max(count, 2);
    vector<Vector3d> points;
    points.reserve(count);
    for (int i = 0; i < count; i++) {
        double t = (double)i / (double)(count - 1);
        points.push_back(evaluate(t));
    }
    return points;
}

BezierGuidingFrameResult build_bezier_guiding_frame(
    const MatrixXd&,
    const MatrixXi&,
    const BoundarySegmentationResult& input,
    const AutomaticLabelingResult& labeling,
    const BezierGuidingFrameConfig& config) {
    BezierGuidingFrameResult result;
    if (!input.valid) {
        result.reason = "boundary input is invalid: " + input.reason;
        return result;
    }
    if (!labeling.valid) {
        result.reason = "labeling input is invalid: " + labeling.reason;
        return result;
    }

    vector<int> side_indices = active_side_indices(labeling);
    if ((int)side_indices.size() < 2 || (int)side_indices.size() > 4) {
        result.reason = "guiding frame requires 2, 3, or 4 labeled sides";
        return result;
    }

    vector<vector<Vector3d>> labeled_samples;
    for (int side_index : side_indices) {
        vector<Vector3d> samples =
            samples_for_side(input, labeling.abstract_sides[side_index]);
        if (samples.empty()) {
            result.reason = "active label side has no guide samples";
            return result;
        }
        labeled_samples.push_back(samples);
    }

    vector<Vector3d> optional_corners;
    if (config.use_virtual_corners &&
        (int)labeling.final_corners.size() >= (int)side_indices.size()) {
        int n = (int)side_indices.size();
        optional_corners.resize(n);
        for (int i = 0; i < n; i++) {
            optional_corners[i] = labeling.final_corners[(i + n - 1) % n].position;
        }
    }

    return fit_frame_from_labeled_samples(
        side_indices, labeled_samples, optional_corners, config);
}

BezierGuidingFrameResult build_bezier_guiding_frame_from_samples(
    const vector<vector<Vector3d>>& labeled_samples,
    const vector<Vector3d>& optional_virtual_corners,
    const BezierGuidingFrameConfig& config) {
    vector<int> side_indices;
    side_indices.reserve(labeled_samples.size());
    for (int i = 0; i < (int)labeled_samples.size(); i++) side_indices.push_back(i);
    return fit_frame_from_labeled_samples(
        side_indices, labeled_samples, optional_virtual_corners, config);
}

bool export_bezier_guiding_frame_debug(
    const string& directory,
    const BezierGuidingFrameResult& result) {
    if (!make_directory(directory)) return false;
    bool ok = true;
    ok = write_reports_csv(path_join(directory, "iteration_report.csv"), result) && ok;
    ok = write_curves_obj(path_join(directory, "frame_curves.obj"), result) && ok;
    ok = write_control_net_obj(path_join(directory, "control_polygons.obj"), result) && ok;
    return ok;
}
