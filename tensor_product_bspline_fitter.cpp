#include "tensor_product_bspline_fitter.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::string;
using std::vector;

namespace {

static double clamp_unit(double x) {
    return std::max(0.0, std::min(1.0, x));
}

static int cp_index(int i, int j, int nv) {
    return i * nv + j;
}

static double greville_abscissa(
    int i,
    int degree,
    const vector<double>& knots) {
    if (degree <= 0) return knots.empty() ? 0.0 : knots[std::min(i, (int)knots.size() - 1)];
    double sum = 0.0;
    int count = 0;
    for (int k = 1; k <= degree; k++) {
        int id = i + k;
        if (id >= 0 && id < (int)knots.size()) {
            sum += knots[id];
            count++;
        }
    }
    return count > 0 ? sum / (double)count : 0.0;
}

static bool is_original_vertex(const vector<bool>& mask, int i) {
    return i < (int)mask.size() && mask[i];
}

static Vector2d normalized_uv(
    const Vector2d& uv,
    const Vector2d& mn,
    const Vector2d& mx) {
    Vector2d d = mx - mn;
    return Vector2d(
        clamp_unit(d.x() > 1e-14 ? (uv.x() - mn.x()) / d.x() : 0.0),
        clamp_unit(d.y() > 1e-14 ? (uv.y() - mn.y()) / d.y() : 0.0));
}

static TensorProductBSplineFitStats compute_error_stats(
    const BSplineSurface3D& surface,
    const MatrixXd& UV,
    const MatrixXd& P,
    const Vector2d& uv_min,
    const Vector2d& uv_max,
    const vector<bool>* vertex_mask) {
    TensorProductBSplineFitStats stats;
    double sum = 0.0;
    double sum2 = 0.0;
    int count = 0;
    for (int i = 0; i < UV.rows(); i++) {
        if (vertex_mask && !is_original_vertex(*vertex_mask, i)) continue;
        Vector2d uv = normalized_uv(UV.row(i).transpose(), uv_min, uv_max);
        double e = (surface.evaluate(uv.x(), uv.y()) - P.row(i).transpose()).norm();
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

static double extension_smoothness(
    const BSplineSurface3D& surface,
    const MatrixXd& UV,
    const MatrixXi& F,
    const vector<bool>& original_face_mask,
    const Vector2d& uv_min,
    const Vector2d& uv_max) {
    std::map<std::pair<int, int>, vector<int>> edge_faces;
    for (int fi = 0; fi < F.rows(); fi++) {
        if (fi < (int)original_face_mask.size() && original_face_mask[fi]) continue;
        for (int e = 0; e < 3; e++) {
            int a = F(fi, e);
            int b = F(fi, (e + 1) % 3);
            if (a > b) std::swap(a, b);
            edge_faces[{a, b}].push_back(fi);
        }
    }
    double sum = 0.0;
    int count = 0;
    for (const auto& kv : edge_faces) {
        if (kv.second.size() != 2) continue;
        int a = kv.first.first;
        int b = kv.first.second;
        Vector2d uva = normalized_uv(UV.row(a).transpose(), uv_min, uv_max);
        Vector2d uvb = normalized_uv(UV.row(b).transpose(), uv_min, uv_max);
        Vector2d mid = 0.5 * (uva + uvb);
        Vector3d du = surface.derivative(mid.x(), mid.y(), 1, 0);
        Vector3d dv = surface.derivative(mid.x(), mid.y(), 0, 1);
        Vector3d n = du.cross(dv);
        double nn = n.norm();
        if (nn <= 1e-12) continue;
        n /= nn;
        Vector2d da = uvb - uva;
        Vector2d perp(-da.y(), da.x());
        double len = perp.norm();
        if (len <= 1e-12) continue;
        perp /= len;
        double eps = 1e-3;
        Vector2d p0 = (mid - eps * perp).cwiseMax(Vector2d::Zero()).cwiseMin(Vector2d::Ones());
        Vector2d p1 = (mid + eps * perp).cwiseMax(Vector2d::Zero()).cwiseMin(Vector2d::Ones());
        Vector3d du0 = surface.derivative(p0.x(), p0.y(), 1, 0);
        Vector3d dv0 = surface.derivative(p0.x(), p0.y(), 0, 1);
        Vector3d du1 = surface.derivative(p1.x(), p1.y(), 1, 0);
        Vector3d dv1 = surface.derivative(p1.x(), p1.y(), 0, 1);
        Vector3d n0 = du0.cross(dv0);
        Vector3d n1 = du1.cross(dv1);
        if (n0.norm() <= 1e-12 || n1.norm() <= 1e-12) continue;
        n0.normalize();
        n1.normalize();
        sum += (n0 - n1).squaredNorm();
        count++;
    }
    return count > 0 ? sum / (double)count : 0.0;
}

static WeakSupportStats compute_weak_support(
    const vector<double>& support,
    double threshold) {
    WeakSupportStats stats;
    stats.control_point_count = (int)support.size();
    if (support.empty()) return stats;
    stats.min_support = *std::min_element(support.begin(), support.end());
    stats.max_support = *std::max_element(support.begin(), support.end());
    stats.mean_support = std::accumulate(support.begin(), support.end(), 0.0) /
                         (double)support.size();
    for (double s : support) {
        if (s < threshold) stats.weak_control_point_count++;
    }
    return stats;
}

static MatrixXd fit_affine_uv_map(
    const MatrixXd& UV,
    const MatrixXd& P,
    const Vector2d& uv_min,
    const Vector2d& uv_max,
    const vector<bool>& original_vertex_mask,
    const vector<double>& sample_weights) {
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, 3, 3> rhs = Eigen::Matrix<double, 3, 3>::Zero();
    int rows = 0;
    auto append_row = [&](int vi) {
        double weight = 1.0;
        if (vi < (int)sample_weights.size() && std::isfinite(sample_weights[vi])) {
            weight = std::max(0.0, sample_weights[vi]);
        }
        if (weight <= 0.0) return;
        Vector2d uv = normalized_uv(UV.row(vi).transpose(), uv_min, uv_max);
        Eigen::Vector3d a(uv.x(), uv.y(), 1.0);
        normal += weight * (a * a.transpose());
        rhs += weight * (a * P.row(vi));
        rows++;
    };
    for (int vi = 0; vi < UV.rows(); vi++) {
        if (!original_vertex_mask.empty() && !is_original_vertex(original_vertex_mask, vi)) continue;
        append_row(vi);
    }
    if (rows < 3) {
        normal.setZero();
        rhs.setZero();
        rows = 0;
        for (int vi = 0; vi < UV.rows(); vi++) append_row(vi);
    }
    if (rows < 3) {
        MatrixXd affine(3, 3);
        affine.setZero();
        affine.row(2) = P.colwise().mean();
        return affine;
    }
    return normal.ldlt().solve(rhs);
}

static double estimate_condition_number(const SparseMatrix<double>& A) {
    if (A.rows() > 250) return 0.0;
    MatrixXd dense(A);
    Eigen::JacobiSVD<MatrixXd> svd(dense);
    if (svd.singularValues().size() == 0) return 0.0;
    double max_sv = svd.singularValues()(0);
    double min_sv = std::numeric_limits<double>::max();
    for (int i = 0; i < svd.singularValues().size(); i++) {
        double s = svd.singularValues()(i);
        if (s > 1e-14) min_sv = std::min(min_sv, s);
    }
    if (min_sv == std::numeric_limits<double>::max()) return std::numeric_limits<double>::infinity();
    return max_sv / min_sv;
}

static bool write_knots_csv(
    const string& filename,
    const vector<double>& u,
    const vector<double>& v) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    out << "direction,index,value\n";
    for (int i = 0; i < (int)u.size(); i++) out << "u," << i << "," << u[i] << "\n";
    for (int i = 0; i < (int)v.size(); i++) out << "v," << i << "," << v[i] << "\n";
    return true;
}

static bool write_control_points_csv(const string& filename, const BSplineSurface3D& surface) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out.precision(17);
    out << "i,j,x,y,z\n";
    for (int i = 0; i < (int)surface.control_grid.size(); i++) {
        for (int j = 0; j < (int)surface.control_grid[i].size(); j++) {
            const Vector3d& p = surface.control_grid[i][j];
            out << i << "," << j << "," << p.x() << "," << p.y() << "," << p.z() << "\n";
        }
    }
    return true;
}

} // namespace

const char* to_string(TensorProductFitBaseline baseline) {
    switch (baseline) {
    case TensorProductFitBaseline::TrimmedOnly: return "TrimmedOnly";
    case TensorProductFitBaseline::ArapOnly: return "ArapOnly";
    case TensorProductFitBaseline::ExtensionOnly: return "ExtensionOnly";
    case TensorProductFitBaseline::LabelingExtension: return "LabelingExtension";
    }
    return "Unknown";
}

TensorProductBSplineFitResult fit_tensor_product_cubic_bspline_surface(
    const TensorProductBSplineFitInput& input,
    const TensorProductBSplineFitConfig& config) {
    TensorProductBSplineFitResult result;
    result.baseline = input.baseline;
    if (input.UV.cols() != 2 || input.positions.cols() != 3 ||
        input.UV.rows() != input.positions.rows() ||
        input.F.cols() != 3) {
        result.reason = "invalid UV/F/position dimensions";
        return result;
    }
    if (config.degree_u != 3 || config.degree_v != 3) {
        result.reason = "baseline fitter currently supports cubic surfaces only";
        return result;
    }
    if (config.control_count_u < 4 || config.control_count_v < 4) {
        result.reason = "cubic tensor-product surface needs at least 4x4 controls";
        return result;
    }
    int nverts = input.UV.rows();
    int nu = config.control_count_u;
    int nv = config.control_count_v;
    int nctrl = nu * nv;

    result.uv_min = input.UV.colwise().minCoeff().transpose();
    result.uv_max = input.UV.colwise().maxCoeff().transpose();
    vector<double> knots_u = make_open_uniform_knot_vector(nu, config.degree_u);
    vector<double> knots_v = make_open_uniform_knot_vector(nv, config.degree_v);

    vector<Triplet<double>> normal_trips;
    MatrixXd rhs = MatrixXd::Zero(nctrl, 3);
    vector<double> support(nctrl, 0.0);
    int fit_rows = 0;
    for (int vi = 0; vi < nverts; vi++) {
        if (config.fit_original_vertices_only && !is_original_vertex(input.original_vertex_mask, vi)) {
            continue;
        }
        double weight = 1.0;
        if (vi < (int)input.sample_weights.size() && std::isfinite(input.sample_weights[vi])) {
            weight = std::max(0.0, input.sample_weights[vi]);
        }
        if (weight <= 0.0) continue;
        Vector2d uv = normalized_uv(input.UV.row(vi).transpose(), result.uv_min, result.uv_max);
        vector<std::pair<int, double>> row;
        row.reserve(nctrl);
        for (int i = 0; i < nu; i++) {
            double bu = bspline_basis(i, config.degree_u, uv.x(), knots_u);
            if (std::abs(bu) <= 1e-14) continue;
            for (int j = 0; j < nv; j++) {
                double bv = bspline_basis(j, config.degree_v, uv.y(), knots_v);
                double b = bu * bv;
                if (std::abs(b) <= 1e-14) continue;
                int id = cp_index(i, j, nv);
                row.push_back({id, b});
                support[id] += weight * b * b;
                rhs.row(id) += weight * b * input.positions.row(vi);
            }
        }
        for (const auto& a : row) {
            for (const auto& b : row) {
                normal_trips.emplace_back(a.first, b.first, weight * a.second * b.second);
            }
        }
        fit_rows++;
    }
    if (fit_rows < nctrl && config.control_net_fairness_weight <= 0.0) {
        result.reason = "not enough fitting rows for unconstrained control grid";
        return result;
    }

    double fair = std::max(0.0, config.control_net_fairness_weight);
    if (fair > 0.0) {
        auto add_fair_row = [&](const vector<std::pair<int, double>>& coeffs) {
            for (const auto& a : coeffs) {
                for (const auto& b : coeffs) {
                    normal_trips.emplace_back(a.first, b.first, fair * a.second * b.second);
                }
            }
        };
        for (int i = 1; i + 1 < nu; i++) {
            for (int j = 0; j < nv; j++) {
                add_fair_row({
                    {cp_index(i - 1, j, nv), 1.0},
                    {cp_index(i, j, nv), -2.0},
                    {cp_index(i + 1, j, nv), 1.0}});
            }
        }
        for (int i = 0; i < nu; i++) {
            for (int j = 1; j + 1 < nv; j++) {
                add_fair_row({
                    {cp_index(i, j - 1, nv), 1.0},
                    {cp_index(i, j, nv), -2.0},
                    {cp_index(i, j + 1, nv), 1.0}});
            }
        }
    }
    double initial = std::max(0.0, config.control_net_initial_weight);
    if (initial > 0.0) {
        MatrixXd affine = fit_affine_uv_map(
            input.UV,
            input.positions,
            result.uv_min,
            result.uv_max,
            input.original_vertex_mask,
            input.sample_weights);
        for (int i = 0; i < nu; i++) {
            double u = clamp_unit(greville_abscissa(i, config.degree_u, knots_u));
            for (int j = 0; j < nv; j++) {
                double v = clamp_unit(greville_abscissa(j, config.degree_v, knots_v));
                Eigen::RowVector3d a;
                a << u, v, 1.0;
                int id = cp_index(i, j, nv);
                normal_trips.emplace_back(id, id, initial);
                rhs.row(id) += initial * (a * affine);
            }
        }
    }
    double reg = std::max(0.0, config.normal_equation_regularization);
    for (int id = 0; id < nctrl; id++) normal_trips.emplace_back(id, id, reg);

    SparseMatrix<double> normal(nctrl, nctrl);
    normal.setFromTriplets(normal_trips.begin(), normal_trips.end());
    result.normal_matrix_nonzeros = (int)normal.nonZeros();
    result.condition_estimate =
        config.estimate_condition_number ? estimate_condition_number(normal) : 0.0;
    result.sample_count = nverts;
    result.fit_sample_count = fit_rows;
    result.weak_support = compute_weak_support(support, config.weak_support_threshold);

    Eigen::SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(normal);
    if (solver.info() != Eigen::Success) {
        result.reason = "sparse normal-equation factorization failed";
        return result;
    }
    MatrixXd ctrl = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        result.reason = "sparse normal-equation solve failed";
        return result;
    }

    vector<vector<Vector3d>> grid(nu, vector<Vector3d>(nv, Vector3d::Zero()));
    for (int i = 0; i < nu; i++) {
        for (int j = 0; j < nv; j++) {
            grid[i][j] = ctrl.row(cp_index(i, j, nv)).transpose();
        }
    }
    result.surface = BSplineSurface3D(
        config.degree_u, config.degree_v, knots_u, knots_v, grid);
    result.all_vertex_error = compute_error_stats(
        result.surface, input.UV, input.positions, result.uv_min, result.uv_max, nullptr);
    result.original_region_error = compute_error_stats(
        result.surface, input.UV, input.positions, result.uv_min, result.uv_max,
        input.original_vertex_mask.empty() ? nullptr : &input.original_vertex_mask);
    result.extension_region_smoothness = extension_smoothness(
        result.surface, input.UV, input.F, input.original_face_mask, result.uv_min, result.uv_max);
    result.valid = true;
    result.reason = "ok";
    if (config.export_debug) {
        export_tensor_product_bspline_fit_debug(config.debug_prefix, result);
    }
    return result;
}

bool export_tensor_product_bspline_fit_debug(
    const string& prefix,
    const TensorProductBSplineFitResult& result) {
    bool ok = true;
    ok = export_bspline_surface_control_net_obj(prefix + "_control_net.obj", result.surface) && ok;
    ok = export_bspline_surface_mesh_obj(prefix + "_sampled_base_mesh.obj", result.surface, 50, 50) && ok;
    ok = write_control_points_csv(prefix + "_control_points.csv", result.surface) && ok;
    ok = write_knots_csv(prefix + "_knots.csv", result.surface.knots_u, result.surface.knots_v) && ok;
    {
        std::ofstream out(prefix + "_report.csv");
        ok = out.is_open() && ok;
        out.precision(17);
        out << "valid,reason,baseline,control_u,control_v,fit_samples,"
            << "all_mean,all_rms,all_max,original_mean,original_rms,original_max,"
            << "extension_smoothness,condition_estimate,normal_nonzeros,"
            << "weak_controls,total_controls,min_support,mean_support,max_support\n";
        out << (result.valid ? 1 : 0) << "," << result.reason << ","
            << to_string(result.baseline) << ","
            << result.surface.control_grid.size() << ","
            << (result.surface.control_grid.empty() ? 0 : result.surface.control_grid[0].size()) << ","
            << result.fit_sample_count << ","
            << result.all_vertex_error.mean_error << ","
            << result.all_vertex_error.rms_error << ","
            << result.all_vertex_error.max_error << ","
            << result.original_region_error.mean_error << ","
            << result.original_region_error.rms_error << ","
            << result.original_region_error.max_error << ","
            << result.extension_region_smoothness << ","
            << result.condition_estimate << ","
            << result.normal_matrix_nonzeros << ","
            << result.weak_support.weak_control_point_count << ","
            << result.weak_support.control_point_count << ","
            << result.weak_support.min_support << ","
            << result.weak_support.mean_support << ","
            << result.weak_support.max_support << "\n";
    }
    return ok;
}
