#ifndef BSPLINE_HEADER
#define BSPLINE_HEADER

#include <Eigen/Dense>

#include <string>
#include <vector>

struct BSplineCurveFitResult;

std::vector<double> make_open_uniform_knot_vector(int control_count, int degree);

double bspline_basis(
    int i,
    int degree,
    double t,
    const std::vector<double>& knots);

double bspline_basis_derivative(
    int i,
    int degree,
    double t,
    int derivative_order,
    const std::vector<double>& knots);

class BSplineCurve3D {
public:
    int degree;
    std::vector<double> knots;
    std::vector<Eigen::Vector3d> control_points;

    BSplineCurve3D();
    BSplineCurve3D(
        int degree_,
        const std::vector<double>& knots_,
        const std::vector<Eigen::Vector3d>& control_points_);

    Eigen::Vector3d evaluate(double t) const;
    Eigen::Vector3d derivative(double t, int order) const;
    std::vector<Eigen::Vector3d> sample(int count) const;
};

struct BSplineCurveFitResult {
    BSplineCurve3D curve;
    double mean_error;
    double max_error;
    bool success;
    std::string message;

    BSplineCurveFitResult();
};

class BSplineSurface3D {
public:
    int degree_u;
    int degree_v;
    std::vector<double> knots_u;
    std::vector<double> knots_v;
    std::vector<std::vector<Eigen::Vector3d>> control_grid;

    BSplineSurface3D();
    BSplineSurface3D(
        int degree_u_,
        int degree_v_,
        const std::vector<double>& knots_u_,
        const std::vector<double>& knots_v_,
        const std::vector<std::vector<Eigen::Vector3d>>& control_grid_);

    Eigen::Vector3d evaluate(double u, double v) const;
    Eigen::Vector3d derivative(
        double u,
        double v,
        int order_u,
        int order_v) const;
};

void sample_bspline_surface(
    const BSplineSurface3D& surface,
    int sample_u,
    int sample_v,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F);

std::vector<double> chord_length_parameters(
    const std::vector<Eigen::Vector3d>& polyline);

BSplineCurveFitResult fit_cubic_bspline_curve_least_squares(
    const std::vector<Eigen::Vector3d>& polyline,
    int num_control_points,
    double fairness_weight);

bool export_bspline_curve_polyline_obj(
    const std::string& filename,
    const BSplineCurve3D& curve,
    int sample_count);

bool export_bspline_surface_mesh_obj(
    const std::string& filename,
    const BSplineSurface3D& surface,
    int sample_u,
    int sample_v);

bool export_bspline_surface_control_net_obj(
    const std::string& filename,
    const BSplineSurface3D& surface);

bool export_control_points_obj(
    const std::string& filename,
    const std::vector<Eigen::Vector3d>& points);

bool export_polyline_obj(
    const std::string& filename,
    const std::vector<Eigen::Vector3d>& points);

bool export_bspline_curve_control_polygon_obj(
    const std::string& filename,
    const BSplineCurve3D& curve);

#endif
