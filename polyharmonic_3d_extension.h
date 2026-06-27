#ifndef POLYHARMONIC_3D_EXTENSION_HEADER
#define POLYHARMONIC_3D_EXTENSION_HEADER

#include "rectangular_domain_extension.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

enum class PolyharmonicContinuityMode {
    G1,
    G2
};

struct Polyharmonic3DExtensionConfig {
    PolyharmonicContinuityMode mode = PolyharmonicContinuityMode::G2;
    double regularization = 1e-8;
    double min_mass = 1e-12;
    double fairness_weight = 0.0;
    double isocurve_fairness_weight = 0.5;
    double isocurve_weight_scale = 1.0;
    double mesh_fairness_weight_scale = 1.0;
    double initial_position_weight = 1e-2;
    double isocurve_tolerance = 1e-7;
    bool include_rectangle_boundary_isocurves = true;
    std::vector<double> labeled_isocurve_u_values;
    std::vector<double> labeled_isocurve_v_values;
    bool export_debug = true;
    std::string debug_prefix = "polyharmonic_3d_extension";
};

struct Polyharmonic3DExtensionResult {
    bool valid = false;
    std::string reason;
    Eigen::MatrixXd extended_vertices;
    Eigen::MatrixXi faces;
    std::vector<bool> original_vertex_mask;
    std::vector<bool> original_face_mask;
    PolyharmonicContinuityMode mode = PolyharmonicContinuityMode::G2;
    int fixed_vertex_count = 0;
    int unknown_vertex_count = 0;
    double min_mass = 0.0;
    double max_mass = 0.0;
    double mean_unknown_displacement = 0.0;
    double max_unknown_displacement = 0.0;
    double residual_norm = 0.0;
    double extension_energy = 0.0;
    double isocurve_fairness_energy = 0.0;
    double mesh_fairness_energy = 0.0;
    double mean_boundary_curvature = 0.0;
    double max_boundary_curvature = 0.0;
    double original_surface_area = 0.0;
    double extended_surface_area = 0.0;
    double surface_area_growth = 0.0;
    double original_bbox_diagonal = 0.0;
    double extended_bbox_diagonal = 0.0;
    double bbox_growth = 0.0;
};

Polyharmonic3DExtensionResult extend_polyharmonic_3d(
    const RectangularDomainExtensionResult& domain,
    const Eigen::MatrixXd& original_positions,
    const Polyharmonic3DExtensionConfig& config = Polyharmonic3DExtensionConfig());

bool export_polyharmonic_3d_extension_debug(
    const std::string& prefix,
    const Polyharmonic3DExtensionResult& result);

const char* to_string(PolyharmonicContinuityMode mode);

#endif
