#ifndef GLOBAL_BSPLINE_OPTIMIZATION_HEADER
#define GLOBAL_BSPLINE_OPTIMIZATION_HEADER

#include "fit_bspline_surface_interior.h"
#include "shared_spline_boundary.h"

#include <Eigen/Dense>

#include <map>
#include <string>
#include <vector>

struct GlobalBSplineOptimizationConfig {
    double fairness_weight;
    double initial_weight;
    bool optimize_shared_boundary_control_points;
    double boundary_fit_weight;
    double boundary_fairness_weight;
    int boundary_fit_sample_count;
    int boundary_drift_sample_count;
    double max_boundary_drift;
    bool rollback_on_excessive_boundary_drift;
    bool compare_with_fixed_boundary;
    int sample_u;
    int sample_v;

    GlobalBSplineOptimizationConfig();
};

struct GlobalBSplineRegionInput {
    int patch_index;
    int region_id;
    std::vector<SurfaceFitSample> samples;
};

struct GlobalBSplineRegionReport {
    int patch_index;
    int region_id;
    bool optimized;
    std::string reason;
    SurfaceFitErrorStats before;
    SurfaceFitErrorStats after;

    GlobalBSplineRegionReport();
};

struct GlobalBSplineOptimizationResult {
    SharedSplineAssembly optimized_assembly;
    SharedSplineSampledMesh sampled_mesh;
    SharedSplineAdjacencyReport adjacency_report;
    std::vector<GlobalBSplineRegionReport> region_reports;
    std::vector<std::string> skipped_regions;
    int total_control_points;
    int variable_control_points;
    int spline_patch_count;
    int source_vertex_count;
    int source_triangle_count;
    int matrix_rows;
    int matrix_cols;
    int matrix_nonzeros;
    double normal_matrix_condition_estimate;
    bool movable_boundary_enabled;
    bool has_fixed_boundary_comparison;
    bool boundary_drift_exceeded;
    bool used_fixed_boundary_fallback;
    int fixed_boundary_variable_control_points;
    SurfaceFitErrorStats fixed_boundary_global_after;
    SurfaceFitErrorStats boundary_drift;
    SharedSplineAssembly fixed_boundary_assembly;
    SharedSplineSampledMesh fixed_boundary_sampled_mesh;
    SharedSplineAdjacencyReport fixed_boundary_adjacency_report;
    SurfaceFitErrorStats global_before;
    SurfaceFitErrorStats global_after;
    bool valid;
    std::string reason;

    GlobalBSplineOptimizationResult();
};

GlobalBSplineOptimizationResult optimize_global_bspline_control_points(
    const SharedSplineAssembly& assembly,
    const std::vector<GlobalBSplineRegionInput>& region_inputs,
    const GlobalBSplineOptimizationConfig& cfg = GlobalBSplineOptimizationConfig());

bool export_global_bspline_optimization_debug(
    const std::string& prefix,
    const GlobalBSplineOptimizationResult& result);

bool export_global_bspline_region_report_csv(
    const std::string& filename,
    const GlobalBSplineOptimizationResult& result);

bool export_global_bspline_summary_csv(
    const std::string& filename,
    const GlobalBSplineOptimizationResult& result);

#endif
