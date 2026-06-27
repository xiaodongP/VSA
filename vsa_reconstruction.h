#ifndef VSA_RECONSTRUCTION_HEADER
#define VSA_RECONSTRUCTION_HEADER

#include <Eigen/Dense>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "feature_barrier.h"
#include "quadric_proxy.h"

using namespace Eigen;
using namespace std;

struct RegionPatch {
    int region_id;

    vector<int> face_ids;
    vector<int> boundary_vertex_ids;
    vector<int> boundary_edge_ids;
    vector<int> interior_vertex_ids;

    vector<EdgeKey> boundary_edges;
    vector<int> ordered_boundary_vertex_ids;

    struct BoundarySample {
        int original_vertex_id;
        EdgeKey edge;
        int segment_index;
        int segment_count;
        double t_from_edge_v0;
        Vector3d position;
        Vector2d uv;

        BoundarySample()
            : original_vertex_id(-1),
              segment_index(0),
              segment_count(1),
              t_from_edge_v0(0.0),
              position(Vector3d::Zero()),
              uv(Vector2d::Zero()) {}

        bool is_original_vertex() const { return original_vertex_id >= 0; }
    };

    vector<BoundarySample> boundary_samples;

    QuadricProxy quadric;

    Vector3d center;
    Vector3d normal;
    Vector3d tangent_u;
    Vector3d tangent_v;

    vector<Vector2d> boundary_uv;
    vector<Vector2d> interior_uv_samples;
    vector<Vector2d> candidate_uv_samples;
    vector<Vector2d> lift_success_uv_samples;
    vector<Vector2d> lift_failed_uv_samples;

    struct InteriorSampleMeta {
        int ix;
        int iy;
        bool is_grid_sample;
        bool near_boundary;
        double boundary_distance;
        int nearest_boundary_segment;
        double nearest_boundary_t;

        InteriorSampleMeta()
            : ix(-1),
              iy(-1),
              is_grid_sample(false),
              near_boundary(false),
              boundary_distance(std::numeric_limits<double>::infinity()),
              nearest_boundary_segment(-1),
              nearest_boundary_t(0.0) {}

        InteriorSampleMeta(int ix_,
                           int iy_,
                           bool is_grid_sample_,
                           bool near_boundary_,
                           double boundary_distance_,
                           int nearest_boundary_segment_,
                           double nearest_boundary_t_)
            : ix(ix_),
              iy(iy_),
              is_grid_sample(is_grid_sample_),
              near_boundary(near_boundary_),
              boundary_distance(boundary_distance_),
              nearest_boundary_segment(nearest_boundary_segment_),
              nearest_boundary_t(nearest_boundary_t_) {}
    };

    vector<InteriorSampleMeta> interior_sample_meta;

    Vector2d bbox_uv_min;
    Vector2d bbox_uv_max;
    Vector3d bbox_3d_min;
    Vector3d bbox_3d_max;
    double bbox_3d_diag;
    double normal_extent;
    double sample_spacing;
    int bbox_grid_resolution_x;
    int bbox_grid_resolution_y;

    bool valid_chart;
    bool simple_boundary_loop;
    bool valid_polygon;
    int boundary_loop_count;
    double polygon_signed_area;
    double polygon_abs_area;
    bool polygon_self_intersect;
    double max_lift_distance;
    double min_lift_normal_alignment;
    double elapsed_chart_ms;
    double elapsed_sampling_ms;

    RegionPatch()
        : region_id(-1),
          center(Vector3d::Zero()),
          normal(Vector3d::UnitZ()),
          tangent_u(Vector3d::UnitX()),
          tangent_v(Vector3d::UnitY()),
          bbox_uv_min(Vector2d::Zero()),
          bbox_uv_max(Vector2d::Zero()),
          bbox_3d_min(Vector3d::Zero()),
          bbox_3d_max(Vector3d::Zero()),
          bbox_3d_diag(0.0),
          normal_extent(0.0),
          sample_spacing(0.0),
          bbox_grid_resolution_x(0),
          bbox_grid_resolution_y(0),
          valid_chart(false),
          simple_boundary_loop(false),
          valid_polygon(false),
          boundary_loop_count(0),
          polygon_signed_area(0.0),
          polygon_abs_area(0.0),
          polygon_self_intersect(false),
          max_lift_distance(1.0),
          min_lift_normal_alignment(0.0),
          elapsed_chart_ms(0.0),
          elapsed_sampling_ms(0.0) {}
};

enum class RegionFallbackType {
    None,
    SkipRegion,
    OriginalRegionMesh,
    FanTriangulation,
    GridTriangulation,
    ConstrainedTriangulation,
    InvalidChart,
    InvalidPolygon,
    TooFewInteriorSamples,
    TooManyFailedLifts
};

enum class ProjectionDropReason {
    None,
    ProjectionFailed,
    NaN,
    DegenerateArea,
    FlippedNormal,
    OutsideDomain,
    SharpBoundaryInvalid,
    RegionMismatch
};

struct ProjectionFilterDebugTriangle {
    int region_id;
    int parent_face_id;
    int parent_region_id;
    ProjectionDropReason reason;
    bool kept;

    ProjectionFilterDebugTriangle()
        : region_id(-1),
          parent_face_id(-1),
          parent_region_id(-1),
          reason(ProjectionDropReason::None),
          kept(false) {}
};

struct ProjectionFilterDebugMesh {
    vector<Vector3d> vertices;
    vector<Vector3i> faces;
    vector<ProjectionFilterDebugTriangle> triangles;
};

struct ReconstructionDebugOptions {
    bool enable_debug_report;
    bool export_region_2d_debug;
    vector<int> debug_region_ids;
    int max_debug_regions;
    string output_prefix;

    ReconstructionDebugOptions()
        : enable_debug_report(true),
          export_region_2d_debug(false),
          max_debug_regions(10),
          output_prefix("reconstruction") {}
};

enum class InteriorProjectionMode {
    QuadricLift,
    OriginalMeshRayCast,
    RayCastThenQuadricFallback
};

struct ReconstructionConfig {
    int max_regions;
    int grid_resolution;
    int max_samples_per_region;
    double max_lift_distance_factor;
    double min_lift_normal_alignment;
    double max_displacement_factor;
    double max_failed_lift_fraction;
    double boundary_subdivision_spacing_factor;
    double boundary_transition_width_factor;
    double adaptive_boundary_band_factor;
    double fast_triangulation_boundary_band_factor;
    double target_sample_spacing;
    double boundary_ring_spacing_factor;
    double max_delaunay_edge_factor;
    double max_triangle_edge_ratio;
    double max_ray_projection_distance_factor;
    double surface_sample_spacing_factor;
    double surface_metric_refine_factor;
    double surface_min_sample_spacing_factor;
    double metric_warp_anisotropy_clamp;
    double proxy_triangle_edge_ratio;
    int max_boundary_subdivisions_per_edge;
    int boundary_ring_layers;
    int surface_metric_refinement_passes;
    InteriorProjectionMode interior_projection_mode;
    bool enable_displacement;
    bool enable_boundary_subdivision;
    bool enable_boundary_transition;
    bool enable_adaptive_boundary_sampling;
    bool enable_global_sample_spacing;
    bool enable_boundary_ring_sampling;
    bool enable_surface_metric_sampling;
    bool enable_surface_spacing_filter;
    bool enable_metric_warped_triangulation;
    bool enable_proxy_triangle_quality_filter;
    bool prefer_full_patch_delaunay;
    bool use_cgal_cdt;
    bool use_simple_grid_triangulation;
    bool fallback_to_original;
    bool export_ply;
    bool verbose;
    ReconstructionDebugOptions debug;

    ReconstructionConfig()
        : max_regions(-1),
          grid_resolution(32),
          max_samples_per_region(8192),
          max_lift_distance_factor(0.75),
          min_lift_normal_alignment(0.0),
          max_displacement_factor(2.0),
          max_failed_lift_fraction(0.5),
          boundary_subdivision_spacing_factor(1.0),
          boundary_transition_width_factor(3.0),
          adaptive_boundary_band_factor(2.5),
          fast_triangulation_boundary_band_factor(1.5),
          target_sample_spacing(0.0),
          boundary_ring_spacing_factor(0.9),
          max_delaunay_edge_factor(3.0),
          max_triangle_edge_ratio(5.0),
          max_ray_projection_distance_factor(2.0),
          surface_sample_spacing_factor(1.0),
          surface_metric_refine_factor(1.20),
          surface_min_sample_spacing_factor(0.30),
          metric_warp_anisotropy_clamp(8.0),
          proxy_triangle_edge_ratio(4.0),
          max_boundary_subdivisions_per_edge(64),
          boundary_ring_layers(1),
          surface_metric_refinement_passes(2),
          interior_projection_mode(InteriorProjectionMode::QuadricLift),
          enable_displacement(false),
          enable_boundary_subdivision(true),
          enable_boundary_transition(false),
          enable_adaptive_boundary_sampling(true),
          enable_global_sample_spacing(true),
          enable_boundary_ring_sampling(true),
          enable_surface_metric_sampling(true),
          enable_surface_spacing_filter(true),
          enable_metric_warped_triangulation(true),
          enable_proxy_triangle_quality_filter(true),
          prefer_full_patch_delaunay(true),
          use_cgal_cdt(true),
          use_simple_grid_triangulation(true),
          fallback_to_original(true),
          export_ply(true),
          verbose(true) {}
};

struct ReconstructionOptions : public ReconstructionConfig {
    ReconstructionOptions() : ReconstructionConfig() {}
};

struct RegionReconstructionStats {
    int region_id;
    int face_count;
    int boundary_vertex_count;
    int interior_sample_count;
    int output_face_count;
    bool chart_valid;
    bool polygon_valid;
    bool skipped;
    bool fallback_used;
    int failed_lifts;
    int failed_displacement_queries;
    double average_displacement;
    double max_displacement;
    string reason;
    RegionFallbackType fallback_type;

    RegionReconstructionStats()
        : region_id(-1),
          face_count(0),
          boundary_vertex_count(0),
          interior_sample_count(0),
          output_face_count(0),
          chart_valid(false),
          polygon_valid(false),
          skipped(false),
          fallback_used(false),
          failed_lifts(0),
          failed_displacement_queries(0),
          average_displacement(0.0),
          max_displacement(0.0),
          fallback_type(RegionFallbackType::None) {}
};

struct RegionReconstructionDebugInfo {
    int region_id;

    int face_count;
    int boundary_vertex_count;
    int boundary_edge_count;
    int boundary_loop_count;

    bool chart_valid;
    bool polygon_valid;
    double polygon_signed_area;
    double polygon_abs_area;
    bool polygon_self_intersect;

    int bbox_grid_resolution_x;
    int bbox_grid_resolution_y;
    int candidate_sample_count;
    int inside_sample_count;
    double inside_ratio;

    int lift_attempt_count;
    int lift_success_count;
    int lift_fail_count;
    double lift_success_ratio;

    int final_boundary_vertex_count;
    int final_interior_vertex_count;
    int final_triangle_count;

    int input_faces;
    int subdivided_triangles;
    int projected_vertices;
    int projection_failed_vertices;
    int kept_triangles;
    int dropped_triangles;
    int dropped_projection_failed;
    int dropped_nan;
    int dropped_degenerate_area;
    int dropped_flipped_normal;
    int dropped_outside_domain;
    int dropped_sharp_boundary_invalid;
    int dropped_region_mismatch;

    RegionFallbackType fallback_type;

    double elapsed_chart_ms;
    double elapsed_sampling_ms;
    double elapsed_lifting_ms;
    double elapsed_triangulation_ms;
    double elapsed_total_ms;

    RegionReconstructionDebugInfo()
        : region_id(-1),
          face_count(0),
          boundary_vertex_count(0),
          boundary_edge_count(0),
          boundary_loop_count(0),
          chart_valid(false),
          polygon_valid(false),
          polygon_signed_area(0.0),
          polygon_abs_area(0.0),
          polygon_self_intersect(false),
          bbox_grid_resolution_x(0),
          bbox_grid_resolution_y(0),
          candidate_sample_count(0),
          inside_sample_count(0),
          inside_ratio(0.0),
          lift_attempt_count(0),
          lift_success_count(0),
          lift_fail_count(0),
          lift_success_ratio(0.0),
          final_boundary_vertex_count(0),
          final_interior_vertex_count(0),
          final_triangle_count(0),
          input_faces(0),
          subdivided_triangles(0),
          projected_vertices(0),
          projection_failed_vertices(0),
          kept_triangles(0),
          dropped_triangles(0),
          dropped_projection_failed(0),
          dropped_nan(0),
          dropped_degenerate_area(0),
          dropped_flipped_normal(0),
          dropped_outside_domain(0),
          dropped_sharp_boundary_invalid(0),
          dropped_region_mismatch(0),
          fallback_type(RegionFallbackType::None),
          elapsed_chart_ms(0.0),
          elapsed_sampling_ms(0.0),
          elapsed_lifting_ms(0.0),
          elapsed_triangulation_ms(0.0),
          elapsed_total_ms(0.0) {}
};

struct ReconstructionLog {
    int num_regions;
    int processed_regions;
    int skipped_regions;
    int fallback_regions;
    int reconstructed_vertices;
    int reconstructed_faces;
    int total_failed_lifts;
    int total_failed_displacement_queries;
    int normal_path_regions;
    int fan_triangulation_regions;
    int original_mesh_fallback_regions;
    int invalid_chart_regions;
    int invalid_polygon_regions;
    int too_few_sample_regions;
    int total_candidate_samples;
    int total_inside_samples;
    int total_successful_lifts;
    double total_reconstruction_time_ms;
    map<int, int> boundary_vertex_map;
    vector<RegionReconstructionStats> region_stats;
    vector<RegionReconstructionDebugInfo> debug_infos;
    ProjectionFilterDebugMesh projection_filter_debug_mesh;

    ReconstructionLog()
        : num_regions(0),
          processed_regions(0),
          skipped_regions(0),
          fallback_regions(0),
          reconstructed_vertices(0),
          reconstructed_faces(0),
          total_failed_lifts(0),
          total_failed_displacement_queries(0),
          normal_path_regions(0),
          fan_triangulation_regions(0),
          original_mesh_fallback_regions(0),
          invalid_chart_regions(0),
          invalid_polygon_regions(0),
          too_few_sample_regions(0),
          total_candidate_samples(0),
          total_inside_samples(0),
          total_successful_lifts(0),
          total_reconstruction_time_ms(0.0) {}
};

struct ReconstructedMesh {
    MatrixXd V;
    MatrixXi F;
    MatrixXi R;
    map<int, int> boundary_vertex_map;
};

class DisplacementQuery {
public:
    virtual ~DisplacementQuery() {}

    virtual bool query(
        int region_id,
        const Vector3d& q,
        const Vector3d& normal,
        Vector3d& out_reference_point
    ) = 0;
};

class NullDisplacementQuery : public DisplacementQuery {
public:
    bool query(
        int region_id,
        const Vector3d& q,
        const Vector3d& normal,
        Vector3d& out_reference_point
    ) override;
};

vector<RegionPatch> extractRegionPatches(
    const MatrixXd& V,
    const MatrixXi& F,
    const MatrixXi& R,
    const vector<QuadricProxy>& QP,
    int num_regions,
    const ReconstructionConfig& cfg);

bool liftToQuadric(
    const RegionPatch& patch,
    const Vector2d& uv,
    Vector3d& out_pos);

bool reconstructQuadricVSA(
    const MatrixXd& V,
    const MatrixXi& F,
    const MatrixXi& R,
    const vector<QuadricProxy>& QP,
    int num_regions,
    const ReconstructionConfig& cfg,
    DisplacementQuery* displacement_query,
    ReconstructedMesh& out_mesh,
    ReconstructionLog& log_out);

bool exportReconstructedMesh(
    const string& filename,
    const ReconstructedMesh& mesh);

bool exportReconstructedMeshPLY(
    const string& filename,
    const ReconstructedMesh& mesh);

bool exportReconstructionLog(
    const string& filename,
    const ReconstructionLog& log);

bool exportReconstructionDebugReport(
    const string& filename,
    const ReconstructionLog& log);

bool exportProjectionFilterDebugTrianglesCSV(
    const string& filename,
    const ReconstructionLog& log);

bool exportProjectionFilterDebugMeshPLY(
    const string& filename,
    const ReconstructionLog& log);

string regionFallbackTypeName(RegionFallbackType type);

string projectionDropReasonName(ProjectionDropReason reason);

bool reconstructAndExportQuadricVSA(
    const MatrixXd& V,
    const MatrixXi& F,
    const MatrixXi& R,
    const vector<QuadricProxy>& QP,
    int num_regions,
    const ReconstructionConfig& cfg,
    const string& base_filename);

#endif
