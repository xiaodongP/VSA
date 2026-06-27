#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/jet.h>
#include <imgui.h>
#include <iostream>
#include <ostream>

#include "HalfedgeBuilder.cpp"

#include "partitioning.h"
#include "distance.h"
#include "proxies.h"
#include "anchors.h"
#include "triangulation.h"
#include "renumbering.h"
#include "vsa_batch.h"
#include "feature_barrier.h"
#include "region_boundary.h"
#include "quad_like_boundary.h"
#include "region_square_parameterization.h"
#include "initial_bspline_surface.h"
#include "fit_bspline_surface_interior.h"
#include "bspline.h"
#include "trimmed_bspline_surface.h"
#include "trimmed_bspline_pipeline.h"
#include "trimmed_mesh_validation.h"
#include "spline_polyscope_renderer.h"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <limits>
#include <string>
#include <sstream>
#include <map>
#include <set>

using namespace Eigen; // to use the classes provided by Eigen library
using namespace std;

MatrixXd V; // matrix storing vertex coordinates of the input mesh (n rows, 3 columns)
MatrixXi F; // incidence relations between faces and edges (f columns)
MatrixXi R; // matrix indicating the partition of each vertex
MatrixXd newV; // matrix storing vertex coordinates of the input mesh (n rows, 3 columns)
MatrixXi newF; // incidence relations between faces and edges (f columns)
MatrixXi newR; // matrix indicating the partition of each vertex
MatrixXd C; // the coloring
MatrixXd Proxies;
MatrixXi Ad; // face adjacency
int p; // number of proxies
MetricMode metric = L21_METRIC; // L2_METRIC=0, L21_METRIC=1, HYBRID_METRIC=2 (第一步过渡版本)
HalfedgeDS* he;
int iterations;
vector<pair<int,double>> global_error_points; //contains the global_distortion_error according to the number of iterations
double error;
double precedent_error;
double treshold;

// ---- Visualization state ----
MatrixXi R_before_smoothing;
MatrixXi R_after_smoothing;
bool has_before_smoothing = false;
bool has_after_smoothing = false;

MatrixXd projectedV;
MatrixXi projectedF;
bool has_projected_mesh = false;

MatrixXd reconstructedV;
MatrixXi reconstructedF;
MatrixXi reconstructedR;
bool has_reconstructed_mesh = false;

VectorXd face_error_values;
bool has_face_error_values = false;
bool g_show_region_id_labels = false;
double g_region_label_offset_scale = 0.02;

// ---- Quadric mode state ----
vector<QuadricProxy> QP;
bool use_quadric = false;

enum class ViewerWorkMode {
  QuadricVSA,
  SplineSurface
};

enum class SplineRenderBackend {
  LibiglSampledMesh,
  Polyscope
};

enum class TrimRenderDebugMode {
  TrimmedOnly,
  OriginalOnly,
  FullSplineOnly,
  AssetOnly,
  ABCPreviewOnly,
  ABCRibbonOnly,
  ABCPreviewAndRibbon,
  TrimmedAndOriginal
};

ViewerWorkMode g_viewer_work_mode = ViewerWorkMode::QuadricVSA;
SplineRenderBackend g_spline_render_backend = SplineRenderBackend::LibiglSampledMesh;
TrimRenderDebugMode g_trim_render_debug_mode = TrimRenderDebugMode::TrimmedOnly;
int g_spline_region_id = 0;
int g_spline_sample_u = 33;
int g_spline_sample_v = 33;
int g_spline_control_count_u = 5;
int g_spline_control_count_v = 5;
double g_spline_fit_fairness_weight = 1e-6;
double g_spline_fit_initial_weight = 1e-8;
double g_trimmed_boundary_fit_weight = 1000.0;
bool g_trimmed_harmonic_boundary_correction = true;
bool g_trimmed_snap_output_boundary = false;
int g_trimmed_spline_grid_u = 45;
int g_trimmed_spline_grid_v = 45;
MatrixXd splineSurfaceV;
MatrixXi splineSurfaceF;
bool has_spline_surface_mesh = false;
MatrixXd g_trim_debug_originalV;
MatrixXi g_trim_debug_originalF;
MatrixXd g_trim_debug_fullSplineV;
MatrixXi g_trim_debug_fullSplineF;
MatrixXd g_trim_debug_assetV;
MatrixXi g_trim_debug_assetF;
MatrixXd g_trim_debug_abcPreviewV;
MatrixXi g_trim_debug_abcPreviewF;
MatrixXd g_trim_debug_abcRibbonV;
MatrixXi g_trim_debug_abcRibbonF;
bool g_has_trim_debug_original = false;
bool g_has_trim_debug_full_spline = false;
bool g_has_trim_debug_asset = false;
bool g_has_trim_debug_abc_preview = false;
bool g_has_trim_debug_abc_ribbon = false;
InitialBSplineSurfacePatch g_last_spline_patch;
bool has_last_spline_patch = false;
string g_last_spline_debug_prefix;
char g_quadric_snapshot_file[260] = "interactive_quadric_vsa_snapshot.qvsa";
char g_polyscope_viewer_exe[260] = "tools/polyscope_spline_viewer/build/spline_polyscope_viewer.exe";
char g_region_snapshot_prefix[260] = "selected_region_snapshot";
char g_trimmed_pipeline_output_dir[260] = "trimmed_bspline_output";
string g_spline_status = "Spline pipeline idle.";

static void set_spline_status(const string& message) {
  g_spline_status = message;
  cout << message << endl;
  cout.flush();
  ofstream out("trimmed_pipeline_viewer_status.txt");
  if (out.is_open()) {
    out << message << "\n";
  }
}

static const char* viewer_mode_name(ViewerWorkMode mode) {
  return mode == ViewerWorkMode::QuadricVSA ? "Quadric VSA" : "Spline Surface";
}

static const char* trim_render_debug_mode_name(TrimRenderDebugMode mode) {
  switch (mode) {
  case TrimRenderDebugMode::TrimmedOnly: return "StandardTrimmedOnly";
  case TrimRenderDebugMode::OriginalOnly: return "OriginalRegionOnly";
  case TrimRenderDebugMode::FullSplineOnly: return "FullUntrimmedSplineOnly";
  case TrimRenderDebugMode::AssetOnly: return "StandardAssetOnly";
  case TrimRenderDebugMode::ABCPreviewOnly: return "ExperimentalABCPreviewOnly";
  case TrimRenderDebugMode::ABCRibbonOnly: return "ExperimentalABCRibbonOnly";
  case TrimRenderDebugMode::ABCPreviewAndRibbon: return "ExperimentalABCPreviewAndRibbon";
  case TrimRenderDebugMode::TrimmedAndOriginal: return "StandardTrimmedAndOriginal";
  }
  return "Unknown";
}

static bool is_data_visible_on_core(
    const igl::opengl::ViewerData& data,
    const igl::opengl::ViewerCore& core) {
  return (data.is_visible & core.id) != 0;
}

static void clear_viewer_to_single_slot(igl::opengl::glfw::Viewer& viewer) {
  while (viewer.data_list.size() > 1) {
    viewer.erase_mesh(viewer.data_list.size() - 1);
  }
  viewer.selected_data_index = 0;
  viewer.data().clear();
  viewer.data().set_visible(true, viewer.core().id);
}

static void print_viewer_data_slots(
    const igl::opengl::glfw::Viewer& viewer,
    const string& context) {
  cout << "[viewer slots] " << context << endl;
  for (size_t i = 0; i < viewer.data_list.size(); i++) {
    const auto& data = viewer.data_list[i];
    cout << "  slot id=" << data.id
         << " index=" << i
         << " mesh name=" << (i == 0 ? "primary" : "secondary")
         << " vertices=" << data.V.rows()
         << " faces=" << data.F.rows()
         << " visible=" << (is_data_visible_on_core(data, viewer.core()) ? "true" : "false")
         << " show_faces=" << (viewer.core().is_set(data.show_faces) ? "true" : "false")
         << " show_lines=" << (viewer.core().is_set(data.show_lines) ? "true" : "false")
         << " show_overlay=" << (viewer.core().is_set(data.show_overlay) ? "true" : "false")
         << " depth_test=" << (viewer.core().depth_test ? "true" : "false")
         << " face_based=" << (data.face_based ? "true" : "false")
         << " double_sided=" << (data.double_sided ? "true" : "false")
         << endl;
  }
}

static bool build_region_mesh_for_display(
    int region_id,
    MatrixXd& regionV,
    MatrixXi& regionF) {
  regionV.resize(0, 3);
  regionF.resize(0, 3);
  if (F.rows() <= 0 || R.rows() != F.rows()) return false;
  map<int, int> global_to_local;
  vector<Vector3d> verts;
  vector<Vector3i> faces;
  for (int fi = 0; fi < F.rows(); fi++) {
    if (R(fi, 0) != region_id) continue;
    Vector3i tri;
    for (int k = 0; k < 3; k++) {
      int gid = F(fi, k);
      auto it = global_to_local.find(gid);
      if (it == global_to_local.end()) {
        int lid = (int)verts.size();
        global_to_local[gid] = lid;
        verts.push_back(V.row(gid).transpose());
        tri(k) = lid;
      } else {
        tri(k) = it->second;
      }
    }
    faces.push_back(tri);
  }
  if (verts.empty() || faces.empty()) return false;
  regionV.resize((int)verts.size(), 3);
  for (int i = 0; i < (int)verts.size(); i++) regionV.row(i) = verts[i].transpose();
  regionF.resize((int)faces.size(), 3);
  for (int i = 0; i < (int)faces.size(); i++) regionF.row(i) = faces[i].transpose();
  return true;
}

static void setup_viewer_mesh_slot(
    igl::opengl::ViewerData& data,
    const MatrixXd& meshV,
    const MatrixXi& meshF,
    const RowVector3d& color,
    bool show_lines,
    bool face_based = false,
    bool double_sided = false) {
  data.clear();
  data.set_mesh(meshV, meshF);
  MatrixXd colors(meshF.rows(), 3);
  for (int i = 0; i < colors.rows(); i++) colors.row(i) = color;
  data.set_colors(colors);
  data.face_based = face_based;
  data.double_sided = double_sided;
  data.show_faces = true;
  data.show_lines = show_lines;
  data.show_overlay = false;
  data.show_overlay_depth = false;
  data.show_custom_labels = false;
  data.clear_labels();
}

static bool apply_trim_render_debug_mode(igl::opengl::glfw::Viewer& viewer) {
  print_viewer_data_slots(viewer, "before trim render mode apply");
  clear_viewer_to_single_slot(viewer);
  if (g_trim_render_debug_mode == TrimRenderDebugMode::OriginalOnly) {
    if (!g_has_trim_debug_original) return false;
    setup_viewer_mesh_slot(
        viewer.data(), g_trim_debug_originalV, g_trim_debug_originalF,
        RowVector3d(0.75, 0.75, 0.75), false);
    viewer.core().align_camera_center(g_trim_debug_originalV, g_trim_debug_originalF);
  } else if (g_trim_render_debug_mode == TrimRenderDebugMode::FullSplineOnly) {
    if (!g_has_trim_debug_full_spline) return false;
    setup_viewer_mesh_slot(
        viewer.data(), g_trim_debug_fullSplineV, g_trim_debug_fullSplineF,
        RowVector3d(0.35, 0.72, 0.95), false);
    viewer.core().align_camera_center(g_trim_debug_fullSplineV, g_trim_debug_fullSplineF);
  } else if (g_trim_render_debug_mode == TrimRenderDebugMode::AssetOnly) {
    if (!g_has_trim_debug_asset) return false;
    setup_viewer_mesh_slot(
        viewer.data(), g_trim_debug_assetV, g_trim_debug_assetF,
        RowVector3d(0.40, 0.58, 0.95), false);
    viewer.core().align_camera_center(g_trim_debug_assetV, g_trim_debug_assetF);
  } else if (g_trim_render_debug_mode == TrimRenderDebugMode::ABCPreviewOnly) {
    if (!g_has_trim_debug_abc_preview) return false;
    setup_viewer_mesh_slot(
        viewer.data(), g_trim_debug_abcPreviewV, g_trim_debug_abcPreviewF,
        RowVector3d(0.95, 0.55, 0.20), false, false, true);
    viewer.core().align_camera_center(g_trim_debug_abcPreviewV, g_trim_debug_abcPreviewF);
  } else if (g_trim_render_debug_mode == TrimRenderDebugMode::ABCRibbonOnly) {
    if (!g_has_trim_debug_abc_ribbon) return false;
    setup_viewer_mesh_slot(
        viewer.data(), g_trim_debug_abcRibbonV, g_trim_debug_abcRibbonF,
        RowVector3d(0.10, 0.70, 0.82), false, false, true);
    viewer.core().align_camera_center(g_trim_debug_abcRibbonV, g_trim_debug_abcRibbonF);
  } else if (g_trim_render_debug_mode == TrimRenderDebugMode::ABCPreviewAndRibbon) {
    if (!g_has_trim_debug_abc_preview || !g_has_trim_debug_abc_ribbon) return false;
    setup_viewer_mesh_slot(
        viewer.data(), g_trim_debug_abcPreviewV, g_trim_debug_abcPreviewF,
        RowVector3d(0.95, 0.55, 0.20), false, false, true);
    int ribbon_id = viewer.append_mesh(true);
    int ribbon_index = viewer.mesh_index(ribbon_id);
    setup_viewer_mesh_slot(
        viewer.data(ribbon_id), g_trim_debug_abcRibbonV, g_trim_debug_abcRibbonF,
        RowVector3d(0.10, 0.70, 0.82), false, false, true);
    viewer.selected_data_index = ribbon_index;
    viewer.core().align_camera_center(g_trim_debug_abcPreviewV, g_trim_debug_abcPreviewF);
  } else if (g_trim_render_debug_mode == TrimRenderDebugMode::TrimmedAndOriginal) {
    if (!has_spline_surface_mesh || !g_has_trim_debug_original) return false;
    setup_viewer_mesh_slot(
        viewer.data(), g_trim_debug_originalV, g_trim_debug_originalF,
        RowVector3d(0.72, 0.72, 0.72), false);
    int trimmed_id = viewer.append_mesh(true);
    int trimmed_index = viewer.mesh_index(trimmed_id);
    setup_viewer_mesh_slot(
        viewer.data(trimmed_id), splineSurfaceV, splineSurfaceF,
        RowVector3d(0.58, 0.42, 0.94), false);
    viewer.selected_data_index = trimmed_index;
    viewer.core().align_camera_center(splineSurfaceV, splineSurfaceF);
  } else {
    if (!has_spline_surface_mesh) return false;
    setup_viewer_mesh_slot(
        viewer.data(), splineSurfaceV, splineSurfaceF,
        RowVector3d(0.58, 0.42, 0.94), false);
    viewer.core().align_camera_center(splineSurfaceV, splineSurfaceF);
  }
  print_viewer_data_slots(
      viewer,
      string("after trim render mode apply: ") +
          trim_render_debug_mode_name(g_trim_render_debug_mode));
  return true;
}

static bool launch_external_polyscope_spline_viewer(const string& prefix) {
  if (prefix.empty()) {
    cout << "[polyscope] No spline debug prefix available. Compute a spline region first." << endl;
    return false;
  }
  string exe(g_polyscope_viewer_exe);
  if (exe.empty()) {
    cout << "[polyscope] viewer executable path is empty." << endl;
    return false;
  }
#ifdef _WIN32
  string cmd = "start \"\" \"" + exe + "\" \"" + prefix + "\"";
#else
  string cmd = "\"" + exe + "\" \"" + prefix + "\" &";
#endif
  int rc = std::system(cmd.c_str());
  if (rc != 0) {
    cout << "[polyscope] launch command failed: " << cmd << endl;
    return false;
  }
  cout << "[polyscope] launched: " << cmd << endl;
  return true;
}

static bool render_current_asset_mesh_in_libigl(igl::opengl::glfw::Viewer& viewer) {
  if (!g_has_trim_debug_asset) {
    if (g_last_spline_debug_prefix.empty()) {
      cout << "[asset mesh] No output directory. Run the standard trimmed B-spline pipeline first." << endl;
      return false;
    }
    string asset_file = g_last_spline_debug_prefix + "/bspline_trimmed_surface_asset.obj";
    g_has_trim_debug_asset =
        igl::readOBJ(asset_file, g_trim_debug_assetV, g_trim_debug_assetF) &&
        g_trim_debug_assetV.rows() > 0 &&
        g_trim_debug_assetF.rows() > 0;
    if (!g_has_trim_debug_asset) {
      cout << "[asset mesh] Cannot read " << asset_file << endl;
      return false;
    }
  }
  g_trim_render_debug_mode = TrimRenderDebugMode::AssetOnly;
  if (!apply_trim_render_debug_mode(viewer)) {
    cout << "[asset mesh] Failed to apply AssetOnly render mode." << endl;
    return false;
  }
  cout << "[asset mesh] Rendered asset-derived mesh "
       << g_trim_debug_assetV.rows() << "V/"
       << g_trim_debug_assetF.rows() << "F" << endl;
  return true;
}

static bool render_current_abc_preview_in_libigl(igl::opengl::glfw::Viewer& viewer) {
  if (!g_has_trim_debug_abc_preview) {
    if (g_last_spline_debug_prefix.empty()) {
      cout << "[ABC preview] No output directory. Run the standard trimmed B-spline pipeline first." << endl;
      return false;
    }
    string preview_file = g_last_spline_debug_prefix + "/bspline_trimmed_surface_abc_preview.obj";
    g_has_trim_debug_abc_preview =
        igl::readOBJ(preview_file, g_trim_debug_abcPreviewV, g_trim_debug_abcPreviewF) &&
        g_trim_debug_abcPreviewV.rows() > 0 &&
        g_trim_debug_abcPreviewF.rows() > 0;
    if (!g_has_trim_debug_abc_preview) {
      cout << "[ABC preview] Cannot read " << preview_file << endl;
      return false;
    }
  }
  g_trim_render_debug_mode = TrimRenderDebugMode::ABCPreviewOnly;
  if (!apply_trim_render_debug_mode(viewer)) {
    cout << "[ABC preview] Failed to apply ABCPreviewOnly render mode." << endl;
    return false;
  }
  cout << "[ABC preview] Rendered EXPERIMENTAL boundary-controlled preview "
       << g_trim_debug_abcPreviewV.rows() << "V/"
       << g_trim_debug_abcPreviewF.rows() << "F" << endl;
  return true;
}

static bool render_current_abc_ribbon_in_libigl(igl::opengl::glfw::Viewer& viewer) {
  if (!g_has_trim_debug_abc_ribbon) {
    if (g_last_spline_debug_prefix.empty()) {
      cout << "[ABC ribbon] No output directory. Run the standard trimmed B-spline pipeline first." << endl;
      return false;
    }
    string ribbon_file = g_last_spline_debug_prefix + "/abc_boundary_ribbon_surfaces.obj";
    g_has_trim_debug_abc_ribbon =
        igl::readOBJ(ribbon_file, g_trim_debug_abcRibbonV, g_trim_debug_abcRibbonF) &&
        g_trim_debug_abcRibbonV.rows() > 0 &&
        g_trim_debug_abcRibbonF.rows() > 0;
    if (!g_has_trim_debug_abc_ribbon) {
      cout << "[ABC ribbon] Cannot read " << ribbon_file << endl;
      return false;
    }
  }
  g_trim_render_debug_mode = TrimRenderDebugMode::ABCRibbonOnly;
  if (!apply_trim_render_debug_mode(viewer)) {
    cout << "[ABC ribbon] Failed to apply ABCRibbonOnly render mode." << endl;
    return false;
  }
  cout << "[ABC ribbon] Rendered EXPERIMENTAL boundary ribbon surfaces "
       << g_trim_debug_abcRibbonV.rows() << "V/"
       << g_trim_debug_abcRibbonF.rows() << "F" << endl;
  return true;
}

static bool render_current_abc_preview_and_ribbon_in_libigl(
    igl::opengl::glfw::Viewer& viewer) {
  if (!g_has_trim_debug_abc_preview) {
    if (!render_current_abc_preview_in_libigl(viewer)) return false;
  }
  if (!g_has_trim_debug_abc_ribbon) {
    if (!render_current_abc_ribbon_in_libigl(viewer)) return false;
  }
  g_trim_render_debug_mode = TrimRenderDebugMode::ABCPreviewAndRibbon;
  if (!apply_trim_render_debug_mode(viewer)) {
    cout << "[ABC preview+ribbon] Failed to apply combined render mode." << endl;
    return false;
  }
  cout << "[ABC preview+ribbon] Rendered preview "
       << g_trim_debug_abcPreviewV.rows() << "V/"
       << g_trim_debug_abcPreviewF.rows() << "F and ribbon "
       << g_trim_debug_abcRibbonV.rows() << "V/"
       << g_trim_debug_abcRibbonF.rows() << "F" << endl;
  return true;
}

// ---- Region boundary structures ----
enum BoundaryType { BT_REGION, BT_MESH, BT_NONMANIFOLD };

struct BoundaryEdge {
    EdgeKey ek;
    BoundaryType type;
    int region_i, region_j;
    int face_i, face_j;
};

static vector<BoundaryEdge> g_boundary_edges;
static bool g_boundary_valid = false;
static vector<pair<int,int>> g_region_pairs;
static int g_current_pair_idx = 0;

// ---- Deterministic region color (same label = same color) ----
static Eigen::RowVector3d region_color_det(int label) {
  if (label < 0) return Eigen::RowVector3d(0.6, 0.6, 0.6);
  double r = sin((double)label * 1.7 + 0.3) * 0.5 + 0.5;
  double g = sin((double)label * 2.3 + 1.1) * 0.5 + 0.5;
  double b = sin((double)label * 3.1 + 2.7) * 0.5 + 0.5;
  r = max(r, 0.15); g = max(g, 0.15); b = max(b, 0.15);
  return Eigen::RowVector3d(r, g, b);
}

// ---- Face colors from labels ----
MatrixXd make_face_colors_from_labels(const MatrixXi& labels, int face_count) {
  MatrixXd colors(face_count, 3);
  if (labels.rows() != face_count || labels.cols() < 1) {
    cout << "WARNING: labels size mismatch (" << labels.rows() << " vs " << face_count
         << "), using gray" << endl;
    for (int i = 0; i < face_count; i++)
      colors.row(i) = RowVector3d(0.6, 0.6, 0.6);
    return colors;
  }
  for (int i = 0; i < face_count; i++)
    colors.row(i) = region_color_det(labels(i, 0));
  return colors;
}

static void clear_region_id_labels(igl::opengl::glfw::Viewer& viewer) {
  viewer.data().clear_labels();
  viewer.data().show_custom_labels = false;
  g_show_region_id_labels = false;
  cout << "[labels] Region id labels hidden." << endl;
}

static bool show_region_id_labels(igl::opengl::glfw::Viewer& viewer) {
  if (V.rows() <= 0 || F.rows() <= 0 || R.rows() != F.rows()) {
    cout << "[labels] Cannot show region ids: invalid current mesh or labels." << endl;
    return false;
  }

  int max_label = -1;
  for (int i = 0; i < R.rows(); i++) {
    max_label = max(max_label, R(i, 0));
  }
  int region_count = max(p, max_label + 1);
  if (region_count <= 0) {
    cout << "[labels] Cannot show region ids: no valid regions." << endl;
    return false;
  }

  vector<Vector3d> weighted_centers(region_count, Vector3d::Zero());
  vector<Vector3d> normal_sums(region_count, Vector3d::Zero());
  vector<double> area_sums(region_count, 0.0);
  vector<int> face_counts(region_count, 0);

  for (int fi = 0; fi < F.rows(); fi++) {
    int region_id = R(fi, 0);
    if (region_id < 0 || region_id >= region_count) continue;
    int i0 = F(fi, 0);
    int i1 = F(fi, 1);
    int i2 = F(fi, 2);
    if (i0 < 0 || i0 >= V.rows() ||
        i1 < 0 || i1 >= V.rows() ||
        i2 < 0 || i2 >= V.rows()) {
      continue;
    }

    Vector3d v0 = V.row(i0).transpose();
    Vector3d v1 = V.row(i1).transpose();
    Vector3d v2 = V.row(i2).transpose();
    Vector3d face_normal_area2 = (v1 - v0).cross(v2 - v0);
    double area = 0.5 * face_normal_area2.norm();
    Vector3d centroid = (v0 + v1 + v2) / 3.0;
    double weight = area > 1e-16 ? area : 1.0;

    weighted_centers[region_id] += weight * centroid;
    area_sums[region_id] += weight;
    normal_sums[region_id] += face_normal_area2;
    face_counts[region_id]++;
  }

  RowVector3d bb_min = V.colwise().minCoeff();
  RowVector3d bb_max = V.colwise().maxCoeff();
  double label_offset = g_region_label_offset_scale * (bb_max - bb_min).norm();

  vector<Vector3d> positions;
  vector<string> strings;
  positions.reserve(region_count);
  strings.reserve(region_count);
  for (int region_id = 0; region_id < region_count; region_id++) {
    if (face_counts[region_id] <= 0 || area_sums[region_id] <= 1e-16) continue;
    Vector3d center = weighted_centers[region_id] / area_sums[region_id];
    Vector3d normal = normal_sums[region_id];
    if (normal.norm() > 1e-12) {
      normal.normalize();
      center += label_offset * normal;
    }
    positions.push_back(center);
    strings.push_back(to_string(region_id));
  }

  if (positions.empty()) {
    cout << "[labels] Cannot show region ids: no non-empty regions." << endl;
    return false;
  }

  MatrixXd label_positions(positions.size(), 3);
  for (int i = 0; i < (int)positions.size(); i++) {
    label_positions.row(i) = positions[i].transpose();
  }
  viewer.data().set_labels(label_positions, strings);
  viewer.data().show_custom_labels = true;
  viewer.data().label_size = 1.35f;
  viewer.data().label_color << 0.02f, 0.02f, 0.02f, 1.0f;
  g_show_region_id_labels = true;

  cout << "[labels] Showing " << strings.size()
       << " region id labels. Use Z or the GUI button to hide/show." << endl;
  return true;
}

static void refresh_region_id_labels_if_enabled(igl::opengl::glfw::Viewer& viewer) {
  if (g_show_region_id_labels) {
    show_region_id_labels(viewer);
  }
}

// ---- Show segmentation view ----
void show_segmentation(igl::opengl::glfw::Viewer& viewer,
                       const MatrixXd& displayV, const MatrixXi& displayF,
                       const MatrixXi& labels, const string& name) {
  viewer.data().clear();
  viewer.data().set_mesh(displayV, displayF);
  MatrixXd colors = make_face_colors_from_labels(labels, displayF.rows());
  viewer.data().set_colors(colors);
  viewer.data().show_lines = true;
  refresh_region_id_labels_if_enabled(viewer);
  cout << "[view] " << name << " (" << displayF.rows() << " faces)" << endl;
}

// ---- Show changed faces after smoothing ----
void show_changed_faces_after_smoothing(igl::opengl::glfw::Viewer& viewer) {
  if (!has_before_smoothing || !has_after_smoothing) {
    cout << "WARNING: No smoothing before/after labels available." << endl;
    return;
  }
  if (R_before_smoothing.rows() != R_after_smoothing.rows() ||
      R_before_smoothing.rows() != F.rows()) {
    cout << "WARNING: Label size mismatch (before=" << R_before_smoothing.rows()
         << " after=" << R_after_smoothing.rows()
         << " faces=" << F.rows() << ")" << endl;
    return;
  }
  viewer.data().clear();
  viewer.data().set_mesh(V, F);
  MatrixXd colors(F.rows(), 3);
  int changed = 0;
  for (int i = 0; i < F.rows(); i++) {
    if (R_before_smoothing(i, 0) != R_after_smoothing(i, 0)) {
      colors.row(i) = RowVector3d(1.0, 0.2, 0.2);  // red highlight
      changed++;
    } else {
      colors.row(i) = RowVector3d(0.85, 0.85, 0.85);  // light gray
    }
  }
  viewer.data().set_colors(colors);
  viewer.data().show_lines = true;
  cout << "[view] Smoothing changed faces: " << changed << " / " << F.rows() << endl;
}

// ---- Show error heatmap ----
void show_error_heatmap(igl::opengl::glfw::Viewer& viewer) {
  if (!has_face_error_values) {
    cout << "WARNING: No per-face error values available." << endl;
    return;
  }
  if (face_error_values.size() != F.rows()) {
    cout << "WARNING: Error values size mismatch (" << face_error_values.size()
         << " vs " << F.rows() << " faces)" << endl;
    return;
  }
  viewer.data().clear();
  viewer.data().set_mesh(V, F);
  MatrixXd Cerr;
  igl::jet(face_error_values, true, Cerr);
  viewer.data().set_colors(Cerr);
  viewer.data().show_lines = true;
  cout << "[view] Error heatmap (min=" << face_error_values.minCoeff()
       << " max=" << face_error_values.maxCoeff() << ")" << endl;
}

// ---- Show projected mesh ----
void show_projected_mesh(igl::opengl::glfw::Viewer& viewer) {
  if (!has_projected_mesh) {
    cout << "WARNING: No projected mesh available." << endl;
    return;
  }
  viewer.data().clear();
  viewer.data().set_mesh(projectedV, projectedF);
  if (projectedF.rows() == R.rows()) {
    MatrixXd colors = make_face_colors_from_labels(R, projectedF.rows());
    viewer.data().set_colors(colors);
  }
  viewer.data().show_lines = true;
  cout << "[view] Projected mesh (" << projectedV.rows() << " verts, "
       << projectedF.rows() << " faces)" << endl;
}

// ---- Show reconstructed mesh ----
void show_reconstructed_mesh(igl::opengl::glfw::Viewer& viewer) {
  if (!has_reconstructed_mesh) {
    cout << "WARNING: No reconstructed mesh available. Press R after fitting regions." << endl;
    return;
  }
  viewer.data().clear();
  viewer.data().set_mesh(reconstructedV, reconstructedF);
  if (reconstructedR.rows() == reconstructedF.rows()) {
    MatrixXd colors = make_face_colors_from_labels(reconstructedR, reconstructedF.rows());
    viewer.data().set_colors(colors);
  }
  viewer.data().show_lines = true;
  cout << "[view] Reconstructed mesh (" << reconstructedV.rows() << " verts, "
       << reconstructedF.rows() << " faces)" << endl;
}

static void enter_quadric_vsa_mode(igl::opengl::glfw::Viewer& viewer) {
  g_viewer_work_mode = ViewerWorkMode::QuadricVSA;
  if (!use_quadric || (int)QP.size() != p) {
    QP.resize(p);
    for (int j = 0; j < p; j++) {
      QP[j] = fit_quadric_region(R, j, F, V);
    }
    use_quadric = true;
  }
  show_segmentation(viewer, V, F, R, "Quadric VSA mode");
  cout << "[mode] Quadric VSA. Regions=" << p
       << " quadric_proxies=" << QP.size() << endl;
}

static bool compute_and_show_spline_surface_for_region(
    igl::opengl::glfw::Viewer& viewer,
    int region_id) {
  g_viewer_work_mode = ViewerWorkMode::SplineSurface;
  if (F.rows() <= 0 || R.rows() != F.rows()) {
    cout << "[spline] Cannot compute: current segmentation is invalid." << endl;
    return false;
  }
  if (region_id < 0 || region_id >= p) {
    cout << "[spline] Cannot compute: region " << region_id
         << " outside [0," << (p - 1) << "]." << endl;
    return false;
  }

  vector<int> face_region_ids(F.rows(), -1);
  for (int i = 0; i < F.rows(); i++) face_region_ids[i] = R(i, 0);

  RegionBoundaryExtractionResult loop_result =
      extract_region_boundary_loop(V, F, face_region_ids, region_id);
  string prefix = "interactive_spline_region_" + to_string(region_id);
  g_last_spline_debug_prefix = prefix;
  export_region_boundary_debug_obj(
      prefix + "_boundary.obj", V, F, face_region_ids, region_id, loop_result);
  if (!loop_result.success) {
    cout << "[spline] Boundary extraction failed for region " << region_id
         << ": " << loop_result.reason << endl;
    return false;
  }

  QuadLikeBoundaryConfig quad_cfg;
  QuadLikeBoundaryResult quad_result =
      split_quad_like_boundary(loop_result.loop, quad_cfg);
  export_quad_like_boundary_pca_debug_obj(
      prefix + "_quad_like_pca.obj", quad_result.debug);
  if (!quad_result.success || !quad_result.boundary.valid) {
    cout << "[spline] Quad-like split failed for region " << region_id
         << ": " << quad_result.reason << endl;
    return false;
  }

  InitialBSplineSurfaceConfig surface_cfg;
  surface_cfg.control_count_u = max(4, g_spline_control_count_u);
  surface_cfg.control_count_v = max(4, g_spline_control_count_v);
  InitialBSplineSurfacePatch patch =
      build_initial_bspline_surface_from_quad_boundary(
          quad_result.boundary, surface_cfg);
  if (!patch.valid) {
    cout << "[spline] Initial B-spline surface failed for region " << region_id
         << ": " << patch.reason << endl;
    return false;
  }

  sample_bspline_surface(
      patch.surface,
      max(4, g_spline_sample_u),
      max(4, g_spline_sample_v),
      splineSurfaceV,
      splineSurfaceF);
  g_last_spline_patch = patch;
  has_last_spline_patch = true;
  has_spline_surface_mesh = splineSurfaceV.rows() > 0 && splineSurfaceF.rows() > 0;
  g_has_trim_debug_asset = false;
  g_has_trim_debug_abc_preview = false;
  export_initial_bspline_surface_debug(
      prefix, patch, max(4, g_spline_sample_u), max(4, g_spline_sample_v));

  viewer.data().clear();
  viewer.data().set_mesh(splineSurfaceV, splineSurfaceF);
  MatrixXd colors(splineSurfaceF.rows(), 3);
  for (int i = 0; i < colors.rows(); i++) {
    colors.row(i) = RowVector3d(0.35, 0.72, 0.95);
  }
  viewer.data().set_colors(colors);
  viewer.data().show_lines = true;
  viewer.core().align_camera_center(splineSurfaceV, splineSurfaceF);

  cout << "[spline] Region " << region_id
       << " sampled surface verts=" << splineSurfaceV.rows()
       << " faces=" << splineSurfaceF.rows()
       << " boundary_max_error=" << patch.boundary_max_error
       << " coons_fit_max_error=" << patch.coons_fit_max_error
       << endl;

  if (g_spline_render_backend == SplineRenderBackend::Polyscope) {
    launch_external_polyscope_spline_viewer(g_last_spline_debug_prefix);
  }
  return true;
}

struct RegionPCAUV {
  MatrixXd UV;
  vector<int> vertex_ids;
  Vector3d origin;
  Vector3d axis_u;
  Vector3d axis_v;
  double min_u;
  double max_u;
  double min_v;
  double max_v;
  bool valid;
  string reason;

  RegionPCAUV()
      : UV(),
        origin(Vector3d::Zero()),
        axis_u(Vector3d::UnitX()),
        axis_v(Vector3d::UnitY()),
        min_u(0.0),
        max_u(1.0),
        min_v(0.0),
        max_v(1.0),
        valid(false) {}
};

static vector<int> collect_region_vertex_ids(
    const MatrixXi& meshF,
    const vector<int>& face_region_ids,
    int region_id) {
  set<int> ids;
  for (int fi = 0; fi < meshF.rows(); fi++) {
    if (face_region_ids[fi] != region_id) continue;
    for (int k = 0; k < 3; k++) ids.insert(meshF(fi, k));
  }
  return vector<int>(ids.begin(), ids.end());
}

static RegionPCAUV compute_region_pca_uv(
    const MatrixXd& meshV,
    const vector<int>& vertex_ids,
    double padding_factor = 0.04) {
  RegionPCAUV result;
  result.UV = MatrixXd::Constant(meshV.rows(), 2, numeric_limits<double>::quiet_NaN());
  result.vertex_ids = vertex_ids;
  if (vertex_ids.size() < 3) {
    result.reason = "too few region vertices for PCA UV";
    return result;
  }

  Vector3d center = Vector3d::Zero();
  for (int vid : vertex_ids) {
    if (vid < 0 || vid >= meshV.rows()) {
      result.reason = "region vertex id out of range";
      return result;
    }
    center += meshV.row(vid).transpose();
  }
  center /= (double)vertex_ids.size();

  Matrix3d cov = Matrix3d::Zero();
  for (int vid : vertex_ids) {
    Vector3d p = meshV.row(vid).transpose() - center;
    cov += p * p.transpose();
  }
  cov /= (double)vertex_ids.size();

  Eigen::SelfAdjointEigenSolver<Matrix3d> eig(cov);
  if (eig.info() != Eigen::Success) {
    result.reason = "PCA eigen solve failed";
    return result;
  }
  result.axis_u = eig.eigenvectors().col(2).normalized();
  result.axis_v = eig.eigenvectors().col(1).normalized();
  result.origin = center;

  vector<Vector2d> raw(vertex_ids.size());
  result.min_u = numeric_limits<double>::infinity();
  result.max_u = -numeric_limits<double>::infinity();
  result.min_v = numeric_limits<double>::infinity();
  result.max_v = -numeric_limits<double>::infinity();
  for (int i = 0; i < (int)vertex_ids.size(); i++) {
    Vector3d p = meshV.row(vertex_ids[i]).transpose() - center;
    raw[i] = Vector2d(p.dot(result.axis_u), p.dot(result.axis_v));
    result.min_u = min(result.min_u, raw[i].x());
    result.max_u = max(result.max_u, raw[i].x());
    result.min_v = min(result.min_v, raw[i].y());
    result.max_v = max(result.max_v, raw[i].y());
  }

  double du = result.max_u - result.min_u;
  double dv = result.max_v - result.min_v;
  if (du <= 1e-14 || dv <= 1e-14) {
    result.reason = "PCA UV bounding box is degenerate";
    return result;
  }
  double pad_u = max(1e-8, padding_factor * du);
  double pad_v = max(1e-8, padding_factor * dv);
  result.min_u -= pad_u;
  result.max_u += pad_u;
  result.min_v -= pad_v;
  result.max_v += pad_v;
  du = result.max_u - result.min_u;
  dv = result.max_v - result.min_v;

  for (int i = 0; i < (int)vertex_ids.size(); i++) {
    int vid = vertex_ids[i];
    result.UV(vid, 0) = (raw[i].x() - result.min_u) / du;
    result.UV(vid, 1) = (raw[i].y() - result.min_v) / dv;
  }

  result.valid = true;
  result.reason = "ok";
  return result;
}

static Vector3d pca_uv_to_plane_point(
    const RegionPCAUV& pca,
    double u,
    double v) {
  double x = pca.min_u + u * (pca.max_u - pca.min_u);
  double y = pca.min_v + v * (pca.max_v - pca.min_v);
  return pca.origin + x * pca.axis_u + y * pca.axis_v;
}

static SurfaceFitErrorStats compute_surface_sample_error_stats(
    const BSplineSurface3D& surface,
    const vector<SurfaceFitSample>& samples) {
  SurfaceFitErrorStats stats;
  double sum = 0.0;
  double sum_sq = 0.0;
  double max_err = 0.0;
  double weight_sum = 0.0;
  for (const SurfaceFitSample& s : samples) {
    if (s.weight <= 0.0) continue;
    double err = (surface.evaluate(s.uv.x(), s.uv.y()) - s.position).norm();
    sum += s.weight * err;
    sum_sq += s.weight * err * err;
    max_err = max(max_err, err);
    weight_sum += s.weight;
  }
  if (weight_sum > 0.0) {
    stats.mean_error = sum / weight_sum;
    stats.rms_error = sqrt(sum_sq / weight_sum);
    stats.max_error = max_err;
  }
  return stats;
}

static bool fit_full_bspline_surface_from_uv_samples(
    const RegionPCAUV& pca,
    const vector<SurfaceFitSample>& samples,
    int control_u,
    int control_v,
    double fairness_weight,
    double initial_weight,
    BSplineSurface3D& surface,
    SurfaceFitErrorStats& before,
    SurfaceFitErrorStats& after,
    string& reason) {
  int nu = max(4, control_u);
  int nv = max(4, control_v);
  int degree_u = 3;
  int degree_v = 3;
  vector<double> knots_u = make_open_uniform_knot_vector(nu, degree_u);
  vector<double> knots_v = make_open_uniform_knot_vector(nv, degree_v);

  vector<vector<Vector3d>> initial_grid(nu, vector<Vector3d>(nv));
  for (int i = 0; i < nu; i++) {
    double u = (double)i / (double)(nu - 1);
    for (int j = 0; j < nv; j++) {
      double v = (double)j / (double)(nv - 1);
      initial_grid[i][j] = pca_uv_to_plane_point(pca, u, v);
    }
  }
  BSplineSurface3D initial_surface(degree_u, degree_v, knots_u, knots_v, initial_grid);
  before = compute_surface_sample_error_stats(initial_surface, samples);

  int unknown_count = nu * nv;
  int fit_rows = 0;
  for (const SurfaceFitSample& s : samples) {
    if (s.weight > 0.0) fit_rows++;
  }
  int fair_rows = fairness_weight > 0.0
      ? (max(0, nu - 2) * nv + nu * max(0, nv - 2))
      : 0;
  int initial_rows = initial_weight > 0.0 ? unknown_count : 0;
  int row_count = fit_rows + fair_rows + initial_rows;
  if (fit_rows < unknown_count / 2) {
    reason = "too few fit samples for trimmed B-spline surface";
    return false;
  }

  MatrixXd A = MatrixXd::Zero(row_count, unknown_count);
  MatrixXd B = MatrixXd::Zero(row_count, 3);
  auto idx = [nv](int i, int j) { return i * nv + j; };
  int row = 0;

  for (const SurfaceFitSample& s : samples) {
    if (s.weight <= 0.0) continue;
    double scale = sqrt(s.weight);
    for (int i = 0; i < nu; i++) {
      double bu = bspline_basis(i, degree_u, s.uv.x(), knots_u);
      if (bu == 0.0) continue;
      for (int j = 0; j < nv; j++) {
        double bv = bspline_basis(j, degree_v, s.uv.y(), knots_v);
        if (bv == 0.0) continue;
        A(row, idx(i, j)) = scale * bu * bv;
      }
    }
    B.row(row) = (scale * s.position).transpose();
    row++;
  }

  if (fairness_weight > 0.0) {
    double scale = sqrt(fairness_weight);
    for (int i = 1; i < nu - 1; i++) {
      for (int j = 0; j < nv; j++) {
        A(row, idx(i - 1, j)) = scale;
        A(row, idx(i, j)) = -2.0 * scale;
        A(row, idx(i + 1, j)) = scale;
        row++;
      }
    }
    for (int i = 0; i < nu; i++) {
      for (int j = 1; j < nv - 1; j++) {
        A(row, idx(i, j - 1)) = scale;
        A(row, idx(i, j)) = -2.0 * scale;
        A(row, idx(i, j + 1)) = scale;
        row++;
      }
    }
  }

  if (initial_weight > 0.0) {
    double scale = sqrt(initial_weight);
    for (int i = 0; i < nu; i++) {
      for (int j = 0; j < nv; j++) {
        A(row, idx(i, j)) = scale;
        B.row(row) = (scale * initial_grid[i][j]).transpose();
        row++;
      }
    }
  }

  MatrixXd X = A.colPivHouseholderQr().solve(B);
  if (X.rows() != unknown_count || X.cols() != 3) {
    reason = "trimmed surface least-squares solve failed";
    return false;
  }

  vector<vector<Vector3d>> fitted_grid(nu, vector<Vector3d>(nv));
  for (int i = 0; i < nu; i++) {
    for (int j = 0; j < nv; j++) {
      fitted_grid[i][j] = X.row(idx(i, j)).transpose();
    }
  }
  surface = BSplineSurface3D(degree_u, degree_v, knots_u, knots_v, fitted_grid);
  after = compute_surface_sample_error_stats(surface, samples);
  reason = "ok";
  return true;
}

static Vector3d region_average_normal(
    const MatrixXd& meshV,
    const MatrixXi& meshF,
    const vector<int>& face_region_ids,
    int region_id) {
  Vector3d normal = Vector3d::Zero();
  for (int fi = 0; fi < meshF.rows(); fi++) {
    if (face_region_ids[fi] != region_id) continue;
    Vector3d p0 = meshV.row(meshF(fi, 0)).transpose();
    Vector3d p1 = meshV.row(meshF(fi, 1)).transpose();
    Vector3d p2 = meshV.row(meshF(fi, 2)).transpose();
    normal += (p1 - p0).cross(p2 - p0);
  }
  if (normal.norm() > 1e-12) normal.normalize();
  return normal;
}

static Vector3d mesh_average_normal(
    const MatrixXd& meshV,
    const MatrixXi& meshF) {
  Vector3d normal = Vector3d::Zero();
  for (int fi = 0; fi < meshF.rows(); fi++) {
    Vector3d p0 = meshV.row(meshF(fi, 0)).transpose();
    Vector3d p1 = meshV.row(meshF(fi, 1)).transpose();
    Vector3d p2 = meshV.row(meshF(fi, 2)).transpose();
    normal += (p1 - p0).cross(p2 - p0);
  }
  if (normal.norm() > 1e-12) normal.normalize();
  return normal;
}

static void flip_triangle_winding(MatrixXi& faces) {
  for (int i = 0; i < faces.rows(); i++) {
    std::swap(faces(i, 1), faces(i, 2));
  }
}

static bool export_mesh_obj(
    const string& filename,
    const MatrixXd& meshV,
    const MatrixXi& meshF) {
  ofstream fout(filename);
  if (!fout.is_open()) return false;
  fout.precision(17);
  fout << "# sampled mesh\n";
  for (int i = 0; i < meshV.rows(); i++) {
    fout << "v " << meshV(i, 0) << " "
         << meshV(i, 1) << " "
         << meshV(i, 2) << "\n";
  }
  for (int i = 0; i < meshF.rows(); i++) {
    fout << "f " << (meshF(i, 0) + 1) << " "
         << (meshF(i, 1) + 1) << " "
         << (meshF(i, 2) + 1) << "\n";
  }
  return true;
}

static bool export_trimmed_fit_report_csv(
    const string& filename,
    int region_id,
    const SurfaceFitErrorStats& before,
    const SurfaceFitErrorStats& after,
    int control_u,
    int control_v,
    int sample_count,
    int trim_vertices,
    int output_vertices,
    int output_faces) {
  ofstream fout(filename);
  if (!fout.is_open()) return false;
  fout << "region_id," << region_id << "\n";
  fout << "control_u," << control_u << "\n";
  fout << "control_v," << control_v << "\n";
  fout << "samples," << sample_count << "\n";
  fout << "trim_vertices," << trim_vertices << "\n";
  fout << "output_vertices," << output_vertices << "\n";
  fout << "output_faces," << output_faces << "\n";
  fout << "before_mean," << before.mean_error << "\n";
  fout << "before_rms," << before.rms_error << "\n";
  fout << "before_max," << before.max_error << "\n";
  fout << "after_mean," << after.mean_error << "\n";
  fout << "after_rms," << after.rms_error << "\n";
  fout << "after_max," << after.max_error << "\n";
  return true;
}

static bool fit_and_show_trimmed_bspline_surface_for_region(
    igl::opengl::glfw::Viewer& viewer,
    int region_id) {
  g_viewer_work_mode = ViewerWorkMode::SplineSurface;
  if (F.rows() <= 0 || R.rows() != F.rows()) {
    cout << "[trimmed spline] Cannot fit: current segmentation is invalid." << endl;
    return false;
  }
  if (region_id < 0 || region_id >= p) {
    cout << "[trimmed spline] Cannot fit: region " << region_id
         << " outside [0," << (p - 1) << "]." << endl;
    return false;
  }

  vector<int> face_region_ids(F.rows(), -1);
  for (int i = 0; i < F.rows(); i++) face_region_ids[i] = R(i, 0);

  string prefix = "interactive_trimmed_bspline_region_" + to_string(region_id);
  g_last_spline_debug_prefix = prefix;

  RegionBoundaryExtractionResult loop_result =
      extract_region_boundary_loop(V, F, face_region_ids, region_id);
  export_region_boundary_debug_obj(
      prefix + "_boundary.obj", V, F, face_region_ids, region_id, loop_result);
  if (!loop_result.success) {
    cout << "[trimmed spline] Boundary extraction failed for region "
         << region_id << ": " << loop_result.reason << endl;
    return false;
  }

  vector<int> region_vertex_ids =
      collect_region_vertex_ids(F, face_region_ids, region_id);
  RegionPCAUV pca = compute_region_pca_uv(V, region_vertex_ids, 0.04);
  if (!pca.valid) {
    cout << "[trimmed spline] PCA UV failed for region "
         << region_id << ": " << pca.reason << endl;
    return false;
  }

  set<int> boundary_ids(loop_result.loop.vertex_ids.begin(),
                        loop_result.loop.vertex_ids.end());
  vector<SurfaceFitSample> samples;
  samples.reserve(region_vertex_ids.size());
  for (int vid : region_vertex_ids) {
    if (!std::isfinite(pca.UV(vid, 0)) || !std::isfinite(pca.UV(vid, 1))) continue;
    SurfaceFitSample s;
    s.uv = pca.UV.row(vid).transpose();
    s.position = V.row(vid).transpose();
    s.weight = boundary_ids.count(vid) ? 8.0 : 1.0;
    samples.push_back(s);
  }
  if (samples.empty()) {
    cout << "[trimmed spline] No valid fit samples for region "
         << region_id << "." << endl;
    return false;
  }

  vector<Vector2d> trim_loop;
  trim_loop.reserve(loop_result.loop.vertex_ids.size());
  for (int vid : loop_result.loop.vertex_ids) {
    if (vid < 0 || vid >= pca.UV.rows()) continue;
    if (!std::isfinite(pca.UV(vid, 0)) || !std::isfinite(pca.UV(vid, 1))) continue;
    trim_loop.push_back(pca.UV.row(vid).transpose());
  }
  if (trim_loop.size() < 3) {
    cout << "[trimmed spline] Trim loop has too few valid UV vertices." << endl;
    return false;
  }

  BSplineSurface3D fitted_surface;
  SurfaceFitErrorStats before;
  SurfaceFitErrorStats after;
  string reason;
  bool fit_ok = fit_full_bspline_surface_from_uv_samples(
      pca,
      samples,
      max(4, g_spline_control_count_u),
      max(4, g_spline_control_count_v),
      max(0.0, g_spline_fit_fairness_weight),
      max(0.0, g_spline_fit_initial_weight),
      fitted_surface,
      before,
      after,
      reason);
  if (!fit_ok) {
    cout << "[trimmed spline] Full surface fit failed for region "
         << region_id << ": " << reason << endl;
    return false;
  }

  TrimmedBSplineSurfacePatch patch;
  patch.surface = fitted_surface;
  patch.outer_trim_polyline = trim_loop;
  int trim_control_count = min(48, max(8, (int)trim_loop.size() / 6));
  patch.outer_trim_curve = fit_trim_curve_2d_from_polyline(
      trim_loop,
      trim_control_count,
      max(0.0, g_spline_fit_fairness_weight));
  patch.valid = true;
  patch.reason = "ok";

  MatrixXd trimUV;
  bool sample_ok = sample_trimmed_bspline_surface(
      patch,
      max(6, g_trimmed_spline_grid_u),
      max(6, g_trimmed_spline_grid_v),
      splineSurfaceV,
      splineSurfaceF,
      &trimUV);
  if (!sample_ok) {
    cout << "[trimmed spline] Trimmed surface sampling failed." << endl;
    return false;
  }

  Vector3d src_normal = region_average_normal(V, F, face_region_ids, region_id);
  Vector3d out_normal = mesh_average_normal(splineSurfaceV, splineSurfaceF);
  bool output_orientation_flipped = false;
  if (src_normal.norm() > 1e-12 &&
      out_normal.norm() > 1e-12 &&
      src_normal.dot(out_normal) < 0.0) {
    flip_triangle_winding(splineSurfaceF);
    output_orientation_flipped = true;
  }

  has_spline_surface_mesh = splineSurfaceV.rows() > 0 && splineSurfaceF.rows() > 0;
  g_has_trim_debug_asset = false;
  g_has_trim_debug_abc_preview = false;
  export_trimmed_bspline_surface_debug(
      prefix,
      patch,
      max(6, g_trimmed_spline_grid_u),
      max(6, g_trimmed_spline_grid_v));
  export_mesh_obj(prefix + "_sampled_surface.obj", splineSurfaceV, splineSurfaceF);
  export_bspline_surface_control_net_obj(prefix + "_control_net.obj", fitted_surface);
  export_trimmed_fit_report_csv(
      prefix + "_report.csv",
      region_id,
      before,
      after,
      max(4, g_spline_control_count_u),
      max(4, g_spline_control_count_v),
      (int)samples.size(),
      (int)trim_loop.size(),
      (int)splineSurfaceV.rows(),
      (int)splineSurfaceF.rows());

  viewer.data().clear();
  viewer.data().set_mesh(splineSurfaceV, splineSurfaceF);
  MatrixXd colors(splineSurfaceF.rows(), 3);
  for (int i = 0; i < colors.rows(); i++) {
    colors.row(i) = RowVector3d(0.95, 0.62, 0.22);
  }
  viewer.data().set_colors(colors);
  viewer.data().show_lines = true;
  viewer.core().align_camera_center(splineSurfaceV, splineSurfaceF);

  cout << "[trimmed spline] Region " << region_id
       << " fitted trimmed B-spline surface"
       << " controls=" << max(4, g_spline_control_count_u)
       << "x" << max(4, g_spline_control_count_v)
       << " samples=" << samples.size()
       << " trim_vertices=" << trim_loop.size()
       << " mesh=" << splineSurfaceV.rows()
       << "V/" << splineSurfaceF.rows() << "F" << endl;
  cout << "  before mean/rms/max="
       << before.mean_error << " / "
       << before.rms_error << " / "
       << before.max_error << endl;
  cout << "  after  mean/rms/max="
       << after.mean_error << " / "
       << after.rms_error << " / "
       << after.max_error << endl;
  cout << "  output_orientation_flipped="
       << (output_orientation_flipped ? "true" : "false") << endl;
  cout << "  trim_curve_fit="
       << (patch.outer_trim_curve.valid ? "ok" : patch.outer_trim_curve.reason)
       << endl;
  cout << "  debug prefix: " << prefix << endl;

  if (g_spline_render_backend == SplineRenderBackend::Polyscope) {
    launch_external_polyscope_spline_viewer(g_last_spline_debug_prefix);
  }
  return true;
}

static bool run_and_show_trimmed_bspline_pipeline_for_region(
    igl::opengl::glfw::Viewer& viewer,
    int region_id,
    bool fast_preview = false) {
  g_viewer_work_mode = ViewerWorkMode::SplineSurface;
  if (F.rows() <= 0 || R.rows() != F.rows()) {
    set_spline_status("[standard trimmed pipeline] Cannot run: current segmentation is invalid.");
    return false;
  }
  if (region_id < 0 || region_id >= p) {
    set_spline_status("[standard trimmed pipeline] Cannot run: region " +
                      to_string(region_id) + " outside [0," +
                      to_string(p - 1) + "].");
    return false;
  }

  vector<int> face_region_ids(F.rows(), -1);
  for (int i = 0; i < F.rows(); i++) face_region_ids[i] = R(i, 0);

  string base_dir(g_trimmed_pipeline_output_dir);
  if (base_dir.empty()) base_dir = "trimmed_bspline_output";
  string output_dir = base_dir + (fast_preview ? "_fast_region_" : "_region_") +
                      to_string(region_id);
  set_spline_status("[standard trimmed pipeline] Running " +
                    string(fast_preview ? "fast preview" : "full") +
                    " for region " + to_string(region_id) +
                    " -> " + output_dir);

  TrimmedBSplinePipelineConfig cfg;
  cfg.output_dir = output_dir;
  cfg.control_count_u = max(4, fast_preview ? min(g_spline_control_count_u, 6) : g_spline_control_count_u);
  cfg.control_count_v = max(4, fast_preview ? min(g_spline_control_count_v, 6) : g_spline_control_count_v);
  cfg.surface_sample_u = max(8, fast_preview ? min(g_trimmed_spline_grid_u, 32) : g_trimmed_spline_grid_u);
  cfg.surface_sample_v = max(8, fast_preview ? min(g_trimmed_spline_grid_v, 32) : g_trimmed_spline_grid_v);
  cfg.extension_sample_u = max(6, fast_preview ? min(g_spline_sample_u, 12) : g_spline_sample_u);
  cfg.extension_sample_v = max(6, fast_preview ? min(g_spline_sample_v, 12) : g_spline_sample_v);
  cfg.fitting_regularization = max(0.0, g_spline_fit_fairness_weight);
  cfg.fitting_initial_weight = max(0.0, g_spline_fit_initial_weight);
  cfg.boundary_fit_weight = max(0.0, g_trimmed_boundary_fit_weight);
  cfg.apply_harmonic_boundary_correction = g_trimmed_harmonic_boundary_correction;
  cfg.snap_output_boundary_to_authoritative = g_trimmed_snap_output_boundary;
  cfg.enable_smoothed_arap = !fast_preview;
  cfg.enable_extension_fairness = !fast_preview;
  cfg.export_debug_artifacts = !fast_preview;
  cfg.run_ablation_baselines = !fast_preview;
  cfg.estimate_condition_number = !fast_preview;

  TrimmedBSplinePipelineResult result =
      run_single_region_trimmed_bspline_pipeline(
          V, F, face_region_ids, region_id, cfg);
  if (!fast_preview) {
    write_trimmed_bspline_reproduction_doc("TRIMMED_BSPLINE_REPRODUCTION.md");
  }
  if (!result.valid) {
    set_spline_status("[standard trimmed pipeline] Failed for region " +
                      to_string(region_id) + ": " + result.reason +
                      ". See " + output_dir + "/pipeline.log");
    return false;
  }

  string mesh_file = output_dir + "/bspline_trimmed_surface.obj";
  if (!igl::readOBJ(mesh_file, splineSurfaceV, splineSurfaceF) ||
      splineSurfaceV.rows() <= 0 || splineSurfaceF.rows() <= 0) {
    set_spline_status("[standard trimmed pipeline] Pipeline succeeded but cannot read " + mesh_file);
    return false;
  }
  MeshValidationReport viewer_mesh_validation =
      validate_trimmed_mesh(splineSurfaceV, splineSurfaceF, nullptr);
  g_has_trim_debug_original =
      build_region_mesh_for_display(region_id, g_trim_debug_originalV, g_trim_debug_originalF);
  g_has_trim_debug_full_spline =
      igl::readOBJ(output_dir + "/bspline_full_surface.obj",
                   g_trim_debug_fullSplineV,
                   g_trim_debug_fullSplineF) &&
      g_trim_debug_fullSplineV.rows() > 0 &&
      g_trim_debug_fullSplineF.rows() > 0;
  g_has_trim_debug_asset =
      igl::readOBJ(output_dir + "/bspline_trimmed_surface_asset.obj",
                   g_trim_debug_assetV,
                   g_trim_debug_assetF) &&
      g_trim_debug_assetV.rows() > 0 &&
      g_trim_debug_assetF.rows() > 0;
  g_has_trim_debug_abc_preview =
      igl::readOBJ(output_dir + "/bspline_trimmed_surface_abc_preview.obj",
                   g_trim_debug_abcPreviewV,
                   g_trim_debug_abcPreviewF) &&
      g_trim_debug_abcPreviewV.rows() > 0 &&
      g_trim_debug_abcPreviewF.rows() > 0;
  g_has_trim_debug_abc_ribbon =
      igl::readOBJ(output_dir + "/abc_boundary_ribbon_surfaces.obj",
                   g_trim_debug_abcRibbonV,
                   g_trim_debug_abcRibbonF) &&
      g_trim_debug_abcRibbonV.rows() > 0 &&
      g_trim_debug_abcRibbonF.rows() > 0;

  has_spline_surface_mesh = true;
  has_last_spline_patch = false;
  g_last_spline_debug_prefix = output_dir;
  g_trim_render_debug_mode = TrimRenderDebugMode::TrimmedOnly;

  if (!apply_trim_render_debug_mode(viewer)) {
    set_spline_status("[standard trimmed pipeline] Pipeline succeeded but render mode failed.");
    return false;
  }

  const TrimmedBSplinePipelineMetrics& m = result.metrics;
  cout << "[standard trimmed pipeline] Region " << region_id
       << " rendered standard trimmed B-spline mesh "
       << splineSurfaceV.rows() << "V/" << splineSurfaceF.rows() << "F" << endl;
  cout << "  viewer mesh validation: duplicate="
       << viewer_mesh_validation.exact_duplicate_faces
       << " geometric_duplicate=" << viewer_mesh_validation.geometric_duplicate_faces
       << " degenerate=" << viewer_mesh_validation.degenerate_faces
       << " bad_winding=" << viewer_mesh_validation.inconsistent_winding_edges
       << " normal_jump_edges=" << viewer_mesh_validation.normal_jump_edges
       << " nonmanifold=" << viewer_mesh_validation.nonmanifold_edges
       << " min_quality=" << viewer_mesh_validation.min_quality << endl;
  cout << "  main output=bspline_trimmed_surface.obj; ABC files are experimental debug previews only" << endl;
  cout << "  labeling=" << m.labeling_configuration
       << " ambiguous=" << (m.ambiguous ? "true" : "false") << endl;
  cout << "  flips=" << m.flipped_triangle_count
       << " arap mean/max=" << m.mean_arap_distortion
       << " / " << m.max_arap_distortion << endl;
  cout << "  fit original rms/max=" << m.original_region_rms_error
       << " / " << m.original_region_max_error
       << " boundary rms/max=" << m.boundary_rms_error
       << " / " << m.boundary_max_error << endl;
  cout << "  weak_controls=" << m.weak_control_point_count
       << " condition=" << m.linear_system_condition_estimate
       << " area_growth=" << m.surface_area_growth_ratio
       << " bbox_growth=" << m.bounding_box_growth_ratio << endl;
  cout << "  outputs: " << output_dir << endl;
  set_spline_status("[standard trimmed pipeline] Done region " + to_string(region_id) +
                    ": " + to_string((int)splineSurfaceV.rows()) + "V/" +
                    to_string((int)splineSurfaceF.rows()) +
                    "F, rendered StandardTrimmedOnly -> " + output_dir);
  return true;
}

static int current_region_count() {
  int max_label = -1;
  for (int i = 0; i < R.rows(); i++) {
    max_label = max(max_label, R(i, 0));
  }
  return max(p, max_label + 1);
}

static TrimmedBSplinePipelineConfig make_viewer_trimmed_pipeline_config(
    const string& output_dir,
    bool fast_preview) {
  TrimmedBSplinePipelineConfig cfg;
  cfg.output_dir = output_dir;
  cfg.control_count_u =
      max(4, fast_preview ? min(g_spline_control_count_u, 6) : g_spline_control_count_u);
  cfg.control_count_v =
      max(4, fast_preview ? min(g_spline_control_count_v, 6) : g_spline_control_count_v);
  cfg.surface_sample_u =
      max(8, fast_preview ? min(g_trimmed_spline_grid_u, 32) : g_trimmed_spline_grid_u);
  cfg.surface_sample_v =
      max(8, fast_preview ? min(g_trimmed_spline_grid_v, 32) : g_trimmed_spline_grid_v);
  cfg.extension_sample_u =
      max(6, fast_preview ? min(g_spline_sample_u, 12) : g_spline_sample_u);
  cfg.extension_sample_v =
      max(6, fast_preview ? min(g_spline_sample_v, 12) : g_spline_sample_v);
  cfg.fitting_regularization = max(0.0, g_spline_fit_fairness_weight);
  cfg.fitting_initial_weight = max(0.0, g_spline_fit_initial_weight);
  cfg.boundary_fit_weight = max(0.0, g_trimmed_boundary_fit_weight);
  cfg.apply_harmonic_boundary_correction = g_trimmed_harmonic_boundary_correction;
  cfg.snap_output_boundary_to_authoritative = g_trimmed_snap_output_boundary;
  cfg.enable_smoothed_arap = !fast_preview;
  cfg.enable_extension_fairness = !fast_preview;
  cfg.export_debug_artifacts = !fast_preview;
  cfg.run_ablation_baselines = !fast_preview;
  cfg.estimate_condition_number = !fast_preview;
  cfg.print_progress_to_console = true;
  return cfg;
}

static void append_mesh(
    const MatrixXd& srcV,
    const MatrixXi& srcF,
    int region_id,
    vector<Vector3d>& dstV,
    vector<Vector3i>& dstF,
    vector<RowVector3d>& dstC) {
  RowVector3d color = region_color_det(region_id);
  int offset = (int)dstV.size();
  for (int i = 0; i < srcV.rows(); i++) {
    dstV.push_back(srcV.row(i).transpose());
  }
  for (int i = 0; i < srcF.rows(); i++) {
    dstF.push_back(Vector3i(
        srcF(i, 0) + offset,
        srcF(i, 1) + offset,
        srcF(i, 2) + offset));
    dstC.push_back(color);
  }
}

static void append_mesh_with_color(
    const MatrixXd& srcV,
    const MatrixXi& srcF,
    const RowVector3d& color,
    vector<Vector3d>& dstV,
    vector<Vector3i>& dstF,
    vector<RowVector3d>& dstC) {
  int offset = (int)dstV.size();
  for (int i = 0; i < srcV.rows(); i++) {
    dstV.push_back(srcV.row(i).transpose());
  }
  for (int i = 0; i < srcF.rows(); i++) {
    dstF.push_back(Vector3i(
        srcF(i, 0) + offset,
        srcF(i, 1) + offset,
        srcF(i, 2) + offset));
    dstC.push_back(color);
  }
}

static bool fit_region_pca_bspline_topology_mesh(
    int region_id,
    const vector<int>& face_region_ids,
    MatrixXd& outV,
    MatrixXi& outF,
    SurfaceFitErrorStats& before,
    SurfaceFitErrorStats& after,
    string& reason) {
  outV.resize(0, 3);
  outF.resize(0, 3);
  vector<int> region_vertex_ids =
      collect_region_vertex_ids(F, face_region_ids, region_id);
  if (region_vertex_ids.size() < 4) {
    reason = "too few vertices for PCA B-spline fallback";
    return false;
  }

  RegionPCAUV pca = compute_region_pca_uv(V, region_vertex_ids, 0.04);
  if (!pca.valid) {
    reason = "PCA UV failed: " + pca.reason;
    return false;
  }

  set<int> boundary_ids;
  RegionBoundaryExtractionResult loop_result =
      extract_region_boundary_loop(V, F, face_region_ids, region_id);
  if (loop_result.success) {
    boundary_ids.insert(
        loop_result.loop.vertex_ids.begin(),
        loop_result.loop.vertex_ids.end());
  }

  vector<SurfaceFitSample> samples;
  samples.reserve(region_vertex_ids.size());
  for (int vid : region_vertex_ids) {
    if (!std::isfinite(pca.UV(vid, 0)) ||
        !std::isfinite(pca.UV(vid, 1))) {
      continue;
    }
    SurfaceFitSample s;
    s.uv = pca.UV.row(vid).transpose();
    s.position = V.row(vid).transpose();
    s.weight = boundary_ids.count(vid) ? 4.0 : 1.0;
    samples.push_back(s);
  }
  if (samples.size() < 4) {
    reason = "too few PCA fit samples";
    return false;
  }

  BSplineSurface3D fitted_surface;
  bool fit_ok = fit_full_bspline_surface_from_uv_samples(
      pca,
      samples,
      max(4, min(g_spline_control_count_u, 6)),
      max(4, min(g_spline_control_count_v, 6)),
      max(1e-8, g_spline_fit_fairness_weight),
      max(1e-6, g_spline_fit_initial_weight),
      fitted_surface,
      before,
      after,
      reason);
  if (!fit_ok) return false;

  map<int, int> global_to_local;
  outV.resize((int)region_vertex_ids.size(), 3);
  for (int li = 0; li < (int)region_vertex_ids.size(); li++) {
    int gid = region_vertex_ids[li];
    global_to_local[gid] = li;
    Vector2d uv = pca.UV.row(gid).transpose();
    outV.row(li) = fitted_surface.evaluate(uv.x(), uv.y()).transpose();
  }

  vector<Vector3i> faces;
  for (int fi = 0; fi < F.rows(); fi++) {
    if (face_region_ids[fi] != region_id) continue;
    auto ia = global_to_local.find(F(fi, 0));
    auto ib = global_to_local.find(F(fi, 1));
    auto ic = global_to_local.find(F(fi, 2));
    if (ia == global_to_local.end() ||
        ib == global_to_local.end() ||
        ic == global_to_local.end()) {
      continue;
    }
    faces.push_back(Vector3i(ia->second, ib->second, ic->second));
  }
  if (faces.empty()) {
    reason = "no local faces for PCA B-spline fallback";
    return false;
  }
  outF.resize((int)faces.size(), 3);
  for (int i = 0; i < (int)faces.size(); i++) outF.row(i) = faces[i].transpose();

  Vector3d src_normal = region_average_normal(V, F, face_region_ids, region_id);
  Vector3d out_normal = mesh_average_normal(outV, outF);
  if (src_normal.norm() > 1e-12 &&
      out_normal.norm() > 1e-12 &&
      src_normal.dot(out_normal) < 0.0) {
    flip_triangle_winding(outF);
  }
  reason = "ok";
  return true;
}

static bool append_original_region_fallback_mesh(
    int region_id,
    vector<Vector3d>& dstV,
    vector<Vector3i>& dstF,
    vector<RowVector3d>& dstC) {
  MatrixXd regionV;
  MatrixXi regionF;
  if (!build_region_mesh_for_display(region_id, regionV, regionF) ||
      regionV.rows() <= 0 || regionF.rows() <= 0) {
    return false;
  }
  int offset = (int)dstV.size();
  for (int i = 0; i < regionV.rows(); i++) {
    dstV.push_back(regionV.row(i).transpose());
  }
  RowVector3d fallback_color(0.95, 0.18, 0.12);
  for (int i = 0; i < regionF.rows(); i++) {
    dstF.push_back(Vector3i(
        regionF(i, 0) + offset,
        regionF(i, 1) + offset,
        regionF(i, 2) + offset));
    dstC.push_back(fallback_color);
  }
  return true;
}

static bool run_and_show_all_trimmed_bspline_regions(
    igl::opengl::glfw::Viewer& viewer,
    bool fast_preview = true) {
  g_viewer_work_mode = ViewerWorkMode::SplineSurface;
  if (F.rows() <= 0 || R.rows() != F.rows()) {
    set_spline_status("[all trimmed pipeline] Cannot run: current segmentation is invalid.");
    return false;
  }

  vector<int> face_region_ids(F.rows(), -1);
  for (int i = 0; i < F.rows(); i++) face_region_ids[i] = R(i, 0);

  string base_dir(g_trimmed_pipeline_output_dir);
  if (base_dir.empty()) base_dir = "trimmed_bspline_output";
  string output_root = base_dir + (fast_preview ? "_all_fast" : "_all");
  int region_count = current_region_count();

  set_spline_status("[all trimmed pipeline] Running " +
                    string(fast_preview ? "fast preview" : "full") +
                    " for " + to_string(region_count) +
                    " regions -> " + output_root);

  vector<Vector3d> merged_vertices;
  vector<Vector3i> merged_faces;
  vector<RowVector3d> merged_colors;
  vector<int> succeeded;
  vector<int> pca_bspline_fallbacks;
  vector<int> original_fallbacks;
  vector<pair<int, string>> standard_pipeline_failures;
  vector<pair<int, string>> hard_failures;

  ofstream summary(output_root + "_summary.csv");
  if (summary.is_open()) {
    summary << "region,status,vertices,faces,original_rms,original_max,boundary_rms,boundary_max,reason\n";
  }

  for (int region_id = 0; region_id < region_count; region_id++) {
    bool has_region_face = false;
    for (int fi = 0; fi < F.rows(); fi++) {
      if (face_region_ids[fi] == region_id) {
        has_region_face = true;
        break;
      }
    }
    if (!has_region_face) continue;

    string output_dir = output_root + "_region_" + to_string(region_id);
    cout << "[all trimmed pipeline] region " << region_id << " start -> "
         << output_dir << endl;
    TrimmedBSplinePipelineConfig cfg =
        make_viewer_trimmed_pipeline_config(output_dir, fast_preview);
    TrimmedBSplinePipelineResult result =
        run_single_region_trimmed_bspline_pipeline(V, F, face_region_ids, region_id, cfg);
    if (!result.valid) {
      standard_pipeline_failures.push_back({region_id, result.reason});
      MatrixXd fallbackV;
      MatrixXi fallbackF;
      SurfaceFitErrorStats fallback_before;
      SurfaceFitErrorStats fallback_after;
      string fallback_reason;
      bool pca_fallback_ok =
          fit_region_pca_bspline_topology_mesh(
              region_id,
              face_region_ids,
              fallbackV,
              fallbackF,
              fallback_before,
              fallback_after,
              fallback_reason);
      if (pca_fallback_ok) {
        append_mesh_with_color(
            fallbackV,
            fallbackF,
            RowVector3d(0.95, 0.68, 0.16),
            merged_vertices,
            merged_faces,
            merged_colors);
        pca_bspline_fallbacks.push_back(region_id);
        string fallback_file =
            output_root + "_pca_bspline_fallback_region_" + to_string(region_id) + ".obj";
        export_mesh_obj(fallback_file, fallbackV, fallbackF);
        cout << "[all trimmed pipeline] region " << region_id
             << " paper pipeline failed: " << result.reason
             << " (rendering PCA B-spline fallback, rms="
             << fallback_after.rms_error << ")" << endl;
        if (summary.is_open()) {
          summary << region_id << ",fallback_pca_bspline,"
                  << fallbackV.rows() << "," << fallbackF.rows() << ","
                  << fallback_after.rms_error << ","
                  << fallback_after.max_error << ",0,0,\""
                  << result.reason << "\"\n";
        }
      } else {
        bool original_ok =
            append_original_region_fallback_mesh(
                region_id, merged_vertices, merged_faces, merged_colors);
        if (original_ok) original_fallbacks.push_back(region_id);
        else hard_failures.push_back(
            {region_id, result.reason + "; PCA fallback: " + fallback_reason});
        cout << "[all trimmed pipeline] region " << region_id
             << " failed: " << result.reason
             << "; PCA fallback failed: " << fallback_reason
             << (original_ok ? " (rendering original region fallback)" : "")
             << endl;
        if (summary.is_open()) {
          summary << region_id
                  << (original_ok ? ",fallback_original," : ",failed,")
                  << "0,0,0,0,0,0,\""
                  << result.reason << "; PCA fallback: "
                  << fallback_reason << "\"\n";
        }
      }
      continue;
    }

    MatrixXd regionV;
    MatrixXi regionF;
    string mesh_file = output_dir + "/bspline_trimmed_surface.obj";
    if (!igl::readOBJ(mesh_file, regionV, regionF) ||
        regionV.rows() <= 0 || regionF.rows() <= 0) {
      string reason = "cannot read " + mesh_file;
      standard_pipeline_failures.push_back({region_id, reason});
      MatrixXd fallbackV;
      MatrixXi fallbackF;
      SurfaceFitErrorStats fallback_before;
      SurfaceFitErrorStats fallback_after;
      string fallback_reason;
      bool pca_fallback_ok =
          fit_region_pca_bspline_topology_mesh(
              region_id,
              face_region_ids,
              fallbackV,
              fallbackF,
              fallback_before,
              fallback_after,
              fallback_reason);
      if (pca_fallback_ok) {
        append_mesh_with_color(
            fallbackV,
            fallbackF,
            RowVector3d(0.95, 0.68, 0.16),
            merged_vertices,
            merged_faces,
            merged_colors);
        pca_bspline_fallbacks.push_back(region_id);
        export_mesh_obj(
            output_root + "_pca_bspline_fallback_region_" + to_string(region_id) + ".obj",
            fallbackV,
            fallbackF);
        cout << "[all trimmed pipeline] region " << region_id
             << " mesh read failed: " << reason
             << " (rendering PCA B-spline fallback, rms="
             << fallback_after.rms_error << ")" << endl;
        if (summary.is_open()) {
          summary << region_id << ",fallback_pca_bspline,"
                  << fallbackV.rows() << "," << fallbackF.rows() << ","
                  << fallback_after.rms_error << ","
                  << fallback_after.max_error << ",0,0,\""
                  << reason << "\"\n";
        }
      } else {
        bool original_ok =
            append_original_region_fallback_mesh(
                region_id, merged_vertices, merged_faces, merged_colors);
        if (original_ok) original_fallbacks.push_back(region_id);
        else hard_failures.push_back(
            {region_id, reason + "; PCA fallback: " + fallback_reason});
        cout << "[all trimmed pipeline] region " << region_id
             << " failed: " << reason
             << "; PCA fallback failed: " << fallback_reason
             << (original_ok ? " (rendering original region fallback)" : "")
             << endl;
        if (summary.is_open()) {
          summary << region_id
                  << (original_ok ? ",fallback_original," : ",failed,")
                  << "0,0,0,0,0,0,\""
                  << reason << "; PCA fallback: "
                  << fallback_reason << "\"\n";
        }
      }
      continue;
    }

    MeshValidationReport validation =
        validate_trimmed_mesh(regionV, regionF, nullptr);
    append_mesh(regionV, regionF, region_id, merged_vertices, merged_faces, merged_colors);
    succeeded.push_back(region_id);
    const TrimmedBSplinePipelineMetrics& m = result.metrics;
    cout << "[all trimmed pipeline] region " << region_id
         << " done: " << regionV.rows() << "V/" << regionF.rows()
         << "F, normal_jump_edges=" << validation.normal_jump_edges
         << ", original_rms=" << m.original_region_rms_error << endl;
    if (summary.is_open()) {
      summary << region_id << ",ok,"
              << regionV.rows() << "," << regionF.rows() << ","
              << m.original_region_rms_error << ","
              << m.original_region_max_error << ","
              << m.boundary_rms_error << ","
              << m.boundary_max_error << ",ok\n";
    }
  }

  if (merged_vertices.empty() || merged_faces.empty()) {
    set_spline_status("[all trimmed pipeline] No region succeeded. See console and " +
                      output_root + "_summary.csv");
    return false;
  }

  splineSurfaceV.resize((int)merged_vertices.size(), 3);
  for (int i = 0; i < (int)merged_vertices.size(); i++) {
    splineSurfaceV.row(i) = merged_vertices[i].transpose();
  }
  splineSurfaceF.resize((int)merged_faces.size(), 3);
  MatrixXd face_colors((int)merged_faces.size(), 3);
  for (int i = 0; i < (int)merged_faces.size(); i++) {
    splineSurfaceF.row(i) = merged_faces[i].transpose();
    face_colors.row(i) = merged_colors[i];
  }

  export_mesh_obj(output_root + "_all_regions_trimmed_bspline.obj",
                  splineSurfaceV,
                  splineSurfaceF);
  {
    ofstream out(output_root + "_failed_regions.txt");
    if (out.is_open()) {
      out << "Succeeded trimmed B-spline regions:\n";
      for (int region_id : succeeded) out << region_id << "\n";
      out << "\nPCA B-spline fallback regions:\n";
      for (int region_id : pca_bspline_fallbacks) out << region_id << "\n";
      out << "\nOriginal fallback regions:\n";
      for (int region_id : original_fallbacks) out << region_id << "\n";
      out << "\nStandard trimmed pipeline failures:\n";
      for (const auto& item : standard_pipeline_failures) {
        out << item.first << ": " << item.second << "\n";
      }
      out << "\nHard failures without visible fallback:\n";
      for (const auto& item : hard_failures) {
        out << item.first << ": " << item.second << "\n";
      }
    }
  }

  has_spline_surface_mesh = true;
  has_last_spline_patch = false;
  g_last_spline_debug_prefix = output_root;
  g_trim_render_debug_mode = TrimRenderDebugMode::TrimmedOnly;
  g_has_trim_debug_original = false;
  g_has_trim_debug_full_spline = false;
  g_has_trim_debug_asset = false;
  g_has_trim_debug_abc_preview = false;
  g_has_trim_debug_abc_ribbon = false;

  clear_viewer_to_single_slot(viewer);
  viewer.data().set_mesh(splineSurfaceV, splineSurfaceF);
  viewer.data().set_colors(face_colors);
  viewer.data().show_faces = true;
  viewer.data().show_lines = false;
  viewer.data().show_overlay = false;
  viewer.core().align_camera_center(splineSurfaceV, splineSurfaceF);
  print_viewer_data_slots(viewer, "after all-region trimmed pipeline");

  string status = "[all trimmed pipeline] Done: " +
      to_string((int)succeeded.size()) + " standard succeeded, " +
      to_string((int)standard_pipeline_failures.size()) + " standard failed, " +
      to_string((int)pca_bspline_fallbacks.size()) + " PCA B-spline fallbacks, " +
      to_string((int)original_fallbacks.size()) + " original fallbacks, " +
      to_string((int)hard_failures.size()) + " hard failures, merged " +
      to_string((int)splineSurfaceV.rows()) + "V/" +
      to_string((int)splineSurfaceF.rows()) + "F -> " + output_root;
  set_spline_status(status);
  if (!pca_bspline_fallbacks.empty()) {
    cout << "[all trimmed pipeline] PCA B-spline fallback regions are rendered in amber:" << endl;
    for (int region_id : pca_bspline_fallbacks) {
      cout << "  region " << region_id << endl;
    }
  }
  if (!original_fallbacks.empty()) {
    cout << "[all trimmed pipeline] original fallback regions are rendered in red:" << endl;
    for (int region_id : original_fallbacks) {
      cout << "  region " << region_id << endl;
    }
  }
  if (!standard_pipeline_failures.empty()) {
    cout << "[all trimmed pipeline] standard pipeline failed regions:" << endl;
    for (const auto& item : standard_pipeline_failures) {
      cout << "  region " << item.first << ": " << item.second << endl;
    }
  }
  if (!hard_failures.empty()) {
    cout << "[all trimmed pipeline] hard failures without visible fallback:" << endl;
    for (const auto& item : hard_failures) {
      cout << "  region " << item.first << ": " << item.second << endl;
    }
  }
  return true;
}

static bool fit_and_show_bspline_surface_for_region(
    igl::opengl::glfw::Viewer& viewer,
    int region_id) {
  g_viewer_work_mode = ViewerWorkMode::SplineSurface;
  if (F.rows() <= 0 || R.rows() != F.rows()) {
    cout << "[spline fit] Cannot fit: current segmentation is invalid." << endl;
    return false;
  }
  if (region_id < 0 || region_id >= p) {
    cout << "[spline fit] Cannot fit: region " << region_id
         << " outside [0," << (p - 1) << "]." << endl;
    return false;
  }

  vector<int> face_region_ids(F.rows(), -1);
  for (int i = 0; i < F.rows(); i++) face_region_ids[i] = R(i, 0);

  string prefix = "interactive_bspline_fit_region_" + to_string(region_id);
  g_last_spline_debug_prefix = prefix;

  RegionBoundaryExtractionResult loop_result =
      extract_region_boundary_loop(V, F, face_region_ids, region_id);
  export_region_boundary_debug_obj(
      prefix + "_boundary.obj", V, F, face_region_ids, region_id, loop_result);
  if (!loop_result.success) {
    cout << "[spline fit] Boundary extraction failed for region " << region_id
         << ": " << loop_result.reason << endl;
    return false;
  }

  QuadLikeBoundaryConfig quad_cfg;
  QuadLikeBoundaryResult quad_result =
      split_quad_like_boundary(loop_result.loop, quad_cfg);
  export_quad_like_boundary_pca_debug_obj(
      prefix + "_quad_like_pca.obj", quad_result.debug);
  if (!quad_result.success || !quad_result.boundary.valid) {
    cout << "[spline fit] Quad-like split failed for region " << region_id
         << ": " << quad_result.reason << endl;
    return false;
  }

  InitialBSplineSurfaceConfig surface_cfg;
  surface_cfg.control_count_u = max(4, g_spline_control_count_u);
  surface_cfg.control_count_v = max(4, g_spline_control_count_v);
  InitialBSplineSurfacePatch initial_patch =
      build_initial_bspline_surface_from_quad_boundary(
          quad_result.boundary, surface_cfg);
  if (!initial_patch.valid) {
    cout << "[spline fit] Initial B-spline surface failed for region "
         << region_id << ": " << initial_patch.reason << endl;
    return false;
  }
  export_initial_bspline_surface_debug(
      prefix + "_initial",
      initial_patch,
      max(4, g_spline_sample_u),
      max(4, g_spline_sample_v));

  RegionSquareParameterizationConfig param_cfg;
  RegionSquareParameterizationResult param_result =
      parameterize_region_to_square(
          V, F, face_region_ids, region_id,
          loop_result.loop, quad_result.boundary, param_cfg);
  export_region_square_parameterization_debug(
      prefix + "_uv", V, F, param_result);
  if (!param_result.valid) {
    cout << "[spline fit] Square parameterization failed for region "
         << region_id << ": " << param_result.reason << endl;
    return false;
  }

  vector<SurfaceFitSample> samples =
      make_region_vertex_fit_samples(V, param_result);
  if (samples.empty()) {
    cout << "[spline fit] No valid region vertex samples for region "
         << region_id << "." << endl;
    return false;
  }

  SurfaceInteriorFitConfig fit_cfg;
  fit_cfg.fairness_weight = max(0.0, g_spline_fit_fairness_weight);
  fit_cfg.initial_weight = max(0.0, g_spline_fit_initial_weight);
  fit_cfg.enable_point_to_plane = false;
  fit_cfg.surface_sample_u = max(4, g_spline_sample_u);
  fit_cfg.surface_sample_v = max(4, g_spline_sample_v);

  SurfaceInteriorFitResult fit_result =
      fit_bspline_surface_interior_control_points(
          initial_patch, samples, fit_cfg);
  export_surface_interior_fit_debug(prefix, fit_result, samples);
  if (!fit_result.valid) {
    cout << "[spline fit] Interior control fit failed for region "
         << region_id << ": " << fit_result.reason << endl;
    cout << "  before mean/rms/max="
         << fit_result.before.mean_error << " / "
         << fit_result.before.rms_error << " / "
         << fit_result.before.max_error << endl;
    cout << "  after mean/rms/max="
         << fit_result.after.mean_error << " / "
         << fit_result.after.rms_error << " / "
         << fit_result.after.max_error << endl;
    return false;
  }

  sample_bspline_surface(
      fit_result.patch.surface,
      max(4, g_spline_sample_u),
      max(4, g_spline_sample_v),
      splineSurfaceV,
      splineSurfaceF);
  bool output_orientation_flipped = false;
  if (param_result.orientation_reflected) {
    flip_triangle_winding(splineSurfaceF);
    output_orientation_flipped = true;
  }
  export_mesh_obj(prefix + "_sampled_surface.obj", splineSurfaceV, splineSurfaceF);
  if (output_orientation_flipped) {
    export_mesh_obj(
        prefix + "_sampled_surface_oriented.obj",
        splineSurfaceV,
        splineSurfaceF);
  }
  g_last_spline_patch = fit_result.patch;
  has_last_spline_patch = true;
  has_spline_surface_mesh = splineSurfaceV.rows() > 0 && splineSurfaceF.rows() > 0;
  g_has_trim_debug_asset = false;
  g_has_trim_debug_abc_preview = false;

  viewer.data().clear();
  viewer.data().set_mesh(splineSurfaceV, splineSurfaceF);
  MatrixXd colors(splineSurfaceF.rows(), 3);
  for (int i = 0; i < colors.rows(); i++) {
    colors.row(i) = RowVector3d(0.18, 0.78, 0.58);
  }
  viewer.data().set_colors(colors);
  viewer.data().show_lines = true;
  viewer.core().align_camera_center(splineSurfaceV, splineSurfaceF);

  cout << "[spline fit] Region " << region_id
       << " fitted B-spline surface"
       << " controls=" << surface_cfg.control_count_u
       << "x" << surface_cfg.control_count_v
       << " samples=" << samples.size()
       << " mesh=" << splineSurfaceV.rows()
       << "V/" << splineSurfaceF.rows() << "F" << endl;
  cout << "  before mean/rms/max="
       << fit_result.before.mean_error << " / "
       << fit_result.before.rms_error << " / "
       << fit_result.before.max_error << endl;
  cout << "  after  mean/rms/max="
       << fit_result.after.mean_error << " / "
       << fit_result.after.rms_error << " / "
       << fit_result.after.max_error << endl;
  cout << "  uv distortion mean/max area="
       << param_result.mean_area_distortion << " / "
       << param_result.max_area_distortion
       << " angle=" << param_result.mean_angle_distortion
       << " / " << param_result.max_angle_distortion << endl;
  cout << "  output_orientation_flipped="
       << (output_orientation_flipped ? "true" : "false") << endl;
  cout << "  debug prefix: " << prefix << endl;

  if (g_spline_render_backend == SplineRenderBackend::Polyscope) {
    launch_external_polyscope_spline_viewer(g_last_spline_debug_prefix);
  }
  return true;
}

static void enter_spline_surface_mode(igl::opengl::glfw::Viewer& viewer) {
  g_viewer_work_mode = ViewerWorkMode::SplineSurface;
  cout << "[mode] Spline Surface. Selected region=" << g_spline_region_id << endl;
  compute_and_show_spline_surface_for_region(viewer, g_spline_region_id);
}

static const string kInteractiveCheckpointFile = "interactive_checkpoint.qvsa";

static string segmentation_preview_filename(const string& snapshot_filename) {
  size_t slash = snapshot_filename.find_last_of("/\\");
  size_t dot = snapshot_filename.find_last_of('.');
  if (dot == string::npos || (slash != string::npos && dot < slash)) {
    return snapshot_filename + "_segmentation.coff";
  }
  return snapshot_filename.substr(0, dot) + "_segmentation.coff";
}

static bool export_current_segmentation_coff(const string& filename) {
  if (V.rows() <= 0 || F.rows() <= 0 || R.rows() != F.rows()) {
    cout << "[segmentation] COFF export failed: invalid current mesh or labels." << endl;
    return false;
  }
  ofstream fout(filename);
  if (!fout.is_open()) {
    cout << "[segmentation] COFF export failed: cannot open " << filename << endl;
    return false;
  }

  fout << "COFF\n";
  fout << V.rows() << " " << F.rows() << " 0\n";
  for (int i = 0; i < V.rows(); i++) {
    fout << V(i, 0) << " " << V(i, 1) << " " << V(i, 2)
         << " 255 255 255 255\n";
  }
  for (int i = 0; i < F.rows(); i++) {
    RowVector3d c = region_color_det(R(i, 0));
    int cr = (int)max(0.0, min(255.0, 255.0 * c(0)));
    int cg = (int)max(0.0, min(255.0, 255.0 * c(1)));
    int cb = (int)max(0.0, min(255.0, 255.0 * c(2)));
    fout << "3 " << F(i, 0) << " " << F(i, 1) << " " << F(i, 2)
         << " " << cr << " " << cg << " " << cb << " 255\n";
  }
  cout << "[segmentation] exported colored COFF: " << filename << endl;
  return true;
}

static bool save_interactive_checkpoint(const string& filename) {
  if (F.rows() <= 0 || R.rows() != F.rows() || p <= 0) {
    cout << "[checkpoint] save failed: invalid current segmentation." << endl;
    return false;
  }

  if (!use_quadric || (int)QP.size() != p) {
    QP.resize(p);
    for (int j = 0; j < p; j++) {
      QP[j] = fit_quadric_region(R, j, F, V);
    }
    use_quadric = true;
  }

  ofstream fout(filename);
  if (!fout.is_open()) {
    cout << "[checkpoint] save failed: cannot open " << filename << endl;
    return false;
  }

  fout.precision(17);
  fout << "QVSA_CHECKPOINT 1\n";
  fout << "vertices " << V.rows() << "\n";
  fout << "faces " << F.rows() << "\n";
  fout << "regions " << p << "\n";
  fout << "use_quadric " << (use_quadric ? 1 : 0) << "\n";
  fout << "feature_barrier_enabled " << (g_feature_barrier_enabled ? 1 : 0) << "\n";
  fout << "feature_angle " << g_feature_angle_threshold << "\n";

  fout << "labels\n";
  for (int i = 0; i < R.rows(); i++) {
    fout << R(i, 0) << "\n";
  }

  fout << "quadric_proxies " << QP.size() << "\n";
  for (int j = 0; j < (int)QP.size(); j++) {
    fout << j;
    for (int k = 0; k < QP[j].coeffs.size(); k++) {
      fout << " " << QP[j].coeffs(k);
    }
    fout << "\n";
  }
  fout << "END\n";
  fout.close();

  cout << "[checkpoint] saved " << filename
       << " faces=" << R.rows()
       << " regions=" << p
       << " quadric_proxies=" << QP.size() << endl;
  return true;
}

static bool save_quadric_vsa_segmentation_result(const string& filename) {
  if (filename.empty()) {
    cout << "[segmentation] save failed: empty snapshot filename." << endl;
    return false;
  }
  bool ok = save_interactive_checkpoint(filename);
  if (ok) {
    export_current_segmentation_coff(segmentation_preview_filename(filename));
  }
  return ok;
}

static string region_snapshot_base_name(int region_id, const string& prefix) {
  return prefix + "_region_" + to_string(region_id);
}

static string region_snapshot_mesh_filename(int region_id, const string& prefix) {
  return region_snapshot_base_name(region_id, prefix) + "_mesh.obj";
}

static bool render_region_snapshot_mesh(
    igl::opengl::glfw::Viewer& viewer,
    int region_id,
    const string& prefix) {
  if (prefix.empty()) {
    cout << "[region snapshot] render failed: empty prefix." << endl;
    return false;
  }
  string mesh_filename = region_snapshot_mesh_filename(region_id, prefix);
  MatrixXd regionV;
  MatrixXi regionF;
  if (!igl::readOBJ(mesh_filename, regionV, regionF)) {
    cout << "[region snapshot] render failed: cannot read " << mesh_filename << endl;
    return false;
  }
  if (regionV.rows() <= 0 || regionF.rows() <= 0) {
    cout << "[region snapshot] render failed: empty mesh in "
         << mesh_filename << endl;
    return false;
  }

  viewer.data().clear();
  viewer.data().set_mesh(regionV, regionF);
  MatrixXd colors(regionF.rows(), 3);
  RowVector3d c = region_color_det(region_id);
  for (int i = 0; i < colors.rows(); i++) colors.row(i) = c;
  viewer.data().set_colors(colors);
  viewer.data().show_lines = true;
  viewer.data().show_custom_labels = false;
  viewer.data().clear_labels();

  Vector3d center = regionV.colwise().mean();
  RowVector3d bb_min = regionV.colwise().minCoeff();
  RowVector3d bb_max = regionV.colwise().maxCoeff();
  double offset = 0.02 * (bb_max - bb_min).norm();
  center.z() += offset;
  MatrixXd labelP(1, 3);
  labelP.row(0) = center.transpose();
  vector<string> labelS(1, "region " + to_string(region_id));
  viewer.data().set_labels(labelP, labelS);
  viewer.data().show_custom_labels = true;
  viewer.data().label_size = 1.35f;
  viewer.data().label_color << 0.02f, 0.02f, 0.02f, 1.0f;

  viewer.core().align_camera_center(regionV, regionF);
  cout << "[region snapshot] rendered " << mesh_filename
       << " verts=" << regionV.rows()
       << " faces=" << regionF.rows() << endl;
  return true;
}

static bool save_region_snapshot(int region_id, const string& prefix) {
  if (prefix.empty()) {
    cout << "[region snapshot] save failed: empty prefix." << endl;
    return false;
  }
  if (V.rows() <= 0 || F.rows() <= 0 || R.rows() != F.rows()) {
    cout << "[region snapshot] save failed: invalid current mesh or labels." << endl;
    return false;
  }
  if (region_id < 0 || region_id >= p) {
    cout << "[region snapshot] save failed: region " << region_id
         << " outside [0," << (p - 1) << "]." << endl;
    return false;
  }

  string base = region_snapshot_base_name(region_id, prefix);
  string mesh_filename = region_snapshot_mesh_filename(region_id, prefix);
  string metadata_filename = base + "_metadata.txt";
  string boundary_filename = base + "_boundary.obj";
  string boundary_loop_filename = base + "_boundary_loop.txt";

  vector<int> region_face_ids;
  map<int, int> vertex_to_local;
  vector<int> local_to_vertex;
  vector<Vector3i> local_faces;
  vector<Vector3i> original_faces;
  region_face_ids.reserve(F.rows());

  for (int fi = 0; fi < F.rows(); fi++) {
    if (R(fi, 0) != region_id) continue;
    region_face_ids.push_back(fi);
    Vector3i local_face;
    Vector3i original_face = F.row(fi);
    for (int k = 0; k < 3; k++) {
      int global_vid = original_face(k);
      if (global_vid < 0 || global_vid >= V.rows()) {
        cout << "[region snapshot] save failed: invalid vertex id "
             << global_vid << " in face " << fi << "." << endl;
        return false;
      }
      auto it = vertex_to_local.find(global_vid);
      if (it == vertex_to_local.end()) {
        int local_id = (int)local_to_vertex.size();
        vertex_to_local[global_vid] = local_id;
        local_to_vertex.push_back(global_vid);
        local_face(k) = local_id;
      } else {
        local_face(k) = it->second;
      }
    }
    local_faces.push_back(local_face);
    original_faces.push_back(original_face);
  }

  if (region_face_ids.empty()) {
    cout << "[region snapshot] save failed: region " << region_id
         << " has no faces." << endl;
    return false;
  }

  ofstream obj(mesh_filename);
  if (!obj.is_open()) {
    cout << "[region snapshot] save failed: cannot open " << mesh_filename << endl;
    return false;
  }
  obj.precision(17);
  obj << "# VSA region snapshot mesh\n";
  obj << "# region_id " << region_id << "\n";
  obj << "# source_vertices " << V.rows() << "\n";
  obj << "# source_faces " << F.rows() << "\n";
  for (int local_id = 0; local_id < (int)local_to_vertex.size(); local_id++) {
    int global_vid = local_to_vertex[local_id];
    obj << "v " << V(global_vid, 0) << " "
        << V(global_vid, 1) << " "
        << V(global_vid, 2) << "\n";
  }
  for (const Vector3i& f : local_faces) {
    obj << "f " << (f(0) + 1) << " "
        << (f(1) + 1) << " "
        << (f(2) + 1) << "\n";
  }
  obj.close();

  ofstream meta(metadata_filename);
  if (!meta.is_open()) {
    cout << "[region snapshot] save failed: cannot open " << metadata_filename << endl;
    return false;
  }
  meta.precision(17);
  meta << "VSA_REGION_SNAPSHOT 1\n";
  meta << "region_id " << region_id << "\n";
  meta << "source_vertices " << V.rows() << "\n";
  meta << "source_faces " << F.rows() << "\n";
  meta << "source_regions " << p << "\n";
  meta << "region_vertices " << local_to_vertex.size() << "\n";
  meta << "region_faces " << region_face_ids.size() << "\n";
  meta << "mesh_obj " << mesh_filename << "\n";
  meta << "boundary_obj " << boundary_filename << "\n";
  meta << "boundary_loop " << boundary_loop_filename << "\n";
  meta << "vertices\n";
  for (int local_id = 0; local_id < (int)local_to_vertex.size(); local_id++) {
    int global_vid = local_to_vertex[local_id];
    meta << local_id << " " << global_vid << " "
         << V(global_vid, 0) << " "
         << V(global_vid, 1) << " "
         << V(global_vid, 2) << "\n";
  }
  meta << "faces\n";
  for (int lf = 0; lf < (int)local_faces.size(); lf++) {
    meta << lf << " " << region_face_ids[lf] << " "
         << local_faces[lf](0) << " "
         << local_faces[lf](1) << " "
         << local_faces[lf](2) << " "
         << original_faces[lf](0) << " "
         << original_faces[lf](1) << " "
         << original_faces[lf](2) << "\n";
  }
  meta << "END\n";
  meta.close();

  vector<int> face_region_ids(F.rows(), -1);
  for (int i = 0; i < F.rows(); i++) face_region_ids[i] = R(i, 0);
  RegionBoundaryExtractionResult loop_result =
      extract_region_boundary_loop(V, F, face_region_ids, region_id);
  export_region_boundary_debug_obj(
      boundary_filename, V, F, face_region_ids, region_id, loop_result);

  ofstream loop_out(boundary_loop_filename);
  if (!loop_out.is_open()) {
    cout << "[region snapshot] warning: cannot open " << boundary_loop_filename << endl;
  } else {
    loop_out.precision(17);
    loop_out << "VSA_REGION_BOUNDARY_LOOP 1\n";
    loop_out << "region_id " << region_id << "\n";
    loop_out << "success " << (loop_result.success ? 1 : 0) << "\n";
    loop_out << "reason " << loop_result.reason << "\n";
    loop_out << "closed " << (loop_result.loop.closed ? 1 : 0) << "\n";
    loop_out << "boundary_loop_count " << loop_result.boundary_loop_count << "\n";
    loop_out << "region_connected " << (loop_result.region_connected ? 1 : 0) << "\n";
    loop_out << "nonmanifold_boundary " << (loop_result.nonmanifold_boundary ? 1 : 0) << "\n";
    loop_out << "broken_chain " << (loop_result.broken_chain ? 1 : 0) << "\n";
    loop_out << "duplicate_edge " << (loop_result.duplicate_edge ? 1 : 0) << "\n";
    loop_out << "vertices " << loop_result.loop.vertex_ids.size() << "\n";
    for (int i = 0; i < (int)loop_result.loop.vertex_ids.size(); i++) {
      const Vector3d& pnt = loop_result.loop.positions[i];
      loop_out << i << " " << loop_result.loop.vertex_ids[i] << " "
               << pnt.x() << " " << pnt.y() << " " << pnt.z() << "\n";
    }
    loop_out << "boundary_edges " << loop_result.boundary_edges.size() << "\n";
    for (const auto& e : loop_result.boundary_edges) {
      loop_out << e[0] << " " << e[1] << "\n";
    }
    loop_out << "END\n";
  }

  cout << "[region snapshot] saved region " << region_id
       << " faces=" << region_face_ids.size()
       << " vertices=" << local_to_vertex.size()
       << " boundary_success=" << (loop_result.success ? "true" : "false")
       << endl;
  cout << "  mesh:     " << mesh_filename << endl;
  cout << "  metadata: " << metadata_filename << endl;
  cout << "  boundary: " << boundary_filename << endl;
  cout << "  loop:     " << boundary_loop_filename << endl;
  return true;
}

static bool load_interactive_checkpoint(const string& filename) {
  ifstream fin(filename);
  if (!fin.is_open()) {
    cout << "[checkpoint] load failed: cannot open " << filename << endl;
    return false;
  }

  string magic;
  int version = 0;
  fin >> magic >> version;
  if (magic != "QVSA_CHECKPOINT" || version != 1) {
    cout << "[checkpoint] load failed: unsupported file format." << endl;
    return false;
  }

  int saved_vertices = -1;
  int saved_faces = -1;
  int saved_regions = -1;
  int saved_use_quadric = 0;
  int saved_feature_enabled = 0;
  double saved_feature_angle = g_feature_angle_threshold;

  string key;
  fin >> key >> saved_vertices;
  if (key != "vertices") return false;
  fin >> key >> saved_faces;
  if (key != "faces") return false;
  fin >> key >> saved_regions;
  if (key != "regions") return false;
  fin >> key >> saved_use_quadric;
  if (key != "use_quadric") return false;
  fin >> key >> saved_feature_enabled;
  if (key != "feature_barrier_enabled") return false;
  fin >> key >> saved_feature_angle;
  if (key != "feature_angle") return false;

  if (saved_faces != F.rows() || saved_vertices != V.rows() || saved_regions <= 0) {
    cout << "[checkpoint] load failed: checkpoint mesh does not match current mesh."
         << " fileVerts=" << saved_vertices << " currentVerts=" << V.rows()
         << " fileFaces=" << saved_faces << " currentFaces=" << F.rows() << endl;
    return false;
  }

  fin >> key;
  if (key != "labels") {
    cout << "[checkpoint] load failed: missing labels block." << endl;
    return false;
  }

  MatrixXi loaded_R(saved_faces, 1);
  int invalid_labels = 0;
  for (int i = 0; i < saved_faces; i++) {
    fin >> loaded_R(i, 0);
    if (loaded_R(i, 0) < 0 || loaded_R(i, 0) >= saved_regions) {
      invalid_labels++;
    }
  }
  if (invalid_labels > 0) {
    cout << "[checkpoint] load failed: invalid labels=" << invalid_labels << endl;
    return false;
  }

  int proxy_count = 0;
  fin >> key >> proxy_count;
  if (key != "quadric_proxies") {
    cout << "[checkpoint] load failed: missing quadric proxy block." << endl;
    return false;
  }

  vector<QuadricProxy> loaded_QP(saved_regions);
  int loaded_proxy_rows = 0;
  for (int row = 0; row < proxy_count; row++) {
    int rid = -1;
    fin >> rid;
    VectorXd coeffs(10);
    for (int k = 0; k < 10; k++) fin >> coeffs(k);
    if (rid >= 0 && rid < saved_regions) {
      loaded_QP[rid] = QuadricProxy(coeffs);
      loaded_proxy_rows++;
    }
  }

  R = loaded_R;
  p = saved_regions;
  use_quadric = true;
  QP = loaded_QP;
  if (loaded_proxy_rows != p || !saved_use_quadric) {
    QP.resize(p);
    for (int j = 0; j < p; j++) {
      QP[j] = fit_quadric_region(R, j, F, V);
    }
  }

  g_feature_barrier_enabled = saved_feature_enabled != 0;
  g_feature_angle_threshold = saved_feature_angle;
  g_feature_edges.clear();
  g_boundary_valid = false;
  has_before_smoothing = false;
  has_after_smoothing = true;
  R_after_smoothing = R;
  has_projected_mesh = false;
  has_reconstructed_mesh = false;

  cout << "[checkpoint] loaded " << filename
       << " faces=" << R.rows()
       << " regions=" << p
       << " quadric_proxies=" << QP.size()
       << " feature_barrier=" << (g_feature_barrier_enabled ? "on" : "off")
       << endl;
  return true;
}

// ---- Helper: RowVectorXd → Vector3d ----
static bool load_and_render_quadric_vsa_segmentation_result(
    igl::opengl::glfw::Viewer& viewer,
    const string& filename) {
  if (filename.empty()) {
    cout << "[segmentation] load failed: empty snapshot filename." << endl;
    return false;
  }
  if (!load_interactive_checkpoint(filename)) return false;
  g_viewer_work_mode = ViewerWorkMode::QuadricVSA;
  show_segmentation(viewer, V, F, R, "Loaded Quadric VSA segmentation");
  return true;
}

static inline Vector3d v3(const Eigen::RowVectorXd& r) {
  return Vector3d(r(0), r(1), r(2));
}

static int fill_unassigned_faces_respecting_features(MatrixXi& labels) {
  int m = labels.rows();
  int assigned = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (int i = 0; i < m; i++) {
      if (labels(i, 0) >= 0) continue;
      for (int k = 0; k < 3; k++) {
        int nb = Ad(i, k);
        if (nb < 0 || nb >= m || labels(nb, 0) < 0) continue;
        if (is_feature_barrier(i, k, F, Ad)) continue;
        labels(i, 0) = labels(nb, 0);
        assigned++;
        changed = true;
        break;
      }
    }
  }

  int crossed_feature_fallbacks = 0;
  for (int i = 0; i < m; i++) {
    if (labels(i, 0) >= 0) continue;
    for (int k = 0; k < 3; k++) {
      int nb = Ad(i, k);
      if (nb >= 0 && nb < m && labels(nb, 0) >= 0) {
        labels(i, 0) = labels(nb, 0);
        assigned++;
        crossed_feature_fallbacks++;
        break;
      }
    }
  }

  if (crossed_feature_fallbacks > 0) {
    cout << "[feature_barrier warning] assigned " << crossed_feature_fallbacks
         << " faces across feature edges because their feature component had no seed." << endl;
  }
  return assigned;
}

static bool apply_final_boundary_check_no_refit() {
  int old_p = p;

  if (g_feature_edges.empty()) {
    compute_feature_edges(F, Ad, g_feature_angle_threshold);
  }
  if (!g_feature_edges.empty()) {
    g_feature_barrier_enabled = true;
  }

  cout << "[final boundary check] begin regions=" << old_p
       << " feature_edges=" << g_feature_edges.size()
       << " barrier=" << (g_feature_barrier_enabled ? "on" : "off")
       << endl;

  BoundaryNormalRelabelReport normal_report =
      relabel_boundary_faces_by_normal(R, F, Ad, p);

  FeatureBarrierEnforceReport report =
      enforce_feature_barrier_final(R, F, Ad, p);

  cout << "[final boundary check] boundary_faces="
       << normal_report.boundary_face_count
       << " relabeled=" << normal_report.relabeled_face_count
       << " feature_violations="
       << report.violating_feature_edges_before << "->"
       << report.violating_feature_edges_after
       << endl;

  if (p != old_p) {
    cout << "[final boundary check] region count changed " << old_p << " -> " << p
         << "; proxies are intentionally not refit." << endl;
  }
  return p == old_p;
}

// ---- Build edge → incident faces mapping ----
static map<EdgeKey, vector<int>> build_edge_to_faces() {
    map<EdgeKey, vector<int>> e2f;
    for (int fi = 0; fi < F.rows(); fi++) {
        for (int j = 0; j < 3; j++) {
            int va = F(fi, j), vb = F(fi, (j+1)%3);
            EdgeKey ek(va, vb);
            e2f[ek].push_back(fi);
        }
    }
    return e2f;
}

// ---- Extract all boundary edges from current R ----
static void extract_region_boundaries() {
    auto e2f = build_edge_to_faces();
    g_boundary_edges.clear();
    g_region_pairs.clear();
    set<pair<int,int>> pair_set;

    for (auto& kv : e2f) {
        auto& ek = kv.first;
        auto& faces = kv.second;
        BoundaryEdge be;
        be.ek = ek;
        be.face_i = faces.size() > 0 ? faces[0] : -1;
        be.face_j = faces.size() > 1 ? faces[1] : -1;

        if (faces.size() == 1) {
            be.type = BT_MESH;
            be.region_i = R(faces[0], 0);
            be.region_j = -1;
            g_boundary_edges.push_back(be);
        } else if (faces.size() == 2) {
            int r0 = R(faces[0], 0);
            int r1 = R(faces[1], 0);
            be.region_i = r0;
            be.region_j = r1;
            if (r0 != r1) {
                be.type = BT_REGION;
                g_boundary_edges.push_back(be);
                pair_set.insert(make_pair(min(r0, r1), max(r0, r1)));
            }
        } else {
            be.type = BT_NONMANIFOLD;
            be.region_i = -1;
            be.region_j = -1;
            g_boundary_edges.push_back(be);
        }
    }

    g_region_pairs.assign(pair_set.begin(), pair_set.end());
    sort(g_region_pairs.begin(), g_region_pairs.end());
    g_boundary_valid = true;

    int rc = 0, mc = 0, nc = 0;
    for (auto& be : g_boundary_edges) {
        if (be.type == BT_REGION) rc++;
        else if (be.type == BT_MESH) mc++;
        else nc++;
    }
    cout << "[boundary] " << g_boundary_edges.size() << " boundary edges: "
         << rc << " region, " << mc << " mesh, " << nc << " nonmanifold" << endl;
    cout << "  Region pairs: " << g_region_pairs.size() << endl;
}

// ---- Show all region boundaries (G key) ----
static void show_region_boundaries(igl::opengl::glfw::Viewer& viewer) {
    if (!g_boundary_valid) extract_region_boundaries();

    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    MatrixXd colors = make_face_colors_from_labels(R, F.rows());
    viewer.data().set_colors(colors);
    viewer.data().show_lines = false;

    int n = g_boundary_edges.size();
    if (n == 0) {
        cout << "[boundary] No boundary edges found." << endl;
        return;
    }

    MatrixXd P1(n, 3), P2(n, 3), EC(n, 3);
    for (int i = 0; i < n; i++) {
        auto& be = g_boundary_edges[i];
        P1.row(i) = V.row(be.ek.v0);
        P2.row(i) = V.row(be.ek.v1);
        if (be.type == BT_REGION)
            EC.row(i) = RowVector3d(1.0, 0.0, 0.0);
        else if (be.type == BT_MESH)
            EC.row(i) = RowVector3d(0.0, 0.4, 1.0);
        else
            EC.row(i) = RowVector3d(1.0, 1.0, 0.0);
    }
    viewer.data().add_edges(P1, P2, EC);
    viewer.data().show_overlay = true;
    viewer.data().line_width = 2.0;

    int rc = 0, mc = 0;
    for (auto& be : g_boundary_edges) {
        if (be.type == BT_REGION) rc++;
        else if (be.type == BT_MESH) mc++;
    }
    cout << "[boundary] Drawn: " << rc << " region (red) + " << mc
         << " mesh (blue) edges (total " << n << ")" << endl;
}

// ---- Show one region pair boundary (H/J/K keys) ----
static void show_one_region_pair_boundary(igl::opengl::glfw::Viewer& viewer, int pair_idx) {
    if (!g_boundary_valid) extract_region_boundaries();
    if (g_region_pairs.empty()) {
        cout << "No region pairs found." << endl;
        return;
    }
    pair_idx = max(0, min(pair_idx, (int)g_region_pairs.size() - 1));
    g_current_pair_idx = pair_idx;

    int ri = g_region_pairs[pair_idx].first;
    int rj = g_region_pairs[pair_idx].second;

    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    MatrixXd colors(F.rows(), 3);
    for (int i = 0; i < F.rows(); i++) {
        int r = R(i, 0);
        if (r == ri)
            colors.row(i) = Eigen::RowVector3d(0.2, 0.6, 1.0);
        else if (r == rj)
            colors.row(i) = Eigen::RowVector3d(1.0, 0.6, 0.2);
        else
            colors.row(i) = Eigen::RowVector3d(0.85, 0.85, 0.85);
    }
    viewer.data().set_colors(colors);
    viewer.data().show_lines = false;

    // Collect edges for this pair
    vector<int> pair_edge_idx;
    for (int i = 0; i < (int)g_boundary_edges.size(); i++) {
        auto& be = g_boundary_edges[i];
        if (be.type != BT_REGION) continue;
        int a = min(be.region_i, be.region_j);
        int b = max(be.region_i, be.region_j);
        if (a == ri && b == rj) pair_edge_idx.push_back(i);
    }

    if (!pair_edge_idx.empty()) {
        MatrixXd P1(pair_edge_idx.size(), 3), P2(pair_edge_idx.size(), 3), EC(pair_edge_idx.size(), 3);
        for (int k = 0; k < (int)pair_edge_idx.size(); k++) {
            auto& be = g_boundary_edges[pair_edge_idx[k]];
            P1.row(k) = V.row(be.ek.v0);
            P2.row(k) = V.row(be.ek.v1);
            EC.row(k) = RowVector3d(1.0, 0.0, 0.0);
        }
        viewer.data().add_edges(P1, P2, EC);
        viewer.data().show_overlay = true;
        viewer.data().line_width = 2.0;
    }

    cout << "[pair " << (pair_idx+1) << "/" << g_region_pairs.size() << "] "
         << "Region " << ri << " <-> " << rj << ": " << pair_edge_idx.size() << " edges" << endl;
}

// ---- Quadric Lloyd: one step (no viewer update) ----
static void quadric_lloyd_step() {
  int m = F.rows();

  // Find best seed per region (min error face)
  vector<int> seeds(p, -1);
  vector<double> best_err(p, 1e18);
  for (int i = 0; i < m; i++) {
    int j = R(i, 0);
    if (j < 0 || j >= p) continue;
    Vector3i f = F.row(i);
    double err = QP[j].triangle_error(v3(V.row(f(0))), v3(V.row(f(1))), v3(V.row(f(2))));
    if (err < best_err[j]) { best_err[j] = err; seeds[j] = i; }
  }

  // Priority queue flooding
  MatrixXi R_new = -MatrixXi::Ones(m, 1);
  priority_queue<pair<double, int>> q;
  for (int i = 0; i < p; i++) {
    if (seeds[i] < 0) continue;
    R_new(seeds[i], 0) = i;
    for (int k = 0; k < 3; k++) {
      if (is_feature_barrier(seeds[i], k, F, Ad)) continue;
      int nb = Ad(seeds[i], k);
      if (nb < 0 || nb >= m) continue;
      Vector3i f = F.row(nb);
      double d = QP[i].triangle_error(v3(V.row(f(0))), v3(V.row(f(1))), v3(V.row(f(2))));
      q.push(make_pair(-d, nb + m * i));
    }
  }
  while (!q.empty()) {
    auto item = q.top(); q.pop();
    int prox = item.second / m;
    int face = item.second % m;
    if (R_new(face, 0) != -1) continue;
    R_new(face, 0) = prox;
    for (int k = 0; k < 3; k++) {
      if (is_feature_barrier(face, k, F, Ad)) continue;
      int nb = Ad(face, k);
      if (nb < 0 || nb >= m || R_new(nb, 0) != -1) continue;
      Vector3i f = F.row(nb);
      double d = QP[prox].triangle_error(v3(V.row(f(0))), v3(V.row(f(1))), v3(V.row(f(2))));
      q.push(make_pair(-d, nb + m * prox));
    }
  }

  fill_unassigned_faces_respecting_features(R_new);

  R = R_new;

  // Refit quadric proxies
  for (int j = 0; j < p; j++)
    QP[j] = fit_quadric_region(R, j, F, V);

  // Compute global error
  error = 0;
  for (int i = 0; i < m; i++) {
    int j = R(i, 0);
    if (j < 0 || j >= p) continue;
    Vector3i f = F.row(i);
    error += QP[j].triangle_error(v3(V.row(f(0))), v3(V.row(f(1))), v3(V.row(f(2))));
  }
}

// ---- Quadric: one iteration with viewer update ----
static void one_iter_quadric(igl::opengl::glfw::Viewer& viewer) {
  quadric_lloyd_step();
  iterations++;
  global_error_points.push_back(make_pair(iterations, error));
  precedent_error = error;
  cout << "Global Error (quadric): " << error << endl;
  show_segmentation(viewer, V, F, R, "Quadric segmentation");
}

// ---- Compute per-face error values ----
static void compute_face_errors() {
  face_error_values.resize(F.rows());
  for (int i = 0; i < F.rows(); i++) {
    int j = R(i, 0);
    if (j < 0 || j >= p) { face_error_values(i) = 0; continue; }
    if (use_quadric && j < (int)QP.size()) {
      Vector3i f = F.row(i);
      face_error_values(i) = QP[j].triangle_error(
          v3(V.row(f(0))), v3(V.row(f(1))), v3(V.row(f(2))));
    } else {
      face_error_values(i) = distance(i, Proxies.row(j), Proxies.row(p + j), V, metric);
    }
  }
  has_face_error_values = true;
}

void debug_regions_vides(MatrixXi R, int p){
  cout<<"Regions vides"<<endl;
  bool trouve_j;
  for (int j=0 ; j<p ; j++){
    trouve_j = false;
    for (int i=0 ; i<R.rows() ; i++){
      if (R(i,0)==j){
        trouve_j = true;
      }
    }
    if (trouve_j == false){
      cout<<j<<endl;
    }
  }
  cout<<"fin"<<endl;
};

MatrixXd covariance(MatrixXd M) {
  Vector3d mean = M.colwise().mean();
  for (int j=0 ; j<M.rows(); j++) { 
    M.row(j) -= mean;
  }
  return M.transpose()*M;
}

pair<Vector3d,Vector3d> compute_ellipse_vectors(int c){
  vector<Vector3d> Radius;
  for(int i = 0; i < F.rows(); i++) {
    if (R(i,0)==c) {
      for (int j=0;j<3;j++) {
        Vector3d q = V.row(F(i,j));
        Vector3d q2 = Proxies.row(R(i,0));
        Radius.push_back(q-q2);
      }
    }
  }
  int k = Radius.size();
  MatrixXd M;
  M.setZero(k,3);
  for (int j=0 ; j<k; j++) { 
    M.row(j) = Radius[j];
  }  
  EigenSolver<MatrixXd> eig(covariance(M)/k);
  MatrixXd ev = eig.eigenvectors().real();
  MatrixXd eva = eig.eigenvalues().real();
  
  Vector3d e1,e2;
  for (int l=0; l<3;l++) {
    if (eva(l) == eva.maxCoeff()) {
      e1 = ev.col(l)*pow(eva(l),0.5);
      eva(l) = -eva(l);
      break;
    }
  }
  for (int l=0; l<3;l++) {
    if (eva(l) == eva.maxCoeff()) {
      e2 = ev.col(l)*pow(eva(l),0.5);
    }
  }
  Vector3d n = Proxies.row(p+c);
  if ( n.dot(e1.cross(e2))<0) {
    return make_pair(-e1,e2);
  }
  return make_pair(e1,e2);
}

void draw_tangent(igl::opengl::glfw::Viewer &viewer) {
   for (int i =0; i<p;i++) {
    viewer.append_mesh();
    viewer.data(0).add_points(Proxies.row(i), Eigen::RowVector3d(1, 0, 0));
    viewer.data(0).add_edges(
        Proxies.row(i),
        Proxies.row(i) + Proxies.row(i+p)/10.0,
        Eigen::RowVector3d(1, 0, 0));
  }
}

void color_scheme(igl::opengl::glfw::Viewer &viewer, MatrixXd V, MatrixXi F) {
  viewer.data().clear();
  int f = F.rows();
  MatrixXd nC(f,1);
  for (int i=0; i<f; i++){
    Vector3d c =(V.row(F(i,0)) + V.row(F(i,1)) + V.row(F(i,2))) / 30.0;
    nC(i,0)=c(1);
  }
  igl::jet(nC,true,C);
  viewer.data().set_mesh(V, F);
  viewer.data().set_colors(C);
}
void draw_anchors(igl::opengl::glfw::Viewer &viewer) {
  vector<vector<int>> anchors = anchor_points(*he, R, V, Proxies,treshold);
  for(size_t i = 0; i < anchors.size(); i++) {
    for(size_t j = 0; j < anchors[i].size(); j++) {
      viewer.data(0).add_points(V.row(anchors[i][j]), Eigen::RowVector3d(1,1,0));
    }
  }
    
}

void triangle_proxy(Vector3d x, Vector3d n, MatrixXd& newV, int k, Vector3d m1, Vector3d m2) {

  int M=20;
  newV.row(M*k) = x;
  for (int i=1; i<M; i++) {
    double t = i*2*M_PI/(M-1);
    newV.row(M*k+i) = x + sin(t)*m1 + cos(t)*m2;
      // result.row(i) <<1,2,3;
  }
}

void draw_prox(igl::opengl::glfw::Viewer &viewer) {
  int M=20;

  MatrixXd newV;
  newV.setZero(M*p,3);
  MatrixXi newF;
  MatrixXi newR0;
  newR0.setZero((M-1)*p,3);
  newF.setZero((M-1)*p,3);

  for(int i = 0; i < p; i++) {
    pair<Vector3d,Vector3d> vec = compute_ellipse_vectors(i);  
    triangle_proxy(Proxies.row(i),Proxies.row(i+p), newV, i, vec.first, vec.second);
    for (int j=1; j<M-1; j++) {
      newF.row((M-1)*i+j-1) << M*i+j,M*i,M*i+j+1;
      newR0((M-1)*i+j-1)=i;
    }
    newF.row((M-1)*(i+1)-1) << M*i+M-1,M*i,M*i+1;
    newR0((M-1)*(i+1)-1)=i;
  }
  viewer.data().clear();
  viewer.data().set_mesh(newV, newF);
  igl::jet(newR0,true,C);
  viewer.data(0).set_colors(C);

}
void one_iter(igl::opengl::glfw::Viewer &viewer) {
  proxy_color(R, Proxies, V,  F, Ad, metric);
  Proxies = new_proxies(R, F, V, p, metric);
  iterations += 1;
  error = global_distortion_error(R,Proxies,V,F,metric);
  cout<<"Global Error : "<<error<<endl;
  global_error_points.push_back(make_pair(iterations,error));
  precedent_error = error;
  igl::jet(R,true,C);
  viewer.data(0).set_colors(C);
}
bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier) {
  cout << "pressed Key: " << key << " " << (unsigned int)key << endl;
  if (key=='1') {
    viewer.data().clear();
  }
  if (key=='2') {
    color_scheme(viewer, V, F);
  }
  if (key=='3') {
    if (use_quadric) one_iter_quadric(viewer);
    else one_iter(viewer);
  }
  if (key=='4') {
    draw_anchors(viewer);
  }
  if (key=='5') {

    vector<vector<int>> anchors = anchor_points(*he, R, V, Proxies,treshold);
    MatrixXi Cr = color_region(R,6,anchors,V,*he);

    // viewer.append_mesh();
    // for(int j = 0; j < Cr.rows(); j++) {
    //   int i = Cr(j,0);
    //   if (i>-1) viewer.data(0).add_points(V.row(j), Eigen::RowVector3d(i%3/2.0,i/9.0, i%2));
    // }
    // return true;
    pair<MatrixXi,MatrixXi> new_F_and_R = triangulation(R,anchors,V,F,*he);
    newF = new_F_and_R.first;
    newR = new_F_and_R.second;

    map<int,int> index = renumber(newF); //modifies F
    newV = new_V(*he,V,Proxies,R,index);
    viewer.data().clear();
    igl::jet(newR,true,C);
    viewer.data().set_mesh(newV, newF);
    viewer.data().set_colors(C);
    cout <<"faces : "<<newF.rows() << endl;

  }
  if (key=='6') {
    color_scheme(viewer, newV, newF);
  }
  if (key=='7') {
    draw_prox(viewer);
  }
  if (key=='8') {
    if (use_quadric) { for (int i=0;i<10;i++) one_iter_quadric(viewer); }
    else { for (int i=0;i<10;i++) one_iter(viewer); }
    cout << "    Done" <<endl;
  }
  if (key=='9') {
    if (use_quadric) { for (int i=0;i<100;i++) one_iter_quadric(viewer); }
    else { for (int i=0;i<100;i++) one_iter(viewer); }
    cout << "    Done" <<endl;
  }
  // E: Error heatmap
  if (key == 'E' || (unsigned int)key == 69) {
    show_error_heatmap(viewer);
    return true;
  }
  // +/= : omega +0.1
  if (key == '+' || key == '=') {
    omega += 0.1;
    cout << "omega = " << omega << endl;
  }
  // - : omega -0.1
  if (key == '-') {
    omega = max(0.0, omega - 0.1);
    cout << "omega = " << omega << endl;
  }
  // M: 切换能量模式 (L2 → L21 → HYBRID → L2)
  if (key == 'M' || (unsigned int)key == 77) {
    if (metric == L2_METRIC) {
      metric = L21_METRIC;
    } else if (metric == L21_METRIC) {
      metric = HYBRID_METRIC;
    } else {
      metric = L2_METRIC;
    }
    cout << "Metric switched to: "
         << (metric == L2_METRIC ? "L2" : metric == L21_METRIC ? "L21" : "HYBRID")
         << " (omega=" << omega << ")" << endl;
  }
  // C: Current segmentation
  if (key == 'C' || (unsigned int)key == 67) {
    show_segmentation(viewer, V, F, R, "Current segmentation");
    return true;
  }
  // Z: Toggle region id labels
  if (key == 'Z' || key == 'z') {
    if (g_show_region_id_labels) {
      clear_region_id_labels(viewer);
    } else {
      show_region_id_labels(viewer);
    }
    return true;
  }
  // B: Before smoothing
  if (key == 'B' || (unsigned int)key == 66) {
    if (has_before_smoothing)
      show_segmentation(viewer, V, F, R_before_smoothing, "Before smoothing");
    else
      cout << "No before-smoothing labels available." << endl;
    return true;
  }
  // A: After smoothing
  if (key == 'A' || (unsigned int)key == 65) {
    if (has_after_smoothing)
      show_segmentation(viewer, V, F, R_after_smoothing, "After smoothing");
    else
      cout << "No after-smoothing labels available." << endl;
    return true;
  }
  // D: Diff (changed faces after smoothing)
  if (key == 'D' || (unsigned int)key == 68) {
    show_changed_faces_after_smoothing(viewer);
    return true;
  }
  // P: Projected mesh
  if (key == 'P' || (unsigned int)key == 80) {
    show_projected_mesh(viewer);
    return true;
  }
  // O: Original mesh (coordinate color)
  if (key == 'O' || (unsigned int)key == 79) {
    color_scheme(viewer, V, F);
    return true;
  }
  // T: Save current segmentation/proxy checkpoint
  if (key == 'T' || key == 't') {
    save_quadric_vsa_segmentation_result(kInteractiveCheckpointFile);
    return true;
  }
  // U: Load saved segmentation/proxy checkpoint
  if (key == 'U' || key == 'u') {
    load_and_render_quadric_vsa_segmentation_result(viewer, kInteractiveCheckpointFile);
    return true;
  }
  // I: Insert one proxy (split worst region)
  if (key == 'I' || (unsigned int)key == 73) {
    // Find worst region by area-normalized error
    vector<double> reg_err(p, 0.0), reg_area(p, 0.0);
    for (int i = 0; i < F.rows(); i++) {
      int j = R(i, 0);
      if (j < 0 || j >= p) continue;
      double e;
      Vector3i f = F.row(i);
      double a = 0.5 * (v3(V.row(f(1)))-v3(V.row(f(0)))).cross(v3(V.row(f(2)))-v3(V.row(f(0)))).norm();
      if (use_quadric && j < (int)QP.size())
        e = QP[j].triangle_error(v3(V.row(f(0))), v3(V.row(f(1))), v3(V.row(f(2))));
      else
        e = distance(i, Proxies.row(j), Proxies.row(p + j), V, metric);
      reg_err[j] += e;
      reg_area[j] += a;
    }
    int worst = 0; double worst_ne = 0;
    for (int j = 0; j < p; j++) {
      double ne = reg_area[j] > 1e-15 ? reg_err[j] / reg_area[j] : 0;
      if (ne > worst_ne) { worst_ne = ne; worst = j; }
    }
    // Find worst face in worst region
    int wf = -1; double wf_err = -1;
    for (int i = 0; i < F.rows(); i++) {
      if (R(i, 0) != worst) continue;
      double e; Vector3i f = F.row(i);
      if (use_quadric && worst < (int)QP.size())
        e = QP[worst].triangle_error(v3(V.row(f(0))), v3(V.row(f(1))), v3(V.row(f(2))));
      else
        e = distance(i, Proxies.row(worst), Proxies.row(p + worst), V, metric);
      if (e > wf_err) { wf_err = e; wf = i; }
    }
    if (wf < 0) { cout << "No valid face to split." << endl; return true; }
    // Insert: reassign worst face to new proxy
    int old_p = p; p++;
    R(wf, 0) = old_p;
    if (use_quadric) {
      QP.resize(p);
      QP[old_p] = fit_quadric_region(R, old_p, F, V);
      QP[worst] = fit_quadric_region(R, worst, F, V);
    } else {
      MatrixXd newP(p * 2, 3); newP.setZero();
      for (int j = 0; j < old_p; j++) {
        newP.row(j) = Proxies.row(j);
        newP.row(p + j) = Proxies.row(old_p + j);
      }
      Vector3d c = (v3(V.row(F(wf,0))) + v3(V.row(F(wf,1))) + v3(V.row(F(wf,2)))) / 3.0;
      Vector3d n = (v3(V.row(F(wf,1)))-v3(V.row(F(wf,0)))).cross(v3(V.row(F(wf,2)))-v3(V.row(F(wf,0))));
      if (n.norm() > 1e-12) n.normalize();
      newP.row(old_p) = c; newP.row(p + old_p) = n;
      Proxies = newP;
    }
    cout << "Inserted proxy " << old_p << " from region " << worst
         << " (norm_err=" << worst_ne << ") -> K=" << p << endl;
    show_segmentation(viewer, V, F, R, "K=" + to_string(p));
    return true;
  }
  // N: Progressive main pipeline (validity-guided in quadric mode)
  if (key == 'N' || key == 'n') {
    double err_thresh = 1e-5;
    int max_p = min((int)F.rows() / 3, 25);
    int max_validity_attempts = 20;
    int consecutive_validity_splits = 0;
    int prev_invalid_count = 0;
    cout << "Progressive insertion: threshold=" << err_thresh << " max_K=" << max_p << endl;
    // Feature barrier: compute edges if enabled but not yet computed
    if (g_feature_barrier_enabled && g_feature_edges.empty()) {
      compute_feature_edges(F, Ad, g_feature_angle_threshold);
      cout << "Feature barrier active: " << g_feature_edges.size() << " edges" << endl;
    }
    for (;;) {
      // Lloyd converge (up to 30 iters)
      if (use_quadric) {
        for (int it = 0; it < 30; it++) {
          MatrixXi prev = R;
          quadric_lloyd_step();
          int ch = 0;
          for (int i = 0; i < F.rows(); i++) if (R(i,0)!=prev(i,0)) ch++;
          if (ch == 0) break;
        }
      } else {
        for (int it = 0; it < 30; it++) {
          MatrixXi prev = R;
          proxy_color(R, Proxies, V, F, Ad, metric);
          Proxies = new_proxies(R, F, V, p, metric);
          int ch = 0;
          for (int i = 0; i < F.rows(); i++) if (R(i,0)!=prev(i,0)) ch++;
          if (ch == 0) break;
        }
      }
      // Compute max region error
      vector<double> re(p,0), ra(p,0);
      for (int i = 0; i < F.rows(); i++) {
        int j = R(i,0);
        if (j<0||j>=p) continue;
        Vector3i f = F.row(i);
        double a = 0.5*(v3(V.row(f(1)))-v3(V.row(f(0)))).cross(v3(V.row(f(2)))-v3(V.row(f(0)))).norm();
        double e;
        if (use_quadric && j<(int)QP.size())
          e = QP[j].triangle_error(v3(V.row(f(0))),v3(V.row(f(1))),v3(V.row(f(2))));
        else
          e = distance(i, Proxies.row(j), Proxies.row(p+j), V, metric);
        re[j] += e; ra[j] += a;
      }
      double max_ne = 0;
      for (int j = 0; j < p; j++) {
        double ne = ra[j]>1e-15 ? re[j]/ra[j] : 0;
        if (ne > max_ne) max_ne = ne;
      }
      // Validity check (quadric mode only)
      int invalid_count = 0;
      vector<ProxyValidityReport> reports;
      if (use_quadric) {
        ProxyValidityConfig full_cfg;
        full_cfg.enable_basic = true;
        full_cfg.enable_degeneracy = true;
        full_cfg.enable_classification = true;
        full_cfg.enable_two_sheet = true;
        reports = check_all_proxies(QP, R, F, V, full_cfg);
        for (auto& rpt : reports)
          if (!rpt.is_valid || rpt.is_suspicious) invalid_count++;
      }
      cout << "  K=" << p << " max_region_err=" << max_ne
           << " invalid=" << invalid_count << endl;
      // Stop conditions
      if (p >= max_p) {
        cout << "  Stopped: max proxies reached" << endl;
        break;
      }
      if (max_ne < err_thresh) {
        cout << "  Stopped: threshold met";
        if (invalid_count > 0) {
          cout << " (invalid proxies still reported=" << invalid_count
               << "; not used as a convergence blocker)";
        }
        cout << endl;
        break;
      }
      // Choose region to split
      int split_region = -1, seed_face = -1;
      string split_mode = "max_error";
      if (use_quadric && invalid_count > 0 &&
          consecutive_validity_splits < max_validity_attempts) {
        int rid = choose_region_to_split_by_validity(reports, 4);
        if (rid >= 0) {
          seed_face = choose_seed_face_for_invalid_region(rid, reports[rid], R, F, V, QP);
          if (seed_face >= 0) {
            split_region = rid;
            split_mode = (!reports[rid].is_valid) ? "invalid_proxy" : "suspicious_proxy";
          }
        }
      }
      if (split_region < 0) {
        // Fallback: max-error region
        int worst = 0; double wne = 0;
        for (int j = 0; j < p; j++) {
          double ne = ra[j]>1e-15 ? re[j]/ra[j] : 0;
          if (ne>wne) { wne=ne; worst=j; }
        }
        split_region = worst;
        seed_face = -1; double wf2e = -1;
        for (int i = 0; i < F.rows(); i++) {
          if (R(i,0)!=split_region) continue;
          double e; Vector3i f = F.row(i);
          if (use_quadric && split_region<(int)QP.size())
            e = QP[split_region].triangle_error(v3(V.row(f(0))),v3(V.row(f(1))),v3(V.row(f(2))));
          else
            e = distance(i, Proxies.row(split_region), Proxies.row(p+split_region), V, metric);
          if (e>wf2e) { wf2e=e; seed_face=i; }
        }
        split_mode = "max_error";
      }
      if (seed_face < 0) break;
      cout << "    split: " << split_mode << " region=" << split_region
           << " seed=" << seed_face << endl;
      // Track consecutive validity splits
      if (split_mode != "max_error") {
        if (invalid_count >= prev_invalid_count) consecutive_validity_splits++;
        else consecutive_validity_splits = 0;
      } else {
        consecutive_validity_splits = 0;
      }
      prev_invalid_count = invalid_count;
      // Insert proxy
      int op = p; p++;
      R(seed_face,0) = op;
      if (use_quadric) {
        QP.resize(p);
        QP[op] = fit_quadric_region(R, op, F, V);
        QP[split_region] = fit_quadric_region(R, split_region, F, V);
      } else {
        MatrixXd nP(p*2,3); nP.setZero();
        for (int j=0;j<op;j++) { nP.row(j)=Proxies.row(j); nP.row(p+j)=Proxies.row(op+j); }
        Vector3d c=(v3(V.row(F(seed_face,0)))+v3(V.row(F(seed_face,1)))+v3(V.row(F(seed_face,2))))/3.0;
        Vector3d n=(v3(V.row(F(seed_face,1)))-v3(V.row(F(seed_face,0)))).cross(v3(V.row(F(seed_face,2)))-v3(V.row(F(seed_face,0))));
        if (n.norm()>1e-12) n.normalize();
        nP.row(op)=c; nP.row(p+op)=n;
        Proxies = nP;
      }
    }
    iterations++;
    int K_after_progressive = p;

    // === Phase 2: Merge ===
    int merge_count = 0;
    int K_after_merge = p;
    bool pipeline_enable_merge = true;
    double merge_rel_thresh = 0.05;
    int max_merge_iters = 50;

    if (pipeline_enable_merge && p >= 2) {
      cout << "\n=== Merge phase ===" << endl;
      ProxyType pt = use_quadric ? QUADRIC_PROXY : PLANE_PROXY;
      merge_count = run_merge_pass(R, QP, Proxies, p, pt, metric,
                                    F, V, Ad, merge_rel_thresh, max_merge_iters);
      K_after_merge = p;
      cout << "  Merged " << merge_count << " pairs. K: "
           << K_after_progressive << " -> " << p << endl;

      // Refit all proxies
      if (use_quadric) {
        QP.resize(p);
        for (int j = 0; j < p; j++)
          QP[j] = fit_quadric_region(R, j, F, V);
      } else {
        Proxies = new_proxies(R, F, V, p, metric);
      }
    }

    // === Phase 3: Final boundary smoothing ===
    int smoothing_changed = 0;
    bool pipeline_enable_smoothing = false;

    if (pipeline_enable_smoothing) {
      cout << "\n=== Final boundary smoothing ===" << endl;
      R_before_smoothing = R;
      has_before_smoothing = true;

      SmoothConfig sc;
      vector<SmoothLogEntry> slog;
      ProxyType pt = use_quadric ? QUADRIC_PROXY : PLANE_PROXY;
      smooth_boundaries(R, F, V, Ad, p, pt, QP, Proxies, metric, sc, slog);

      R_after_smoothing = R;
      has_after_smoothing = true;

      for (int i = 0; i < F.rows(); i++)
        if (R_before_smoothing(i,0) != R_after_smoothing(i,0)) smoothing_changed++;
      cout << "  Changed " << smoothing_changed << " faces." << endl;

      // Refit all proxies after smoothing
      if (use_quadric) {
        for (int j = 0; j < p; j++)
          QP[j] = fit_quadric_region(R, j, F, V);
      } else {
        Proxies = new_proxies(R, F, V, p, metric);
      }
    } else {
      cout << "\n=== Final boundary smoothing skipped in main pipeline ===" << endl;
    }

    // === Compute final max_region_error ===
    {
      vector<double> fre(p,0), fra(p,0);
      for (int i = 0; i < F.rows(); i++) {
        int j = R(i,0);
        if (j<0||j>=p) continue;
        Vector3i f = F.row(i);
        double a = 0.5*(v3(V.row(f(1)))-v3(V.row(f(0)))).cross(v3(V.row(f(2)))-v3(V.row(f(0)))).norm();
        double e;
        if (use_quadric && j<(int)QP.size())
          e = QP[j].triangle_error(v3(V.row(f(0))),v3(V.row(f(1))),v3(V.row(f(2))));
        else
          e = distance(i, Proxies.row(j), Proxies.row(p+j), V, metric);
        fre[j] += e; fra[j] += a;
      }
      double final_max_ne = 0;
      for (int j = 0; j < p; j++) {
        double ne = fra[j]>1e-15 ? fre[j]/fra[j] : 0;
        if (ne > final_max_ne) final_max_ne = ne;
      }
      // Count invalid (if quadric)
      int final_invalid = 0;
      if (use_quadric) {
        ProxyValidityConfig full_cfg;
        full_cfg.enable_basic = true;
        full_cfg.enable_degeneracy = true;
        full_cfg.enable_classification = true;
        full_cfg.enable_two_sheet = true;
        auto reports = check_all_proxies(QP, R, F, V, full_cfg);
        for (auto& rpt : reports)
          if (!rpt.is_valid || rpt.is_suspicious) final_invalid++;
      }

      // === Pipeline Summary ===
      cout << "\n[Pipeline Summary]" << endl;
      cout << "  K_after_progressive  = " << K_after_progressive << endl;
      cout << "  K_after_merge        = " << K_after_merge << endl;
      cout << "  final_K              = " << p << endl;
      cout << "  merge_enabled        = " << (pipeline_enable_merge ? "true" : "false") << endl;
      cout << "  merge_count          = " << merge_count << endl;
      cout << "  boundary_smoothing   = " << (pipeline_enable_smoothing ? "true" : "false") << endl;
      cout << "  smoothing_changed    = " << smoothing_changed << endl;
      cout << "  final_max_region_err = " << final_max_ne << endl;
      if (use_quadric)
        cout << "  final_invalid        = " << final_invalid << endl;
    }

    // Final boundary ownership check. This is deliberately after the main
    // pipeline and does not refit proxies.
    bool proxies_still_match = apply_final_boundary_check_no_refit();
    if (!proxies_still_match) {
      cout << "[final boundary check] proxies no longer match final regions; "
           << "final segmentation is shown without proxy refit." << endl;
    }
    show_segmentation(viewer, V, F, R, "Pipeline K=" + to_string(p));
    return true;
  }
  // Y: Run merge pass only
  if (key == 'Y' || key == 'y') {
    if (p < 2) {
      cout << "[merge] skipped: need at least 2 regions." << endl;
      return true;
    }

    ProxyType pt = use_quadric ? QUADRIC_PROXY : PLANE_PROXY;
    double merge_rel_thresh = 0.05;
    int max_merge_iters = 50;
    int old_p = p;
    cout << "[merge] begin K=" << old_p
         << " relative_threshold=" << merge_rel_thresh
         << " max_iters=" << max_merge_iters << endl;

    int merge_count = run_merge_pass(R, QP, Proxies, p, pt, metric,
                                     F, V, Ad, merge_rel_thresh, max_merge_iters);

    if (use_quadric) {
      QP.resize(p);
      for (int j = 0; j < p; j++)
        QP[j] = fit_quadric_region(R, j, F, V);
    } else {
      Proxies = new_proxies(R, F, V, p, metric);
    }

    cout << "[merge] merged=" << merge_count
         << " K: " << old_p << " -> " << p << endl;
    show_segmentation(viewer, V, F, R, "After merge K=" + to_string(p));
    return true;
  }
  // L: Run final feature barrier / boundary ownership check only
  if (key == 'L' || key == 'l') {
    bool proxies_still_match = apply_final_boundary_check_no_refit();
    if (!proxies_still_match) {
      cout << "[final boundary check] proxies no longer match final regions; "
           << "final segmentation is shown without proxy refit." << endl;
    }
    show_segmentation(viewer, V, F, R, "After feature barrier K=" + to_string(p));
    return true;
  }
  // F: Feature edge barrier toggle
  if (key == 'F' || key == 'f') {
    if (g_feature_edges.empty()) {
      // First press: compute feature edges
      compute_feature_edges(F, Ad, g_feature_angle_threshold);
      g_feature_barrier_enabled = true;
      cout << "Feature barrier ENABLED (" << g_feature_edges.size() << " edges, threshold="
           << g_feature_angle_threshold << " deg)" << endl;
    } else if (g_feature_barrier_enabled) {
      g_feature_barrier_enabled = false;
      cout << "Feature barrier DISABLED (edges kept, " << g_feature_edges.size() << " edges)" << endl;
    } else {
      g_feature_barrier_enabled = true;
      cout << "Feature barrier RE-ENABLED (" << g_feature_edges.size() << " edges)" << endl;
    }
    // Show/hide feature edges
    if (g_feature_barrier_enabled && !g_feature_edges.empty()) {
      MatrixXd P1, P2;
      get_feature_edge_points(F, V, g_feature_edges, P1, P2);
      viewer.data().add_edges(P1, P2, RowVector3d(1, 0, 0));
      viewer.data().show_lines = false;
      viewer.data().show_overlay = true;
      viewer.data().line_width = 2.0;
      cout << "Feature edges shown in red" << endl;
    } else {
      // Clear overlay to hide feature edges
      viewer.data().clear_edges();
      viewer.data().show_lines = true;
      cout << "Feature edges hidden" << endl;
    }
    return true;
  }
  // Q: Toggle quadric/plane mode
  if (key == 'Q' || (unsigned int)key == 81) {
    use_quadric = !use_quadric;
    if (use_quadric) {
      QP.resize(p);
      for (int j = 0; j < p; j++)
        QP[j] = fit_quadric_region(R, j, F, V);
      cout << "Switched to QUADRIC mode. Fitted " << p << " quadric proxies." << endl;

      // ========== DIAGNOSTICS ==========
      int m = F.rows();

      // 1. Unique labels
      set<int> labels;
      for (int i = 0; i < m; i++) labels.insert(R(i, 0));
      cout << "\n--- DIAGNOSTICS ---" << endl;
      cout << "Unique labels: " << labels.size() << "  (expected K=" << p << ")" << endl;
      for (int l : labels) {
        if (l < 0) cout << "  WARNING: label " << l << " < 0" << endl;
        if (l >= p) cout << "  WARNING: label " << l << " >= K=" << p << endl;
      }

      // 2. Label histogram
      cout << "\nLabel histogram:" << endl;
      for (int l : labels) {
        int cnt = 0;
        for (int i = 0; i < m; i++) if (R(i,0)==l) cnt++;
        cout << "  label " << l << ": " << cnt << " faces" << endl;
      }

      // 3. Connected components per region (BFS via Ad)
      cout << "\nConnected components:" << endl;
      for (int l : labels) {
        vector<bool> visited(m, false);
        int cc = 0;
        for (int start = 0; start < m; start++) {
          if (R(start,0) != l || visited[start]) continue;
          cc++;
          queue<int> bfs;
          bfs.push(start); visited[start] = true;
          while (!bfs.empty()) {
            int fi = bfs.front(); bfs.pop();
            for (int k = 0; k < 3; k++) {
              int nb = Ad(fi, k);
              if (nb < 0 || nb >= m || visited[nb] || R(nb,0) != l) continue;
              visited[nb] = true; bfs.push(nb);
            }
          }
        }
        cout << "  region " << l << ": " << cc << " connected component(s)" << endl;
      }

      // 4. Proxy coefficients
      cout << "\nProxy coefficients (normalized):" << endl;
      for (int l : labels) {
        if (l < 0 || l >= (int)QP.size()) continue;
        VectorXd c = QP[l].coeffs;
        double cn = c.norm();
        if (cn > 1e-15) c /= cn;
        cout << "  proxy " << l << " (||c||=" << cn << "):" << endl;
        cout << "    C0=" << c(0) << " C1=" << c(1) << " C2=" << c(2) << " C3=" << c(3) << endl;
        cout << "    C4=" << c(4) << " C5=" << c(5) << " C6=" << c(6)
             << " C7=" << c(7) << " C8=" << c(8) << " C9=" << c(9) << endl;
        cout << "    diag check: C4-C7=" << c(4)-c(7) << " C4-C9=" << c(4)-c(9)
             << " C7-C9=" << c(7)-c(9) << endl;
        cout << "    off-diag: |C5|=" << abs(c(5)) << " |C6|=" << abs(c(6)) << " |C8|=" << abs(c(8)) << endl;
        cout << "    linear: |C1|=" << abs(c(1)) << " |C2|=" << abs(c(2)) << " |C3|=" << abs(c(3)) << endl;
      }

      // 5. Per-region error stats
      cout << "\nPer-region stats:" << endl;
      for (int l : labels) {
        if (l < 0 || l >= (int)QP.size()) continue;
        int fc = 0;
        double area = 0, err_sum = 0;
        double d_max = 0, d_sum = 0, d_sq_sum = 0;
        for (int i = 0; i < m; i++) {
          if (R(i,0) != l) continue;
          fc++;
          Vector3i f = F.row(i);
          Vector3d v0 = v3(V.row(f(0))), v1 = v3(V.row(f(1))), v2 = v3(V.row(f(2)));
          double a = 0.5 * (v1-v0).cross(v2-v0).norm();
          area += a;
          double te = QP[l].triangle_error(v0, v1, v2);
          err_sum += te;
          double d0 = QP[l].point_distance(v0);
          double d1 = QP[l].point_distance(v1);
          double d2 = QP[l].point_distance(v2);
          double db = QP[l].point_distance((v0+v1+v2)/3.0);
          for (double d : {d0,d1,d2,db}) {
            d_sum += d; d_sq_sum += d*d;
            if (d > d_max) d_max = d;
          }
        }
        int n_samples = fc * 4;
        double d_mean = n_samples > 0 ? d_sum / n_samples : 0;
        double d_rms = n_samples > 0 ? sqrt(d_sq_sum / n_samples) : 0;
        double norm_err = area > 1e-15 ? err_sum / area : 0;
        cout << "  region " << l << ": faces=" << fc << " area=" << area
             << " norm_err=" << norm_err
             << " mean_dist=" << d_mean << " rms_dist=" << d_rms
             << " max_dist=" << d_max << endl;
      }

      // 6. Error margin analysis (K=2 specific)
      if (p == 2 && labels.size() == 2) {
        cout << "\nError margin analysis (K=2):" << endl;
        vector<double> margins;
        for (int i = 0; i < m; i++) {
          Vector3i f = F.row(i);
          Vector3d v0 = v3(V.row(f(0))), v1 = v3(V.row(f(1))), v2 = v3(V.row(f(2)));
          double e0 = QP[0].triangle_error(v0, v1, v2);
          double e1 = QP[1].triangle_error(v0, v1, v2);
          margins.push_back(abs(e0 - e1));
        }
        sort(margins.begin(), margins.end());
        double sum_m = 0;
        for (double m : margins) sum_m += m;
        int n_tiny8 = 0, n_tiny6 = 0;
        for (double mg : margins) {
          if (mg < 1e-8) n_tiny8++;
          if (mg < 1e-6) n_tiny6++;
        }
        cout << "  mean margin=" << sum_m / margins.size() << endl;
        cout << "  median margin=" << margins[margins.size()/2] << endl;
        cout << "  min margin=" << margins.front() << endl;
        cout << "  max margin=" << margins.back() << endl;
        cout << "  margin<1e-8: " << n_tiny8 << "/" << margins.size()
             << " (" << 100.0*n_tiny8/margins.size() << "%)" << endl;
        cout << "  margin<1e-6: " << n_tiny6 << "/" << margins.size()
             << " (" << 100.0*n_tiny6/margins.size() << "%)" << endl;
      }

      cout << "--- END DIAGNOSTICS ---\n" << endl;
      // ========== END DIAGNOSTICS ==========

    } else {
      // Safety: check R for invalid labels before calling new_proxies
      {
        int bad = 0;
        for (int i = 0; i < R.rows(); i++) {
          if (R(i,0) < 0 || R(i,0) >= p) {
            if (bad == 0) cout << "WARNING: invalid labels in R before new_proxies:" << endl;
            if (bad < 5) cout << "  R(" << i << ")=" << R(i,0) << " (p=" << p << ")" << endl;
            bad++;
          }
        }
        if (bad > 0) cout << "  total " << bad << " invalid labels" << endl;
      }
      Proxies = new_proxies(R, F, V, p, metric);
      cout << "Switched to PLANE mode. Fitted " << p << " plane proxies." << endl;
    }
    show_segmentation(viewer, V, F, R, use_quadric ? "Quadric mode" : "Plane mode");
    return true;
  }
  // V: Compute and show error heatmap
  if (key == 'V' || (unsigned int)key == 86) {
    compute_face_errors();
    show_error_heatmap(viewer);
    return true;
  }
  // W: Boundary smoothing
  if (key == 'W' || (unsigned int)key == 87) {
    R_before_smoothing = R;
    has_before_smoothing = true;
    SmoothConfig sc;
    sc.ring = 2;
    sc.ringSize = 2;
    sc.lambda = 6.0;
    sc.maxPasses = 2;
    sc.skipSharpBoundary = false;
    sc.protectFeatureBoundaryFaces = true;
    sc.rejectBoundaryLengthIncrease = true;
    vector<SmoothLogEntry> slog;
    ProxyType pt = use_quadric ? QUADRIC_PROXY : PLANE_PROXY;
    BoundarySmoothingResult sres =
      smooth_boundaries(R, F, V, Ad, p, pt, QP, Proxies, metric, sc, slog);
    export_smooth_log(slog, "boundary_smoothing_log.csv");
    R_after_smoothing = R;
    has_after_smoothing = true;
    int invalid_labels = 0;
    for (int i = 0; i < R.rows(); i++) {
      if (R(i, 0) < 0 || R(i, 0) >= p) invalid_labels++;
    }
    cout << "[W] graph-cut smoothing changed "
         << sres.total_relabeled_faces
         << " faces. regions=" << p
         << " invalid_labels=" << invalid_labels
         << " (final boundary check is on L, not W)" << endl;
    show_segmentation(viewer, V, F, R, "After smoothing");
    return true;
  }
  // X: Project vertices onto proxies
  if (key == 'X' || (unsigned int)key == 88) {
    ProxyType pt = use_quadric ? QUADRIC_PROXY : PLANE_PROXY;
    vector<ClassifiedType> ptypes;
    if (use_quadric) {
      auto cr = classify_all_proxies(QP, R, F, V, 0.1);
      for (auto& r : cr) ptypes.push_back(r.type);
    }
    ProjectionConfig pcfg;
    ProjectionLog plog;
    projectedV = project_vertices(V, F, R, p, pt, QP, Proxies, ptypes, pcfg, plog);
    projectedF = F;
    has_projected_mesh = true;
    show_projected_mesh(viewer);
    cout << "  Projected: success=" << plog.success_count
         << " fallback=" << plog.fallback_count
         << " failure=" << plog.failure_count << endl;
    return true;
  }
  // R: Reconstruct, export, and show MVP quadric reconstruction
  if (key == 'R' || key == 'r') {
    if (g_viewer_work_mode == ViewerWorkMode::SplineSurface) {
      compute_and_show_spline_surface_for_region(viewer, g_spline_region_id);
      return true;
    }

    if (!use_quadric || (int)QP.size() != p) {
      QP.resize(p);
      for (int j = 0; j < p; j++)
        QP[j] = fit_quadric_region(R, j, F, V);
      use_quadric = true;
      cout << "Fitted " << p << " quadric proxies for reconstruction." << endl;
    }

    ReconstructionConfig rcfg;
    rcfg.enable_displacement = false;
    rcfg.interior_projection_mode = InteriorProjectionMode::OriginalMeshRayCast;
    rcfg.grid_resolution = 32;
    rcfg.max_samples_per_region = 8192;
    string base = "interactive_quadric_p" + to_string(p);
    rcfg.debug.output_prefix = base;

    NullDisplacementQuery null_query;
    ReconstructedMesh mesh;
    ReconstructionLog rlog;
    bool ok = reconstructQuadricVSA(V, F, R, QP, p, rcfg, &null_query, mesh, rlog);
    if (!ok) {
      cout << "Reconstruction failed: no mesh generated." << endl;
      return true;
    }

    reconstructedV = mesh.V;
    reconstructedF = mesh.F;
    reconstructedR = mesh.R;
    has_reconstructed_mesh = true;

    exportReconstructedMesh(base + "_reconstructed_quadric_only.obj", mesh);
    exportReconstructedMesh(base + "_reconstructed.obj", mesh);
    if (rcfg.export_ply) {
      exportReconstructedMeshPLY(base + "_reconstructed_quadric_only.ply", mesh);
      exportReconstructedMeshPLY(base + "_reconstructed.ply", mesh);
    }
    exportReconstructionLog(base + "_reconstruction_quadric_only_log.csv", rlog);
    if (rcfg.debug.enable_debug_report) {
      exportReconstructionDebugReport(base + "_reconstruction_debug_report.csv", rlog);
      exportProjectionFilterDebugTrianglesCSV(
          base + "_projection_filter_debug_triangles.csv", rlog);
      exportProjectionFilterDebugMeshPLY(
          base + "_projection_filter_debug_mesh.ply", rlog);
    }

    show_reconstructed_mesh(viewer);
    return true;
  }
  // G/g: Show all region boundaries
  if (key == 'G' || key == 'g') {
    g_boundary_valid = false;  // force re-extract
    show_region_boundaries(viewer);
    return true;
  }
  // H/h: Show current region pair boundary
  if (key == 'H' || key == 'h') {
    show_one_region_pair_boundary(viewer, g_current_pair_idx);
    return true;
  }
  // J/j: Next region pair
  if (key == 'J' || key == 'j') {
    if (!g_boundary_valid) extract_region_boundaries();
    if (g_region_pairs.empty()) {
      cout << "No region pairs. Press G first to extract boundaries." << endl;
      return true;
    }
    g_current_pair_idx = (g_current_pair_idx + 1) % g_region_pairs.size();
    show_one_region_pair_boundary(viewer, g_current_pair_idx);
    return true;
  }
  // K/k: Previous region pair
  if (key == 'K' || key == 'k') {
    if (!g_boundary_valid) extract_region_boundaries();
    if (g_region_pairs.empty()) {
      cout << "No region pairs. Press G first to extract boundaries." << endl;
      return true;
    }
    g_current_pair_idx = (g_current_pair_idx - 1 + g_region_pairs.size()) % g_region_pairs.size();
    show_one_region_pair_boundary(viewer, g_current_pair_idx);
    return true;
  }
  if (key == 'S' || (unsigned int)key == 83){
    if (use_quadric) {
      vector<double> errors;
      while (fabs(error - precedent_error)>0.0001){
        precedent_error = error;
        quadric_lloyd_step();
        iterations++;
        global_error_points.push_back(make_pair(iterations, error));
        cout << error << endl;
        if (vector_contains(errors, error)) {
          cout << "cycle !" << endl;
          break;
        }
        errors.push_back(error);
        MatrixXd colors = make_face_colors_from_labels(R, F.rows());
        viewer.data().set_colors(colors);
      }
      cout << "    Done" << endl;
    } else {
      vector<double> errors;
      while (fabs(error - precedent_error)>0.0001){
        precedent_error = error;
        proxy_color(R, Proxies, V,  F, Ad, metric);
        Proxies = new_proxies(R, F, V, p, metric);
        iterations += 1;
        error = global_distortion_error(R,Proxies,V,F,metric);
        cout << error << endl;

        if (vector_contains(errors,error)){
          cout<<"cycle !"<<endl;
          break;
        }

        global_error_points.push_back(make_pair(iterations,error));
        errors.push_back(error);

        igl::jet(R,true,C);
        viewer.data(0).set_colors(C);
      }
      cout << "    Done" <<endl;
    }
  }
  return false;
}



// ------------ main program ----------------

// Simple CLI argument helper
static bool has_arg(int argc, char *argv[], const string& flag) {
  for (int i = 1; i < argc; i++)
    if (string(argv[i]) == flag) return true;
  return false;
}

static string get_arg(int argc, char *argv[], const string& flag, const string& def) {
  for (int i = 1; i < argc - 1; i++)
    if (string(argv[i]) == flag) return argv[i + 1];
  return def;
}

static vector<int> parse_int_list(const string& s) {
  vector<int> values;
  string item;
  stringstream ss(s);
  while (getline(ss, item, ',')) {
    if (!item.empty()) values.push_back(stoi(item));
  }
  return values;
}

static ReconstructionConfig parse_reconstruction_config(int argc, char *argv[]) {
  ReconstructionConfig cfg;
  cfg.enable_displacement = has_arg(argc, argv, "--reconstruct-displacement");
  cfg.export_ply = !has_arg(argc, argv, "--reconstruct-no-ply");
  cfg.fallback_to_original = !has_arg(argc, argv, "--reconstruct-skip-failed");
  cfg.debug.enable_debug_report = !has_arg(argc, argv, "--reconstruct-no-debug-report");
  cfg.debug.export_region_2d_debug = has_arg(argc, argv, "--reconstruct-debug-2d");
  cfg.enable_boundary_subdivision = !has_arg(argc, argv, "--reconstruct-no-boundary-subdivision");
  cfg.enable_boundary_transition = has_arg(argc, argv, "--reconstruct-boundary-transition") &&
      !has_arg(argc, argv, "--reconstruct-no-boundary-transition");
  cfg.enable_adaptive_boundary_sampling = !has_arg(argc, argv, "--reconstruct-no-adaptive-boundary");
  cfg.enable_global_sample_spacing = !has_arg(argc, argv, "--reconstruct-local-spacing");
  cfg.enable_boundary_ring_sampling = !has_arg(argc, argv, "--reconstruct-no-boundary-ring");
  cfg.enable_surface_metric_sampling = !has_arg(argc, argv, "--reconstruct-no-surface-metric-sampling");
  cfg.enable_surface_spacing_filter = !has_arg(argc, argv, "--reconstruct-no-surface-spacing-filter");
  cfg.enable_metric_warped_triangulation = !has_arg(argc, argv, "--reconstruct-no-metric-warped-cdt");
  cfg.enable_proxy_triangle_quality_filter = !has_arg(argc, argv, "--reconstruct-no-proxy-triangle-filter");
  cfg.prefer_full_patch_delaunay = !has_arg(argc, argv, "--reconstruct-fast-grid-triangulation");
  cfg.use_cgal_cdt = !has_arg(argc, argv, "--reconstruct-no-cgal-cdt");
  if (has_arg(argc, argv, "--reconstruct-grid"))
    cfg.grid_resolution = stoi(get_arg(argc, argv, "--reconstruct-grid", "64"));
  if (has_arg(argc, argv, "--reconstruct-max-samples"))
    cfg.max_samples_per_region = stoi(
      get_arg(argc, argv, "--reconstruct-max-samples", "32768"));
  if (has_arg(argc, argv, "--reconstruct-min-lift-normal-alignment"))
    cfg.min_lift_normal_alignment = stod(
      get_arg(argc, argv, "--reconstruct-min-lift-normal-alignment", "0.0"));
  if (has_arg(argc, argv, "--reconstruct-target-spacing"))
    cfg.target_sample_spacing = stod(
      get_arg(argc, argv, "--reconstruct-target-spacing", "0"));
  if (has_arg(argc, argv, "--reconstruct-boundary-factor"))
    cfg.boundary_subdivision_spacing_factor = stod(
      get_arg(argc, argv, "--reconstruct-boundary-factor", "1.0"));
  if (has_arg(argc, argv, "--reconstruct-max-boundary-subdivisions"))
    cfg.max_boundary_subdivisions_per_edge = stoi(
      get_arg(argc, argv, "--reconstruct-max-boundary-subdivisions", "64"));
  if (has_arg(argc, argv, "--reconstruct-transition-width"))
    cfg.boundary_transition_width_factor = stod(
      get_arg(argc, argv, "--reconstruct-transition-width", "3.0"));
  if (has_arg(argc, argv, "--reconstruct-adaptive-band"))
    cfg.adaptive_boundary_band_factor = stod(
      get_arg(argc, argv, "--reconstruct-adaptive-band", "2.5"));
  if (has_arg(argc, argv, "--reconstruct-fast-boundary-band"))
    cfg.fast_triangulation_boundary_band_factor = stod(
      get_arg(argc, argv, "--reconstruct-fast-boundary-band", "1.5"));
  if (has_arg(argc, argv, "--reconstruct-boundary-ring-factor"))
    cfg.boundary_ring_spacing_factor = stod(
      get_arg(argc, argv, "--reconstruct-boundary-ring-factor", "0.9"));
  if (has_arg(argc, argv, "--reconstruct-boundary-ring-layers"))
    cfg.boundary_ring_layers = stoi(
      get_arg(argc, argv, "--reconstruct-boundary-ring-layers", "1"));
  if (has_arg(argc, argv, "--reconstruct-max-delaunay-edge-factor"))
    cfg.max_delaunay_edge_factor = stod(
      get_arg(argc, argv, "--reconstruct-max-delaunay-edge-factor", "3.0"));
  if (has_arg(argc, argv, "--reconstruct-max-triangle-edge-ratio"))
    cfg.max_triangle_edge_ratio = stod(
      get_arg(argc, argv, "--reconstruct-max-triangle-edge-ratio", "5.0"));
  if (has_arg(argc, argv, "--reconstruct-ray-distance-factor"))
    cfg.max_ray_projection_distance_factor = stod(
      get_arg(argc, argv, "--reconstruct-ray-distance-factor", "2.0"));
  if (has_arg(argc, argv, "--reconstruct-surface-spacing-factor"))
    cfg.surface_sample_spacing_factor = stod(
      get_arg(argc, argv, "--reconstruct-surface-spacing-factor", "1.0"));
  if (has_arg(argc, argv, "--reconstruct-surface-refine-factor"))
    cfg.surface_metric_refine_factor = stod(
      get_arg(argc, argv, "--reconstruct-surface-refine-factor", "1.20"));
  if (has_arg(argc, argv, "--reconstruct-surface-min-spacing-factor"))
    cfg.surface_min_sample_spacing_factor = stod(
      get_arg(argc, argv, "--reconstruct-surface-min-spacing-factor", "0.30"));
  if (has_arg(argc, argv, "--reconstruct-surface-refine-passes"))
    cfg.surface_metric_refinement_passes = stoi(
      get_arg(argc, argv, "--reconstruct-surface-refine-passes", "2"));
  if (has_arg(argc, argv, "--reconstruct-metric-warp-anisotropy-clamp"))
    cfg.metric_warp_anisotropy_clamp = stod(
      get_arg(argc, argv, "--reconstruct-metric-warp-anisotropy-clamp", "8.0"));
  if (has_arg(argc, argv, "--reconstruct-proxy-triangle-edge-ratio"))
    cfg.proxy_triangle_edge_ratio = stod(
      get_arg(argc, argv, "--reconstruct-proxy-triangle-edge-ratio", "4.0"));
  if (has_arg(argc, argv, "--reconstruct-projection-mode")) {
    string mode = get_arg(argc, argv, "--reconstruct-projection-mode", "quadric");
    if (mode == "ray" || mode == "original-ray" || mode == "original_mesh_ray") {
      cfg.interior_projection_mode = InteriorProjectionMode::OriginalMeshRayCast;
    } else if (mode == "ray-quadric-fallback" || mode == "ray_quadric_fallback") {
      cfg.interior_projection_mode = InteriorProjectionMode::RayCastThenQuadricFallback;
    } else {
      cfg.interior_projection_mode = InteriorProjectionMode::QuadricLift;
    }
  }
  if (has_arg(argc, argv, "--reconstruct-max-regions"))
    cfg.max_regions = stoi(get_arg(argc, argv, "--reconstruct-max-regions", "-1"));
  if (has_arg(argc, argv, "--reconstruct-debug-regions"))
    cfg.debug.debug_region_ids = parse_int_list(
      get_arg(argc, argv, "--reconstruct-debug-regions", ""));
  if (has_arg(argc, argv, "--reconstruct-debug-max-regions"))
    cfg.debug.max_debug_regions = stoi(
      get_arg(argc, argv, "--reconstruct-debug-max-regions", "10"));
  return cfg;
}

static void draw_vsa_mode_controls(igl::opengl::glfw::Viewer& viewer) {
  if (!ImGui::CollapsingHeader("VSA Modes", ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }

  auto command_button = [&](const char* label, unsigned char key, float width = -1.0f) {
    if (ImGui::Button(label, ImVec2(width, 0))) {
      key_down(viewer, key, 0);
    }
  };
  auto same_line_button = [&](const char* label, unsigned char key, float width) {
    ImGui::SameLine();
    command_button(label, key, width);
  };

  ImGui::Text("Current: %s", viewer_mode_name(g_viewer_work_mode));
  float w = ImGui::GetContentRegionAvail().x;
  float pstyle = ImGui::GetStyle().FramePadding.x;
  if (ImGui::Button("Quadric VSA", ImVec2((w - pstyle) / 2.0f, 0))) {
    enter_quadric_vsa_mode(viewer);
  }
  ImGui::SameLine(0, pstyle);
  if (ImGui::Button("Spline Surface", ImVec2((w - pstyle) / 2.0f, 0))) {
    enter_spline_surface_mode(viewer);
  }

  ImGui::Separator();
  ImGui::InputInt("Spline region", &g_spline_region_id);
  if (p > 0) {
    g_spline_region_id = max(0, min(g_spline_region_id, p - 1));
  }
  ImGui::SliderInt("Sample U", &g_spline_sample_u, 4, 96);
  ImGui::SliderInt("Sample V", &g_spline_sample_v, 4, 96);
  ImGui::SliderInt("Control U", &g_spline_control_count_u, 4, 12);
  ImGui::SliderInt("Control V", &g_spline_control_count_v, 4, 12);
  ImGui::InputDouble("Fit fairness", &g_spline_fit_fairness_weight, 0.0, 0.0, "%.2e");
  ImGui::InputDouble("Fit initial", &g_spline_fit_initial_weight, 0.0, 0.0, "%.2e");
  ImGui::InputDouble("Trim boundary fit weight", &g_trimmed_boundary_fit_weight, 0.0, 0.0, "%.2e");
  ImGui::Checkbox("Harmonic boundary correction", &g_trimmed_harmonic_boundary_correction);
  ImGui::Checkbox("Snap output boundary", &g_trimmed_snap_output_boundary);
  ImGui::SliderInt("Trim grid U", &g_trimmed_spline_grid_u, 8, 128);
  ImGui::SliderInt("Trim grid V", &g_trimmed_spline_grid_v, 8, 128);

  int backend = static_cast<int>(g_spline_render_backend);
  const char* backends[] = {
      "libigl sampled mesh",
      "Polyscope"};
  if (ImGui::Combo("Spline renderer", &backend, backends, 2)) {
    g_spline_render_backend = static_cast<SplineRenderBackend>(backend);
  }
  int trim_mode = static_cast<int>(g_trim_render_debug_mode);
  const char* trim_modes[] = {
      "Standard trimmed only",
      "Original region only",
      "Full untrimmed spline",
      "Standard asset sampled mesh",
      "Experimental ABC preview",
      "Experimental ABC ribbon",
      "Experimental ABC preview+ribbon",
      "Standard trimmed + original"};
  if (ImGui::Combo("Trim render mode", &trim_mode, trim_modes, 8)) {
    g_trim_render_debug_mode = static_cast<TrimRenderDebugMode>(trim_mode);
    if (has_spline_surface_mesh) apply_trim_render_debug_mode(viewer);
  }
  if (ImGui::Button("Apply trim render mode", ImVec2(-1, 0))) {
    apply_trim_render_debug_mode(viewer);
  }

  if (ImGui::Button("Compute selected spline region", ImVec2(-1, 0))) {
    compute_and_show_spline_surface_for_region(viewer, g_spline_region_id);
  }
  if (ImGui::Button("Fit selected region B-spline surface", ImVec2(-1, 0))) {
    fit_and_show_bspline_surface_for_region(viewer, g_spline_region_id);
  }
  if (ImGui::Button("Fit selected region trimmed B-spline", ImVec2(-1, 0))) {
    fit_and_show_trimmed_bspline_surface_for_region(viewer, g_spline_region_id);
  }
  ImGui::InputText(
      "Trimmed pipeline output",
      g_trimmed_pipeline_output_dir,
      sizeof(g_trimmed_pipeline_output_dir));
  if (ImGui::Button("Fast standard trimmed pipeline", ImVec2(-1, 0))) {
    run_and_show_trimmed_bspline_pipeline_for_region(viewer, g_spline_region_id, true);
  }
  if (ImGui::Button("Run standard trimmed B-spline pipeline", ImVec2(-1, 0))) {
    run_and_show_trimmed_bspline_pipeline_for_region(viewer, g_spline_region_id, false);
  }
  if (ImGui::Button("Fast all trimmed regions", ImVec2(-1, 0))) {
    run_and_show_all_trimmed_bspline_regions(viewer, true);
  }
  if (ImGui::Button("Run all trimmed regions", ImVec2(-1, 0))) {
    run_and_show_all_trimmed_bspline_regions(viewer, false);
  }
  ImGui::InputText(
      "Region snapshot prefix",
      g_region_snapshot_prefix,
      sizeof(g_region_snapshot_prefix));
  if (ImGui::Button("Save selected region snapshot", ImVec2(-1, 0))) {
    string prefix(g_region_snapshot_prefix);
    if (save_region_snapshot(g_spline_region_id, prefix)) {
      render_region_snapshot_mesh(viewer, g_spline_region_id, prefix);
    }
  }
  if (ImGui::Button("Render saved region snapshot", ImVec2(-1, 0))) {
    render_region_snapshot_mesh(
        viewer,
        g_spline_region_id,
        string(g_region_snapshot_prefix));
  }
  if (ImGui::Button("Open current spline in Polyscope", ImVec2(-1, 0))) {
    launch_external_polyscope_spline_viewer(g_last_spline_debug_prefix);
  }
  if (ImGui::Button("Render standard asset mesh only", ImVec2(-1, 0))) {
    render_current_asset_mesh_in_libigl(viewer);
  }
  if (ImGui::Button("Render experimental ABC preview", ImVec2(-1, 0))) {
    render_current_abc_preview_in_libigl(viewer);
  }
  if (ImGui::Button("Render experimental ABC ribbon", ImVec2(-1, 0))) {
    render_current_abc_ribbon_in_libigl(viewer);
  }
  if (ImGui::Button("Render experimental ABC preview + ribbon", ImVec2(-1, 0))) {
    render_current_abc_preview_and_ribbon_in_libigl(viewer);
  }
  if (ImGui::Button("Open asset mesh in Polyscope", ImVec2(-1, 0))) {
    launch_external_polyscope_spline_viewer(g_last_spline_debug_prefix);
  }
  ImGui::InputText("Polyscope exe", g_polyscope_viewer_exe, sizeof(g_polyscope_viewer_exe));
  ImGui::Text("Polyscope viewer reads exported OBJ debug files.");
  if (has_spline_surface_mesh) {
    ImGui::Text("Spline mesh: %d V / %d F",
                (int)splineSurfaceV.rows(),
                (int)splineSurfaceF.rows());
  }
  ImGui::Separator();
  ImGui::TextWrapped("%s", g_spline_status.c_str());

  if (ImGui::CollapsingHeader("Quadric VSA Commands", ImGuiTreeNodeFlags_DefaultOpen)) {
    float full = ImGui::GetContentRegionAvail().x;
    float half = (full - pstyle) / 2.0f;
    float third = (full - 2.0f * pstyle) / 3.0f;

    ImGui::Text("Iteration");
    command_button("One iter [3]", '3', third);
    same_line_button("10 iter [8]", '8', third);
    same_line_button("100 iter [9]", '9', third);
    command_button("Converge [S]", 'S', half);
    same_line_button("Toggle quadric/plane [Q]", 'Q', half);

    ImGui::Text("Pipeline");
    command_button("Insert proxy [I]", 'I', third);
    same_line_button("Progressive pipeline [N]", 'N', third);
    same_line_button("Merge pass [Y]", 'Y', third);
    command_button("Final feature check [L]", 'L', half);
    same_line_button("Feature barrier [F]", 'F', half);

    ImGui::Text("Geometry");
    command_button("Reconstruct [R]", 'R', third);
    same_line_button("Smooth boundary [W]", 'W', third);
    same_line_button("Project vertices [X]", 'X', third);
    command_button("Error heatmap [V]", 'V', half);
    same_line_button("Stored error heatmap [E]", 'E', half);

    ImGui::Text("Views");
    command_button("Original [O]", 'O', third);
    same_line_button("Current labels [C]", 'C', third);
    same_line_button("Draw proxies [7]", '7', third);
    command_button("Before smoothing [B]", 'B', third);
    same_line_button("After smoothing [A]", 'A', third);
    same_line_button("Smoothing diff [D]", 'D', third);
    string region_label_button =
        g_show_region_id_labels ? "Hide region IDs [Z]" : "Show region IDs [Z]";
    command_button(region_label_button.c_str(), 'Z', full);

    ImGui::Text("Boundaries");
    command_button("All boundaries [G]", 'G', half);
    same_line_button("Current pair [H]", 'H', half);
    command_button("Next pair [J]", 'J', half);
    same_line_button("Previous pair [K]", 'K', half);

    ImGui::Text("Session");
    ImGui::InputText("Snapshot file", g_quadric_snapshot_file, sizeof(g_quadric_snapshot_file));
    if (ImGui::Button("Save segmentation result", ImVec2(half, 0))) {
      save_quadric_vsa_segmentation_result(string(g_quadric_snapshot_file));
    }
    ImGui::SameLine();
    if (ImGui::Button("Load and render result", ImVec2(half, 0))) {
      load_and_render_quadric_vsa_segmentation_result(
          viewer, string(g_quadric_snapshot_file));
    }
    command_button("Save default [T]", 'T', half);
    same_line_button("Load default [U]", 'U', half);
    command_button("Metric mode [M]", 'M', third);
    same_line_button("Omega + [+]", '+', third);
    same_line_button("Omega - [-]", '-', third);
  }
}

int main(int argc, char *argv[])
{
  // Progressive mode: detect --progressive flag
  if (has_arg(argc, argv, "--progressive")) {
    string proxy_str = get_arg(argc, argv, "--proxy-type", "plane");
    ProxyType ptype = (proxy_str == "quadric") ? QUADRIC_PROXY : PLANE_PROXY;
    int init_p = stoi(get_arg(argc, argv, "--init-proxies", "3"));
    int target_p = stoi(get_arg(argc, argv, "--target-proxies", "50"));
    double err_thresh = stod(get_arg(argc, argv, "--error-threshold", "0"));
    int lloyd_iter = stoi(get_arg(argc, argv, "--max-iter", "50"));
    unsigned int seed = (unsigned int)stoul(get_arg(argc, argv, "--seed", "42"));
    string model = (argc >= 2 && string(argv[1])[0] != '-') ? argv[1] : "smooth_bunny";

    // Proxy validity config
    ProxyValidityConfig validity_cfg;
    bool check_validity = has_arg(argc, argv, "--check-proxy-validity");
    bool classify_proxies = has_arg(argc, argv, "--classify-proxies");
    if (check_validity) {
      validity_cfg.enable_basic = true;
    }

    double classify_eps = 0.1;
    if (has_arg(argc, argv, "--classification-eps"))
      classify_eps = stod(get_arg(argc, argv, "--classification-eps", "0.1"));

    bool enable_merge = has_arg(argc, argv, "--enable-merge");
    bool enable_feature_barrier_cli = has_arg(argc, argv, "--feature-barrier");
    if (has_arg(argc, argv, "--feature-angle"))
      g_feature_angle_threshold = stod(get_arg(argc, argv, "--feature-angle", "30"));
    if (enable_feature_barrier_cli)
      g_feature_barrier_enabled = true;

    // Boundary smoothing config
    bool enable_smooth = has_arg(argc, argv, "--smooth-boundary");
    SmoothConfig smooth_cfg;
    if (has_arg(argc, argv, "--smooth-ring"))
      smooth_cfg.ring = stoi(get_arg(argc, argv, "--smooth-ring", "2"));
    if (has_arg(argc, argv, "--smooth-lambda"))
      smooth_cfg.lambda = stod(get_arg(argc, argv, "--smooth-lambda", "1.0"));

    // Projection config
    bool enable_projection = has_arg(argc, argv, "--project-output");
    ProjectionConfig proj_cfg;
    bool enable_reconstruction = has_arg(argc, argv, "--reconstruct-output");
    ReconstructionConfig recon_cfg = parse_reconstruction_config(argc, argv);

    // Validity-guided insertion config
    bool validity_guided = has_arg(argc, argv, "--validity-guided-insertion");
    if (validity_guided) {
      validity_cfg.enable_basic = true;
      validity_cfg.enable_degeneracy = true;
      validity_cfg.enable_classification = true;
      validity_cfg.enable_two_sheet = true;
    }
    int max_validity_split_attempts = stoi(
        get_arg(argc, argv, "--max-validity-split-attempts", "20"));
    int min_faces_to_split = stoi(
        get_arg(argc, argv, "--min-faces-to-split", "4"));
    bool export_validity_each_step = has_arg(argc, argv, "--export-validity-each-step");

    cout << "=== Progressive VSA ===" << endl;
    cout << "  model:           " << model << endl;
    cout << "  proxy_type:      " << (ptype == QUADRIC_PROXY ? "quadric" : "plane") << endl;
    cout << "  init_proxies:    " << init_p << endl;
    cout << "  target_proxies:  " << target_p << endl;
    cout << "  error_threshold: " << err_thresh << endl;
    cout << "  lloyd_iter:      " << lloyd_iter << endl;
    cout << "  seed:            " << seed << endl;
    if (check_validity) {
      cout << "  validity_check:  basic (Layer 1)" << endl;
    }
    if (validity_guided) {
      cout << "  validity_guided: enabled (all 4 layers)" << endl;
      cout << "  max_validity_split_attempts: " << max_validity_split_attempts << endl;
      cout << "  min_faces_to_split: " << min_faces_to_split << endl;
    }
    if (classify_proxies) {
      cout << "  classify:        enabled (eps=" << classify_eps << ")" << endl;
    }
    if (enable_merge) {
      cout << "  merge:           enabled" << endl;
    }
    if (enable_feature_barrier_cli) {
      cout << "  feature_barrier: enabled angle=" << g_feature_angle_threshold << endl;
    }
    if (enable_smooth) {
      cout << "  smooth_boundary: ring=" << smooth_cfg.ring
           << " lambda=" << smooth_cfg.lambda << endl;
    }
    if (enable_projection) {
      cout << "  projection:      enabled" << endl;
    }
    if (enable_reconstruction) {
      cout << "  reconstruction:  enabled"
           << " grid=" << recon_cfg.grid_resolution
           << " displacement=" << (recon_cfg.enable_displacement ? "true" : "false")
           << " fallback=" << (recon_cfg.fallback_to_original ? "true" : "false")
           << " fast_band=" << recon_cfg.fast_triangulation_boundary_band_factor
           << " global_spacing=" << (recon_cfg.enable_global_sample_spacing ? "true" : "false")
           << " boundary_ring=" << (recon_cfg.enable_boundary_ring_sampling ? "true" : "false")
           << " full_delaunay=" << (recon_cfg.prefer_full_patch_delaunay ? "true" : "false")
           << " max_edge_factor=" << recon_cfg.max_delaunay_edge_factor
           << " max_edge_ratio=" << recon_cfg.max_triangle_edge_ratio
           << " lift_align=" << recon_cfg.min_lift_normal_alignment
           << " debug_report=" << (recon_cfg.debug.enable_debug_report ? "true" : "false")
           << " debug_2d=" << (recon_cfg.debug.export_region_2d_debug ? "true" : "false")
           << endl;
    }
    if (export_validity_each_step) {
      cout << "  export_validity_each_step: enabled" << endl;
    }

    MatrixXi R_out;
    vector<IterationStats> stats;
    vector<InsertionStep> ins_log;
    vector<MergeStep> merge_log;
    vector<SmoothLogEntry> smooth_log;
    run_vsa_progressive(model, init_p, ptype, target_p, err_thresh,
                        lloyd_iter, seed, R_out, stats, ins_log,
                        merge_log, smooth_log, enable_merge,
                        enable_smooth, smooth_cfg, validity_cfg,
                        classify_proxies, classify_eps,
                        enable_projection, proj_cfg,
                        validity_guided, max_validity_split_attempts,
                        min_faces_to_split, export_validity_each_step,
                        enable_reconstruction, recon_cfg);
    return 0;
  }

  // Batch mode: detect --proxy-type flag
  if (has_arg(argc, argv, "--proxy-type")) {
    string proxy_str = get_arg(argc, argv, "--proxy-type", "plane");
    ProxyType ptype = (proxy_str == "quadric") ? QUADRIC_PROXY : PLANE_PROXY;
    int target_p = stoi(get_arg(argc, argv, "--target-proxies", "50"));
    int max_iter = stoi(get_arg(argc, argv, "--max-iter", "50"));
    unsigned int seed = (unsigned int)stoul(get_arg(argc, argv, "--seed", "42"));
    string model = (argc >= 2 && string(argv[1])[0] != '-') ? argv[1] : "smooth_bunny";
    bool enable_reconstruction = has_arg(argc, argv, "--reconstruct-output");
    ReconstructionConfig recon_cfg = parse_reconstruction_config(argc, argv);
    bool enable_feature_barrier_cli = has_arg(argc, argv, "--feature-barrier");
    if (has_arg(argc, argv, "--feature-angle"))
      g_feature_angle_threshold = stod(get_arg(argc, argv, "--feature-angle", "30"));
    if (enable_feature_barrier_cli)
      g_feature_barrier_enabled = true;

    cout << "=== Batch VSA ===" << endl;
    cout << "  model:       " << model << endl;
    cout << "  proxy_type:  " << (ptype == QUADRIC_PROXY ? "quadric" : "plane") << endl;
    cout << "  num_proxies: " << target_p << endl;
    cout << "  max_iter:    " << max_iter << endl;
    cout << "  seed:        " << seed << endl;
    if (enable_reconstruction) {
      cout << "  reconstruction: enabled"
           << " grid=" << recon_cfg.grid_resolution
           << " displacement=" << (recon_cfg.enable_displacement ? "true" : "false")
           << " fallback=" << (recon_cfg.fallback_to_original ? "true" : "false")
           << " fast_band=" << recon_cfg.fast_triangulation_boundary_band_factor
           << " global_spacing=" << (recon_cfg.enable_global_sample_spacing ? "true" : "false")
           << " boundary_ring=" << (recon_cfg.enable_boundary_ring_sampling ? "true" : "false")
           << " full_delaunay=" << (recon_cfg.prefer_full_patch_delaunay ? "true" : "false")
           << " max_edge_factor=" << recon_cfg.max_delaunay_edge_factor
           << " max_edge_ratio=" << recon_cfg.max_triangle_edge_ratio
           << " lift_align=" << recon_cfg.min_lift_normal_alignment
           << " debug_report=" << (recon_cfg.debug.enable_debug_report ? "true" : "false")
           << " debug_2d=" << (recon_cfg.debug.export_region_2d_debug ? "true" : "false")
           << endl;
    }
    if (enable_feature_barrier_cli) {
      cout << "  feature_barrier: enabled angle=" << g_feature_angle_threshold << endl;
    }

    MatrixXi R_out;
    vector<IterationStats> stats;
    run_vsa_batch(model, target_p, ptype, max_iter, seed, R_out, stats,
                  enable_reconstruction, recon_cfg);
    return 0;
  }

  // srand ( time(NULL) );
  p = 180;
  // metric 和 omega 已在全局声明处设置默认值 (L21_METRIC, 0.5)
  treshold = 0.4;
  string file = "data/sphere_large.off";
  if (argc>=2) {
    string w = argv[1];
    file = "data/" + w + ".off";
  }
  if (argc>=3) {
    p = atoi(argv[2]);
  }
  if (argc>=4) {
    treshold = atof(argv[3]);
  }

  // 解析 metric mode (argv[5])：l2 / l21 / hybrid
  if (argc>=6) {
    string m = argv[5];
    if (m == "l2") {
      metric = L2_METRIC;
    } else if (m == "l21") {
      metric = L21_METRIC;
    } else if (m == "hybrid") {
      metric = HYBRID_METRIC;
    } else {
      cout << "Unknown metric mode '" << m << "', using default L2,1" << endl;
    }
  }

  // 解析 omega (argv[6])，仅 hybrid 模式时有效
  if (argc>=7) {
    omega = atof(argv[6]);
  }

  cout << "Config: p=" << p << " treshold=" << treshold
       << " metric=" << (metric == L2_METRIC ? "L2" : metric == L21_METRIC ? "L21" : "HYBRID")
       << " omega=" << omega << endl;

  igl::readOFF(file, V, F); // Load an input mesh in OFF format
  HalfedgeBuilder* builder=new HalfedgeBuilder();  
  HalfedgeDS he2 = builder->createMesh(V.rows(), F); 
  he = &he2;
  //  print the number of mesh elements
  cout << "Vertices: " << V.rows() << endl;
  cout << "Faces:    " << F.rows() << endl;

  // Face adjacency
  cout << "Computing face constants..." << endl;
  Ad = face_adjacency(F,V.rows());
  initialize_normals_areas(F,V);
  cout << "   ...done" << endl;

  //coloring 
  // Partition_faces.setZero(F.rows(),1);
  // MatrixXd C;
  // tcolor(Partition_faces);
  // igl::jet(Partition_faces,true,C);

  // coloring adjacency 
  // MatrixXd Cf;
  // fcolor(Cf,Ad);
  // MatrixXd C;
  // igl::jet(Cf,true,C);

  // coloring distance 
  // MatrixXd Cf;
  // distance_color(Cf,F,V,0);
  // MatrixXd C;
  // igl::jet(Cf,true,C);

  // coloring proxies
  if (argc>=5) {
    string w = argv[4];
    if (w=="f") {
      cout << "furthest init" <<endl;
      initial_partition2(p, R, V, F, Ad, metric);
    }
    else {
      cout << "random init" <<endl;
      // 旧行为兼容：非 "f" 的 init mode 曾强制 L2，现不再覆盖 metric
      initial_partition(p, R, V, F, Ad, metric);
    }
  }
  else {
    cout << "random init" <<endl;
    initial_partition(p, R, V, F, Ad, metric);
  }
  cout << "... done" <<endl;
  cout << "... done" <<endl;

  Proxies = new_proxies(R, F, V, p, metric);
  iterations = 1;
  error = global_distortion_error(R,Proxies,V,F,metric);
  precedent_error = error - 1 ; 
  global_error_points.push_back(make_pair(iterations,error));
  igl::jet(R,true,C);
  igl::opengl::glfw::Viewer viewer; // create the 3d viewer
  igl::opengl::glfw::imgui::ImGuiPlugin imgui_plugin;
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  viewer.plugins.push_back(&imgui_plugin);
  imgui_plugin.widgets.push_back(&menu);
  menu.callback_draw_viewer_window = [&viewer, &menu]() {
    ImGuiIO& io = ImGui::GetIO();
    float panel_width = 390.0f;
    float panel_height = std::min(680.0f, std::max(320.0f, io.DisplaySize.y - 16.0f));
    ImGui::SetNextWindowPos(ImVec2(8.0f, 8.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panel_width, panel_height), ImGuiCond_Always);
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoSavedSettings;
    ImGui::Begin("VSA Controls", nullptr, flags);
    ImGui::BeginChild(
        "VSA Controls Scroll",
        ImVec2(0.0f, 0.0f),
        false,
        ImGuiWindowFlags_HorizontalScrollbar);
    draw_vsa_mode_controls(viewer);
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Viewer Settings")) {
      menu.draw_viewer_menu();
    }
    ImGui::EndChild();
    ImGui::End();
  };

  //showing normals
  // viewer.append_mesh();
  // for (int j=0;j<F.rows();j++) {
  //   Vector3d center = triangle_center(F.row(j),V);
  //   Vector3d norm = triangle_normal(F.row(j),V);
  //   viewer.data(0).add_edges(
  //       center.transpose(),
  //       center.transpose()+norm.transpose()/10.0,
  //       Eigen::RowVector3d(1, 0, 0));
  // }

  viewer.callback_key_down = &key_down; // for dealing with keyboard events
  viewer.data().set_mesh(V, F); // load a face-based representation of the input 3d shape
  // viewer.data().set_colors(C);
  color_scheme(viewer, V, F);
  viewer.launch(); // run the editor


  cout<<"\n_______Erreurs par itération_______\n"<<endl;
  pair<int,double> item;
  for (int i=0 ; i<global_error_points.size() ; i++){
    item = global_error_points[i];
    cout<<item.second<<endl;
  }
}

