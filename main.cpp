#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/jet.h>
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
#include <ctime>
#include <string>
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

VectorXd face_error_values;
bool has_face_error_values = false;

// ---- Quadric mode state ----
vector<QuadricProxy> QP;
bool use_quadric = false;

// ---- Feature barrier state ----
double g_feature_angle_threshold = 30.0;  // degrees

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

// ---- Show segmentation view ----
void show_segmentation(igl::opengl::glfw::Viewer& viewer,
                       const MatrixXd& displayV, const MatrixXi& displayF,
                       const MatrixXi& labels, const string& name) {
  viewer.data().clear();
  viewer.data().set_mesh(displayV, displayF);
  MatrixXd colors = make_face_colors_from_labels(labels, displayF.rows());
  viewer.data().set_colors(colors);
  viewer.data().show_lines = true;
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

// ---- Helper: RowVectorXd → Vector3d ----
static inline Vector3d v3(const Eigen::RowVectorXd& r) {
  return Vector3d(r(0), r(1), r(2));
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
      if (nb < 0) continue;
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
      if (nb < 0 || R_new(nb, 0) != -1) continue;
      Vector3i f = F.row(nb);
      double d = QP[prox].triangle_error(v3(V.row(f(0))), v3(V.row(f(1))), v3(V.row(f(2))));
      q.push(make_pair(-d, nb + m * prox));
    }
  }

  // Assign any remaining -1 faces to a neighbor's proxy
  for (int i = 0; i < m; i++) {
    if (R_new(i, 0) >= 0) continue;
    for (int k = 0; k < 3; k++) {
      int nb = Ad(i, k);
      if (nb >= 0 && R_new(nb, 0) >= 0) {
        R_new(i, 0) = R_new(nb, 0);
        break;
      }
    }
  }

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
  // N: Progressive auto-insert until error threshold (validity-guided in quadric mode)
  if (key == 'N' || key == 'n') {
    double err_thresh = 2e-5;
    int max_p = min((int)F.rows() / 3, 30);
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
      if (max_ne < err_thresh && invalid_count == 0) {
        cout << "  Stopped: threshold met, no invalid proxies" << endl;
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
    bool pipeline_enable_smoothing = true;

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

    show_segmentation(viewer, V, F, R, "Pipeline K=" + to_string(p));
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
    vector<SmoothLogEntry> slog;
    ProxyType pt = use_quadric ? QUADRIC_PROXY : PLANE_PROXY;
    smooth_boundaries(R, F, V, Ad, p, pt, QP, Proxies, metric, sc, slog);
    R_after_smoothing = R;
    has_after_smoothing = true;
    // Refit proxies after smoothing
    if (use_quadric) {
      for (int j = 0; j < p; j++)
        QP[j] = fit_quadric_region(R, j, F, V);
    } else {
      Proxies = new_proxies(R, F, V, p, metric);
    }
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
    if (enable_smooth) {
      cout << "  smooth_boundary: ring=" << smooth_cfg.ring
           << " lambda=" << smooth_cfg.lambda << endl;
    }
    if (enable_projection) {
      cout << "  projection:      enabled" << endl;
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
                        min_faces_to_split, export_validity_each_step);
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

    cout << "=== Batch VSA ===" << endl;
    cout << "  model:       " << model << endl;
    cout << "  proxy_type:  " << (ptype == QUADRIC_PROXY ? "quadric" : "plane") << endl;
    cout << "  num_proxies: " << target_p << endl;
    cout << "  max_iter:    " << max_iter << endl;
    cout << "  seed:        " << seed << endl;

    MatrixXi R_out;
    vector<IterationStats> stats;
    run_vsa_batch(model, target_p, ptype, max_iter, seed, R_out, stats);
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

