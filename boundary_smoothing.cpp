// Windows headers must come before Eigen to avoid byte/std::byte conflict
#include <igl/opengl/glfw/Viewer.h>
#include "boundary_smoothing.h"
#include "maxflow.h"
#include "feature_barrier.h"
#include "distance.h"
#include <queue>
#include <set>
#include <fstream>
#include <cmath>
#include <algorithm>

static inline Vector3d v3(const Eigen::RowVectorXd& r) {
    return Vector3d(r(0), r(1), r(2));
}

static const double INF_CAP = 1e12;
static const double EPS_NORM = 1e-15;

// Edge length between two vertices
static double edge_length(const MatrixXd& V, int v0, int v1) {
    return (v3(V.row(v0)) - v3(V.row(v1))).norm();
}

// Average edge length of the entire mesh
static double avg_edge_length(const MatrixXi& F, const MatrixXd& V) {
    double total = 0;
    int count = 0;
    for (int i = 0; i < F.rows(); i++) {
        total += edge_length(V, F(i,0), F(i,1));
        total += edge_length(V, F(i,1), F(i,2));
        total += edge_length(V, F(i,2), F(i,0));
        count += 3;
    }
    return (count > 0) ? total / count : 1.0;
}

// Compute shared edge length between two adjacent faces
static double shared_edge_length(int fi, int fj,
                                  const MatrixXi& F, const MatrixXd& V) {
    // Find the two shared vertices
    vector<int> shared;
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            if (F(fi, a) == F(fj, b)) {
                shared.push_back(F(fi, a));
                break;
            }
        }
        if ((int)shared.size() == 2) break;
    }
    if ((int)shared.size() != 2) return 0;
    return edge_length(V, shared[0], shared[1]);
}

// Triangle error for a face against a quadric proxy
static double face_error_quadric(int fi, const QuadricProxy& proxy,
                                  const MatrixXi& F, const MatrixXd& V) {
    Vector3i f = F.row(fi);
    return proxy.triangle_error(v3(V.row(f(0))), v3(V.row(f(1))), v3(V.row(f(2))));
}

// Triangle error for a face against a plane proxy
static double face_error_plane(int fi, const MatrixXd& Proxies,
                                int num_proxies, int proxy_id,
                                const MatrixXi& F, const MatrixXd& V,
                                MetricMode metric) {
    if (proxy_id < 0 || proxy_id >= num_proxies) return INF_CAP;
    Vector3d center = v3(Proxies.row(proxy_id));
    Vector3d normal = v3(Proxies.row(num_proxies + proxy_id));
    Vector3i f = F.row(fi);
    Vector3d v0 = v3(V.row(f(0)));
    Vector3d v1 = v3(V.row(f(1)));
    Vector3d v2 = v3(V.row(f(2)));
    Vector3d tri_center = (v0 + v1 + v2) / 3.0;
    if (metric == L2_METRIC) {
        return (tri_center - center).squaredNorm();
    }
    Vector3d n = (v1 - v0).cross(v2 - v0);
    if (n.norm() > 1e-12) n.normalize();
    else return 0;
    return pow(n.dot(normal), 2);
}

// Compute total boundary length between two regions
double compute_pair_boundary_length(const MatrixXi& R, const MatrixXi& F,
                                     const MatrixXd& V, const MatrixXi& Ad,
                                     int ri, int rj) {
    double len = 0;
    int m = R.rows();
    for (int fi = 0; fi < m; fi++) {
        if (R(fi, 0) != ri) continue;
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb <= 0 || nb >= m) continue;
            if (R(nb, 0) == rj) {
                len += shared_edge_length(fi, nb, F, V);
            }
        }
    }
    return len;
}

// Build region adjacency pairs
static set<pair<int,int>> build_region_pairs(const MatrixXi& R,
                                              const MatrixXi& F,
                                              const MatrixXi& Ad,
                                              int num_proxies) {
    set<pair<int,int>> pairs;
    int m = R.rows();
    for (int fi = 0; fi < m; fi++) {
        int ri = R(fi, 0);
        if (ri < 0 || ri >= num_proxies) continue;
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb <= 0 || nb >= m) continue;
            int rj = R(nb, 0);
            if (rj < 0 || rj >= num_proxies || rj == ri) continue;
            if (is_feature_barrier(fi, k, F, Ad)) continue;
            pairs.insert(make_pair(min(ri, rj), max(ri, rj)));
        }
    }
    return pairs;
}

// Find boundary faces between two regions
static set<int> find_boundary_faces(const MatrixXi& R, const MatrixXi& Ad,
                                     int ri, int rj) {
    set<int> faces;
    int m = R.rows();
    for (int fi = 0; fi < m; fi++) {
        if (R(fi, 0) != ri && R(fi, 0) != rj) continue;
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb <= 0 || nb >= m) continue;
            if ((R(fi, 0) == ri && R(nb, 0) == rj) ||
                (R(fi, 0) == rj && R(nb, 0) == ri)) {
                faces.insert(fi);
                faces.insert(nb);
            }
        }
    }
    return faces;
}

// k-ring expansion from a set of seed faces
static set<int> k_ring_expand(const set<int>& seeds, const MatrixXi& Ad,
                               int ring, int m) {
    set<int> result = seeds;
    vector<int> frontier(seeds.begin(), seeds.end());
    for (int r = 0; r < ring; r++) {
        vector<int> next;
        for (int fi : frontier) {
            for (int k = 0; k < 3; k++) {
                int nb = Ad(fi, k);
                if (nb <= 0 || nb >= m) continue;
                if (result.count(nb)) continue;
                result.insert(nb);
                next.push_back(nb);
            }
        }
        frontier = next;
    }
    return result;
}

// Smooth a single region pair using graph cut
static SmoothLogEntry smooth_pair(MatrixXi& R, const MatrixXi& F,
                                   const MatrixXd& V, const MatrixXi& Ad,
                                   int ri, int rj, int num_proxies,
                                   ProxyType proxy_type,
                                   const vector<QuadricProxy>& QP,
                                   const MatrixXd& Proxies, MetricMode metric,
                                   const SmoothConfig& cfg,
                                   double avg_el) {
    SmoothLogEntry log;
    log.region_i = ri;
    log.region_j = rj;
    log.relabeled_count = 0;
    int m = R.rows();

    // Before boundary length
    log.boundary_length_before = compute_pair_boundary_length(R, F, V, Ad, ri, rj);

    // Step A: Find boundary faces
    set<int> boundary = find_boundary_faces(R, Ad, ri, rj);
    if (boundary.empty()) {
        log.boundary_length_after = log.boundary_length_before;
        return log;
    }

    // Step B: k-ring expansion → fuzzy region Vf
    set<int> Vf = k_ring_expand(boundary, Ad, cfg.ring, m);

    // Filter Vf: only keep faces belonging to ri or rj, or adjacent to them
    set<int> Vf_filtered;
    for (int fi : Vf) {
        int reg = R(fi, 0);
        if (reg == ri || reg == rj) {
            Vf_filtered.insert(fi);
        } else {
            // Check if adjacent to ri or rj face
            for (int k = 0; k < 3; k++) {
                int nb = Ad(fi, k);
                if (nb > 0 && nb < m && (R(nb, 0) == ri || R(nb, 0) == rj)) {
                    Vf_filtered.insert(fi);
                    break;
                }
            }
        }
    }
    Vf = Vf_filtered;

    // Step C: Separate V0, V1, and Vf_proper
    set<int> V0, V1, Vf_proper;
    for (int fi : Vf) {
        int reg = R(fi, 0);
        if (reg == ri) V0.insert(fi);
        else if (reg == rj) V1.insert(fi);
        else Vf_proper.insert(fi);
    }
    // Vf_proper faces that belong to neither ri nor rj are also in fuzzy region
    // They get unary terms based on error

    log.fuzzy_count = (int)Vf.size();
    log.v0_count = (int)V0.size();
    log.v1_count = (int)V1.size();

    if (Vf.empty()) {
        log.boundary_length_after = log.boundary_length_before;
        return log;
    }

    // Map face index to graph node index
    map<int, int> face_to_node;
    vector<int> node_to_face;
    for (int fi : Vf) {
        int nid = (int)node_to_face.size();
        face_to_node[fi] = nid;
        node_to_face.push_back(fi);
    }

    int N = (int)node_to_face.size();

    // Step D: Build graph
    MaxFlow<double> mf;
    mf.add_node(N);

    // Unary terms (t-edges)
    // add_tedge(i, cap_source, cap_sink): SOURCE=label 0 (ri), SINK=label 1 (rj)
    // D(0) = cost of label 0, D(1) = cost of label 1
    // t-link: add_tedge(i, D(1), D(0))
    for (int nid = 0; nid < N; nid++) {
        int fi = node_to_face[nid];
        int reg = R(fi, 0);
        double d_label0, d_label1; // unary costs

        if (reg == ri) {
            // V0: must stay label 0 (= ri)
            d_label0 = 0;
            d_label1 = INF_CAP;
        } else if (reg == rj) {
            // V1: must stay label 1 (= rj)
            d_label0 = INF_CAP;
            d_label1 = 0;
        } else {
            // Fuzzy face from another region: compute errors
            double d0, d1;
            if (proxy_type == QUADRIC_PROXY) {
                d0 = face_error_quadric(fi, QP[ri], F, V);
                d1 = face_error_quadric(fi, QP[rj], F, V);
            } else {
                d0 = face_error_plane(fi, Proxies, num_proxies, ri, F, V, metric);
                d1 = face_error_plane(fi, Proxies, num_proxies, rj, F, V, metric);
            }
            double sum = d0 + d1 + 1e-10;
            d_label0 = d0 / sum;
            d_label1 = d1 / sum;
        }

        mf.add_tedge(nid, d_label1, d_label0);
    }

    // Pairwise terms (n-edges)
    set<pair<int,int>> added_edges;
    for (int nid = 0; nid < N; nid++) {
        int fi = node_to_face[nid];
        for (int k = 0; k < 3; k++) {
            if (is_feature_barrier(fi, k, F, Ad)) continue;
            int nb = Ad(fi, k);
            if (nb <= 0 || nb >= m) continue;
            auto it = face_to_node.find(nb);
            if (it == face_to_node.end()) continue;
            int nid2 = it->second;
            if (nid >= nid2) continue; // avoid duplicate

            pair<int,int> key(nid, nid2);
            if (added_edges.count(key)) continue;
            added_edges.insert(key);

            double sel = shared_edge_length(fi, nb, F, V);
            double smooth_cost = cfg.lambda * sel / (sel + avg_el);
            mf.add_edge(nid, nid2, smooth_cost, smooth_cost);
        }
    }

    // Step E: Solve
    log.cut_cost = mf.maxflow();

    // Apply labels
    for (int nid = 0; nid < N; nid++) {
        int fi = node_to_face[nid];
        int reg = R(fi, 0);
        int label = mf.what_segment(nid); // SOURCE=0=ri, SINK=1=rj
        int new_reg = (label == MaxFlow<double>::SOURCE) ? ri : rj;
        if (new_reg != reg) {
            R(fi, 0) = new_reg;
            log.relabeled_count++;
        }
    }

    // After boundary length
    log.boundary_length_after = compute_pair_boundary_length(R, F, V, Ad, ri, rj);

    return log;
}

void smooth_boundaries(MatrixXi& R, const MatrixXi& F, const MatrixXd& V,
                       const MatrixXi& Ad, int num_proxies,
                       ProxyType proxy_type,
                       const vector<QuadricProxy>& QP,
                       const MatrixXd& Proxies, MetricMode metric,
                       const SmoothConfig& cfg,
                       vector<SmoothLogEntry>& log_out) {
    log_out.clear();
    double avg_el = avg_edge_length(F, V);

    set<pair<int,int>> pairs = build_region_pairs(R, F, Ad, num_proxies);

    cout << "\n=== Boundary Smoothing ===" << endl;
    cout << "  " << pairs.size() << " adjacent pairs, ring=" << cfg.ring
         << " lambda=" << cfg.lambda << endl;

    for (auto& [ri, rj] : pairs) {
        SmoothLogEntry entry = smooth_pair(R, F, V, Ad, ri, rj, num_proxies,
            proxy_type, QP, Proxies, metric, cfg, avg_el);
        log_out.push_back(entry);

        if (entry.relabeled_count > 0) {
            cout << "  pair (" << ri << "," << rj << "): "
                 << entry.relabeled_count << " faces relabeled, "
                 << "boundary " << entry.boundary_length_before
                 << " -> " << entry.boundary_length_after << endl;
        }
    }
}

void export_smooth_log(const vector<SmoothLogEntry>& log, const string& filename) {
    ofstream fout(filename);
    if (!fout.is_open()) { cerr << "Cannot write " << filename << endl; return; }
    fout << "region_i,region_j,fuzzy_count,v0_count,v1_count,"
         << "cut_cost,relabeled_count,boundary_before,boundary_after" << endl;
    for (const auto& e : log) {
        fout << e.region_i << "," << e.region_j << ","
             << e.fuzzy_count << "," << e.v0_count << "," << e.v1_count << ","
             << e.cut_cost << "," << e.relabeled_count << ","
             << e.boundary_length_before << "," << e.boundary_length_after << endl;
    }
    fout.close();
    cout << "Exported smoothing log: " << filename << endl;
}
