// Windows headers must come before Eigen to avoid byte/std::byte conflict
#include <igl/opengl/glfw/Viewer.h>
#include "boundary_smoothing.h"
#include "maxflow.h"
#include "feature_barrier.h"
#include "proxies.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>

static inline Vector3d v3(const Eigen::RowVectorXd& r) {
    return Vector3d(r(0), r(1), r(2));
}

static Vector3d tri_normal_area_weighted(const MatrixXi& F, const MatrixXd& V, int fi) {
    Vector3d a = v3(V.row(F(fi, 0)));
    Vector3d b = v3(V.row(F(fi, 1)));
    Vector3d c = v3(V.row(F(fi, 2)));
    return (b - a).cross(c - a);
}

static double edge_length(const MatrixXd& V, int v0, int v1) {
    return (v3(V.row(v0)) - v3(V.row(v1))).norm();
}

static double avg_edge_length(const MatrixXi& F, const MatrixXd& V) {
    double total = 0.0;
    int count = 0;
    for (int i = 0; i < F.rows(); i++) {
        total += edge_length(V, F(i,0), F(i,1));
        total += edge_length(V, F(i,1), F(i,2));
        total += edge_length(V, F(i,2), F(i,0));
        count += 3;
    }
    return (count > 0) ? total / count : 1.0;
}

static double shared_edge_length(int fi, int fj,
                                 const MatrixXi& F, const MatrixXd& V) {
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
    if ((int)shared.size() != 2) return 0.0;
    return edge_length(V, shared[0], shared[1]);
}

static double face_error_quadric(int fi, const QuadricProxy& proxy,
                                 const MatrixXi& F, const MatrixXd& V) {
    Vector3i f = F.row(fi);
    return proxy.triangle_error(v3(V.row(f(0))), v3(V.row(f(1))), v3(V.row(f(2))));
}

static double face_error_plane(int fi, const MatrixXd& Proxies,
                               int num_proxies, int proxy_id,
                               const MatrixXi& F, const MatrixXd& V,
                               MetricMode metric) {
    if (proxy_id < 0 || proxy_id >= num_proxies) return 1e12;
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
    else return 0.0;
    return pow(n.dot(normal), 2);
}

static double face_proxy_error(int fi, int region_id, ProxyType proxy_type,
                               const vector<QuadricProxy>& QP,
                               const MatrixXd& Proxies, int num_proxies,
                               const MatrixXi& F, const MatrixXd& V,
                               MetricMode metric) {
    if (region_id < 0 || region_id >= num_proxies) return 1e12;
    if (proxy_type == QUADRIC_PROXY) {
        if (region_id >= (int)QP.size()) return 1e12;
        return face_error_quadric(fi, QP[region_id], F, V);
    }
    return face_error_plane(fi, Proxies, num_proxies, region_id, F, V, metric);
}

double compute_pair_boundary_length(const MatrixXi& R, const MatrixXi& F,
                                    const MatrixXd& V, const MatrixXi& Ad,
                                    int ri, int rj) {
    double len = 0.0;
    int m = R.rows();
    for (int fi = 0; fi < m; fi++) {
        if (R(fi, 0) != ri) continue;
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb < 0 || nb >= m) continue;
            if (R(nb, 0) == rj) {
                len += shared_edge_length(fi, nb, F, V);
            }
        }
    }
    return len;
}

static int count_region_boundary_edges(const MatrixXi& R, const MatrixXi& Ad) {
    int count = 0;
    int m = R.rows();
    for (int fi = 0; fi < m; fi++) {
        int ri = R(fi, 0);
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb < 0 || nb >= m || nb < fi) continue;
            if (R(nb, 0) != ri) count++;
        }
    }
    return count;
}

static double total_region_boundary_length(const MatrixXi& R, const MatrixXi& F,
                                           const MatrixXd& V, const MatrixXi& Ad) {
    double len = 0.0;
    int m = R.rows();
    for (int fi = 0; fi < m; fi++) {
        int ri = R(fi, 0);
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb < 0 || nb >= m || nb < fi) continue;
            if (R(nb, 0) != ri) len += shared_edge_length(fi, nb, F, V);
        }
    }
    return len;
}

static set<pair<int,int>> build_region_pairs(const MatrixXi& R,
                                             const MatrixXi& Ad,
                                             int num_proxies) {
    set<pair<int,int>> pairs;
    int m = R.rows();
    for (int fi = 0; fi < m; fi++) {
        int ri = R(fi, 0);
        if (ri < 0 || ri >= num_proxies) continue;
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb < 0 || nb >= m) continue;
            int rj = R(nb, 0);
            if (rj < 0 || rj >= num_proxies || rj == ri) continue;
            pairs.insert(make_pair(min(ri, rj), max(ri, rj)));
        }
    }
    return pairs;
}

static set<int> find_boundary_faces(const MatrixXi& R, const MatrixXi& Ad,
                                    int ri, int rj) {
    set<int> faces;
    int m = R.rows();
    for (int fi = 0; fi < m; fi++) {
        int reg = R(fi, 0);
        if (reg != ri && reg != rj) continue;
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb < 0 || nb >= m) continue;
            int nbr = R(nb, 0);
            if ((reg == ri && nbr == rj) || (reg == rj && nbr == ri)) {
                faces.insert(fi);
                faces.insert(nb);
            }
        }
    }
    return faces;
}

static set<int> k_ring_expand_pair(const set<int>& seeds,
                                   const MatrixXi& R,
                                   const MatrixXi& F,
                                   const MatrixXi& Ad,
                                   int ri, int rj,
                                   int ring, int m) {
    set<int> result = seeds;
    vector<int> frontier(seeds.begin(), seeds.end());
    for (int r = 0; r < ring; r++) {
        vector<int> next;
        for (int fi : frontier) {
            for (int k = 0; k < 3; k++) {
                if (is_feature_barrier(fi, k, F, Ad)) continue;
                int nb = Ad(fi, k);
                if (nb < 0 || nb >= m) continue;
                int nr = R(nb, 0);
                if (nr != ri && nr != rj) continue;
                if (result.count(nb)) continue;
                result.insert(nb);
                next.push_back(nb);
            }
        }
        frontier.swap(next);
        if (frontier.empty()) break;
    }
    return result;
}

static void build_hard_constraints(const set<int>& Vf,
                                   const MatrixXi& R,
                                   const MatrixXi& F,
                                   const MatrixXi& Ad,
                                   int ri, int rj,
                                   set<int>& V0,
                                   set<int>& V1) {
    V0.clear();
    V1.clear();
    int m = R.rows();
    for (int fi : Vf) {
        for (int k = 0; k < 3; k++) {
            if (is_feature_barrier(fi, k, F, Ad)) continue;
            int nb = Ad(fi, k);
            if (nb < 0 || nb >= m) continue;
            if (Vf.count(nb)) continue;
            int reg = R(nb, 0);
            if (reg == ri) V0.insert(nb);
            else if (reg == rj) V1.insert(nb);
        }
    }
}

static void build_feature_fixed_faces(const MatrixXi& R,
                                      const MatrixXi& F,
                                      const MatrixXi& Ad,
                                      int ri, int rj,
                                      set<int>& F0,
                                      set<int>& F1) {
    F0.clear();
    F1.clear();
    if (!g_feature_barrier_enabled || g_feature_edges.empty()) return;

    int m = R.rows();
    for (int fi = 0; fi < m; fi++) {
        int reg = R(fi, 0);
        if (reg != ri && reg != rj) continue;
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb < 0 || nb >= m) continue;
            int nr = R(nb, 0);
            if (!((reg == ri && nr == rj) || (reg == rj && nr == ri))) continue;
            if (!is_feature_barrier(fi, k, F, Ad)) continue;

            if (reg == ri) F0.insert(fi);
            else F1.insert(fi);
            if (nr == ri) F0.insert(nb);
            else F1.insert(nb);
        }
    }
}

static bool average_region_normal(const MatrixXi& R, const MatrixXi& F,
                                  const MatrixXd& V, int region,
                                  Vector3d& out_normal) {
    Vector3d n = Vector3d::Zero();
    for (int fi = 0; fi < R.rows(); fi++) {
        if (R(fi, 0) != region) continue;
        n += tri_normal_area_weighted(F, V, fi);
    }
    double len = n.norm();
    if (len < 1e-12) return false;
    out_normal = n / len;
    return true;
}

static double region_normal_angle_deg(const MatrixXi& R, const MatrixXi& F,
                                      const MatrixXd& V, int ri, int rj) {
    Vector3d ni, nj;
    if (!average_region_normal(R, F, V, ri, ni) ||
        !average_region_normal(R, F, V, rj, nj)) {
        return 0.0;
    }
    double c = max(-1.0, min(1.0, ni.dot(nj)));
    return acos(c) * 180.0 / M_PI;
}

static int count_faces_in_region(const MatrixXi& R, int region) {
    int count = 0;
    for (int fi = 0; fi < R.rows(); fi++) {
        if (R(fi, 0) == region) count++;
    }
    return count;
}

static int effective_ring(const SmoothConfig& cfg) {
    return max(0, max(cfg.ring, cfg.ringSize));
}

static SmoothLogEntry smooth_pair(MatrixXi& R, const MatrixXi& F,
                                  const MatrixXd& V, const MatrixXi& Ad,
                                  int ri, int rj, int num_proxies,
                                  ProxyType proxy_type,
                                  const vector<QuadricProxy>& QP,
                                  const MatrixXd& Proxies, MetricMode metric,
                                  const SmoothConfig& cfg,
                                  double avg_el,
                                  int pass,
                                  set<int>& changed_regions) {
    SmoothLogEntry log;
    log.pass = pass;
    log.region_i = ri;
    log.region_j = rj;
    int m = R.rows();

    log.boundary_length_before = compute_pair_boundary_length(R, F, V, Ad, ri, rj);
    log.region_normal_angle_deg = region_normal_angle_deg(R, F, V, ri, rj);
    log.feature_barrier_edge_count = count_feature_edges_on_boundary(R, F, Ad, ri, rj);

    // Do not skip an entire region pair just because part of its boundary
    // contains feature edges. Feature edges are still respected below: BFS and
    // pairwise n-links never cross is_feature_barrier(...). Skipping the whole
    // pair made almost all real model pairs no-ops whenever the common boundary
    // had even one detected feature edge.
    if (cfg.skipSharpBoundary &&
        log.region_normal_angle_deg > cfg.sharpAngleThresholdDeg) {
        log.skipped_sharp = true;
        log.boundary_length_after = log.boundary_length_before;
        return log;
    }

    set<int> boundary = find_boundary_faces(R, Ad, ri, rj);
    log.boundary_seed_count = (int)boundary.size();
    if (boundary.empty()) {
        log.skipped_empty = true;
        log.boundary_length_after = log.boundary_length_before;
        return log;
    }

    set<int> Vf = k_ring_expand_pair(boundary, R, F, Ad, ri, rj,
                                     effective_ring(cfg), m);
    set<int> V0, V1;
    build_hard_constraints(Vf, R, F, Ad, ri, rj, V0, V1);
    set<int> F0, F1;
    if (cfg.protectFeatureBoundaryFaces) {
        build_feature_fixed_faces(R, F, Ad, ri, rj, F0, F1);
    }

    log.fuzzy_count = (int)Vf.size();
    log.v0_count = (int)V0.size();
    log.v1_count = (int)V1.size();
    log.feature_fixed_i = (int)F0.size();
    log.feature_fixed_j = (int)F1.size();

    if (Vf.empty()) {
        log.skipped_empty = true;
        log.boundary_length_after = log.boundary_length_before;
        return log;
    }
    if ((V0.empty() && F0.empty()) || (V1.empty() && F1.empty())) {
        log.skipped_constraints = true;
        log.boundary_length_after = log.boundary_length_before;
        return log;
    }

    set<int> Vlocal = Vf;
    Vlocal.insert(V0.begin(), V0.end());
    Vlocal.insert(V1.begin(), V1.end());

    map<int, int> face_to_node;
    vector<int> node_to_face;
    for (int fi : Vlocal) {
        int nid = (int)node_to_face.size();
        face_to_node[fi] = nid;
        node_to_face.push_back(fi);
    }

    int N = (int)node_to_face.size();
    log.graph_nodes = N;
    MaxFlow<double> mf;
    mf.add_node(N);

    for (int nid = 0; nid < N; nid++) {
        int fi = node_to_face[nid];
        double d_label0 = 0.0;
        double d_label1 = 0.0;

        if (V0.count(fi) || F0.count(fi)) {
            d_label0 = 0.0;
            d_label1 = cfg.infCost;
        } else if (V1.count(fi) || F1.count(fi)) {
            d_label0 = cfg.infCost;
            d_label1 = 0.0;
        } else {
            double d0 = face_proxy_error(fi, ri, proxy_type, QP, Proxies,
                                         num_proxies, F, V, metric);
            double d1 = face_proxy_error(fi, rj, proxy_type, QP, Proxies,
                                         num_proxies, F, V, metric);
            double denom = max(d0 + d1, 1e-12);
            d_label0 = d0 / denom;
            d_label1 = d1 / denom;
        }

        // MaxFlow::add_tedge uses source-cap/sink-cap. For graph-cut costs:
        // SOURCE=label0, SINK=label1, so pass D(1), D(0).
        mf.add_tedge(nid, d_label1, d_label0);
    }

    set<pair<int,int>> added_edges;
    for (int nid = 0; nid < N; nid++) {
        int fi = node_to_face[nid];
        for (int k = 0; k < 3; k++) {
            if (is_feature_barrier(fi, k, F, Ad)) continue;
            int nb = Ad(fi, k);
            if (nb < 0 || nb >= m) continue;
            auto it = face_to_node.find(nb);
            if (it == face_to_node.end()) continue;
            int nid2 = it->second;
            if (nid >= nid2) continue;
            pair<int,int> key(nid, nid2);
            if (added_edges.count(key)) continue;
            added_edges.insert(key);

            double sel = shared_edge_length(fi, nb, F, V);
            double w = (sel > 0.0) ? sel / (sel + avg_el) : 0.0;
            double smooth_cost = cfg.lambda * w;
            mf.add_edge(nid, nid2, smooth_cost, smooth_cost);
            log.graph_edges++;
        }
    }

    log.cut_cost = mf.maxflow();

    vector<pair<int,int>> proposed;
    int ri_count = count_faces_in_region(R, ri);
    int rj_count = count_faces_in_region(R, rj);
    int ri_delta = 0;
    int rj_delta = 0;

    for (int fi : Vf) {
        auto it = face_to_node.find(fi);
        if (it == face_to_node.end()) continue;
        if (F0.count(fi) || F1.count(fi)) continue;
        int reg = R(fi, 0);
        if (reg != ri && reg != rj) continue;

        int label = mf.what_segment(it->second);
        int new_reg = (label == MaxFlow<double>::SOURCE) ? ri : rj;
        if (new_reg == reg) continue;

        proposed.push_back(make_pair(fi, new_reg));
        if (reg == ri && new_reg == rj) {
            ri_delta--;
            rj_delta++;
            log.relabeled_i_to_j++;
        } else if (reg == rj && new_reg == ri) {
            rj_delta--;
            ri_delta++;
            log.relabeled_j_to_i++;
        }
    }

    if (ri_count + ri_delta <= 0 || rj_count + rj_delta <= 0) {
        log.skipped_empty_region = true;
        log.relabeled_i_to_j = 0;
        log.relabeled_j_to_i = 0;
        log.boundary_length_after = log.boundary_length_before;
        return log;
    }

    vector<int> old_labels;
    old_labels.reserve(proposed.size());
    for (const auto& mv : proposed) {
        old_labels.push_back(R(mv.first, 0));
        R(mv.first, 0) = mv.second;
    }
    log.relabeled_count = (int)proposed.size();

    log.boundary_length_after = compute_pair_boundary_length(R, F, V, Ad, ri, rj);
    if (cfg.rejectBoundaryLengthIncrease &&
        log.boundary_length_after > log.boundary_length_before + 1e-9) {
        for (int i = 0; i < (int)proposed.size(); i++) {
            R(proposed[i].first, 0) = old_labels[i];
        }
        log.rejected_boundary_length_increase = true;
        log.boundary_length_after = log.boundary_length_before;
        log.relabeled_count = 0;
        log.relabeled_i_to_j = 0;
        log.relabeled_j_to_i = 0;
        return log;
    }

    if (log.relabeled_count > 0) {
        changed_regions.insert(ri);
        changed_regions.insert(rj);
    }
    return log;
}

BoundarySmoothingResult smooth_boundaries(MatrixXi& R, const MatrixXi& F,
                       const MatrixXd& V, const MatrixXi& Ad, int num_proxies,
                       ProxyType proxy_type,
                       vector<QuadricProxy>& QP,
                       MatrixXd& Proxies, MetricMode metric,
                       const SmoothConfig& cfg,
                       vector<SmoothLogEntry>& log_out) {
    log_out.clear();
    BoundarySmoothingResult result;
    if (R.rows() == 0 || F.rows() == 0 || num_proxies <= 0) {
        return result;
    }

    double avg_el = avg_edge_length(F, V);
    result.boundary_edge_count_before = count_region_boundary_edges(R, Ad);
    result.boundary_length_before = total_region_boundary_length(R, F, V, Ad);

    int passes = max(1, cfg.maxPasses);
    set<int> changed_regions;

    if (cfg.enableDebugOutput) {
        cout << "\n=== Boundary Smoothing (graph cut) ===" << endl;
        cout << "  ringSize=" << effective_ring(cfg)
             << " lambda=" << cfg.lambda
             << " maxPasses=" << passes
             << " skipSharp=" << (cfg.skipSharpBoundary ? "true" : "false")
             << " sharpAngleDeg=" << cfg.sharpAngleThresholdDeg << endl;
    }

    for (int pass = 0; pass < passes; pass++) {
        set<pair<int,int>> pairs = build_region_pairs(R, Ad, num_proxies);
        if (cfg.enableDebugOutput) {
            cout << "  pass " << (pass + 1) << "/" << passes
                 << ": adjacentPairs=" << pairs.size() << endl;
        }

        for (const auto& pr : pairs) {
            SmoothLogEntry entry = smooth_pair(R, F, V, Ad, pr.first, pr.second,
                num_proxies, proxy_type, QP, Proxies, metric, cfg, avg_el,
                pass + 1, changed_regions);
            log_out.push_back(entry);
            result.pair_logs.push_back(entry);
            result.region_pairs_processed++;
            result.sharp_pairs_skipped += entry.skipped_sharp ? 1 : 0;
            result.regions_becoming_empty += entry.skipped_empty_region ? 1 : 0;
            result.total_relabeled_faces += entry.relabeled_count;
            result.relabeled_i_to_j += entry.relabeled_i_to_j;
            result.relabeled_j_to_i += entry.relabeled_j_to_i;

            if (cfg.enableDebugOutput) {
                cout << "[BoundarySmoothPair] pass=" << entry.pass
                     << " pair=(" << entry.region_i << "," << entry.region_j << ")"
                     << " seeds=" << entry.boundary_seed_count
                     << " Vf=" << entry.fuzzy_count
                     << " V0=" << entry.v0_count
                     << " V1=" << entry.v1_count
                     << " nodes=" << entry.graph_nodes
                     << " edges=" << entry.graph_edges
                     << " angleDeg=" << entry.region_normal_angle_deg
                     << " featureEdges=" << entry.feature_barrier_edge_count
                     << " featureFixed=(" << entry.feature_fixed_i
                     << "," << entry.feature_fixed_j << ")"
                     << " relabel=" << entry.relabeled_count
                     << " iToJ=" << entry.relabeled_i_to_j
                     << " jToI=" << entry.relabeled_j_to_i
                     << " boundary=" << entry.boundary_length_before
                     << "->" << entry.boundary_length_after
                     << " cut=" << entry.cut_cost;
                if (entry.skipped_sharp) cout << " skipped=sharp";
                else if (entry.skipped_empty) cout << " skipped=empty";
                else if (entry.skipped_constraints) cout << " skipped=constraints";
                else if (entry.skipped_empty_region) cout << " skipped=empty_region";
                else if (entry.rejected_boundary_length_increase) cout << " rejected=boundary_length_increase";
                cout << endl;
            }
        }
    }

    if (cfg.refitProxiesAfterSmoothing && !changed_regions.empty()) {
        if (proxy_type == QUADRIC_PROXY) {
            if ((int)QP.size() < num_proxies) QP.resize(num_proxies);
            for (int rid : changed_regions) {
                if (rid >= 0 && rid < num_proxies) {
                    QP[rid] = fit_quadric_region(R, rid, F, V);
                }
            }
        } else {
            Proxies = new_proxies(R, F, V, num_proxies, metric);
        }
    }

    result.boundary_edge_count_after = count_region_boundary_edges(R, Ad);
    result.boundary_length_after = total_region_boundary_length(R, F, V, Ad);

    if (cfg.enableDebugOutput) {
        cout << "[BoundarySmoothSummary]" << endl;
        cout << "region_pairs_processed=" << result.region_pairs_processed << endl;
        cout << "sharp_pairs_skipped=" << result.sharp_pairs_skipped << endl;
        cout << "total_relabeled_faces=" << result.total_relabeled_faces << endl;
        cout << "relabeled_i_to_j=" << result.relabeled_i_to_j << endl;
        cout << "relabeled_j_to_i=" << result.relabeled_j_to_i << endl;
        cout << "boundary_edge_count_before=" << result.boundary_edge_count_before << endl;
        cout << "boundary_edge_count_after=" << result.boundary_edge_count_after << endl;
        cout << "boundary_length_before=" << result.boundary_length_before << endl;
        cout << "boundary_length_after=" << result.boundary_length_after << endl;
        cout << "regions_becoming_empty=" << result.regions_becoming_empty << endl;
        cout << "changed_regions_refit=" << changed_regions.size()
             << " refit=" << (cfg.refitProxiesAfterSmoothing ? "true" : "false")
             << endl;
    }

    return result;
}

void export_smooth_log(const vector<SmoothLogEntry>& log, const string& filename) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Cannot write " << filename << endl;
        return;
    }
    fout << "pass,region_i,region_j,boundary_seed_count,fuzzy_count,v0_count,v1_count,"
         << "graph_nodes,graph_edges,cut_cost,relabeled_count,relabeled_i_to_j,"
         << "relabeled_j_to_i,feature_fixed_i,feature_fixed_j,"
         << "boundary_before,boundary_after,region_normal_angle_deg,"
         << "feature_barrier_edge_count,skipped_sharp,skipped_empty,"
         << "skipped_constraints,skipped_empty_region,rejected_boundary_length_increase\n";
    for (const auto& e : log) {
        fout << e.pass << ","
             << e.region_i << "," << e.region_j << ","
             << e.boundary_seed_count << ","
             << e.fuzzy_count << "," << e.v0_count << "," << e.v1_count << ","
             << e.graph_nodes << "," << e.graph_edges << ","
             << e.cut_cost << "," << e.relabeled_count << ","
             << e.relabeled_i_to_j << "," << e.relabeled_j_to_i << ","
             << e.feature_fixed_i << "," << e.feature_fixed_j << ","
             << e.boundary_length_before << "," << e.boundary_length_after << ","
             << e.region_normal_angle_deg << ","
             << e.feature_barrier_edge_count << ","
             << (e.skipped_sharp ? 1 : 0) << ","
             << (e.skipped_empty ? 1 : 0) << ","
             << (e.skipped_constraints ? 1 : 0) << ","
             << (e.skipped_empty_region ? 1 : 0) << ","
             << (e.rejected_boundary_length_increase ? 1 : 0) << "\n";
    }
    fout.close();
    cout << "Exported smoothing log: " << filename << endl;
}
