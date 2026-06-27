#ifndef FEATURE_BARRIER_CPP
#define FEATURE_BARRIER_CPP

#include "feature_barrier.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

// Global definitions
set<EdgeKey> g_feature_edges;
bool g_feature_barrier_enabled = false;
double g_feature_angle_threshold = 30.0;

// Face_normal extern from distance.cpp
extern MatrixXd Face_normal;

static EdgeKey shared_edge_of_faces(int fi, int nb, const MatrixXi& F) {
    int a = F(fi,0), b = F(fi,1), c = F(fi,2);
    int na = F(nb,0), nbv = F(nb,1), nc = F(nb,2);
    if ((a==na||a==nbv||a==nc) && (b==na||b==nbv||b==nc))
        return EdgeKey(a,b);
    if ((a==na||a==nbv||a==nc) && (c==na||c==nbv||c==nc))
        return EdgeKey(a,c);
    return EdgeKey(b,c);
}

void compute_feature_edges(const MatrixXi& F, const MatrixXi& Ad,
                           double angle_threshold_deg) {
    g_feature_edges.clear();
    int m = F.rows();
    double threshold_rad = angle_threshold_deg * M_PI / 180.0;

    for (int fi = 0; fi < m; fi++) {
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb < 0 || nb >= m) continue;
            // Only process each edge once
            if (nb < fi) continue;

            double cos_angle = Face_normal.row(fi).dot(Face_normal.row(nb));
            // Clamp for numerical safety
            if (cos_angle > 1.0) cos_angle = 1.0;
            if (cos_angle < -1.0) cos_angle = -1.0;
            double angle = acos(cos_angle);

            if (angle > threshold_rad) {
                EdgeKey ek = shared_edge_of_faces(fi, nb, F);
                g_feature_edges.insert(ek);
            }
        }
    }
    cout << "Feature edges detected: " << g_feature_edges.size()
         << " (threshold=" << angle_threshold_deg << " degrees)" << endl;
}

VectorXi build_feature_groups(const MatrixXi& F, const MatrixXi& Ad,
                              const set<EdgeKey>& feature_edges) {
    int m = F.rows();
    VectorXi groups = -VectorXi::Ones(m, 1);
    int gid = 0;

    // Temporarily set global state for is_feature_barrier
    bool old_enabled = g_feature_barrier_enabled;
    set<EdgeKey> old_edges = g_feature_edges;  // backup
    g_feature_edges = feature_edges;
    g_feature_barrier_enabled = true;

    for (int fi = 0; fi < m; fi++) {
        if (groups(fi) != -1) continue;
        // BFS from fi
        queue<int> q;
        q.push(fi);
        while (!q.empty()) {
            int f = q.front(); q.pop();
            if (groups(f) != -1) continue;
            groups(f) = gid;
            for (int k = 0; k < 3; k++) {
                int nb = Ad(f, k);
                if (nb < 0 || nb >= m || groups(nb) != -1) continue;
                if (is_feature_barrier(f, k, F, Ad)) continue;
                q.push(nb);
            }
        }
        gid++;
    }

    // Restore global state
    g_feature_edges = old_edges;
    g_feature_barrier_enabled = old_enabled;

    cout << "Feature groups: " << gid << endl;
    return groups;
}

int count_feature_edges_on_boundary(const MatrixXi& R, const MatrixXi& F,
                                     const MatrixXi& Ad, int ri, int rj) {
    if (g_feature_edges.empty()) return 0;
    int m = R.rows();
    int count = 0;
    for (int fi = 0; fi < m; fi++) {
        if (R(fi, 0) != ri) continue;
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb < 0 || nb >= m || R(nb, 0) != rj) continue;
            if (is_feature_barrier(fi, k, F, Ad)) count++;
        }
    }
    return count;
}

void get_feature_edge_points(const MatrixXi& F, const MatrixXd& V,
                              const set<EdgeKey>& feature_edges,
                              MatrixXd& P1, MatrixXd& P2) {
    int n = (int)feature_edges.size();
    P1.resize(n, 3);
    P2.resize(n, 3);
    int i = 0;
    for (const auto& ek : feature_edges) {
        P1.row(i) = V.row(ek.v0);
        P2.row(i) = V.row(ek.v1);
        i++;
    }
}

void export_feature_edges_log(const set<EdgeKey>& feature_edges,
                               const MatrixXi& F, const MatrixXd& V,
                               const string& filename) {
    ofstream ofs(filename);
    ofs << "edge_id,v0,v1,x0,y0,z0,x1,y1,z1\n";
    int eid = 0;
    for (const auto& ek : feature_edges) {
        ofs << eid << ","
            << ek.v0 << "," << ek.v1 << ","
            << V(ek.v0,0) << "," << V(ek.v0,1) << "," << V(ek.v0,2) << ","
            << V(ek.v1,0) << "," << V(ek.v1,1) << "," << V(ek.v1,2) << "\n";
        eid++;
    }
    ofs.close();
    cout << "Exported " << eid << " feature edges to " << filename << endl;
}

BoundaryNormalRelabelReport relabel_boundary_faces_by_normal(
    MatrixXi& R,
    const MatrixXi& F,
    const MatrixXi& Ad,
    int num_regions,
    double min_improvement,
    double min_candidate_alignment,
    int max_iterations,
    bool verbose) {
    BoundaryNormalRelabelReport report;
    if (R.rows() == 0 || Face_normal.rows() != R.rows() || num_regions <= 0) {
        return report;
    }

    int m = R.rows();
    auto valid_label = [&](int label) {
        return label >= 0 && label < num_regions;
    };

    for (int iter = 0; iter < max_iterations; iter++) {
        vector<pair<int, int>> relabels;

        for (int fi = 0; fi < m; fi++) {
            int current = R(fi, 0);
            if (!valid_label(current)) continue;

            Vector3d nf = Face_normal.row(fi);
            if (nf.norm() < 1e-12) continue;
            nf.normalize();

            bool is_boundary = false;
            map<int, vector<int>> neighbor_faces_by_label;
            for (int k = 0; k < 3; k++) {
                int nb = Ad(fi, k);
                if (nb < 0 || nb >= m) continue;
                int label = R(nb, 0);
                if (!valid_label(label)) continue;
                neighbor_faces_by_label[label].push_back(nb);
                if (label != current) is_boundary = true;
            }
            if (!is_boundary) continue;
            if (iter == 0) report.boundary_face_count++;

            auto label_score = [&](int label) {
                auto it = neighbor_faces_by_label.find(label);
                if (it == neighbor_faces_by_label.end()) return -1.0;
                double best = -1.0;
                for (int nb : it->second) {
                    Vector3d nn = Face_normal.row(nb);
                    if (nn.norm() < 1e-12) continue;
                    nn.normalize();
                    best = max(best, nf.dot(nn));
                }
                return best;
            };

            double current_score = label_score(current);
            int best_label = current;
            double best_score = current_score;

            for (const auto& kv : neighbor_faces_by_label) {
                int candidate = kv.first;
                if (candidate == current) continue;
                double score = label_score(candidate);
                if (score > best_score) {
                    best_score = score;
                    best_label = candidate;
                }
            }

            if (best_label == current) continue;
            if (best_score < min_candidate_alignment) continue;
            if (best_score - current_score < min_improvement) continue;

            relabels.push_back(make_pair(fi, best_label));
        }

        if (relabels.empty()) break;
        for (const auto& mv : relabels) {
            R(mv.first, 0) = mv.second;
        }
        report.relabeled_face_count += (int)relabels.size();
        report.iterations = iter + 1;
    }

    if (verbose) {
        cout << "[boundary normal cleanup] boundary_faces="
             << report.boundary_face_count
             << " relabeled=" << report.relabeled_face_count
             << " iterations=" << report.iterations
             << " min_improvement=" << min_improvement
             << " min_candidate_alignment=" << min_candidate_alignment
             << endl;
    }

    return report;
}

static int count_same_region_feature_crossings(const MatrixXi& R,
                                               const MatrixXi& F,
                                               const MatrixXi& Ad) {
    if (!g_feature_barrier_enabled || g_feature_edges.empty()) return 0;
    int count = 0;
    int m = R.rows();
    for (int fi = 0; fi < m; fi++) {
        int ri = R(fi, 0);
        if (ri < 0) continue;
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb < 0 || nb >= m || nb < fi) continue;
            if (R(nb, 0) != ri) continue;
            if (is_feature_barrier(fi, k, F, Ad)) count++;
        }
    }
    return count;
}

static void repair_invalid_labels_for_final_barrier(MatrixXi& R,
                                                    const MatrixXi& F,
                                                    const MatrixXi& Ad,
                                                    int& num_regions) {
    int m = R.rows();
    auto valid_label = [&](int label) {
        return label >= 0 && label < num_regions;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (int fi = 0; fi < m; fi++) {
            if (valid_label(R(fi, 0))) continue;
            for (int k = 0; k < 3; k++) {
                int nb = Ad(fi, k);
                if (nb < 0 || nb >= m) continue;
                if (!valid_label(R(nb, 0))) continue;
                if (is_feature_barrier(fi, k, F, Ad)) continue;
                R(fi, 0) = R(nb, 0);
                changed = true;
                break;
            }
        }
    }

    for (int fi = 0; fi < m; fi++) {
        if (valid_label(R(fi, 0))) continue;

        int new_region = num_regions++;
        queue<int> q;
        q.push(fi);
        R(fi, 0) = new_region;

        while (!q.empty()) {
            int fcur = q.front();
            q.pop();
            for (int k = 0; k < 3; k++) {
                int nb = Ad(fcur, k);
                if (nb < 0 || nb >= m) continue;
                if (valid_label(R(nb, 0))) continue;
                if (R(nb, 0) == new_region) continue;
                if (is_feature_barrier(fcur, k, F, Ad)) continue;
                R(nb, 0) = new_region;
                q.push(nb);
            }
        }
    }
}

static int split_remaining_feature_edge_violations(MatrixXi& R,
                                                   const MatrixXi& F,
                                                   const MatrixXi& Ad,
                                                   int& num_regions) {
    int m = R.rows();
    int split_count = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (int fi = 0; fi < m; fi++) {
            int ri = R(fi, 0);
            if (ri < 0) continue;
            for (int k = 0; k < 3; k++) {
                int nb = Ad(fi, k);
                if (nb < 0 || nb >= m || nb < fi) continue;
                if (R(nb, 0) != ri) continue;
                if (!is_feature_barrier(fi, k, F, Ad)) continue;

                R(nb, 0) = num_regions++;
                split_count++;
                changed = true;
            }
        }
    }
    return split_count;
}

FeatureBarrierEnforceReport enforce_feature_barrier_final(
    MatrixXi& R,
    const MatrixXi& F,
    const MatrixXi& Ad,
    int& num_regions,
    bool verbose) {
    FeatureBarrierEnforceReport report;
    report.old_region_count = num_regions;
    report.new_region_count = num_regions;

    if (!g_feature_barrier_enabled || g_feature_edges.empty() || R.rows() == 0) {
        return report;
    }

    repair_invalid_labels_for_final_barrier(R, F, Ad, num_regions);
    report.old_region_count = num_regions;

    int m = R.rows();
    report.violating_feature_edges_before =
        count_same_region_feature_crossings(R, F, Ad);

    MatrixXi R_new = -MatrixXi::Ones(m, 1);
    vector<int> component_count(max(0, num_regions), 0);
    int next_region = 0;

    for (int fi = 0; fi < m; fi++) {
        if (R_new(fi, 0) >= 0) continue;
        int old_region = R(fi, 0);
        if (old_region < 0) continue;

        int new_region = next_region++;
        if (old_region >= 0 && old_region < (int)component_count.size()) {
            component_count[old_region]++;
        }

        queue<int> q;
        q.push(fi);
        R_new(fi, 0) = new_region;

        while (!q.empty()) {
            int fcur = q.front();
            q.pop();
            for (int k = 0; k < 3; k++) {
                int nb = Ad(fcur, k);
                if (nb < 0 || nb >= m) continue;
                if (R_new(nb, 0) >= 0) continue;
                if (R(nb, 0) != old_region) continue;
                if (is_feature_barrier(fcur, k, F, Ad)) continue;

                R_new(nb, 0) = new_region;
                q.push(nb);
            }
        }
    }

    for (int fi = 0; fi < m; fi++) {
        if (R_new(fi, 0) < 0) {
            R_new(fi, 0) = next_region++;
        }
    }

    for (int c : component_count) {
        if (c > 1) report.split_region_count++;
    }

    R = R_new;
    num_regions = next_region;
    int forced_single_face_splits =
        split_remaining_feature_edge_violations(R, F, Ad, num_regions);
    report.new_region_count = num_regions;
    report.added_region_count = max(0, report.new_region_count - report.old_region_count);
    report.violating_feature_edges_after =
        count_same_region_feature_crossings(R, F, Ad);

    if (verbose) {
        cout << "[feature_barrier final] regions "
             << report.old_region_count << " -> " << report.new_region_count
             << ", split_regions=" << report.split_region_count
             << ", added=" << report.added_region_count
             << ", forced_edge_splits=" << forced_single_face_splits
             << ", violations " << report.violating_feature_edges_before
             << " -> " << report.violating_feature_edges_after
             << endl;
    }

    return report;
}

#endif
