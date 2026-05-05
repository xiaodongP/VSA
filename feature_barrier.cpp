#ifndef FEATURE_BARRIER_CPP
#define FEATURE_BARRIER_CPP

#include "feature_barrier.h"
#include <cmath>
#include <fstream>
#include <iostream>

// Global definitions
set<EdgeKey> g_feature_edges;
bool g_feature_barrier_enabled = false;

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
                if (nb < 0 || groups(nb) != -1) continue;
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
            if (nb < 0 || R(nb, 0) != rj) continue;
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

#endif
