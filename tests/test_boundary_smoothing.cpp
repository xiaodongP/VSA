#include <igl/opengl/glfw/Viewer.h>
#include "boundary_smoothing.h"
#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

using namespace Eigen;
using namespace std;

static MatrixXi build_adjacency(const MatrixXi& F) {
    MatrixXi Ad = -MatrixXi::Ones(F.rows(), 3);
    map<pair<int,int>, pair<int,int>> edge_owner;

    for (int fi = 0; fi < F.rows(); fi++) {
        for (int k = 0; k < 3; k++) {
            int a = F(fi, k);
            int b = F(fi, (k + 1) % 3);
            pair<int,int> e(min(a, b), max(a, b));
            auto it = edge_owner.find(e);
            if (it == edge_owner.end()) {
                edge_owner[e] = make_pair(fi, k);
            } else {
                int fj = it->second.first;
                int kj = it->second.second;
                Ad(fi, k) = fj;
                Ad(fj, kj) = fi;
            }
        }
    }
    return Ad;
}

static bool expect_true(bool ok, const string& msg) {
    if (ok) {
        cout << "  OK: " << msg << endl;
        return true;
    }
    cerr << "  FAIL: " << msg << endl;
    return false;
}

int main() {
    MatrixXd V(10, 3);
    int idx = 0;
    for (int y = 0; y <= 1; y++) {
        for (int x = 0; x <= 4; x++) {
            V.row(idx++) << (double)x, (double)y, 0.0;
        }
    }

    MatrixXi F(8, 3);
    for (int cell = 0; cell < 4; cell++) {
        int v00 = cell;
        int v10 = cell + 1;
        int v01 = 5 + cell;
        int v11 = 5 + cell + 1;
        F.row(2 * cell) << v00, v10, v11;
        F.row(2 * cell + 1) << v00, v11, v01;
    }

    MatrixXi Ad = build_adjacency(F);
    MatrixXi R(8, 1);
    R << 0, 0, 1, 0, 1, 1, 1, 1;
    const int intentionally_wrong_face = 2;
    const int hard_left_face = 0;
    const int hard_right_face = 6;

    MatrixXd Proxies(4, 3);
    Proxies.row(0) << 0.75, 0.5, 0.0;
    Proxies.row(1) << 3.25, 0.5, 0.0;
    Proxies.row(2) << 0.0, 0.0, 1.0;
    Proxies.row(3) << 0.0, 0.0, 1.0;

    vector<QuadricProxy> QP;
    SmoothConfig cfg;
    cfg.ring = 0;
    cfg.ringSize = 0;
    cfg.lambda = 0.0;
    cfg.skipSharpBoundary = false;
    cfg.refitProxiesAfterSmoothing = false;
    cfg.enableDebugOutput = false;

    vector<SmoothLogEntry> logs;
    BoundarySmoothingResult result =
        smooth_boundaries(R, F, V, Ad, 2, PLANE_PROXY, QP, Proxies,
                          L2_METRIC, cfg, logs);

    bool ok = true;
    ok &= expect_true(result.region_pairs_processed > 0,
                      "processed at least one adjacent region pair");
    ok &= expect_true(R(intentionally_wrong_face, 0) == 0,
                      "boundary fuzzy face relabeled to lower-error proxy");
    ok &= expect_true(R(hard_left_face, 0) == 0,
                      "left hard-constraint face stayed in region 0");
    ok &= expect_true(R(hard_right_face, 0) == 1,
                      "right hard-constraint face stayed in region 1");

    if (!ok) return 1;
    cout << "Boundary smoothing synthetic test passed." << endl;
    return 0;
}
