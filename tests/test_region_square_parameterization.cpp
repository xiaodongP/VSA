#include "region_square_parameterization.h"
#include "quad_like_boundary.h"
#include "region_boundary.h"

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            cerr << "  FAIL: " << msg << endl; \
            g_fail++; \
        } else { \
            cout << "  OK:   " << msg << endl; \
            g_pass++; \
        } \
    } while (0)

static bool file_nonempty(const string& filename) {
    std::ifstream fin(filename, std::ios::binary | std::ios::ate);
    return fin.is_open() && fin.tellg() > 0;
}

static void build_grid_mesh(int nx, int ny, MatrixXd& V, MatrixXi& F) {
    V.resize((nx + 1) * (ny + 1), 3);
    auto vid = [nx](int x, int y) {
        return y * (nx + 1) + x;
    };
    for (int y = 0; y <= ny; y++) {
        for (int x = 0; x <= nx; x++) {
            double z = 0.04 * x + 0.02 * y;
            V.row(vid(x, y)) << (double)x, (double)y, z;
        }
    }

    F.resize(nx * ny * 2, 3);
    int f = 0;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int v00 = vid(x, y);
            int v10 = vid(x + 1, y);
            int v01 = vid(x, y + 1);
            int v11 = vid(x + 1, y + 1);
            F.row(f++) << v00, v10, v11;
            F.row(f++) << v00, v11, v01;
        }
    }
}

static bool build_boundary_data(
    const MatrixXd& V,
    const MatrixXi& F,
    const vector<int>& labels,
    RegionBoundaryLoop& loop,
    QuadLikeBoundary& quad) {
    RegionBoundaryExtractionResult boundary =
        extract_region_boundary_loop(V, F, labels, 1);
    if (!boundary.success) {
        cerr << "  boundary extraction failed: " << boundary.reason << endl;
        return false;
    }
    QuadLikeBoundaryConfig qcfg;
    qcfg.max_corner_candidates = 16;
    qcfg.min_quality_score = 0.20;
    QuadLikeBoundaryResult qresult =
        split_quad_like_boundary(boundary.loop, qcfg);
    if (!qresult.success) {
        cerr << "  quad split failed: " << qresult.reason << endl;
        return false;
    }
    loop = boundary.loop;
    quad = qresult.boundary;
    return true;
}

static void test_square_parameterization_success() {
    cout << "\n=== Square parameterization success ===" << endl;
    MatrixXd V;
    MatrixXi F;
    build_grid_mesh(3, 3, V, F);
    vector<int> labels(F.rows(), 1);

    RegionBoundaryLoop loop;
    QuadLikeBoundary quad;
    CHECK(build_boundary_data(V, F, labels, loop, quad),
          "boundary loop and quad sides are available");

    RegionSquareParameterizationConfig cfg;
    RegionSquareParameterizationResult result =
        parameterize_region_to_square(V, F, labels, 1, loop, quad, cfg);

    cout << "  reason=" << result.reason << endl;
    cout << "  min_signed_area=" << result.min_signed_area
         << " max_signed_area=" << result.max_signed_area << endl;
    cout << "  mean_area_distortion=" << result.mean_area_distortion
         << " max_area_distortion=" << result.max_area_distortion << endl;
    cout << "  mean_angle_distortion=" << result.mean_angle_distortion
         << " max_angle_distortion=" << result.max_angle_distortion << endl;

    CHECK(result.valid, "parameterization succeeds");
    CHECK(result.UV.rows() == V.rows() && result.UV.cols() == 2,
          "UV has one row per mesh vertex");
    CHECK(result.uv_in_range, "all region UV coordinates are in square range");
    CHECK(!result.has_flips, "no flipped parameter triangles");
    CHECK(result.triangle_stats.size() == (size_t)F.rows(),
          "one triangle stat per region face");
    CHECK(result.min_signed_area > 1e-10, "all signed areas are positive");
    CHECK(result.mean_angle_distortion >= 0.0, "angle distortion is finite");
    CHECK(export_region_square_parameterization_debug(
              "region_square_parameterization_success", V, F, result),
          "debug exports succeed");
    CHECK(file_nonempty("region_square_parameterization_success_uv.obj") &&
              file_nonempty("region_square_parameterization_success_signed_area.csv") &&
              file_nonempty("region_square_parameterization_success_distortion_heatmap.csv") &&
              file_nonempty("region_square_parameterization_success_correspondence.obj"),
          "debug export files are non-empty");
}

static void test_flip_detection_failure() {
    cout << "\n=== Flip detection failure ===" << endl;
    MatrixXd V;
    MatrixXi F;
    build_grid_mesh(3, 3, V, F);
    vector<int> labels(F.rows(), 1);

    RegionBoundaryLoop loop;
    QuadLikeBoundary quad;
    CHECK(build_boundary_data(V, F, labels, loop, quad),
          "boundary data for flipped case is available");

    std::swap(F(0, 1), F(0, 2));
    RegionSquareParameterizationConfig cfg;
    RegionSquareParameterizationResult result =
        parameterize_region_to_square(V, F, labels, 1, loop, quad, cfg);

    cout << "  reason=" << result.reason << endl;
    cout << "  min_signed_area=" << result.min_signed_area << endl;

    CHECK(!result.valid, "flipped UV triangle causes failure");
    CHECK(result.has_flips, "flip flag is set");
    CHECK(result.reason.find("flipped") != string::npos ||
              result.reason.find("degenerate") != string::npos,
          "failure reason mentions flipped or degenerate triangles");
    CHECK(export_region_square_parameterization_debug(
              "region_square_parameterization_flipped", V, F, result),
          "flipped debug exports succeed");
    CHECK(file_nonempty("region_square_parameterization_flipped_uv.obj") &&
              file_nonempty("region_square_parameterization_flipped_signed_area.csv") &&
              file_nonempty("region_square_parameterization_flipped_distortion_heatmap.csv") &&
              file_nonempty("region_square_parameterization_flipped_correspondence.obj"),
          "flipped debug export files are non-empty");
}

int main() {
    cout << "========================================" << endl;
    cout << "  Region Square Parameterization Tests" << endl;
    cout << "========================================" << endl;

    test_square_parameterization_success();
    test_flip_detection_failure();

    cout << "\n========================================" << endl;
    cout << "  Passed: " << g_pass << "  Failed: " << g_fail << endl;
    if (g_fail == 0)
        cout << "  ALL TESTS PASSED" << endl;
    else
        cout << "  SOME TESTS FAILED" << endl;
    cout << "========================================" << endl;

    return g_fail;
}
