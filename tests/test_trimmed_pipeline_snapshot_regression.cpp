#include "trimmed_bspline_pipeline.h"

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using std::cout;
using std::endl;
using std::string;
using std::vector;

namespace {

bool read_coff_mesh(const string& filename, MatrixXd& V, MatrixXi& F, string& reason) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        reason = "cannot open COFF file: " + filename;
        return false;
    }
    string magic;
    in >> magic;
    if (magic != "COFF" && magic != "OFF") {
        reason = "unsupported mesh magic: " + magic;
        return false;
    }
    int nv = 0, nf = 0, ne = 0;
    in >> nv >> nf >> ne;
    if (nv <= 0 || nf <= 0) {
        reason = "empty COFF mesh";
        return false;
    }
    V.resize(nv, 3);
    for (int i = 0; i < nv; i++) {
        double x = 0.0, y = 0.0, z = 0.0;
        in >> x >> y >> z;
        V.row(i) << x, y, z;
        string rest;
        std::getline(in, rest);
    }
    F.resize(nf, 3);
    for (int i = 0; i < nf; i++) {
        int n = 0, a = 0, b = 0, c = 0;
        in >> n >> a >> b >> c;
        if (n != 3) {
            reason = "non-triangle face in COFF";
            return false;
        }
        F.row(i) << a, b, c;
        string rest;
        std::getline(in, rest);
    }
    return true;
}

bool read_qvsa_labels(
    const string& filename,
    int expected_vertices,
    int expected_faces,
    vector<int>& labels,
    int& region_count,
    string& reason) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        reason = "cannot open QVSA file: " + filename;
        return false;
    }
    string magic;
    int version = 0;
    in >> magic >> version;
    if (magic != "QVSA_CHECKPOINT" || version != 1) {
        reason = "unsupported QVSA checkpoint";
        return false;
    }
    string key;
    int nv = -1, nf = -1;
    in >> key >> nv;
    if (key != "vertices") return false;
    in >> key >> nf;
    if (key != "faces") return false;
    in >> key >> region_count;
    if (key != "regions") return false;
    if (nv != expected_vertices || nf != expected_faces) {
        std::ostringstream ss;
        ss << "mesh mismatch: qvsa " << nv << "V/" << nf
           << "F, coff " << expected_vertices << "V/" << expected_faces << "F";
        reason = ss.str();
        return false;
    }
    int dummy_int = 0;
    double dummy_double = 0.0;
    in >> key >> dummy_int;
    if (key != "use_quadric") return false;
    in >> key >> dummy_int;
    if (key != "feature_barrier_enabled") return false;
    in >> key >> dummy_double;
    if (key != "feature_angle") return false;
    in >> key;
    if (key != "labels") {
        reason = "missing labels block";
        return false;
    }
    labels.assign(nf, -1);
    for (int i = 0; i < nf; i++) in >> labels[i];
    return true;
}

vector<int> parse_regions(int argc, char** argv) {
    if (argc <= 1) return {5, 6, 7, 8, 9, 11, 12, 13, 15, 16};
    vector<int> regions;
    for (int i = 1; i < argc; i++) {
        regions.push_back(std::stoi(argv[i]));
    }
    return regions;
}

TrimmedBSplinePipelineConfig regression_config(int region_id) {
    TrimmedBSplinePipelineConfig cfg;
    cfg.output_dir = "trimmed_snapshot_regression_region_" + std::to_string(region_id);
    cfg.control_count_u = 6;
    cfg.control_count_v = 6;
    cfg.surface_sample_u = 32;
    cfg.surface_sample_v = 32;
    cfg.extension_sample_u = 12;
    cfg.extension_sample_v = 12;
    cfg.enable_smoothed_arap = false;
    cfg.enable_extension_fairness = false;
    cfg.export_debug_artifacts = true;
    cfg.run_ablation_baselines = false;
    cfg.estimate_condition_number = false;
    cfg.print_progress_to_console = true;
    return cfg;
}

} // namespace

int main(int argc, char** argv) {
    MatrixXd V;
    MatrixXi F;
    vector<int> labels;
    int region_count = 0;
    string reason;
    if (!read_coff_mesh("interactive_quadric_vsa_snapshot_segmentation.coff", V, F, reason)) {
        std::cerr << "Failed to read snapshot COFF: " << reason << endl;
        return 1;
    }
    if (!read_qvsa_labels(
            "interactive_quadric_vsa_snapshot.qvsa",
            (int)V.rows(),
            (int)F.rows(),
            labels,
            region_count,
            reason)) {
        std::cerr << "Failed to read snapshot labels: " << reason << endl;
        return 1;
    }

    int failed = 0;
    vector<int> regions = parse_regions(argc, argv);
    cout << "Snapshot mesh: " << V.rows() << "V/" << F.rows()
         << "F, regions=" << region_count << endl;
    for (int region_id : regions) {
        cout << "\n=== Snapshot region " << region_id << " ===" << endl;
        TrimmedBSplinePipelineResult result =
            run_single_region_trimmed_bspline_pipeline(
                V, F, labels, region_id, regression_config(region_id));
        cout << "region " << region_id << " result="
             << (result.valid ? "ok" : "failed")
             << " reason=" << result.reason << endl;
        if (!result.valid) failed++;
    }
    cout << "\nSnapshot regression failures: " << failed
         << " / " << regions.size() << endl;
    return failed == 0 ? 0 : 1;
}
