// Windows headers must come before Eigen to avoid byte/std::byte conflict
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>

#include "vsa_batch.h"
#include "proxy_validity.h"
#include "proxy_classification.h"
#include "proxy_projection.h"
#include "feature_barrier.h"
#include "distance.h"
#include "partitioning.h"
#include "proxies.h"
#include <queue>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>

// Helper: row vector → Vector3d
static inline Vector3d v3(const Eigen::RowVectorXd& r) {
    return Vector3d(r(0), r(1), r(2));
}

// Helper: triangle area from face index
static double face_area(int fi, const MatrixXi& F, const MatrixXd& V) {
    Vector3i f = F.row(fi);
    return 0.5 * (v3(V.row(f(1))) - v3(V.row(f(0))))
                .cross(v3(V.row(f(2))) - v3(V.row(f(0)))).norm();
}

// ============================================================
// Quadric flooding: priority-queue Dijkstra with triangle_error
// ============================================================

static double face_quadric_error(int fi, const QuadricProxy& proxy,
                                 const MatrixXi& F, const MatrixXd& V) {
    Vector3i f = F.row(fi);
    return proxy.triangle_error(v3(V.row(f(0))), v3(V.row(f(1))), v3(V.row(f(2))));
}

static void flood_regions_quadric(MatrixXi& R, const vector<int>& seeds,
                                   const vector<QuadricProxy>& QP,
                                   const MatrixXd& V, const MatrixXi& F,
                                   const MatrixXi& Ad) {
    int m = F.rows();
    int p = (int)seeds.size();
    R = -MatrixXi::Ones(m, 1);

    priority_queue<pair<double, int>> q;

    for (int i = 0; i < p; i++) {
        R(seeds[i], 0) = i;
        for (int k = 0; k < 3; k++) {
            if (is_feature_barrier(seeds[i], k, F, Ad)) continue;
            int nb = Ad(seeds[i], k);
            if (nb < 0) continue;
            double d = face_quadric_error(nb, QP[i], F, V);
            q.push(make_pair(-d, nb + m * i));
        }
    }

    while (!q.empty()) {
        auto item = q.top(); q.pop();
        int prox = item.second / m;
        int face = item.second % m;
        if (R(face, 0) != -1) continue;
        R(face, 0) = prox;
        for (int k = 0; k < 3; k++) {
            if (is_feature_barrier(face, k, F, Ad)) continue;
            int nb = Ad(face, k);
            if (nb < 0 || R(nb, 0) != -1) continue;
            double d = face_quadric_error(nb, QP[prox], F, V);
            q.push(make_pair(-d, nb + m * prox));
        }
    }

    // Assign any remaining -1 faces to a neighbor's proxy
    for (int i = 0; i < m; i++) {
        if (R(i, 0) >= 0) continue;
        for (int k = 0; k < 3; k++) {
            int nb = Ad(i, k);
            if (nb >= 0 && R(nb, 0) >= 0) {
                R(i, 0) = R(nb, 0);
                break;
            }
        }
    }
}

// ============================================================
// Quadric seed selection: per region, pick face with min error
// ============================================================

static vector<int> find_best_seeds_quadric(const MatrixXi& R,
                                            const vector<QuadricProxy>& QP,
                                            const MatrixXi& F, const MatrixXd& V,
                                            int num_proxies) {
    vector<int> seeds(num_proxies, -1);
    vector<double> best(num_proxies, 1e18);

    for (int i = 0; i < F.rows(); i++) {
        int j = R(i, 0);
        if (j < 0 || j >= num_proxies) continue;
        double d = face_quadric_error(i, QP[j], F, V);
        if (d < best[j]) {
            best[j] = d;
            seeds[j] = i;
        }
    }
    return seeds;
}

// ============================================================
// Per-region quadric fitting
// ============================================================

static vector<QuadricProxy> fit_all_quadric_proxies(const MatrixXi& R,
                                                      const MatrixXi& F,
                                                      const MatrixXd& V,
                                                      int num_proxies) {
    vector<QuadricProxy> QP(num_proxies);
    for (int j = 0; j < num_proxies; j++) {
        QP[j] = fit_quadric_region(R, j, F, V);
    }
    return QP;
}

// ============================================================
// Statistics computation (quadric mode)
// ============================================================

static IterationStats compute_stats_quadric(const MatrixXi& R,
                                             const MatrixXi& F,
                                             const MatrixXd& V,
                                             const vector<QuadricProxy>& QP,
                                             int num_proxies,
                                             const MatrixXi& prev_R) {
    IterationStats st;
    st.num_proxies = num_proxies;
    st.region_face_counts.assign(num_proxies, 0);
    st.region_errors.assign(num_proxies, 0.0);
    st.total_error = 0.0;
    st.changed_faces = 0;
    st.max_region_error = 0.0;

    int m = F.rows();
    vector<double> region_area(num_proxies, 0.0);

    for (int i = 0; i < m; i++) {
        int j = R(i, 0);
        if (j < 0 || j >= num_proxies) continue;
        st.region_face_counts[j]++;

        double terr = face_quadric_error(i, QP[j], F, V);
        st.total_error += terr;
        st.region_errors[j] += terr;
        region_area[j] += face_area(i, F, V);

        if (i < prev_R.rows() && R(i, 0) != prev_R(i, 0))
            st.changed_faces++;
    }

    for (int j = 0; j < num_proxies; j++) {
        if (region_area[j] > 1e-15)
            st.region_errors[j] /= region_area[j];
        if (st.region_errors[j] > st.max_region_error)
            st.max_region_error = st.region_errors[j];
    }
    return st;
}

// ============================================================
// Statistics computation (plane mode)
// ============================================================

static IterationStats compute_stats_plane(const MatrixXi& R,
                                           const MatrixXi& F,
                                           const MatrixXd& V,
                                           const MatrixXd& Proxies,
                                           MetricMode metric,
                                           int num_proxies,
                                           const MatrixXi& prev_R) {
    IterationStats st;
    st.num_proxies = num_proxies;
    st.region_face_counts.assign(num_proxies, 0);
    st.region_errors.assign(num_proxies, 0.0);
    st.total_error = 0.0;
    st.changed_faces = 0;
    st.max_region_error = 0.0;

    int m = F.rows();
    vector<double> region_area(num_proxies, 0.0);

    for (int i = 0; i < m; i++) {
        int j = R(i, 0);
        if (j < 0 || j >= num_proxies) continue;
        st.region_face_counts[j]++;
        double d = distance(i, Proxies.row(j), Proxies.row(j + num_proxies), V, metric);
        st.total_error += d;
        st.region_errors[j] += d;
        region_area[j] += get_area(i);

        if (i < prev_R.rows() && R(i, 0) != prev_R(i, 0))
            st.changed_faces++;
    }

    for (int j = 0; j < num_proxies; j++) {
        if (region_area[j] > 1e-15)
            st.region_errors[j] /= region_area[j];
        if (st.region_errors[j] > st.max_region_error)
            st.max_region_error = st.region_errors[j];
    }
    return st;
}

// ============================================================
// Print one iteration's log
// ============================================================

static void print_iteration_log(const IterationStats& st) {
    cout << "[iter " << st.iter
         << "] proxies=" << st.num_proxies
         << " total_err=" << st.total_error
         << " changed=" << st.changed_faces
         << " max_region_err=" << st.max_region_error << endl;

    for (int j = 0; j < st.num_proxies; j++) {
        cout << "  region " << j
             << ": faces=" << st.region_face_counts[j]
             << " norm_err=" << st.region_errors[j] << endl;
    }
}

// ============================================================
// Export: colored segmentation mesh (COFF format)
// ============================================================

static Vector3d region_color(int j) {
    double r = (sin(j * 1.7 + 0.3) * 0.5 + 0.5);
    double g = (sin(j * 2.3 + 1.1) * 0.5 + 0.5);
    double b = (sin(j * 3.1 + 2.7) * 0.5 + 0.5);
    return Vector3d(r, g, b);
}

static void export_colored_mesh(const MatrixXd& V, const MatrixXi& F,
                                 const MatrixXi& R, int num_proxies,
                                 const string& filename) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Cannot open " << filename << " for writing" << endl;
        return;
    }
    fout << "COFF" << endl;
    fout << V.rows() << " " << F.rows() << " 0" << endl;

    for (int i = 0; i < V.rows(); i++)
        fout << V(i,0) << " " << V(i,1) << " " << V(i,2) << " 255 255 255 255" << endl;

    for (int i = 0; i < F.rows(); i++) {
        int j = R(i, 0);
        Vector3d c = region_color(j);
        int cr = (int)(c(0) * 255);
        int cg = (int)(c(1) * 255);
        int cb = (int)(c(2) * 255);
        fout << "3 " << F(i,0) << " " << F(i,1) << " " << F(i,2)
             << " " << cr << " " << cg << " " << cb << " 255" << endl;
    }
    fout.close();
    cout << "Exported colored mesh: " << filename << endl;
}

// ============================================================
// Export: proxies.json (quadric)
// ============================================================

static void export_proxies_json_quadric(const vector<QuadricProxy>& QP,
                                         const MatrixXi& R, const MatrixXi& F,
                                         const MatrixXd& V, int num_proxies,
                                         const string& filename) {
    ofstream fout(filename);
    if (!fout.is_open()) { cerr << "Cannot write " << filename << endl; return; }

    fout << "{" << endl;
    fout << "  \"proxy_type\": \"quadric\"," << endl;
    fout << "  \"num_proxies\": " << num_proxies << "," << endl;
    fout << "  \"proxies\": [" << endl;

    for (int j = 0; j < num_proxies; j++) {
        fout << "    {" << endl;
        fout << "      \"coeffs\": [";
        VectorXd c = QP[j].coeffs;
        for (int k = 0; k < 10; k++) {
            fout << c(k);
            if (k < 9) fout << ", ";
        }
        fout << "]," << endl;

        int fc = 0;
        double area_sum = 0, err_sum = 0;
        for (int f = 0; f < F.rows(); f++) {
            if (R(f, 0) != j) continue;
            fc++;
            area_sum += face_area(f, F, V);
            err_sum += face_quadric_error(f, QP[j], F, V);
        }
        double norm_err = (area_sum > 1e-15) ? err_sum / area_sum : 0.0;

        fout << "      \"num_faces\": " << fc << "," << endl;
        fout << "      \"region_error\": " << norm_err << endl;
        fout << "    }";
        if (j < num_proxies - 1) fout << ",";
        fout << endl;
    }
    fout << "  ]" << endl;
    fout << "}" << endl;
    fout.close();
    cout << "Exported proxies: " << filename << endl;
}

// ============================================================
// Export: proxies.json (plane)
// ============================================================

static void export_proxies_json_plane(const MatrixXd& Proxies, int num_proxies,
                                       const MatrixXi& R, const MatrixXi& F,
                                       const MatrixXd& V, MetricMode metric,
                                       const string& filename) {
    ofstream fout(filename);
    if (!fout.is_open()) { cerr << "Cannot write " << filename << endl; return; }

    fout << "{" << endl;
    fout << "  \"proxy_type\": \"plane\"," << endl;
    fout << "  \"num_proxies\": " << num_proxies << "," << endl;
    fout << "  \"proxies\": [" << endl;

    for (int j = 0; j < num_proxies; j++) {
        Vector3d X = Proxies.row(j);
        Vector3d N = Proxies.row(j + num_proxies);

        int fc = 0;
        double area_sum = 0, err_sum = 0;
        for (int f = 0; f < F.rows(); f++) {
            if (R(f, 0) != j) continue;
            fc++;
            area_sum += get_area(f);
            err_sum += distance(f, X, N, V, metric);
        }
        double norm_err = (area_sum > 1e-15) ? err_sum / area_sum : 0.0;

        fout << "    {" << endl;
        fout << "      \"center\": [" << X(0) << ", " << X(1) << ", " << X(2) << "]," << endl;
        fout << "      \"normal\": [" << N(0) << ", " << N(1) << ", " << N(2) << "]," << endl;
        fout << "      \"num_faces\": " << fc << "," << endl;
        fout << "      \"region_error\": " << norm_err << endl;
        fout << "    }";
        if (j < num_proxies - 1) fout << ",";
        fout << endl;
    }
    fout << "  ]" << endl;
    fout << "}" << endl;
    fout.close();
    cout << "Exported proxies: " << filename << endl;
}

// ============================================================
// Export: iteration_log.csv
// ============================================================

static void export_iteration_log(const vector<IterationStats>& stats,
                                  int num_proxies,
                                  const string& filename) {
    ofstream fout(filename);
    if (!fout.is_open()) { cerr << "Cannot write " << filename << endl; return; }

    fout << "iter,num_proxies,total_error,changed_faces,max_region_error";
    // Use final num_proxies for header, but iterate per-stat for data
    for (int j = 0; j < num_proxies; j++)
        fout << ",region_" << j << "_faces,region_" << j << "_error";
    fout << endl;

    for (const auto& st : stats) {
        fout << st.iter << "," << st.num_proxies << ","
             << st.total_error << "," << st.changed_faces << ","
             << st.max_region_error;
        for (size_t j = 0; j < st.region_face_counts.size(); j++)
            fout << "," << st.region_face_counts[j] << "," << st.region_errors[j];
        fout << endl;
    }
    fout.close();
    cout << "Exported iteration log: " << filename << endl;
}

// ============================================================
// Main batch runner
// ============================================================

int run_vsa_batch(const string& model_name,
                  int num_proxies, ProxyType proxy_type,
                  int max_iter, unsigned int seed,
                  MatrixXi& R_out,
                  vector<IterationStats>& stats_out) {

    srand(seed);

    string file = "data/" + model_name + ".off";
    MatrixXd V;
    MatrixXi F;
    igl::readOFF(file, V, F);
    cout << "Loaded: " << file << "  (" << V.rows() << " verts, " << F.rows() << " faces)" << endl;

    MatrixXi Ad = face_adjacency(F, V.rows());
    initialize_normals_areas(F, V);

    MetricMode metric = L21_METRIC;

    // Step 1: Initial partition (always plane-based)
    cout << "Initial partition (plane-based, p=" << num_proxies << ")..." << endl;
    MatrixXi R;
    initial_partition(num_proxies, R, V, F, Ad, metric);
    cout << "  done" << endl;

    stats_out.clear();

    if (proxy_type == QUADRIC_PROXY) {
        cout << "Fitting initial quadric proxies..." << endl;
        vector<QuadricProxy> QP = fit_all_quadric_proxies(R, F, V, num_proxies);

        MatrixXi prev_R = R;
        IterationStats st0 = compute_stats_quadric(R, F, V, QP, num_proxies, prev_R);
        st0.iter = 0;
        print_iteration_log(st0);
        stats_out.push_back(st0);

        for (int it = 1; it <= max_iter; it++) {
            prev_R = R;

            vector<int> seeds = find_best_seeds_quadric(R, QP, F, V, num_proxies);
            flood_regions_quadric(R, seeds, QP, V, F, Ad);
            QP = fit_all_quadric_proxies(R, F, V, num_proxies);

            IterationStats st = compute_stats_quadric(R, F, V, QP, num_proxies, prev_R);
            st.iter = it;
            print_iteration_log(st);
            stats_out.push_back(st);

            if (st.changed_faces == 0) {
                cout << "  Converged (no changed faces) at iter " << it << endl;
                break;
            }
        }

        string base = model_name + "_quadric_p" + to_string(num_proxies);
        export_colored_mesh(V, F, R, num_proxies, base + "_segmentation.coff");
        export_proxies_json_quadric(QP, R, F, V, num_proxies, base + "_proxies.json");
        export_iteration_log(stats_out, num_proxies, base + "_log.csv");
        R_out = R;
    }
    else { // PLANE_PROXY
        cout << "Fitting initial plane proxies..." << endl;
        MatrixXd Proxies = new_proxies(R, F, V, num_proxies, metric);

        MatrixXi prev_R = R;
        IterationStats st0 = compute_stats_plane(R, F, V, Proxies, metric, num_proxies, prev_R);
        st0.iter = 0;
        print_iteration_log(st0);
        stats_out.push_back(st0);

        for (int it = 1; it <= max_iter; it++) {
            prev_R = R;

            proxy_color(R, Proxies, V, F, Ad, metric);
            Proxies = new_proxies(R, F, V, num_proxies, metric);

            IterationStats st = compute_stats_plane(R, F, V, Proxies, metric, num_proxies, prev_R);
            st.iter = it;
            print_iteration_log(st);
            stats_out.push_back(st);

            if (st.changed_faces == 0) {
                cout << "  Converged (no changed faces) at iter " << it << endl;
                break;
            }
        }

        string base = model_name + "_plane_p" + to_string(num_proxies);
        export_colored_mesh(V, F, R, num_proxies, base + "_segmentation.coff");
        export_proxies_json_plane(Proxies, num_proxies, R, F, V, metric, base + "_proxies.json");
        export_iteration_log(stats_out, num_proxies, base + "_log.csv");
        R_out = R;
    }

    cout << "Batch VSA complete." << endl;
    return 0;
}

// ============================================================
// Lloyd convergence helper: run up to max_iter Lloyd steps.
// Returns final stats. Works for both plane and quadric.
// ============================================================

static IterationStats run_lloyd_to_convergence(
    MatrixXi& R, const MatrixXd& V, const MatrixXi& F, const MatrixXi& Ad,
    ProxyType proxy_type, MetricMode metric,
    int num_proxies, int max_iter,
    vector<QuadricProxy>& QP, MatrixXd& Proxies,
    vector<IterationStats>& stats_out, int& global_iter)
{
    IterationStats st;
    st.changed_faces = -1;

    for (int it = 0; it < max_iter; it++) {
        MatrixXi prev_R = R;
        global_iter++;

        if (proxy_type == QUADRIC_PROXY) {
            vector<int> seeds = find_best_seeds_quadric(R, QP, F, V, num_proxies);
            flood_regions_quadric(R, seeds, QP, V, F, Ad);
            QP = fit_all_quadric_proxies(R, F, V, num_proxies);
            st = compute_stats_quadric(R, F, V, QP, num_proxies, prev_R);
        } else {
            proxy_color(R, Proxies, V, F, Ad, metric);
            Proxies = new_proxies(R, F, V, num_proxies, metric);
            st = compute_stats_plane(R, F, V, Proxies, metric, num_proxies, prev_R);
        }
        st.iter = global_iter;
        print_iteration_log(st);
        stats_out.push_back(st);

        if (st.changed_faces == 0) break;
    }
    return st;
}

// ============================================================
// Find worst region and its worst face (for splitting)
// ============================================================

struct SplitInfo {
    int region_id;
    double region_error;
    int seed_face;
    double seed_face_error;
};

static SplitInfo find_split_candidate(
    const MatrixXi& R, const MatrixXi& F, const MatrixXd& V,
    ProxyType proxy_type, const vector<QuadricProxy>& QP,
    const MatrixXd& Proxies, MetricMode metric,
    int num_proxies)
{
    // Compute per-region area-normalized error
    vector<double> region_err(num_proxies, 0.0);
    vector<double> region_area(num_proxies, 0.0);

    for (int i = 0; i < F.rows(); i++) {
        int j = R(i, 0);
        if (j < 0 || j >= num_proxies) continue;
        double e;
        if (proxy_type == QUADRIC_PROXY)
            e = face_quadric_error(i, QP[j], F, V);
        else
            e = distance(i, Proxies.row(j), Proxies.row(j + num_proxies), V, metric);
        region_err[j] += e;
        region_area[j] += face_area(i, F, V);
    }

    // Find worst region
    SplitInfo info;
    info.region_id = 0;
    info.region_error = 0.0;
    for (int j = 0; j < num_proxies; j++) {
        double norm_err = (region_area[j] > 1e-15) ? region_err[j] / region_area[j] : 0.0;
        if (norm_err > info.region_error) {
            info.region_error = norm_err;
            info.region_id = j;
        }
    }

    // Find worst face in that region
    info.seed_face = -1;
    info.seed_face_error = -1.0;
    for (int i = 0; i < F.rows(); i++) {
        if (R(i, 0) != info.region_id) continue;
        double e;
        if (proxy_type == QUADRIC_PROXY)
            e = face_quadric_error(i, QP[info.region_id], F, V);
        else
            e = distance(i, Proxies.row(info.region_id),
                         Proxies.row(info.region_id + num_proxies), V, metric);
        if (e > info.seed_face_error) {
            info.seed_face_error = e;
            info.seed_face = i;
        }
    }
    return info;
}

// ============================================================
// Insert one proxy: split worst region, add new seed, refit
// ============================================================

static void insert_one_proxy(
    MatrixXi& R, const MatrixXi& F, const MatrixXd& V,
    ProxyType proxy_type, int& num_proxies,
    vector<QuadricProxy>& QP, MatrixXd& Proxies,
    const SplitInfo& split)
{
    int old_p = num_proxies;
    int new_p = old_p + 1;
    num_proxies = new_p;

    // The worst face in the worst region becomes seed for the new proxy.
    // Reassign that single face to the new proxy id.
    R(split.seed_face, 0) = old_p;

    if (proxy_type == QUADRIC_PROXY) {
        // Fit the new proxy for the split-off face
        QP.push_back(fit_quadric_region(R, old_p, F, V));
        // Also refit the region that was split
        QP[split.region_id] = fit_quadric_region(R, split.region_id, F, V);
    } else {
        // Grow Proxies matrix: append one center + one normal row
        MatrixXd newProxies(new_p * 2, 3);
        newProxies.setZero();
        // Copy existing rows
        for (int j = 0; j < old_p; j++) {
            newProxies.row(j) = Proxies.row(j);
            newProxies.row(new_p + j) = Proxies.row(old_p + j);
        }
        // New proxy: plane of the seed face
        newProxies.row(old_p) = v3(V.row(F(split.seed_face, 0)) +
                                    V.row(F(split.seed_face, 1)) +
                                    V.row(F(split.seed_face, 2))) / 3.0;
        Vector3d n = v3(V.row(F(split.seed_face, 1)) - V.row(F(split.seed_face, 0)))
                    .cross(v3(V.row(F(split.seed_face, 2)) - V.row(F(split.seed_face, 0))));
        if (n.norm() > 1e-12) n.normalize();
        newProxies.row(new_p + old_p) = n;
        Proxies = newProxies;
    }
}

// ============================================================
// Export: insertion_log.csv
// ============================================================

static void export_insertion_log(const vector<InsertionStep>& log,
                                  const string& filename) {
    ofstream fout(filename);
    if (!fout.is_open()) { cerr << "Cannot write " << filename << endl; return; }
    fout << "step,K_before,K_after,split_mode,split_region,split_region_error,"
         << "seed_face,seed_face_error,total_error_after_lloyd,"
         << "validity_priority,invalid_reason,detected_quadric_type,"
         << "condition_H,condition_Q,"
         << "max_region_error_before,max_region_error_after,"
         << "invalid_proxy_count_before,invalid_proxy_count_after,"
         << "stop_reason" << endl;
    for (const auto& s : log) {
        fout << s.step << ","
             << s.num_proxies_before << ","
             << (s.num_proxies_before + 1) << ","
             << s.split_mode << ","
             << s.split_region << ","
             << s.split_region_error << ","
             << s.seed_face << ","
             << s.seed_face_error << ","
             << s.total_error_after_lloyd << ","
             << s.validity_priority << ","
             << "\"" << s.invalid_reason << "\","
             << "\"" << s.detected_quadric_type << "\","
             << s.condition_H << ","
             << s.condition_Q << ","
             << s.max_region_error_before << ","
             << s.max_region_error_after << ","
             << s.invalid_proxy_count_before << ","
             << s.invalid_proxy_count_after << ","
             << "\"" << s.stop_reason << "\"" << endl;
    }
    fout.close();
    cout << "Exported insertion log: " << filename << endl;
}

// ============================================================
// Proxy merging
// ============================================================

// Build set of adjacent region pairs (ri < rj)
static set<pair<int,int>> build_region_adjacency(
    const MatrixXi& R, const MatrixXi& Ad, int num_proxies)
{
    set<pair<int,int>> adj;
    int m = R.rows();
    for (int fi = 0; fi < m; fi++) {
        int ri = R(fi, 0);
        if (ri < 0 || ri >= num_proxies) continue;
        for (int k = 0; k < 3; k++) {
            int nb = Ad(fi, k);
            if (nb <= 0 || nb >= m) continue;
            int rj = R(nb, 0);
            if (rj < 0 || rj >= num_proxies || rj == ri) continue;
            adj.insert(make_pair(min(ri, rj), max(ri, rj)));
        }
    }
    return adj;
}

struct MergeCandidate {
    int ri, rj;
    double E_i, E_j, E_t;
    double delta;
};

// Find merge candidates: adjacent pairs where merging doesn't increase error beyond epsilon
static vector<MergeCandidate> find_merge_candidates(
    const MatrixXi& R, const MatrixXi& F, const MatrixXd& V, const MatrixXi& Ad,
    const vector<QuadricProxy>& QP, int num_proxies, double epsilon,
    vector<MergeStep>& all_evaluated)
{
    set<pair<int,int>> adj = build_region_adjacency(R, Ad, num_proxies);
    vector<MergeCandidate> accepted;

    for (auto& [ri, rj] : adj) {
        // Feature barrier: don't merge across feature edges
        if (g_feature_barrier_enabled && !g_feature_edges.empty() &&
            count_feature_edges_on_boundary(R, F, Ad, ri, rj) > 0) {
            continue;
        }

        MergeStep ms;
        ms.region_i = ri;
        ms.region_j = rj;
        ms.epsilon = epsilon;
        ms.accepted = false;

        // Compute individual region errors
        ms.E_i = quadric_region_error(R, ri, F, V, QP[ri]);
        ms.E_j = quadric_region_error(R, rj, F, V, QP[rj]);

        // Create temporary R with merged regions
        MatrixXi R_tmp = R;
        for (int fi = 0; fi < R_tmp.rows(); fi++) {
            if (R_tmp(fi, 0) == rj) R_tmp(fi, 0) = ri;
        }

        // Fit proxy for merged region
        QuadricProxy merged = fit_quadric_region(R_tmp, ri, F, V);
        ms.E_t = quadric_region_error(R_tmp, ri, F, V, merged);

        double delta = abs(ms.E_t - (ms.E_i + ms.E_j));

        if (delta < epsilon) {
            MergeCandidate mc;
            mc.ri = ri; mc.rj = rj;
            mc.E_i = ms.E_i; mc.E_j = ms.E_j;
            mc.E_t = ms.E_t; mc.delta = delta;
            accepted.push_back(mc);
            ms.accepted = true;
            ms.reject_reason = "";
        } else {
            ms.reject_reason = "delta=" + to_string(delta) + " >= epsilon=" + to_string(epsilon);
        }
        all_evaluated.push_back(ms);
    }

    // Sort by delta ascending (best merge first)
    sort(accepted.begin(), accepted.end(),
         [](const MergeCandidate& a, const MergeCandidate& b) {
             return a.delta < b.delta;
         });
    return accepted;
}

// Execute merge: reassign faces, remove proxy, renumber
static void execute_merge(MatrixXi& R, vector<QuadricProxy>& QP,
                           int& num_proxies, int ri, int rj,
                           const MatrixXi& F, const MatrixXd& V)
{
    // Reassign all rj faces to ri
    for (int fi = 0; fi < R.rows(); fi++) {
        if (R(fi, 0) == rj) R(fi, 0) = ri;
    }
    // Remove proxy at index rj
    QP.erase(QP.begin() + rj);
    // Renumber: all labels > rj get decremented
    for (int fi = 0; fi < R.rows(); fi++) {
        if (R(fi, 0) > rj) R(fi, 0)--;
    }
    num_proxies--;
    // Refit merged region
    QP[ri] = fit_quadric_region(R, ri, F, V);
}

// Plane mode merge: compute centroid and normal of union
static void execute_merge_plane(MatrixXi& R, MatrixXd& Proxies,
                                 int& num_proxies, int ri, int rj,
                                 const MatrixXi& F, const MatrixXd& V)
{
    // Reassign all rj faces to ri
    for (int fi = 0; fi < R.rows(); fi++) {
        if (R(fi, 0) == rj) R(fi, 0) = ri;
    }
    // Renumber labels > rj
    for (int fi = 0; fi < R.rows(); fi++) {
        if (R(fi, 0) > rj) R(fi, 0)--;
    }

    // Rebuild Proxies matrix without rj
    int old_p = num_proxies;
    num_proxies--;
    MatrixXd newProxies(num_proxies * 2, 3);
    newProxies.setZero();
    for (int j = 0; j < old_p; j++) {
        int dest = (j < rj) ? j : (j - 1);
        newProxies.row(dest) = Proxies.row(j);
        newProxies.row(num_proxies + dest) = Proxies.row(old_p + j);
    }
    Proxies = newProxies;

    // Refit ri's centroid and normal from all faces in the merged region
    Vector3d centroid = Vector3d::Zero();
    Vector3d normal = Vector3d::Zero();
    int count = 0;
    for (int fi = 0; fi < F.rows(); fi++) {
        if (R(fi, 0) != ri) continue;
        count++;
        Vector3d v0 = v3(V.row(F(fi, 0)));
        Vector3d v1 = v3(V.row(F(fi, 1)));
        Vector3d v2 = v3(V.row(F(fi, 2)));
        centroid += (v0 + v1 + v2) / 3.0;
        Vector3d n = (v1 - v0).cross(v2 - v0);
        if (n.norm() > 1e-12) normal += n.normalized();
    }
    if (count > 0) centroid /= count;
    if (normal.norm() > 1e-12) normal.normalize();
    Proxies.row(ri) = centroid;
    Proxies.row(num_proxies + ri) = normal;
}

// Plane mode merge candidate evaluation
static double plane_region_error(int region_id, const MatrixXi& R,
                                  const MatrixXi& F, const MatrixXd& V,
                                  const MatrixXd& Proxies, int num_proxies,
                                  MetricMode metric)
{
    double total_err = 0, total_area = 0;
    Vector3d center = v3(Proxies.row(region_id));
    Vector3d normal = v3(Proxies.row(num_proxies + region_id));
    for (int fi = 0; fi < F.rows(); fi++) {
        if (R(fi, 0) != region_id) continue;
        Vector3d v0 = v3(V.row(F(fi, 0)));
        Vector3d v1 = v3(V.row(F(fi, 1)));
        Vector3d v2 = v3(V.row(F(fi, 2)));
        double area = 0.5 * (v1 - v0).cross(v2 - v0).norm();
        Vector3d tri_center = (v0 + v1 + v2) / 3.0;
        double d;
        if (metric == L2_METRIC) {
            d = (tri_center - center).squaredNorm();
        } else {
            Vector3d n = (v1 - v0).cross(v2 - v0);
            if (n.norm() > 1e-12) n.normalize();
            else continue;
            d = pow(n.dot(normal), 2);
        }
        total_err += area * d;
        total_area += area;
    }
    return (total_area > 1e-15) ? total_err / total_area : 0;
}

static vector<pair<int,int>> find_merge_candidates_plane(
    const MatrixXi& R, const MatrixXi& F, const MatrixXd& V, const MatrixXi& Ad,
    const MatrixXd& Proxies, int num_proxies, MetricMode metric, double epsilon,
    vector<MergeStep>& all_evaluated)
{
    set<pair<int,int>> adj = build_region_adjacency(R, Ad, num_proxies);
    vector<pair<double,pair<int,int>>> scored;

    for (auto& [ri, rj] : adj) {
        // Feature barrier: don't merge across feature edges
        if (g_feature_barrier_enabled && !g_feature_edges.empty() &&
            count_feature_edges_on_boundary(R, F, Ad, ri, rj) > 0) {
            continue;
        }

        MergeStep ms;
        ms.region_i = ri; ms.region_j = rj;
        ms.epsilon = epsilon; ms.accepted = false;

        ms.E_i = plane_region_error(ri, R, F, V, Proxies, num_proxies, metric);
        ms.E_j = plane_region_error(rj, R, F, V, Proxies, num_proxies, metric);

        // Temporary merge: compute centroid and normal of union
        Vector3d centroid = Vector3d::Zero();
        Vector3d normal = Vector3d::Zero();
        int count = 0;
        for (int fi = 0; fi < F.rows(); fi++) {
            if (R(fi, 0) != ri && R(fi, 0) != rj) continue;
            count++;
            Vector3d v0 = v3(V.row(F(fi, 0)));
            Vector3d v1 = v3(V.row(F(fi, 1)));
            Vector3d v2 = v3(V.row(F(fi, 2)));
            centroid += (v0 + v1 + v2) / 3.0;
            Vector3d n = (v1 - v0).cross(v2 - v0);
            if (n.norm() > 1e-12) normal += n.normalized();
        }
        if (count > 0) centroid /= count;
        if (normal.norm() > 1e-12) normal.normalize();

        // Compute merged error with temporary proxy
        double E_t = 0, total_area = 0;
        for (int fi = 0; fi < F.rows(); fi++) {
            if (R(fi, 0) != ri && R(fi, 0) != rj) continue;
            Vector3d v0 = v3(V.row(F(fi, 0)));
            Vector3d v1 = v3(V.row(F(fi, 1)));
            Vector3d v2 = v3(V.row(F(fi, 2)));
            double area = 0.5 * (v1 - v0).cross(v2 - v0).norm();
            Vector3d tri_center = (v0 + v1 + v2) / 3.0;
            double d;
            if (metric == L2_METRIC) {
                d = (tri_center - centroid).squaredNorm();
            } else {
                Vector3d n = (v1 - v0).cross(v2 - v0);
                if (n.norm() > 1e-12) n.normalize();
                else continue;
                d = pow(n.dot(normal), 2);
            }
            E_t += area * d;
            total_area += area;
        }
        ms.E_t = (total_area > 1e-15) ? E_t / total_area : 0;

        double delta = abs(ms.E_t - (ms.E_i + ms.E_j));
        if (delta < epsilon) {
            scored.push_back({delta, {ri, rj}});
            ms.accepted = true;
            ms.reject_reason = "";
        } else {
            ms.reject_reason = "delta=" + to_string(delta) + " >= epsilon";
        }
        all_evaluated.push_back(ms);
    }

    sort(scored.begin(), scored.end());
    vector<pair<int,int>> result;
    for (auto& [d, p] : scored) result.push_back(p);
    return result;
}

// ============================================================
// Export: merge_log.csv
// ============================================================

static void export_merge_log(const vector<MergeStep>& log, const string& filename) {
    ofstream fout(filename);
    if (!fout.is_open()) { cerr << "Cannot write " << filename << endl; return; }
    fout << "step,region_i,region_j,E_i,E_j,E_t,epsilon,accepted,reject_reason" << endl;
    for (const auto& m : log) {
        fout << m.step << "," << m.region_i << "," << m.region_j << ","
             << m.E_i << "," << m.E_j << "," << m.E_t << ","
             << m.epsilon << "," << (m.accepted ? "true" : "false") << ","
             << "\"" << m.reject_reason << "\"" << endl;
    }
    fout.close();
    cout << "Exported merge log: " << filename << endl;
}

// ============================================================
// Progressive VSA runner
// ============================================================

int run_vsa_progressive(const string& model_name,
                        int init_proxies, ProxyType proxy_type,
                        int target_proxies, double error_threshold,
                        int lloyd_iter_per_insert, unsigned int seed,
                        MatrixXi& R_out,
                        vector<IterationStats>& stats_out,
                        vector<InsertionStep>& insertion_log_out,
                        vector<MergeStep>& merge_log_out,
                        vector<SmoothLogEntry>& smooth_log_out,
                        bool enable_merge,
                        bool enable_smooth,
                        SmoothConfig smooth_cfg,
                        const ProxyValidityConfig& validity_cfg,
                        bool enable_classify,
                        double classify_eps,
                        bool enable_projection,
                        ProjectionConfig proj_cfg,
                        bool validity_guided,
                        int max_validity_split_attempts,
                        int min_faces_to_split,
                        bool export_validity_each_step)
{
    srand(seed);
    stats_out.clear();
    insertion_log_out.clear();
    merge_log_out.clear();
    smooth_log_out.clear();

    string file = "data/" + model_name + ".off";
    MatrixXd V;
    MatrixXi F;
    igl::readOFF(file, V, F);
    cout << "Loaded: " << file << "  (" << V.rows() << " verts, " << F.rows() << " faces)" << endl;

    MatrixXi Ad = face_adjacency(F, V.rows());
    initialize_normals_areas(F, V);
    MetricMode metric = L21_METRIC;

    int num_proxies = max(init_proxies, 2);
    int global_iter = 0;

    cout << "=== Progressive VSA ===" << endl;
    cout << "  init_proxies:  " << num_proxies << endl;
    cout << "  target:        " << target_proxies << " proxies / error < " << error_threshold << endl;
    cout << "  lloyd/insert:  " << lloyd_iter_per_insert << endl;

    // Step 1: Initial partition with init_proxies
    cout << "\n[init] Partition with " << num_proxies << " proxies..." << endl;
    MatrixXi R;
    initial_partition(num_proxies, R, V, F, Ad, metric);

    // Fit initial proxies
    vector<QuadricProxy> QP;
    MatrixXd Proxies;
    if (proxy_type == QUADRIC_PROXY) {
        QP = fit_all_quadric_proxies(R, F, V, num_proxies);
    } else {
        Proxies = new_proxies(R, F, V, num_proxies, metric);
    }

    // Step 2: Initial Lloyd convergence
    cout << "\n[init] Lloyd convergence (" << lloyd_iter_per_insert << " iters)..." << endl;
    IterationStats st = run_lloyd_to_convergence(
        R, V, F, Ad, proxy_type, metric, num_proxies,
        lloyd_iter_per_insert, QP, Proxies, stats_out, global_iter);

    // Validity check + repair after initial Lloyd
    if (proxy_type == QUADRIC_PROXY && validity_cfg.enable_basic) {
        for (int repair = 0; repair < 3; repair++) {
            cout << "\n[validity] Checking " << num_proxies << " proxies (round " << (repair+1) << ")..." << endl;
            auto reports = check_all_proxies(QP, R, F, V, validity_cfg);
            vector<int> invalid;
            for (auto& rpt : reports) {
                print_proxy_validity(rpt);
                if (!rpt.is_valid) invalid.push_back(rpt.proxy_id);
            }
            if (invalid.empty()) break;
            cout << "[validity] " << invalid.size() << " invalid proxies, inserting repairs..." << endl;
            for (int rid : invalid) {
                int wf = find_worst_face_in_region(QP[rid], rid, R, F, V);
                if (wf < 0) continue;
                SplitInfo split{rid, reports[rid].region_error, wf, 0.0};
                insert_one_proxy(R, F, V, proxy_type, num_proxies, QP, Proxies, split);
                cout << "  repaired proxy " << rid << " -> new proxy, total=" << num_proxies << endl;
            }
            st = run_lloyd_to_convergence(R, V, F, Ad, proxy_type, metric,
                num_proxies, lloyd_iter_per_insert, QP, Proxies, stats_out, global_iter);
        }
    }

    // Step 3: Progressive insertion loop
    int prev_invalid_count = 0;
    int consecutive_validity_splits = 0;

    for (int step = 1; ; step++) {
        // === Validity check ===
        vector<ProxyValidityReport> reports;
        int invalid_count = 0;
        if (validity_guided && proxy_type == QUADRIC_PROXY) {
            ProxyValidityConfig full_cfg = validity_cfg;
            full_cfg.enable_basic = true;
            full_cfg.enable_degeneracy = true;
            full_cfg.enable_classification = true;
            full_cfg.enable_two_sheet = true;

            reports = check_all_proxies(QP, R, F, V, full_cfg);
            for (auto& rpt : reports) {
                if (!rpt.is_valid || rpt.is_suspicious) invalid_count++;
            }
            cout << "[Progressive] K=" << num_proxies
                 << " max_region_error=" << st.max_region_error
                 << " invalid_proxy_count=" << invalid_count << endl;

            if (export_validity_each_step) {
                export_proxy_validity_log(reports,
                    model_name + "_validity_K" + to_string(num_proxies) + ".csv");
            }
        }

        // === Stopping criteria ===
        bool stop = false;
        string stop_reason;

        if (num_proxies >= (int)(F.rows() / 2)) {
            stop = true;
            stop_reason = "too_many_proxies";
        }
        if (!stop && target_proxies > 0 && num_proxies >= target_proxies) {
            if (!validity_guided || invalid_count == 0) {
                stop = true;
                stop_reason = "target_reached";
            } else {
                cout << "  [validity] Target reached but " << invalid_count
                     << " invalid proxies remain, continuing." << endl;
            }
        }
        if (!stop && error_threshold > 0 && st.max_region_error < error_threshold) {
            if (!validity_guided || invalid_count == 0) {
                stop = true;
                stop_reason = "error_threshold_met";
            } else {
                cout << "  [validity] Error threshold met but " << invalid_count
                     << " invalid proxies remain, continuing." << endl;
            }
        }
        if (stop) {
            cout << "\n[stop] " << stop_reason << " (K=" << num_proxies << ")" << endl;
            if (!insertion_log_out.empty())
                insertion_log_out.back().stop_reason = stop_reason;
            break;
        }

        // === Choose split ===
        string split_mode = "max_error";
        SplitInfo split;
        split.region_id = -1;
        split.seed_face = -1;
        double split_condition_H = 0, split_condition_Q = 0;
        int split_priority = 0;
        string split_invalid_reason;
        string split_quadric_type;

        if (validity_guided && invalid_count > 0 && proxy_type == QUADRIC_PROXY &&
            consecutive_validity_splits < max_validity_split_attempts) {
            int region_id = choose_region_to_split_by_validity(reports, min_faces_to_split);
            if (region_id >= 0) {
                auto& rpt = reports[region_id];
                int seed = choose_seed_face_for_invalid_region(
                    region_id, rpt, R, F, V, QP);
                if (seed >= 0) {
                    split.region_id = region_id;
                    split.region_error = rpt.region_error;
                    split.seed_face = seed;
                    split.seed_face_error = face_quadric_error(seed, QP[region_id], F, V);
                    split_condition_H = rpt.h_condition;
                    split_condition_Q = rpt.q_condition;
                    split_priority = rpt.priority;
                    split_invalid_reason = rpt.invalid_reason_str;
                    split_quadric_type = quadric_type_name(rpt.quadric_type);

                    if (!rpt.is_valid) split_mode = "invalid_proxy";
                    else split_mode = "suspicious_proxy";
                }
            }
        }

        if (split.seed_face < 0) {
            // Fallback to max-error split
            split = find_split_candidate(
                R, F, V, proxy_type, QP, Proxies, metric, num_proxies);
            split_mode = "max_error";
        }

        if (split.seed_face < 0) {
            cout << "\n[stop] No valid split face found." << endl;
            break;
        }

        cout << "\n=== Insertion step " << step << " ===" << endl;
        cout << "  selected split mode = " << split_mode << endl;
        cout << "  selected region = " << split.region_id
             << " (norm_err=" << split.region_error << ")" << endl;
        cout << "  selected seed face = " << split.seed_face
             << " (face_err=" << split.seed_face_error << ")" << endl;
        if (split_mode != "max_error") {
            cout << "  reason = " << split_invalid_reason
                 << " (priority=" << split_priority << ")" << endl;
        }

        // Build log entry
        InsertionStep ilog;
        ilog.step = step;
        ilog.num_proxies_before = num_proxies;
        ilog.split_region = split.region_id;
        ilog.split_region_error = split.region_error;
        ilog.seed_face = split.seed_face;
        ilog.seed_face_error = split.seed_face_error;
        ilog.split_mode = split_mode;
        ilog.invalid_reason = split_invalid_reason;
        ilog.detected_quadric_type = split_quadric_type;
        ilog.validity_priority = split_priority;
        ilog.condition_H = split_condition_H;
        ilog.condition_Q = split_condition_Q;
        ilog.max_region_error_before = st.max_region_error;
        ilog.invalid_proxy_count_before = invalid_count;

        // Insert proxy
        insert_one_proxy(R, F, V, proxy_type, num_proxies, QP, Proxies, split);
        cout << "  proxies: " << (num_proxies - 1) << " -> " << num_proxies << endl;

        // Lloyd convergence after insertion
        cout << "  Lloyd convergence..." << endl;
        st = run_lloyd_to_convergence(
            R, V, F, Ad, proxy_type, metric, num_proxies,
            lloyd_iter_per_insert, QP, Proxies, stats_out, global_iter);

        ilog.total_error_after_lloyd = st.total_error;
        ilog.max_region_error_after = st.max_region_error;

        // Track consecutive validity splits
        if (split_mode != "max_error") {
            if (invalid_count >= prev_invalid_count) {
                consecutive_validity_splits++;
            } else {
                consecutive_validity_splits = 0;
            }
        } else {
            consecutive_validity_splits = 0;
        }
        prev_invalid_count = invalid_count;

        // Post-insertion validity count for logging
        if (validity_guided && proxy_type == QUADRIC_PROXY) {
            ProxyValidityConfig full_cfg = validity_cfg;
            full_cfg.enable_basic = true;
            full_cfg.enable_degeneracy = true;
            full_cfg.enable_classification = true;
            full_cfg.enable_two_sheet = true;
            auto post_reports = check_all_proxies(QP, R, F, V, full_cfg);
            int post_invalid = 0;
            for (auto& rpt : post_reports)
                if (!rpt.is_valid || rpt.is_suspicious) post_invalid++;
            ilog.invalid_proxy_count_after = post_invalid;
        }

        insertion_log_out.push_back(ilog);

        cout << "  After insertion: total_err=" << st.total_error
             << " max_region_err=" << st.max_region_error
             << " invalid_proxies=" << ilog.invalid_proxy_count_after << endl;

        if (consecutive_validity_splits >= max_validity_split_attempts) {
            cout << "[warning] " << max_validity_split_attempts
                 << " consecutive validity splits without improvement" << endl;
        }
    }

    // Post-loop merge phase
    if (enable_merge) {
        int merge_step = 0;
        for (;;) {
            double epsilon = 0.5 * st.max_region_error;
            if (epsilon <= 0 || num_proxies < 2) break;

            vector<MergeStep> evaluated;
            bool merged_any = false;

            if (proxy_type == QUADRIC_PROXY) {
                auto candidates = find_merge_candidates(
                    R, F, V, Ad, QP, num_proxies, epsilon, evaluated);
                if (!candidates.empty()) {
                    auto& best = candidates[0];
                    merge_step++;
                    cout << "\n=== Merge step " << merge_step << " ===" << endl;
                    cout << "  merging regions " << best.ri << " + " << best.rj
                         << " (E_i=" << best.E_i << " E_j=" << best.E_j
                         << " E_t=" << best.E_t << " delta=" << best.delta
                         << " epsilon=" << epsilon << ")" << endl;
                    execute_merge(R, QP, num_proxies, best.ri, best.rj, F, V);
                    cout << "  proxies: " << (num_proxies + 1) << " -> " << num_proxies << endl;

                    // Record accepted merge
                    MergeStep accepted;
                    accepted.step = merge_step;
                    accepted.region_i = best.ri; accepted.region_j = best.rj;
                    accepted.E_i = best.E_i; accepted.E_j = best.E_j;
                    accepted.E_t = best.E_t; accepted.epsilon = epsilon;
                    accepted.accepted = true;
                    merge_log_out.push_back(accepted);

                    // Lloyd convergence after merge
                    st = run_lloyd_to_convergence(R, V, F, Ad, proxy_type, metric,
                        num_proxies, lloyd_iter_per_insert, QP, Proxies,
                        stats_out, global_iter);
                    cout << "  After merge Lloyd: total_err=" << st.total_error
                         << " max_region_err=" << st.max_region_error << endl;
                    merged_any = true;
                }
            } else {
                auto candidates = find_merge_candidates_plane(
                    R, F, V, Ad, Proxies, num_proxies, metric, epsilon, evaluated);
                if (!candidates.empty()) {
                    auto& [ri, rj] = candidates[0];
                    merge_step++;
                    cout << "\n=== Merge step " << merge_step << " ===" << endl;
                    cout << "  merging regions " << ri << " + " << rj << endl;
                    execute_merge_plane(R, Proxies, num_proxies, ri, rj, F, V);
                    cout << "  proxies: " << (num_proxies + 1) << " -> " << num_proxies << endl;

                    MergeStep accepted;
                    accepted.step = merge_step;
                    accepted.region_i = ri; accepted.region_j = rj;
                    accepted.E_i = 0; accepted.E_j = 0;
                    accepted.E_t = 0; accepted.epsilon = epsilon;
                    accepted.accepted = true;
                    merge_log_out.push_back(accepted);

                    st = run_lloyd_to_convergence(R, V, F, Ad, proxy_type, metric,
                        num_proxies, lloyd_iter_per_insert, QP, Proxies,
                        stats_out, global_iter);
                    merged_any = true;
                }
            }

            // Record rejected candidates
            for (auto& ms : evaluated) {
                if (!ms.accepted) {
                    ms.step = merge_step;
                    merge_log_out.push_back(ms);
                }
            }

            if (!merged_any) break;
        }
    }

    // Export
    string ptype_str = (proxy_type == QUADRIC_PROXY) ? "quadric" : "plane";
    string base = model_name + "_progressive_" + ptype_str + "_p" + to_string(num_proxies);

    // Boundary smoothing
    if (enable_smooth) {
        export_colored_mesh(V, F, R, num_proxies, base + "_before_smoothing.coff");
        smooth_boundaries(R, F, V, Ad, num_proxies, proxy_type,
                          QP, Proxies, metric, smooth_cfg, smooth_log_out);
        if (!smooth_log_out.empty())
            export_smooth_log(smooth_log_out, base + "_boundary_smoothing_log.csv");
    }

    export_colored_mesh(V, F, R, num_proxies, base + "_segmentation.coff");
    if (proxy_type == QUADRIC_PROXY)
        export_proxies_json_quadric(QP, R, F, V, num_proxies, base + "_proxies.json");
    else
        export_proxies_json_plane(Proxies, num_proxies, R, F, V, metric, base + "_proxies.json");
    export_iteration_log(stats_out, num_proxies, base + "_log.csv");
    export_insertion_log(insertion_log_out, base + "_insertion_log.csv");
    if (enable_merge && !merge_log_out.empty())
        export_merge_log(merge_log_out, base + "_merge_log.csv");

    // Final validity report
    if (proxy_type == QUADRIC_PROXY && validity_cfg.enable_basic) {
        auto final_reports = check_all_proxies(QP, R, F, V, validity_cfg);
        cout << "\n=== Final Proxy Validity ===" << endl;
        for (auto& rpt : final_reports) print_proxy_validity(rpt);
        export_proxy_validity_log(final_reports, base + "_proxy_validity_log.csv");
    }

    // Quadric proxy classification (post-processing only)
    vector<ClassifiedType> proxy_types;
    if (enable_classify && proxy_type == QUADRIC_PROXY) {
        cout << "\n=== Quadric Proxy Classification (eps=" << classify_eps << ") ===" << endl;
        auto class_reports = classify_all_proxies(QP, R, F, V, classify_eps);
        for (auto& rpt : class_reports) print_classification_report(rpt);
        export_proxy_types_json(class_reports, base + "_proxy_types.json");

        for (auto& rpt : class_reports) proxy_types.push_back(rpt.type);

        // Summary
        map<string,int> type_counts;
        for (auto& rpt : class_reports) type_counts[rpt.type_name]++;
        cout << "\n  Classification summary:" << endl;
        for (auto& [name, cnt] : type_counts)
            cout << "    " << name << ": " << cnt << endl;
    }

    // Proxy projection (post-processing)
    if (enable_projection) {
        cout << "\n=== Proxy Projection ===" << endl;

        // If classification wasn't run but we're in quadric mode, classify now
        if (proxy_type == QUADRIC_PROXY && proxy_types.empty()) {
            auto class_reports = classify_all_proxies(QP, R, F, V, 0.1);
            for (auto& rpt : class_reports) proxy_types.push_back(rpt.type);
        }
        // Ensure proxy_types has correct size
        while ((int)proxy_types.size() < num_proxies)
            proxy_types.push_back(TYPE_GENERAL_QUADRIC);

        ProjectionLog proj_log;
        MatrixXd V_proj = project_vertices(V, F, R, num_proxies, proxy_type,
                                            QP, Proxies, proxy_types, proj_cfg, proj_log);

        export_obj(V_proj, F, base + "_projected_mesh.obj");
        export_projection_log(proj_log, num_proxies, base + "_projection_log.csv");

        cout << "  vertices: " << proj_log.total_vertices
             << " (interior=" << proj_log.interior_count
             << " boundary=" << proj_log.boundary_count << ")" << endl;
        cout << "  success=" << proj_log.success_count
             << " fallback=" << proj_log.fallback_count
             << " failure=" << proj_log.failure_count << endl;
    }

    R_out = R;
    cout << "\nProgressive VSA complete. Final proxies: " << num_proxies << endl;
    return 0;
}

// ============================================================
// Merge pass (reuses existing static functions)
// ============================================================

int run_merge_pass(MatrixXi& R, vector<QuadricProxy>& QP, MatrixXd& Proxies,
                   int& num_proxies, ProxyType proxy_type, MetricMode metric,
                   const MatrixXi& F, const MatrixXd& V, const MatrixXi& Ad,
                   double relative_threshold, int max_iterations) {
    if (num_proxies < 2) return 0;
    int merge_count = 0;

    for (int it = 0; it < max_iterations; it++) {
        // Compute max region error for epsilon
        vector<double> re(num_proxies, 0), ra(num_proxies, 0);
        for (int i = 0; i < F.rows(); i++) {
            int j = R(i, 0);
            if (j < 0 || j >= num_proxies) continue;
            double e;
            if (proxy_type == QUADRIC_PROXY)
                e = face_quadric_error(i, QP[j], F, V);
            else
                e = distance(i, Proxies.row(j), Proxies.row(j + num_proxies), V, metric);
            re[j] += e;
            ra[j] += face_area(i, F, V);
        }
        double max_ne = 0;
        for (int j = 0; j < num_proxies; j++) {
            double ne = ra[j] > 1e-15 ? re[j] / ra[j] : 0;
            if (ne > max_ne) max_ne = ne;
        }

        double epsilon = relative_threshold * max_ne;
        if (epsilon < 1e-15) break;

        bool merged = false;
        if (proxy_type == QUADRIC_PROXY) {
            vector<MergeStep> evaluated;
            auto candidates = find_merge_candidates(
                R, F, V, Ad, QP, num_proxies, epsilon, evaluated);
            if (!candidates.empty()) {
                auto& best = candidates[0];
                cout << "  merge: region " << best.ri << " + " << best.rj
                     << " (delta=" << best.delta << " < eps=" << epsilon << ")" << endl;
                execute_merge(R, QP, num_proxies, best.ri, best.rj, F, V);
                merge_count++;
                merged = true;
            }
        } else {
            vector<MergeStep> evaluated;
            auto candidates = find_merge_candidates_plane(
                R, F, V, Ad, Proxies, num_proxies, metric, epsilon, evaluated);
            if (!candidates.empty()) {
                auto& [ri, rj] = candidates[0];
                cout << "  merge: region " << ri << " + " << rj << endl;
                execute_merge_plane(R, Proxies, num_proxies, ri, rj, F, V);
                merge_count++;
                merged = true;
            }
        }
        if (!merged) break;
    }
    return merge_count;
}
