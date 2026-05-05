#include "proxy_classification.h"
#include <cmath>
#include <fstream>
#include <algorithm>

static inline Vector3d v3(const Eigen::RowVectorXd& r) {
    return Vector3d(r(0), r(1), r(2));
}

string classified_type_name(ClassifiedType t) {
    switch (t) {
        case TYPE_PLANE:                   return "Plane";
        case TYPE_SPHERE:                  return "Sphere";
        case TYPE_CIRCULAR_CYLINDER:       return "CircularCylinder";
        case TYPE_ELLIPSOID:               return "Ellipsoid";
        case TYPE_HYPERBOLOID_ONE_SHEET:   return "HyperboloidOneSheet";
        case TYPE_HYPERBOLOID_TWO_SHEETS:  return "HyperboloidTwoSheets";
        case TYPE_PARABOLOID:              return "Paraboloid";
        case TYPE_DEGENERATE:              return "Degenerate";
        case TYPE_GENERAL_QUADRIC:         return "GeneralQuadric";
        default:                           return "Unknown";
    }
}

static void compute_region_stats(int region_id, const MatrixXi& R,
                                  const MatrixXi& F, const MatrixXd& V,
                                  const QuadricProxy& proxy,
                                  int& num_faces, double& region_error) {
    num_faces = 0;
    region_error = 0;
    double total_area = 0;
    for (int i = 0; i < F.rows(); i++) {
        if (R(i, 0) != region_id) continue;
        num_faces++;
        Vector3i f = F.row(i);
        Vector3d v0 = v3(V.row(f(0)));
        Vector3d v1 = v3(V.row(f(1)));
        Vector3d v2 = v3(V.row(f(2)));
        double area = 0.5 * (v1 - v0).cross(v2 - v0).norm();
        region_error += proxy.triangle_error(v0, v1, v2);
        total_area += area;
    }
    if (total_area > 1e-15) region_error /= total_area;
}

static bool eig_approx_equal(double a, double b, double tol) {
    double max_abs = max(abs(a), abs(b));
    if (max_abs < 1e-15) return true;
    return abs(a - b) / max_abs < tol;
}

static ClassificationReport classify_one_proxy(const QuadricProxy& proxy,
                                                int proxy_id,
                                                const MatrixXi& R,
                                                const MatrixXi& F,
                                                const MatrixXd& V,
                                                double eps) {
    ClassificationReport rpt;
    rpt.proxy_id = proxy_id;
    rpt.type = TYPE_GENERAL_QUADRIC;
    rpt.confidence = 1.0;
    rpt.eigenvalues = Vector3d::Zero();
    rpt.eigenvectors = Matrix3d::Identity();

    compute_region_stats(proxy_id, R, F, V, proxy, rpt.num_faces, rpt.region_error);

    Matrix3d Q = proxy.quadraticMatrix();
    SelfAdjointEigenSolver<Matrix3d> eig(Q);

    if (eig.info() != Success) {
        rpt.type = TYPE_DEGENERATE;
        rpt.reason = "eigenvalue decomposition failed";
        rpt.confidence = 0.0;
        rpt.type_name = classified_type_name(rpt.type);
        return rpt;
    }

    Vector3d evals = eig.eigenvalues();   // ascending
    Matrix3d evecs = eig.eigenvectors();  // columns
    rpt.eigenvalues = evals;
    rpt.eigenvectors = evecs;

    double max_abs = max({abs(evals(0)), abs(evals(1)), abs(evals(2))});

    // All eigenvalues essentially zero → Plane
    if (max_abs < 1e-10) {
        rpt.type = TYPE_PLANE;
        rpt.confidence = 1.0;
        rpt.reason = "all eigenvalues zero (|Q| ~ 0)";
        rpt.type_name = classified_type_name(rpt.type);
        return rpt;
    }

    double tol = eps * max_abs;

    // Count zero / positive / negative eigenvalues
    int n_zero = 0, zero_idx = -1;
    int n_pos = 0, n_neg = 0;
    for (int i = 0; i < 3; i++) {
        if (abs(evals(i)) < tol) { n_zero++; zero_idx = i; }
        else if (evals(i) > 0) n_pos++;
        else n_neg++;
    }

    // --- Degenerate: rank <= 1 ---
    if (n_zero >= 2) {
        rpt.type = TYPE_DEGENERATE;
        rpt.confidence = 0.5;
        rpt.reason = to_string(n_zero) + " zero eigenvalues (rank <= 1)";
        rpt.type_name = classified_type_name(rpt.type);
        return rpt;
    }

    // --- One zero eigenvalue: cylinder or paraboloid ---
    if (n_zero == 1) {
        Vector3d e_zero = evecs.col(zero_idx);
        Vector3d linear(proxy.coeffs(1), proxy.coeffs(2), proxy.coeffs(3));
        double lin_proj = abs(e_zero.dot(linear));

        double ea = -1, eb = -1;
        for (int i = 0; i < 3; i++) {
            if (i == zero_idx) continue;
            if (ea < 0) ea = evals(i);
            else eb = evals(i);
        }
        bool same_sign = (ea * eb > 0);

        if (lin_proj > tol) {
            // Paraboloid: nonzero linear term along zero-eigenvalue direction
            rpt.type = TYPE_PARABOLOID;
            rpt.confidence = min(1.0, lin_proj / (tol + 1e-15) * 0.1);
            if (same_sign)
                rpt.reason = "1 zero eigenvalue + nonzero linear term: elliptic paraboloid";
            else
                rpt.reason = "1 zero eigenvalue + nonzero linear term: hyperbolic paraboloid";
        } else {
            // Cylinder: zero linear term along zero-eigenvalue direction
            if (same_sign && eig_approx_equal(ea, eb, eps)) {
                rpt.type = TYPE_CIRCULAR_CYLINDER;
                double ratio = abs(ea - eb) / max(abs(ea), abs(eb));
                rpt.confidence = max(0.0, 1.0 - ratio / (eps + 1e-15));
                rpt.reason = "1 zero eigenvalue + 2 equal same-sign eigenvalues";
            } else if (same_sign) {
                rpt.type = TYPE_GENERAL_QUADRIC;
                rpt.confidence = 0.5;
                rpt.reason = "1 zero eigenvalue + 2 unequal same-sign eigenvalues (elliptic cylinder)";
            } else {
                rpt.type = TYPE_GENERAL_QUADRIC;
                rpt.confidence = 0.5;
                rpt.reason = "1 zero eigenvalue + opposite-sign eigenvalues (hyperbolic cylinder)";
            }
        }
        rpt.type_name = classified_type_name(rpt.type);
        return rpt;
    }

    // --- All eigenvalues nonzero ---
    bool all_same_sign = (n_pos == 3 || n_neg == 3);

    if (all_same_sign) {
        bool all_equal = eig_approx_equal(evals(0), evals(1), eps)
                      && eig_approx_equal(evals(1), evals(2), eps);
        if (all_equal) {
            rpt.type = TYPE_SPHERE;
            double min_abs = min({abs(evals(0)), abs(evals(1)), abs(evals(2))});
            double ratio = max_abs / (min_abs + 1e-15);
            rpt.confidence = max(0.0, 2.0 - ratio);
            rpt.reason = "all eigenvalues same sign and equal";
        } else {
            rpt.type = TYPE_ELLIPSOID;
            rpt.confidence = 0.8;
            rpt.reason = "all eigenvalues same sign, not all equal";
        }
    } else {
        // Mixed signs → hyperboloid
        Vector3d b(proxy.coeffs(1), proxy.coeffs(2), proxy.coeffs(3));
        bool invertible = false;
        Matrix3d Q_inv;
        Q.computeInverseWithCheck(Q_inv, invertible);

        if (!invertible) {
            rpt.type = TYPE_GENERAL_QUADRIC;
            rpt.confidence = 0.3;
            rpt.reason = "mixed eigenvalue signs but Q not invertible";
        } else {
            Vector3d center = -0.5 * Q_inv * b;
            double f_center = proxy.coeffs(0) + b.dot(center)
                            + center.transpose() * Q * center;

            int minority_sign = (n_pos == 1) ? 1 : -1;
            if ((minority_sign > 0 && f_center > 0) || (minority_sign < 0 && f_center < 0)) {
                rpt.type = TYPE_HYPERBOLOID_TWO_SHEETS;
                rpt.reason = "mixed signs, f(center) matches minority: two sheets";
            } else {
                rpt.type = TYPE_HYPERBOLOID_ONE_SHEET;
                rpt.reason = "mixed signs, f(center) matches majority: one sheet";
            }
            rpt.confidence = 0.8;
        }
    }

    rpt.type_name = classified_type_name(rpt.type);
    return rpt;
}

vector<ClassificationReport> classify_all_proxies(
    const vector<QuadricProxy>& QP,
    const MatrixXi& R, const MatrixXi& F, const MatrixXd& V,
    double eps)
{
    vector<ClassificationReport> reports;
    for (int i = 0; i < (int)QP.size(); i++) {
        reports.push_back(classify_one_proxy(QP[i], i, R, F, V, eps));
    }
    return reports;
}

void print_classification_report(const ClassificationReport& rpt) {
    cout << "  proxy " << rpt.proxy_id
         << ": type=" << rpt.type_name
         << " eigenvalues=[" << rpt.eigenvalues(0) << ", "
         << rpt.eigenvalues(1) << ", " << rpt.eigenvalues(2) << "]"
         << " confidence=" << rpt.confidence
         << " faces=" << rpt.num_faces
         << " err=" << rpt.region_error << endl;
    cout << "    reason: " << rpt.reason << endl;
}

void export_proxy_types_json(const vector<ClassificationReport>& reports,
                              const string& filename) {
    ofstream fout(filename);
    if (!fout.is_open()) { cerr << "Cannot write " << filename << endl; return; }

    fout << "{" << endl;
    fout << "  \"num_proxies\": " << reports.size() << "," << endl;
    fout << "  \"proxies\": [" << endl;

    for (size_t i = 0; i < reports.size(); i++) {
        const auto& r = reports[i];
        fout << "    {" << endl;
        fout << "      \"id\": " << r.proxy_id << "," << endl;
        fout << "      \"type\": \"" << r.type_name << "\"," << endl;
        fout << "      \"eigenvalues\": ["
             << r.eigenvalues(0) << ", " << r.eigenvalues(1) << ", " << r.eigenvalues(2) << "],"
             << endl;
        fout << "      \"confidence\": " << r.confidence << "," << endl;
        fout << "      \"reason\": \"" << r.reason << "\"," << endl;
        fout << "      \"num_faces\": " << r.num_faces << "," << endl;
        fout << "      \"region_error\": " << r.region_error << endl;
        fout << "    }";
        if (i + 1 < reports.size()) fout << ",";
        fout << endl;
    }

    fout << "  ]" << endl;
    fout << "}" << endl;
    fout.close();
    cout << "Exported proxy types: " << filename << endl;
}
