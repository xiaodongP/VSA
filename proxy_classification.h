#ifndef PROXY_CLASSIFICATION_HEADER
#define PROXY_CLASSIFICATION_HEADER

#include <Eigen/Dense>
#include <string>
#include <vector>
#include "quadric_proxy.h"

using namespace Eigen;
using namespace std;

enum ClassifiedType {
    TYPE_PLANE = 0,
    TYPE_SPHERE,
    TYPE_CIRCULAR_CYLINDER,
    TYPE_ELLIPSOID,
    TYPE_HYPERBOLOID_ONE_SHEET,
    TYPE_HYPERBOLOID_TWO_SHEETS,
    TYPE_PARABOLOID,
    TYPE_DEGENERATE,
    TYPE_GENERAL_QUADRIC
};

struct ClassificationReport {
    int proxy_id;
    ClassifiedType type;
    string type_name;
    Vector3d eigenvalues;      // sorted ascending
    Matrix3d eigenvectors;     // columns = eigenvectors
    double confidence;         // [0, 1]
    string reason;
    int num_faces;
    double region_error;
};

string classified_type_name(ClassifiedType t);

vector<ClassificationReport> classify_all_proxies(
    const vector<QuadricProxy>& QP,
    const MatrixXi& R, const MatrixXi& F, const MatrixXd& V,
    double eps = 0.1);

void export_proxy_types_json(const vector<ClassificationReport>& reports,
                              const string& filename);

void print_classification_report(const ClassificationReport& rpt);

#endif
