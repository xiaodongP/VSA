#ifndef PROXY_PROJECTION_HEADER
#define PROXY_PROJECTION_HEADER

#include <Eigen/Dense>
#include <string>
#include <vector>
#include "quadric_proxy.h"
#include "proxy_classification.h"

#ifndef PROXY_TYPE_DEFINED
#define PROXY_TYPE_DEFINED
enum ProxyType { PLANE_PROXY, QUADRIC_PROXY };
#endif

using namespace Eigen;
using namespace std;

struct ProjectionConfig {
    int max_newton_iter;      // default 20
    double newton_tol;        // default 1e-10
    double newton_step_max;   // max step size, default 1.0
    ProjectionConfig()
        : max_newton_iter(20), newton_tol(1e-10), newton_step_max(1.0) {}
};

struct ProjectionLog {
    int total_vertices;
    int interior_count;
    int boundary_count;
    int success_count;     // Newton converged or exact projection
    int fallback_count;    // 1st-order fallback used
    int failure_count;     // kept original position
    ProjectionLog()
        : total_vertices(0), interior_count(0), boundary_count(0),
          success_count(0), fallback_count(0), failure_count(0) {}
};

// Project all vertices onto their proxy surfaces.
// Returns projected vertex positions (same rows as V).
MatrixXd project_vertices(const MatrixXd& V, const MatrixXi& F,
                           const MatrixXi& R, int num_proxies,
                           ProxyType proxy_type,
                           const vector<QuadricProxy>& QP,
                           const MatrixXd& Proxies,
                           const vector<ClassifiedType>& proxy_types,
                           const ProjectionConfig& cfg,
                           ProjectionLog& log_out);

// Export mesh as OBJ (1-indexed faces)
void export_obj(const MatrixXd& V, const MatrixXi& F, const string& filename);

// Export projection log as CSV
void export_projection_log(const ProjectionLog& log, int num_proxies,
                            const string& filename);

#endif
