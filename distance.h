#ifndef DISTANCE_HEADER
#define DISTANCE_HEADER

#include <igl/opengl/glfw/Viewer.h>

using namespace Eigen;
using namespace std;

//******************Metric mode enum******************
// 第一步过渡版本：支持 L2 / L2,1 / HYBRID 三种度量模式
// 注意：这不是论文完整版本，仅是 hybrid energy 的第一步接入
enum MetricMode { L2_METRIC = 0, L21_METRIC = 1, HYBRID_METRIC = 2 };

// Hybrid energy weight: E_hybrid = Ed + omega * En
// 仅在 metric == HYBRID_METRIC 时使用
extern double omega;

//******************Distortion error******************
double orthogonal_distance(Vector3d X, Vector3d N, Vector3d M);
Vector3d get_center(int i);
Vector3d get_normal(int i);
double get_area(int i);

double distance_L_2(int i, Vector3d X, Vector3d N, MatrixXd V);
double distance_L_2_1(int i, Vector3d N);
double distance_hybrid(int i, Vector3d X, Vector3d N, MatrixXd V, double om);

double distance(int i, Vector3d X, Vector3d N, MatrixXd V, MetricMode metric);

double global_distortion_error(MatrixXi R, MatrixXd Proxies, MatrixXd V, MatrixXi F, MetricMode metric);

double distance_projection(MatrixXd V, MatrixXd Proxies, int anchor1, int anchor2, int v, int r1, int r2);

void initialize_normals_areas(MatrixXi F, MatrixXd V);

#endif
