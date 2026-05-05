#include <igl/opengl/glfw/Viewer.h>
#include <queue>
#include <map>

#include "anchors.h"

using namespace Eigen;
using namespace std;

MatrixXi face_adjacency(MatrixXi F, int n);
void tcolor(MatrixXi &R);
void fcolor(MatrixXd &Cf, MatrixXi Ad);
void distance_color(MatrixXd &Cf, MatrixXi F, MatrixXd V, MetricMode metric);
void initial_partition(int p, MatrixXi &R, MatrixXd V, MatrixXi F, MatrixXi Ad, MetricMode metric);
void initial_partition2(int p, MatrixXi &R, MatrixXd V, MatrixXi F, MatrixXi Ad, MetricMode metric);
VectorXi find_best_triangles(MatrixXi R, MatrixXd Proxies, MatrixXd V, MatrixXi F, MetricMode metric);
void proxy_color(MatrixXi &R, MatrixXd Proxies, MatrixXd V, MatrixXi F, MatrixXi Ad, MetricMode metric);




