#include <igl/opengl/glfw/Viewer.h>
#include "distance.h"  // for MetricMode enum

using namespace Eigen;
using namespace std;

MatrixXd new_proxies_L_2(MatrixXi R, MatrixXi F, MatrixXd V, int k);
MatrixXd new_proxies_L_2_1(MatrixXi R, MatrixXi F, MatrixXd V, int k);
MatrixXd new_proxies_hybrid_plane(MatrixXi R, MatrixXi F, MatrixXd V, int k, double om);
MatrixXd new_proxies(MatrixXi R, MatrixXi F, MatrixXd V, int k, MetricMode metric);