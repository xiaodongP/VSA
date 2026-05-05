#include "proxies.h"
#include "distance.h"

Vector3d g(Vector3d v1,Vector3d v2,Vector3d v3){

  return (1./3.)*(v1+v2+v3);

};

MatrixXd M(Vector3d v1,Vector3d v2,Vector3d v3){

  MatrixXd M = MatrixXd::Zero(3,3);
  M.row(0) = v2-v1;
  M.row(1) = v3-v1;

  return M;

}; 

Vector3d new_Xi_L_2 (MatrixXi R, int i, MatrixXi F, MatrixXd V){

  Vector3d Xi(0.,0.,0.);
  double w = 0.;

  Vector3i T;
  Vector3d v1;
  Vector3d v2;
  Vector3d v3;
  Vector3d gT;
  double s;

  for (int f=0 ; f<R.rows() ; f++){
    //we only add the triangles that belong to the region i
    if (R(f,0) == i){

      T = F.row(f);
      v1 = V.row(T(0));
      v2 = V.row(T(1));
      v3 = V.row(T(2));

      gT = g(v1,v2,v3);
      s = get_area(f);

      Xi += s*gT;
      w += s;
    }
  }
  
  return Xi/w;

};

Vector3d new_Ni_L_2 (MatrixXi R, int i, MatrixXi F, MatrixXd V){

  //Compute the Covariance Matrix Ci
  MatrixXd Ci = MatrixXd::Zero(3,3); 
  double w = 0.;
  MatrixXd Xi = new_Xi_L_2(R,i,F,V);

  MatrixXd A(3,3);
  A(0,0) = 10; A(0,1) = 7; A(0,2) = 0;
  A(1,0) = 7; A(1,1) = 10; A(1,2) = 0;
  A(2,0) = 0; A(2,1) = 0.; A(2,2) = 0;

  Vector3i T;
  Vector3d v1;
  Vector3d v2;
  Vector3d v3;
  Vector3d gT;
  MatrixXd MT;
  double s;

  for (int f=0 ; f<R.rows() ; f++){
    //we only add the triangles that belong to the region i
    if (R(f,0) == i){

      T = F.row(f);
      v1 = V.row(T(0));
      v2 = V.row(T(1));
      v3 = V.row(T(2));

      gT = g(v1,v2,v3);
      MT = M(v1,v2,v3);
      s = get_area(f);
      
      Ci += (2./72.)*s*MT*A*MT.transpose() + s*gT*gT.transpose();
      w += s;
    } 
  }
  
  Ci = Ci - w*Xi*Xi.transpose();
  
  //Find the eigenvector of the min eigenvalue of Ci
  EigenSolver<MatrixXd> es(Ci);

  MatrixXd valp;
  MatrixXd vectp;
  valp = es.eigenvalues().real();
  vectp = es.eigenvectors().real();

  double min_valp;
  MatrixXd::Index minRow, minCol;
  Vector3d Ni;
  min_valp = valp.minCoeff(&minRow,&minCol);
  Ni = vectp.col(minRow);

  return Ni.normalized();

};

Vector3d new_Xi_L_2_1 (MatrixXi R, int i, MatrixXi F, MatrixXd V){

  return new_Xi_L_2(R,i,F,V);

};


MatrixXd compute_N (MatrixXi R, MatrixXi F, MatrixXd V, int p) {
  MatrixXd N;
  N.setZero(p,3);

  Vector3i T;
  Vector3d v1;
  Vector3d v2;
  Vector3d v3;
  double s;
  Vector3d nT;

  for (int f=0 ; f<R.rows() ; f++){
    //we only add the triangles that belong to the region i
    int i = R(f,0);

    s = get_area(f);
    nT = get_normal(f); 

    N.row(i) += s*nT;
  }

  return N;
}


Vector3d new_Ni_L_2_1 (MatrixXi R, int i, MatrixXi F, MatrixXd V){

  Vector3d Ni(0.,0.,0.);

  Vector3i T;
  Vector3d v1;
  Vector3d v2;
  Vector3d v3;
  double s;
  Vector3d nT;

  for (int f=0 ; f<R.rows() ; f++){
    //we only add the triangles that belong to the region i
    if (R(f,0) == i){
      s = get_area(f);
      nT = get_normal(f); 
      Ni += s*nT;
    }
  }

  return Ni.normalized();

};

//k is the number of regions/proxies of the partition R
MatrixXd new_proxies_L_2(MatrixXi R, MatrixXi F, MatrixXd V, int k){

  MatrixXd P(2*k,3);

  Vector3d Xi;
  Vector3d Ni;

  for (int i=0 ; i<k ; i++){
    Xi = new_Xi_L_2(R,i,F,V);
    Ni = new_Ni_L_2(R,i,F,V);
    P.row(i) = Xi;
    P.row(k+i) = Ni.normalized();
  }

  return P;

};

MatrixXd new_proxies_L_2_1(MatrixXi R, MatrixXi F, MatrixXd V, int k){

  MatrixXd P(2*k,3);

  Vector3d Xi;
  MatrixXd N = compute_N(R,F,V,k);

  for (int i=0 ; i<k ; i++){
    Xi = new_Xi_L_2_1(R,i,F,V);
    P.row(i) = Xi;
    P.row(k+i) = N.row(i).normalized();
  }

  return P;

};

// ============================================================
// Hybrid plane proxy fitting — 与 hybrid energy 一致的 fitting
// 仍然是 plane proxy (X, N)，不是 quadric，不是 2012 论文的离散系统
// ============================================================

// Helper: 构建 region i 的 L2 协方差矩阵
// 复用 new_Ni_L_2 中的矩阵逻辑，使得 Ed_total = N^T * Ci * N
MatrixXd compute_region_covariance_L2(MatrixXi R, int i, MatrixXi F, MatrixXd V, Vector3d Xi) {
  MatrixXd Ci = MatrixXd::Zero(3,3);

  MatrixXd Amat(3,3);
  Amat << 10, 7, 0,
          7, 10, 0,
          0,  0, 0;

  double total_area = 0.;
  for (int f = 0; f < R.rows(); f++) {
    if (R(f,0) == i) {
      Vector3i T = F.row(f);
      Vector3d v1 = V.row(T(0));
      Vector3d v2 = V.row(T(1));
      Vector3d v3 = V.row(T(2));
      Vector3d gT = g(v1, v2, v3);
      MatrixXd MT = M(v1, v2, v3);
      double s = get_area(f);
      Ci += (2.0/72.0) * s * MT * Amat * MT.transpose() + s * gT * gT.transpose();
      total_area += s;
    }
  }
  Ci -= total_area * Xi * Xi.transpose();
  return Ci;
}

// Helper: B = sum_t |t| * n_t for region i
Vector3d compute_region_normal_sum(MatrixXi R, int i) {
  Vector3d B = Vector3d::Zero();
  for (int f = 0; f < R.rows(); f++) {
    if (R(f,0) == i) {
      B += get_area(f) * get_normal(f);
    }
  }
  return B;
}

// Exact hybrid plane energy for one region: sum_t [Ed(t) + om * En(t)]
double region_hybrid_energy(MatrixXi R, int region_id, Vector3d X, Vector3d N, MatrixXd V, double om) {
  double E = 0.;
  for (int f = 0; f < R.rows(); f++) {
    if (R(f,0) == region_id) {
      E += distance_L_2(f, X, N, V) + om * distance_L_2_1(f, N);
    }
  }
  return E;
}

// Projected gradient descent on unit sphere + backtracking line search
// 目标: min ||N||=1  Ed(N) + omega * En(N)
// grad = 2*Ci*N - 2*omega*B, projected to tangent space
Vector3d optimize_hybrid_normal(MatrixXi R, int i, MatrixXi F, MatrixXd V,
                                Vector3d X, double om) {
  // Build Ci and B
  Vector3d Xi = new_Xi_L_2(R, i, F, V);
  MatrixXd Ci = compute_region_covariance_L2(R, i, F, V, Xi);
  Vector3d B = compute_region_normal_sum(R, i);

  // Initial N: try L2 and L21, pick lower hybrid energy
  Vector3d N_l2 = new_Ni_L_2(R, i, F, V);
  Vector3d N_l21 = new_Ni_L_2_1(R, i, F, V);
  if (N_l2.norm() > 1e-10) N_l2.normalize();
  if (N_l21.norm() > 1e-10) N_l21.normalize();

  double E_l2  = (N_l2.norm()  > 1e-10) ? region_hybrid_energy(R, i, X, N_l2,  V, om) : 1e18;
  double E_l21 = (N_l21.norm() > 1e-10) ? region_hybrid_energy(R, i, X, N_l21, V, om) : 1e18;

  Vector3d N = (E_l2 <= E_l21) ? N_l2 : N_l21;
  if (N.norm() < 1e-10) N = Vector3d(0, 0, 1);
  N.normalize();

  double E_curr = region_hybrid_energy(R, i, X, N, V, om);

  // Projected gradient descent
  int max_iter = 50;
  double grad_tol = 1e-8;

  for (int iter = 0; iter < max_iter; iter++) {
    // grad = 2*Ci*N - 2*omega*B
    Vector3d grad = 2.0 * Ci * N - 2.0 * om * B;
    // Project to tangent space of unit sphere
    Vector3d grad_t = grad - grad.dot(N) * N;

    if (grad_t.norm() < grad_tol) break;

    // Backtracking line search
    double step = 1.0;
    bool accepted = false;
    while (step > 1e-10) {
      Vector3d N_new = N - step * grad_t;
      if (N_new.norm() < 1e-10) { step *= 0.5; continue; }
      N_new.normalize();
      double E_new = region_hybrid_energy(R, i, X, N_new, V, om);
      if (E_new < E_curr) {
        N = N_new;
        E_curr = E_new;
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) break;
  }

  return N;
}

// Hybrid plane proxy fitting — 对每个 region 独立优化
// X: area-weighted barycenter (过渡实现，重点在法向更新)
// N: projected gradient descent for Ed + omega*En
MatrixXd new_proxies_hybrid_plane(MatrixXi R, MatrixXi F, MatrixXd V, int k, double om) {
  MatrixXd P(2*k, 3);

  for (int i = 0; i < k; i++) {
    Vector3d Xi = new_Xi_L_2(R, i, F, V);
    Vector3d Ni = optimize_hybrid_normal(R, i, F, V, Xi, om);
    P.row(i) = Xi;
    P.row(k+i) = Ni.normalized();
  }

  return P;
}

MatrixXd new_proxies(MatrixXi R, MatrixXi F, MatrixXd V, int k, MetricMode metric){
  if (metric == L2_METRIC){
    return new_proxies_L_2(R,F,V,k);
  }
  else if (metric == L21_METRIC){
    return new_proxies_L_2_1(R,F,V,k);
  }
  else if (metric == HYBRID_METRIC) {
    // 使用与 hybrid energy 一致的 plane fitting
    // projected gradient descent on unit sphere + backtracking
    return new_proxies_hybrid_plane(R, F, V, k, omega);
  }
  else {
    cout<<"wrong metric parameter"<<endl;
    return new_proxies_L_2(R,F,V,k);
  }
};
