#ifndef VOXEL_MAP_HPP
#define VOXEL_MAP_HPP
#include "balm/tools.hpp"
#include <Eigen/Eigenvalues>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace lidar_ba {
using namespace std;





struct pointVar 
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d pnt;
  Eigen::Matrix3d var;
};

using PVec = std::vector<pointVar>;
using PVecPtr = std::shared_ptr<std::vector<pointVar>>;

class PointCluster {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix3d P=Eigen::Matrix3d::Zero();
  Eigen::Vector3d v=Eigen::Vector3d::Zero();
  int N=0;
  void clear(){P.setZero();v.setZero();N=0;}
  void push(const Eigen::Vector3d &x){++N;P+=x*x.transpose();v+=x;}
  Eigen::Matrix3d cov() const { if(N<=0)return Eigen::Matrix3d::Zero(); const auto c=v/double(N);return P/double(N)-c*c.transpose(); }
  PointCluster& operator+=(const PointCluster &s){P+=s.P;v+=s.v;N+=s.N;return *this;}
  PointCluster& operator-=(const PointCluster &s){P-=s.P;v-=s.v;N-=s.N;return *this;}
  void transform(const PointCluster &s,const BAState &x){N=s.N;v=x.R*s.v+N*x.p;const auto rp=x.R*s.v*x.p.transpose();P=x.R*s.P*x.R.transpose()+rp+rp.transpose()+N*x.p*x.p.transpose();}
};

void down_sampling_pvec(PVec &pvec, double voxel_size, pcl::PointCloud<PointType> &pl_keep)
{
  std::unordered_map<VOXEL_LOC, pair<pointVar, int>> feat_map;
  float loc_xyz[3];
  for(pointVar &pv: pvec)
  {
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = pv.pnt[j] / voxel_size;
      if(loc_xyz[j] < 0)
        loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter == feat_map.end())
    {
      feat_map[position] = make_pair(pv, 1);
    }
    else
    {
      pair<pointVar, int> &pp = iter->second;
      pp.first.pnt = (pp.first.pnt * pp.second + pv.pnt) / (pp.second + 1);
      pp.first.var = (pp.first.var * pp.second + pv.var) / (pp.second + 1);
      pp.second += 1;
    }
  }

  pcl::PointCloud<PointType>().swap(pl_keep);
  pl_keep.reserve(feat_map.size());
  PointType ap;
  for(auto iter=feat_map.begin(); iter!=feat_map.end(); ++iter)
  {
    pointVar &pv = iter->second.first;
    ap.x = pv.pnt[0]; ap.y = pv.pnt[1]; ap.z = pv.pnt[2];
    ap.normal_x = pv.var(0, 0);
    ap.normal_y = pv.var(1, 1);
    ap.normal_z = pv.var(2, 2);
    pl_keep.push_back(ap);
  }
 
}

struct Plane
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 6, 6> plane_var;
  float radius = 0;
  bool is_plane = false;

  Plane()
  {
    plane_var.setZero();
  }

};

Eigen::Vector4d min_point = Eigen::Vector4d::Constant(5);
double min_eigen_value = 0.0025;
inline int max_layer = 2;
inline int max_points = 100;
inline double voxel_size = 1.0;
inline int min_ba_point = 20;
inline std::vector<double> plane_eigen_value_thre{1,1,1};

void Bf_var(const pointVar &pv, Eigen::Matrix<double, 9, 9> &bcov, const Eigen::Vector3d &vec)
{
  Eigen::Matrix<double, 6, 3> Bi;
  // Eigen::Vector3d &vec = pv.world;
  Bi << 2*vec(0),        0,        0,
          vec(1),   vec(0),        0,
          vec(2),        0,   vec(0),
               0, 2*vec(1),        0,
               0,   vec(2),   vec(1),
               0,        0, 2*vec(2);
  Eigen::Matrix<double, 6, 3> Biup = Bi * pv.var;
  bcov.block<6, 6>(0, 0) = Biup * Bi.transpose();
  bcov.block<6, 3>(0, 6) = Biup;
  bcov.block<3, 6>(6, 0) = Biup.transpose();
  bcov.block<3, 3>(6, 6) = pv.var;
}

// The LiDAR BA factor in optimization
class LidarFactor
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<PointCluster> sig_vecs;
  std::vector<std::vector<PointCluster>> plvec_voxels;
  std::vector<double> coeffs;
  LBA_PLV(3) eig_values; LBA_PLM(3) eig_vectors; 
  std::vector<PointCluster> pcr_adds;
  int win_size;

  LidarFactor(int _w): win_size(_w){}

  void push_voxel(std::vector<PointCluster> &vec_orig, PointCluster &fix, double coe, Eigen::Vector3d &eig_value, Eigen::Matrix3d &eig_vector, PointCluster &pcr_add)
  {
    plvec_voxels.push_back(vec_orig);
    sig_vecs.push_back(fix);
    coeffs.push_back(coe);
    eig_values.push_back(eig_value);
    eig_vectors.push_back(eig_vector);
    pcr_adds.push_back(pcr_add);
  }

  void acc_evaluate2(const std::vector<BAState> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero(); JacT.setZero(); residual = 0;
    std::vector<PointCluster> sig_tran(win_size);
    const int kk = 0;

    LBA_PLV(3) viRiTuk(win_size);
    LBA_PLM(3) viRiTukukT(win_size);

    std::vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    for(int a=head; a<end; a++)
    {
      std::vector<PointCluster> &sig_orig = plvec_voxels[a];
      double coe = coeffs[a];

      // PointCluster sig = sig_vecs[a];
      // for(int i=0; i<win_size; i++)
      // if(sig_orig[i].N != 0)
      // {
      //   sig_tran[i].transform(sig_orig[i], xs[i]);
      //   sig += sig_tran[i];
      // }
      
      // const Eigen::Vector3d &vBar = sig.v / sig.N;
      // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      // const Eigen::Vector3d &lmbd = saes.eigenvalues();
      // const Eigen::Matrix3d &U = saes.eigenvectors();
      // int NN = sig.N;

      Eigen::Vector3d lmbd = eig_values[a];
      Eigen::Matrix3d U = eig_vectors[a];
      int NN = pcr_adds[a].N;
      Eigen::Vector3d vBar = pcr_adds[a].v / NN;
      
      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};
      Eigen::Vector3d &uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for(int i=0; i<3; i++)
        if(i != kk)
          umumT += 2.0/(lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for(int i=0; i<win_size; i++)
      // for(int i=1; i<win_size; i++)
      if(sig_orig[i].N != 0)
      {
        Eigen::Matrix3d Pi = sig_orig[i].P;
        Eigen::Vector3d vi = sig_orig[i].v;
        Eigen::Matrix3d Ri = xs[i].R;
        double ni = sig_orig[i].N;

        Eigen::Matrix3d vihat = hat(vi);
        Eigen::Vector3d RiTuk = Ri.transpose() * uk;
        Eigen::Matrix3d RiTukhat = hat(RiTuk);

        Eigen::Vector3d PiRiTuk = Pi * RiTuk;
        viRiTuk[i] = vihat * RiTuk;
        viRiTukukT[i] = viRiTuk[i] * uk.transpose();
        
        Eigen::Vector3d ti_v = xs[i].p - vBar;
        double ukTti_v = uk.dot(ti_v);

        Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
        Eigen::Vector3d combo2 = Ri*vi + ni*ti_v;
        Auk[i].block<3, 3>(0, 0) = (Ri*Pi + ti_v*vi.transpose()) * RiTukhat - Ri*combo1;
        Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
        Auk[i] /= NN;

        const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
        JacT.block<6, 1>(6*i, 0) += coe * jjt;

        const Eigen::Matrix3d &HRt = 2.0/NN * (1.0-ni/NN) * viRiTukukT[i];
        Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
        Hb.block<3, 3>(0, 0) += 2.0/NN * (combo1 - RiTukhat*Pi) * RiTukhat - 2.0/NN/NN * viRiTuk[i] * viRiTuk[i].transpose() - 0.5*hat(jjt.block<3, 1>(0, 0));
        Hb.block<3, 3>(0, 3) += HRt;
        Hb.block<3, 3>(3, 0) += HRt.transpose();
        Hb.block<3, 3>(3, 3) += 2.0/NN * (ni - ni*ni/NN) * ukukT;

        Hess.block<6, 6>(6*i, 6*i) += coe * Hb;
      }

      for(int i=0; i<win_size-1; i++)
      // for(int i=1; i<win_size-1; i++)
      if(sig_orig[i].N != 0)
      {
        double ni = sig_orig[i].N;
        for(int j=i+1; j<win_size; j++)
        if(sig_orig[j].N != 0)
        {
          double nj = sig_orig[j].N;
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
          Hb.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTuk[i] * viRiTuk[j].transpose();
          Hb.block<3, 3>(0, 3) += -2.0*nj/NN/NN * viRiTukukT[i];
          Hb.block<3, 3>(3, 0) += -2.0*ni/NN/NN * viRiTukukT[j].transpose();
          Hb.block<3, 3>(3, 3) += -2.0*ni*nj/NN/NN * ukukT;

          Hess.block<6, 6>(6*i, 6*j) += coe * Hb;
        }
      }
      
      residual += coe * lmbd[kk];
    }

    for(int i=1; i<win_size; i++)
      for(int j=0; j<i; j++)
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();
    
  }

  void evaluate_only_residual(const std::vector<BAState> &xs, int head, int end, double &residual)
  {
    residual = 0;
    // std::vector<PointCluster> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value

    // int gps_size = plvec_voxels.size();
    PointCluster pcr;

    for(int a=head; a<end; a++)
    {
      const std::vector<PointCluster> &sig_orig = plvec_voxels[a];
      PointCluster sig = sig_vecs[a];

      for(int i=0; i<win_size; i++)
      if(sig_orig[i].N != 0)
      {
        pcr.transform(sig_orig[i], xs[i]);
        sig += pcr;
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      // Eigen::Matrix3d cmt = sig.P/sig.N - vBar * vBar.transpose();
      // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P - sig.v * vBar.transpose());
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      Eigen::Vector3d lmbd = saes.eigenvalues();

      // centers[a] = vBar;
      eig_values[a] = saes.eigenvalues();
      eig_vectors[a] = saes.eigenvectors();
      pcr_adds[a] = sig;
      // Ns[a] = sig.N;

      residual += coeffs[a] * lmbd[kk];
    }
    
  }

  void clear()
  {
    sig_vecs.clear(); plvec_voxels.clear();
    eig_values.clear(); eig_vectors.clear();
    pcr_adds.clear(); coeffs.clear();
  }

  ~LidarFactor(){}

};

struct InertialState
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
};

struct ImuPreintegration
{
  std::shared_ptr<gtsam::PreintegratedImuMeasurements> pim;
};

// The LM optimizer for LiDAR BA
class Lidar_BA_Optimizer
{
public:
  int win_size, jac_leng, thd_num = 2;
  double last_gravity_sigma = 5.0 * M_PI / 180.0;

  double lidar_linearize(std::vector<BAState> &x_stats, LidarFactor &voxhess,
                         Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    // int thd_num = 4;
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    LBA_PLM(-1) hessians(thd_num); 
    LBA_PLV(-1) jacobins(thd_num);

    for(int i=0; i<thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    std::vector<double> resis(tthd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;

    std::vector<thread*> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    // for(int i=0; i<tthd_num; i++)
    for(int i=1; i<tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part*i, part*(i+1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    for(int i=0; i<tthd_num; i++)
    {
      if(i != 0) mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double lidar_cost(std::vector<BAState> &x_stats, LidarFactor &voxhess)
  {
    double residual1 = 0;
    // voxhess.evaluate_only_residual(x_stats, 0, voxhess.plvec_voxels.size(), residual1);

    // int thd_num = 2;
    std::vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < thd_num)
    {
      return std::numeric_limits<double>::infinity();
    }
    std::vector<thread*> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for(int i=1; i<thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part*i, part*(i+1), ref(residuals[i]));

    for(int i=0; i<thd_num; i++)
    {
      if(i != 0) 
        mthreads[i]->join();
      else
        voxhess.evaluate_only_residual(x_stats, part*i, part*(i+1), residuals[i]);
      residual1 += residuals[i];
      delete mthreads[i];
    }

    return residual1;
  }

  bool damping_iter(std::vector<BAState> &x_stats, LidarFactor &voxhess,
                    Eigen::MatrixXd* hess, std::vector<double> &resis,
                    int max_iter = 3, bool is_display = false)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;

    double u = 0.01, v = 2;
    Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng);
    hess->resize(jac_leng, jac_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    std::vector<BAState> x_stats_temp = x_stats;

    bool is_converge = true;

    // double tt1 = ros_now_sec();
    // for(int i=0; i<10; i++)
    for(int i=0; i<max_iter; i++)
    {
      if(is_calc_hess)
      {
        residual1 = lidar_linearize(x_stats, voxhess, Hess, JacT);
        *hess = Hess;
      }

      if(i == 0)
        resis.push_back(residual1);

      Hess.topRows(6).setZero();
      Hess.leftCols(6).setZero();
      Hess.block<6, 6>(0, 0).setIdentity();
      JacT.head(6).setZero();
      
      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u*D).ldlt().solve(-JacT);

      for(int j=0; j<win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(6*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(6*j+3, 0);
      }
      double q1 = 0.5*dxi.dot(u*D*dxi-JacT);

      residual2 = lidar_cost(x_stats_temp, voxhess);

      q = (residual1-residual2);
      if(is_display)
        printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

      if(q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
        is_converge = false;
      }

      if(fabs((residual1-residual2)/residual1)<1e-6)
        break;
    }
    resis.push_back(residual2);
    return is_converge;
  }

  static int state_offset(int i) { return 15*i; }

  static gtsam::Pose3 to_gtsam_pose(const BAState &pose)
  {
    return gtsam::Pose3(gtsam::Rot3(pose.R),
                        gtsam::Point3(pose.p.x(), pose.p.y(), pose.p.z()));
  }

  double add_imu_terms(const std::vector<BAState> &poses,
                       const std::vector<InertialState> &states,
                       const std::vector<ImuPreintegration> &preintegrations,
                       const gtsam::Unit3 &gravity_up,
                       double gravity_magnitude,
                       double bg_cov, double ba_cov,
                       Eigen::MatrixXd *Hess, Eigen::VectorXd *JacT) const
  {
    const int gravity_offset = 15*win_size;
    const int dimension = gravity_offset + 2;
    double cost = 0.0;
    Eigen::Matrix<double, 3, 2> D_up;
    const Eigen::Vector3d up = gravity_up.point3(D_up);
    const Eigen::Vector3d gravity = -gravity_magnitude * up;
    const Eigen::Matrix<double, 3, 2> D_gravity = -gravity_magnitude * D_up;

    for(size_t i=0; i<preintegrations.size() && i+1<poses.size(); ++i)
    {
      const auto &pim = *preintegrations[i].pim;
      pim.p().n_gravity = gravity;
      const gtsam::Pose3 pose_i = to_gtsam_pose(poses[i]);
      const gtsam::Pose3 pose_j = to_gtsam_pose(poses[i+1]);
      const gtsam::imuBias::ConstantBias bias_i(states[i].ba, states[i].bg);

      Eigen::Matrix<double, 9, 6> H_pose_i, H_bias_i, H_pose_j;
      Eigen::Matrix<double, 9, 3> H_vel_i, H_vel_j;
      const Eigen::Matrix<double, 9, 1> residual = pim.computeErrorAndJacobians(
          pose_i, states[i].v, pose_j, states[i+1].v, bias_i,
          H_pose_i, H_vel_i, H_pose_j, H_vel_j, H_bias_i);

      Eigen::MatrixXd J = Eigen::MatrixXd::Zero(9, dimension);
      const int oi = state_offset(static_cast<int>(i));
      const int oj = state_offset(static_cast<int>(i+1));
      J.block<9,3>(0,oi) = H_pose_i.leftCols<3>();
      J.block<9,3>(0,oi+3) = H_pose_i.rightCols<3>() * poses[i].R.transpose();
      J.block<9,3>(0,oi+6) = H_vel_i;
      J.block<9,3>(0,oi+9) = H_bias_i.rightCols<3>();
      J.block<9,3>(0,oi+12) = H_bias_i.leftCols<3>();
      J.block<9,3>(0,oj) = H_pose_j.leftCols<3>();
      J.block<9,3>(0,oj+3) = H_pose_j.rightCols<3>() * poses[i+1].R.transpose();
      J.block<9,3>(0,oj+6) = H_vel_j;

      const gtsam::NavState nav_i(pose_i, states[i].v);
      const gtsam::NavState nav_j(pose_j, states[i+1].v);
      const Eigen::Matrix<double,9,1> corrected = pim.biasCorrectedDelta(bias_i);
      const Eigen::Matrix<double,9,1> xi = nav_i.correctPIM(
          corrected, pim.deltaTij(), gravity, pim.p().omegaCoriolis,
          pim.p().use2ndOrderCoriolis);
      Eigen::Matrix<double,9,9> D_predict_xi, D_error_predict;
      const gtsam::NavState predicted = nav_i.retract(xi, boost::none, D_predict_xi);
      nav_j.localCoordinates(predicted, boost::none, D_error_predict);
      Eigen::Matrix<double,9,2> D_xi_gravity = Eigen::Matrix<double,9,2>::Zero();
      D_xi_gravity.block<3,2>(3,0) =
          0.5 * pim.deltaTij() * pim.deltaTij() * poses[i].R.transpose() * D_gravity;
      D_xi_gravity.block<3,2>(6,0) =
          pim.deltaTij() * poses[i].R.transpose() * D_gravity;
      J.block<9,2>(0,gravity_offset) = D_error_predict * D_predict_xi * D_xi_gravity;

      const Eigen::Matrix<double,9,9> information =
          pim.preintMeasCov().ldlt().solve(Eigen::Matrix<double,9,9>::Identity());
      cost += 0.5 * residual.dot(information * residual);
      if(Hess && JacT)
      {
        Hess->noalias() += J.transpose() * information * J;
        JacT->noalias() += J.transpose() * information * residual;
      }

      Eigen::Matrix<double,6,1> bias_residual;
      bias_residual << states[i+1].bg-states[i].bg,
                       states[i+1].ba-states[i].ba;
      Eigen::MatrixXd J_bias = Eigen::MatrixXd::Zero(6, dimension);
      J_bias.block<3,3>(0,oi+9) = -Eigen::Matrix3d::Identity();
      J_bias.block<3,3>(0,oj+9) = Eigen::Matrix3d::Identity();
      J_bias.block<3,3>(3,oi+12) = -Eigen::Matrix3d::Identity();
      J_bias.block<3,3>(3,oj+12) = Eigen::Matrix3d::Identity();
      Eigen::Matrix<double,6,1> weights;
      weights.head<3>().setConstant(1.0/(bg_cov*pim.deltaTij()));
      weights.tail<3>().setConstant(1.0/(ba_cov*pim.deltaTij()));
      const Eigen::Matrix<double,6,6> bias_information = weights.asDiagonal();
      cost += 0.5 * bias_residual.dot(bias_information*bias_residual);
      if(Hess && JacT)
      {
        Hess->noalias() += J_bias.transpose()*bias_information*J_bias;
        JacT->noalias() += J_bias.transpose()*bias_information*bias_residual;
      }
    }
    return cost;
  }

  double add_gravity_prior(const gtsam::Unit3 &gravity_up,
                           const gtsam::Unit3 &gravity_prior,
                           double sigma,
                           Eigen::MatrixXd *Hess, Eigen::VectorXd *JacT) const
  {
    Eigen::Matrix2d H_current;
    const Eigen::Vector2d residual = gravity_prior.errorVector(
        gravity_up, boost::none, H_current);
    const double weight = 1.0/(sigma*sigma);
    if(Hess && JacT)
    {
      const int offset = 15*win_size;
      Hess->block<2,2>(offset,offset).noalias() +=
          weight * H_current.transpose()*H_current;
      JacT->segment<2>(offset).noalias() +=
          weight * H_current.transpose()*residual;
    }
    return 0.5*weight*residual.squaredNorm();
  }

  double joint_linearize(std::vector<BAState> &poses,
                             const std::vector<InertialState> &states,
                             LidarFactor &voxhess,
                             const std::vector<ImuPreintegration> &preintegrations,
                             const gtsam::Unit3 &gravity_up,
                             const gtsam::Unit3 &gravity_prior,
                             double gravity_magnitude, double gravity_sigma,
                             double bg_cov, double ba_cov,
                             Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    const int dimension=15*win_size+2;
    Hess=Eigen::MatrixXd::Zero(dimension,dimension);
    JacT=Eigen::VectorXd::Zero(dimension);
    Eigen::MatrixXd lidar_hessian(6*win_size,6*win_size);
    Eigen::VectorXd lidar_gradient(6*win_size);
    const double lidar_cost=lidar_linearize(poses,voxhess,lidar_hessian,lidar_gradient);
    for(int i=0;i<win_size;++i)
    {
      const int si=state_offset(i);
      JacT.segment<6>(si)+=lidar_gradient.segment<6>(6*i);
      for(int j=0;j<win_size;++j)
        Hess.block<6,6>(si,state_offset(j))+=lidar_hessian.block<6,6>(6*i,6*j);
    }
    const double imu_cost=add_imu_terms(poses,states,preintegrations,gravity_up,
                                        gravity_magnitude,bg_cov,ba_cov,&Hess,&JacT);
    const double gravity_cost=add_gravity_prior(gravity_up,gravity_prior,gravity_sigma,
                                                &Hess,&JacT);
    return lidar_cost+imu_cost+gravity_cost;
  }

  double joint_cost(std::vector<BAState> &poses,
                        const std::vector<InertialState> &states,
                        LidarFactor &voxhess,
                        const std::vector<ImuPreintegration> &preintegrations,
                        const gtsam::Unit3 &gravity_up,
                        const gtsam::Unit3 &gravity_prior,
                        double gravity_magnitude, double gravity_sigma,
                        double bg_cov, double ba_cov)
  {
    const double lidar=lidar_cost(poses,voxhess);
    const double imu=add_imu_terms(poses,states,preintegrations,gravity_up,
                                   gravity_magnitude,bg_cov,ba_cov,nullptr,nullptr);
    const double gravity=add_gravity_prior(gravity_up,gravity_prior,gravity_sigma,
                                           nullptr,nullptr);
    return lidar+imu+gravity;
  }

  bool damping_iter_joint(std::vector<BAState> &poses,
                          std::vector<InertialState> &states,
                          gtsam::Unit3 &gravity_up,
                          const gtsam::Unit3 &gravity_prior,
                          LidarFactor &voxhess,
                          const std::vector<ImuPreintegration> &preintegrations,
                          double gravity_magnitude, double gravity_sigma,
                          double bg_cov, double ba_cov, bool gravity_optimize,
                          int max_iter=5)
  {
    win_size=voxhess.win_size;
    jac_leng=6*win_size;
    const int dimension=15*win_size+2;
    double u=0.01,v=2.0;
    bool accepted=false;
    Eigen::MatrixXd Hess(dimension,dimension),D(dimension,dimension);
    Eigen::VectorXd gradient(dimension),dx(dimension);
    double current=0.0;
    for(int iter=0;iter<max_iter;++iter)
    {
      current=joint_linearize(poses,states,voxhess,preintegrations,gravity_up,
          gravity_prior,gravity_magnitude,gravity_sigma,bg_cov,ba_cov,Hess,gradient);
      Hess.topRows(15).setZero(); Hess.leftCols(15).setZero();
      Hess.block(0,0,15,15).setIdentity(); gradient.head(15).setZero();
      if(!gravity_optimize)
      {
        Hess.bottomRows(2).setZero(); Hess.rightCols(2).setZero();
        Hess.bottomRightCorner(2,2).setIdentity(); gradient.tail(2).setZero();
      }
      D.setZero(); D.diagonal()=Hess.diagonal();
      dx=(Hess+u*D).ldlt().solve(-gradient);

      std::vector<BAState> trial_poses=poses;
      std::vector<InertialState> trial_states=states;
      for(int i=0;i<win_size;++i)
      {
        const int o=state_offset(i);
        trial_poses[i].R=poses[i].R*Exp(dx.segment<3>(o));
        trial_poses[i].p=poses[i].p+dx.segment<3>(o+3);
        trial_states[i].v=states[i].v+dx.segment<3>(o+6);
        trial_states[i].bg=states[i].bg+dx.segment<3>(o+9);
        trial_states[i].ba=states[i].ba+dx.segment<3>(o+12);
      }
      const gtsam::Unit3 trial_gravity=gravity_optimize
          ? gravity_up.retract(dx.tail<2>()) : gravity_up;
      const double trial=joint_cost(trial_poses,trial_states,voxhess,
          preintegrations,trial_gravity,gravity_prior,gravity_magnitude,
          gravity_sigma,bg_cov,ba_cov);
      const double predicted=0.5*dx.dot(u*D*dx-gradient);
      const double reduction=current-trial;
      if(reduction>0.0)
      {
        poses=std::move(trial_poses); states=std::move(trial_states);
        gravity_up=trial_gravity; accepted=true;
        double ratio=reduction/predicted;
        double scale=1.0-std::pow(2.0*ratio-1.0,3);
        u*=std::max(1.0/3.0,scale); v=2.0;
      }
      else
      {
        u*=v; v*=2.0;
      }
      if(std::abs(reduction/current)<1e-6) break;
    }
    return accepted;
  }

};

struct Keyframe
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BAState x0;
  pcl::PointCloud<PointType>::Ptr plptr;
  int exist;
  int id, mp;
  float jour;

  Keyframe(BAState &_x0): x0(_x0), exist(0)
  {
    plptr.reset(new pcl::PointCloud<PointType>());
  }

  void generate(pcl::PointCloud<PointType> &pl_send, Eigen::Matrix3d rot = Eigen::Matrix3d::Identity(), Eigen::Vector3d tra = Eigen::Vector3d(0, 0, 0))
  {
    Eigen::Vector3d v3;
    for(PointType ap: plptr->points)
    {
      v3 << ap.x, ap.y, ap.z;
      v3 = rot * v3 + tra;
      ap.x = v3[0]; ap.y = v3[1]; ap.z = v3[2];
      pl_send.push_back(ap);
    }
  }

};

// The sldingwindow in each voxel nodes
class SlideWindow
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<PVec> points;
  std::vector<PointCluster> pcrs_local;

  SlideWindow(int wdsize)
  {
    pcrs_local.resize(wdsize);
    points.resize(wdsize);
    for(int i=0; i<wdsize; i++)
      points[i].reserve(20);
  }

  void resize(int wdsize)
  {
    if(points.size() != wdsize)
    {
      points.resize(wdsize);
      pcrs_local.resize(wdsize);
    }
  }

  void clear()
  {
    int wdsize = points.size();
    for(int i=0; i<wdsize; i++)
    {
      points[i].clear();
      pcrs_local[i].clear();
    }
  }

};

// The octotree map for odometry and local mapping
// You can re-write it in your own project
inline int* mp = nullptr;
inline int thread_num = 2;
class OctoTree
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SlideWindow* sw = nullptr;
  PointCluster pcr_add;
  Eigen::Matrix<double, 9, 9> cov_add;

  PointCluster pcr_fix;
  PVec point_fix;

  int layer, octo_state, wdsize;
  OctoTree* leaves[8];
  double voxel_center[3];
  double jour = 0;
  float quater_length;

  Plane plane;
  bool isexist = false;

  Eigen::Vector3d eig_value;
  Eigen::Matrix3d eig_vector;

  int last_num = 0, opt_state = -1;
  std::mutex mVox;

  OctoTree(int _l, int _w) : layer(_l), wdsize(_w), octo_state(0)
  {
    for(int i=0; i<8; i++) leaves[i] = nullptr;
    cov_add.setZero();

    // ins = 255.0*rand()/(RAND_MAX + 1.0f);
  }

  inline void push(int ord, const pointVar &pv, const Eigen::Vector3d &pw, std::vector<SlideWindow*> &sws)
  {
    mVox.lock();
    if(sw == nullptr)
    {
      if(sws.size() != 0)
      {
        sw = sws.back();
        sws.pop_back();
        sw->resize(wdsize);
      }
      else
        sw = new SlideWindow(wdsize);
    }
    if(!isexist) isexist = true;

    int mord = mp[ord];
    if(layer < max_layer)
      sw->points[mord].push_back(pv);
    sw->pcrs_local[mord].push(pv.pnt);
    pcr_add.push(pw);
    Eigen::Matrix<double, 9, 9> Bi;
    Bf_var(pv, Bi, pw);
    cov_add += Bi;
    mVox.unlock();
  }

  inline void push_fix(pointVar &pv)
  {
    if(layer < max_layer)
      point_fix.push_back(pv);
    pcr_fix.push(pv.pnt);
    pcr_add.push(pv.pnt);
    Eigen::Matrix<double, 9, 9> Bi;
    Bf_var(pv, Bi, pv.pnt);
    cov_add += Bi;
  }

  inline void push_fix_novar(pointVar &pv)
  {
    if(layer < max_layer)
      point_fix.push_back(pv);
    pcr_fix.push(pv.pnt);
    pcr_add.push(pv.pnt);
  }

  inline bool plane_judge(Eigen::Vector3d &eig_values)
  {
    // return (eig_values[0] < min_eigen_value);
    return (eig_values[0] < min_eigen_value && (eig_values[0]/eig_values[2])<plane_eigen_value_thre[layer]);
  }

  void allocate(int ord, const pointVar &pv, const Eigen::Vector3d &pw, std::vector<SlideWindow*> &sws)
  {
    if(octo_state == 0)
    {
      push(ord, pv, pw, sws);
    }
    else
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pw[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize); 
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->allocate(ord, pv, pw, sws);
    }

  }

  void allocate_fix(pointVar &pv)
  {
    if(octo_state == 0)
    {
      push_fix_novar(pv);
    }
    else if(layer < max_layer)
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pv.pnt[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->allocate_fix(pv);
    }
  }

  void fix_divide(std::vector<SlideWindow*> &sws)
  {
    for(pointVar &pv: point_fix)
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pv.pnt[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->push_fix(pv);
    }

  }

  void subdivide(int si, BAState &xx, std::vector<SlideWindow*> &sws)
  {
    for(pointVar &pv: sw->points[mp[si]])
    {
      Eigen::Vector3d pw = xx.R * pv.pnt + xx.p;
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(pw[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer+1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->push(si, pv, pw, sws);
    }
  }

  void plane_update()
  {
    plane.center = pcr_add.v / pcr_add.N;
    int l = 0;
    Eigen::Vector3d u[3] = {eig_vector.col(0), eig_vector.col(1), eig_vector.col(2)};
    double nv = 1.0 / pcr_add.N;

    Eigen::Matrix<double, 3, 9> u_c; u_c.setZero();
    for(int k=0; k<3; k++)
    if(k != l)
    {
      Eigen::Matrix3d ukl = u[k] * u[l].transpose();
      Eigen::Matrix<double, 1, 9> fkl;
      fkl.head(6) << ukl(0, 0), ukl(1, 0)+ukl(0, 1), ukl(2, 0)+ukl(0, 2), 
                     ukl(1, 1), ukl(1, 2)+ukl(2, 1),           ukl(2, 2);
      fkl.tail(3) = -(u[k].dot(plane.center) * u[l] + u[l].dot(plane.center) * u[k]);
      
      u_c += nv / (eig_value[l]-eig_value[k]) * u[k] * fkl;
    }

    Eigen::Matrix<double, 3, 9> Jc = u_c * cov_add;
    plane.plane_var.block<3, 3>(0, 0) = Jc * u_c.transpose();
    Eigen::Matrix3d Jc_N = nv * Jc.block<3, 3>(0, 6);
    plane.plane_var.block<3, 3>(0, 3) = Jc_N;
    plane.plane_var.block<3, 3>(3, 0) = Jc_N.transpose();
    plane.plane_var.block<3, 3>(3, 3) = nv * nv * cov_add.block<3, 3>(6, 6);
    plane.normal = u[0];
    plane.radius = eig_value[2];
  }

  void recut(int win_count, std::vector<BAState> &x_buf, std::vector<SlideWindow*> &sws)
  {
    if(octo_state == 0)
    {
      if(layer >= 0)
      {
        opt_state = -1;
        if(pcr_add.N <= min_point[layer])
        {
          plane.is_plane = false; return;
        }
        if(!isexist || sw == nullptr) return;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
        eig_value  = saes.eigenvalues();
        eig_vector = saes.eigenvectors();
        plane.is_plane = plane_judge(eig_value);

        if(plane.is_plane)
        {
          return;
        }
        else if(layer >= max_layer)
          return;
      }
      
      if(pcr_fix.N != 0)
      {
        fix_divide(sws);
        // point_fix.clear();
        PVec().swap(point_fix);
      }

      for(int i=0; i<win_count; i++)
        subdivide(i, x_buf[i], sws);

      sw->clear();
      sws.push_back(sw);
      sw = nullptr;
      octo_state = 1;
    }

    for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut(win_count, x_buf, sws);

  }

  void margi(int win_count, int mgsize, std::vector<BAState> &x_buf, const LidarFactor &vox_opt)
  {
    if(octo_state == 0 && layer>=0)
    {
      if(!isexist || sw == nullptr) return;
      mVox.lock();
      std::vector<PointCluster> pcrs_world(wdsize);
      // pcr_add = pcr_fix;
      // for(int i=0; i<win_count; i++)
      // if(sw->pcrs_local[mp[i]].N != 0)
      // {
      //   pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
      //   pcr_add += pcrs_world[i];
      // }

      if(opt_state >= int(vox_opt.pcr_adds.size()))
      {
        printf("Error: opt_state: %d %zu\n", opt_state, vox_opt.pcr_adds.size());
        mVox.unlock();
        return;
      }

      if(opt_state >= 0)
      {
        pcr_add = vox_opt.pcr_adds[opt_state];
        eig_value  = vox_opt.eig_values[opt_state];
        eig_vector = vox_opt.eig_vectors[opt_state];
        opt_state = -1;
        
        for(int i=0; i<mgsize; i++)
        if(sw->pcrs_local[mp[i]].N != 0)
        {
          pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
        }
      }
      else
      {
        pcr_add = pcr_fix;
        for(int i=0; i<win_count; i++)
        if(sw->pcrs_local[mp[i]].N != 0)
        {
          pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
          pcr_add += pcrs_world[i];
        }

        if(plane.is_plane)
        {
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
          eig_value = saes.eigenvalues();
          eig_vector = saes.eigenvectors();
        }
        
      }

      if(pcr_fix.N < max_points && plane.is_plane)
      if(pcr_add.N - last_num >= 5 || last_num <= 10)
      {
        plane_update();
        last_num = pcr_add.N;
      }

      if(pcr_fix.N < max_points)
      {
        for(int i=0; i<mgsize; i++)
        if(pcrs_world[i].N != 0)
        {
          pcr_fix += pcrs_world[i];
          for(pointVar pv: sw->points[mp[i]])
          {
            pv.pnt = x_buf[i].R * pv.pnt + x_buf[i].p;
            point_fix.push_back(pv);
          }
        }

      }
      else
      {
        for(int i=0; i<mgsize; i++)
          if(pcrs_world[i].N != 0)
            pcr_add -= pcrs_world[i];
        
        if(point_fix.size() != 0)
          PVec().swap(point_fix);
      }

      for(int i=0; i<mgsize; i++)
      if(sw->pcrs_local[mp[i]].N != 0)
      {
        sw->pcrs_local[mp[i]].clear();
        sw->points[mp[i]].clear();
      }
      
      if(pcr_fix.N >= pcr_add.N)
        isexist = false;
      else
        isexist = true;
      
      mVox.unlock();
    }
    else
    {
      isexist = false;
      for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
      {
        leaves[i]->margi(win_count, mgsize, x_buf, vox_opt);
        isexist = isexist || leaves[i]->isexist;
      }
    }

  }

  // Extract the LiDAR factor
  void tras_opt(LidarFactor &vox_opt)
  {
    if(octo_state == 0)
    {
      if(layer >= 0 && isexist && plane.is_plane && sw!=nullptr)
      {
        if(eig_value[0]/eig_value[1] > 0.12) return;

        double coe = 1;
        std::vector<PointCluster> pcrs(wdsize);
        for(int i=0; i<wdsize; i++)
          pcrs[i] = sw->pcrs_local[mp[i]];
        opt_state = vox_opt.plvec_voxels.size();
        vox_opt.push_voxel(pcrs, pcr_fix, coe, eig_value, eig_vector, pcr_add);
      }

    }
    else
    {
      for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt);
    }


  }

  int match(Eigen::Vector3d &wld, Plane* &pla, double &max_prob, Eigen::Matrix3d &var_wld, double &sigma_d, OctoTree* &oc)
  {
    int flag = 0;
    if(octo_state == 0)
    {
      if(plane.is_plane)
      {
        float dis_to_plane = fabs(plane.normal.dot(wld - plane.center));
        float dis_to_center = (plane.center - wld).squaredNorm();
        float range_dis = (dis_to_center - dis_to_plane * dis_to_plane);
        if(range_dis <= 3*3*plane.radius)
        {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = wld - plane.center;
          J_nq.block<1, 3>(0, 3) = -plane.normal;
          double sigma_l = J_nq * plane.plane_var * J_nq.transpose();
          sigma_l += plane.normal.transpose() * var_wld * plane.normal;
          if(dis_to_plane < 3 * sqrt(sigma_l))
          {
            // float prob = 1 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
            // if(prob > max_prob)
            {
              oc = this;
              sigma_d = sigma_l;
              // max_prob = prob;
              pla = &plane;
            }

            flag = 1;
          }
        }
      }
    }
    else
    {
      int xyz[3] = {0, 0, 0};
      for(int k=0; k<3; k++)
        if(wld[k] > voxel_center[k]) xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];

      // for(int i=0; i<8; i++)
      // if(leaves[i] != nullptr)
      // {
      //   int flg = leaves[i]->match(wld, pla, max_prob, var_wld);
      //   if(i == leafnum)
      //     flag = flg;
      // }

      if(leaves[leafnum] != nullptr)
        flag = leaves[leafnum]->match(wld, pla, max_prob, var_wld, sigma_d, oc);

      // for(int i=0; i<8; i++)
      //   if(leaves[i] != nullptr)
      //     leaves[i]->match(pv, pla, max_prob, var_wld);
    }

    return flag;
  }

  void tras_ptr(std::vector<OctoTree*> &octos_release)
  {
    if(octo_state == 1)
    {
      for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
      {
        octos_release.push_back(leaves[i]);
        leaves[i]->tras_ptr(octos_release);
      }
    }
  }

  // ~OctoTree()
  // {
  //   for(int i=0; i<8; i++)
  //   if(leaves[i] != nullptr)
  //   {
  //     delete leaves[i];
  //     leaves[i] = nullptr;
  //   }
  // }

  // Extract the point cloud map for debug
  void tras_display(int win_count, pcl::PointCloud<PointType> &pl_fixd, pcl::PointCloud<PointType> &pl_wind, std::vector<BAState> &x_buf)
  {
    if(octo_state == 0)
    {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
      Eigen::Matrix3d eig_vectors = saes.eigenvectors();
      Eigen::Vector3d eig_values  = saes.eigenvalues();

      PointType ap; 
      // ap.intensity = ins;

      if(plane.is_plane)
      {
        // if(pcr_add.N-pcr_fix.N < min_ba_point) return;
        // if(eig_value[0]/eig_value[1] > 0.1)
        //   return;

        // for(pointVar &pv: point_fix)
        // {
        //   Eigen::Vector3d pvec = pv.pnt;
        //   ap.x = pvec[0];
        //   ap.y = pvec[1];
        //   ap.z = pvec[2];
        //   ap.normal_x = sqrt(eig_values[0]);
        //   ap.normal_y = sqrt(eig_values[2] / eig_values[0]);
        //   ap.normal_z = pcr_add.N;
        //   ap.curvature = pcr_add.N - pcr_fix.N;
        //   pl_fixd.push_back(ap);
        // }

        for(int i=0; i<win_count; i++)
        for(pointVar &pv: sw->points[mp[i]])
        {
          Eigen::Vector3d pvec = x_buf[i].R * pv.pnt + x_buf[i].p;
          ap.x = pvec[0]; ap.y = pvec[1]; ap.z = pvec[2];
          // ap.normal_x = sqrt(eig_values[0]);
          // ap.normal_y = sqrt(eig_values[2] / eig_values[0]);
          // ap.normal_z = pcr_add.N;
          // ap.curvature = pcr_add.N - pcr_fix.N;
          pl_wind.push_back(ap);
        }
      }

    }
    else
    {
      for(int i=0; i<8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_display(win_count, pl_fixd, pl_wind, x_buf);
    }

  }

  bool inside(Eigen::Vector3d &wld)
  {
    double hl = quater_length * 2;
    return (wld[0] >= voxel_center[0] - hl &&
            wld[0] <= voxel_center[0] + hl &&
            wld[1] >= voxel_center[1] - hl &&
            wld[1] <= voxel_center[1] + hl &&
            wld[2] >= voxel_center[2] - hl &&
            wld[2] <= voxel_center[2] + hl);
  }

  void clear_slwd(std::vector<SlideWindow*> &sws)
  {
    if(octo_state != 0)
    {
      for(int i=0; i<8; i++)
      if(leaves[i] != nullptr)
      {
        leaves[i]->clear_slwd(sws);
      }
    }

    if(sw != nullptr)
    {
      sw->clear();
      sws.push_back(sw);
      sw = nullptr;
    }

  }

};

void cut_voxel(std::unordered_map<VOXEL_LOC, OctoTree*> &feat_map, PVecPtr pvec, int win_count, std::unordered_map<VOXEL_LOC, OctoTree*> &feat_tem_map, int wdsize, LBA_PLV(3) &pwld, std::vector<SlideWindow*> &sws)
{
  int plsize = pvec->size();
  for(int i=0; i<plsize; i++)
  {
    pointVar &pv = (*pvec)[i];
    Eigen::Vector3d &pw = pwld[i];
    float loc[3];
    for(int j=0; j<3; j++)
    {
      loc[j] = pw[j] / voxel_size;
      if(loc[j] < 0) loc[j] -= 1;
    }

    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->allocate(win_count, pv, pw, sws);
      iter->second->isexist = true;
      if(feat_tem_map.find(position) == feat_map.end())
        feat_tem_map[position] = iter->second;
    }
    else
    {
      OctoTree* ot = new OctoTree(0, wdsize);
      ot->allocate(win_count, pv, pw, sws);
      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      feat_map[position] = ot;
      feat_tem_map[position] = ot;
    }
  }
  
}

// Cut the current scan into corresponding voxel in multi thread
void cut_voxel_multi(std::unordered_map<VOXEL_LOC, OctoTree*> &feat_map, PVecPtr pvec, int win_count, std::unordered_map<VOXEL_LOC, OctoTree*> &feat_tem_map, int wdsize, LBA_PLV(3) &pwld, std::vector<std::vector<SlideWindow*>> &sws)
{
  std::unordered_map<OctoTree*, std::vector<int>> map_pvec;
  int plsize = pvec->size();
  for(int i=0; i<plsize; i++)
  {
    pointVar &pv = (*pvec)[i];
    Eigen::Vector3d &pw = pwld[i];
    float loc[3];
    for(int j=0; j<3; j++)
    {
      // loc[j] = pv.world[j] / voxel_size;
      loc[j] = pw[j] / voxel_size;
      if(loc[j] < 0) loc[j] -= 1;
    }

    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    auto iter = feat_map.find(position);
    OctoTree* ot = nullptr;
    if(iter != feat_map.end())
    {
      iter->second->isexist = true;
      if(feat_tem_map.find(position) == feat_map.end())
        feat_tem_map[position] = iter->second;
      ot = iter->second;
    }
    else
    {
      ot = new OctoTree(0, wdsize);
      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      feat_map[position] = ot;
      feat_tem_map[position] = ot;
    }

    map_pvec[ot].push_back(i);
  }

  // for(auto iter=map_pvec.begin(); iter!=map_pvec.end(); iter++)
  // {
  //   for(int i: iter->second)
  //   {
  //     iter->first->allocate(win_count, (*pvec)[i], pwld[i], sws);
  //   }
  // }

  std::vector<pair<OctoTree *const, std::vector<int>>*> octs; octs.reserve(map_pvec.size());
  for(auto iter=map_pvec.begin(); iter!=map_pvec.end(); iter++)
    octs.push_back(&(*iter));

  int thd_num = sws.size();
  int g_size = octs.size();
  if(g_size < thd_num)
  {
    for (auto *entry : octs)
      for (int k : entry->second)
        entry->first->allocate(win_count, (*pvec)[k], pwld[k], sws[0]);
    return;
  }
  std::vector<thread*> mthreads(thd_num);
  double part = 1.0 * g_size / thd_num;

  int swsize = sws[0].size() / thd_num;
  for(int i=1; i<thd_num; i++)
  {
    sws[i].insert(sws[i].end(), sws[0].end() - swsize, sws[0].end());
    sws[0].erase(sws[0].end() - swsize, sws[0].end());
  }

  for(int i=1; i<thd_num; i++)
  {
    mthreads[i] = new thread
    (
      [&](int head, int tail, std::vector<SlideWindow*> &sw)
      {
        for(int j=head; j<tail; j++)
        {
          for(int k: octs[j]->second)
            octs[j]->first->allocate(win_count, (*pvec)[k], pwld[k], sw);
        }
      }, part*i, part*(i+1), ref(sws[i])
    );
  }

  for(int i=0; i<thd_num; i++)
  {
    if(i == 0)
    {
      for(int j=0; j<int(part); j++)
        for(int k: octs[j]->second)
          octs[j]->first->allocate(win_count, (*pvec)[k], pwld[k], sws[0]);
    }
    else
    {
      mthreads[i]->join();
      delete mthreads[i];
    }
    
  }

}

void cut_voxel(std::unordered_map<VOXEL_LOC, OctoTree*> &feat_map, PVec &pvec, int wdsize, double jour)
{
  for(pointVar &pv: pvec)
  {
    float loc[3];
    for(int j=0; j<3; j++)
    {
      loc[j] = pv.pnt[j] / voxel_size;
      if(loc[j] < 0) loc[j] -= 1;
    }

    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->allocate_fix(pv);
    }
    else
    {
      OctoTree* ot = new OctoTree(0, wdsize);
      ot->push_fix_novar(pv);
      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->jour = jour;
      feat_map[position] = ot;
    }
  }
  
}

// Match the point with the plane in the voxel map
int match(std::unordered_map<VOXEL_LOC, OctoTree*> &feat_map, Eigen::Vector3d &wld, Plane* &pla, Eigen::Matrix3d &var_wld, double &sigma_d, OctoTree* &oc)
{
  int flag = 0;

  float loc[3];
  for(int j=0; j<3; j++)
  {
    loc[j] = wld[j] / voxel_size;
    if(loc[j] < 0) loc[j] -= 1;
  }
  VOXEL_LOC position(loc[0], loc[1], loc[2]);
  auto iter = feat_map.find(position);
  if(iter != feat_map.end())
  {
    double max_prob = 0;
    flag = iter->second->match(wld, pla, max_prob, var_wld, sigma_d, oc);
    // iter->second->match_end(pv, pla, max_prob);
    if(flag && pla==nullptr)
    {
      printf("pla null max_prob: %lf %ld %ld %ld\n", max_prob, iter->first.x, iter->first.y, iter->first.z);
    }
  }

  return flag;
}


} // namespace lidar_ba
#endif
