#pragma once

#include "balm/tools.hpp"
#include "lidarBA.h"
#include <Eigen/Cholesky>

namespace lidar_ba {

inline Eigen::Matrix3d jr_inv_local(const Eigen::Matrix3d &R) {
  Eigen::AngleAxisd aa(R);
  const double a=aa.angle();
  if(a<1e-9) return Eigen::Matrix3d::Identity();
  const Eigen::Vector3d u=aa.axis();
  const double c=a/(2.0*std::tan(a/2.0));
  return c*Eigen::Matrix3d::Identity()+(1.0-c)*u*u.transpose()+0.5*a*hat(u);
}

inline Eigen::Matrix3d jr_local(const Eigen::Vector3d &v) {
  const double a=v.norm();
  if(a<1e-9) return Eigen::Matrix3d::Identity();
  const Eigen::Vector3d u=v/a;
  const Eigen::Matrix3d K=hat(u);
  const double s=std::sin(a)/a;
  return s*Eigen::Matrix3d::Identity()+(1.0-s)*u*u.transpose()-(1.0-std::cos(a))/a*K;
}

struct InertialState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
};

struct ImuPreintegration {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix3d R_delta = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p_delta = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_delta = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_bg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d p_bg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d p_ba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d v_bg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d v_ba = Eigen::Matrix3d::Zero();
  Eigen::Matrix<double,9,9> covariance = Eigen::Matrix<double,9,9>::Zero();
  Eigen::Vector3d bg_hat = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba_hat = Eigen::Vector3d::Zero();
  double dt = 0.0;

  ImuPreintegration() = default;
  ImuPreintegration(const Eigen::Vector3d &bg, const Eigen::Vector3d &ba)
      : bg_hat(bg), ba_hat(ba) {}

  void integrate(const std::vector<ImuSample, Eigen::aligned_allocator<ImuSample>> &samples,
                 double gyr_cov, double acc_cov) {
    for (size_t k=1; k<samples.size(); ++k) {
      const double h=samples[k].t-samples[k-1].t;
      if (h<=0.0) continue;
      const Eigen::Vector3d w=0.5*(samples[k-1].gyr+samples[k].gyr)-bg_hat;
      const Eigen::Vector3d a=0.5*(samples[k-1].acc+samples[k].acc)-ba_hat;
      add(w,a,h,gyr_cov,acc_cov);
    }
  }

  void add(const Eigen::Vector3d &w, const Eigen::Vector3d &a, double h,
           double gyr_cov, double acc_cov) {
    dt += h;
    const Eigen::Matrix3d Ri=Exp(w*h);
    const Eigen::Matrix3d Jr=jr_local(w*h);
    const Eigen::Matrix3d Rdt=h*R_delta;
    const Eigen::Matrix3d Rdt2=0.5*h*h*R_delta;
    const Eigen::Matrix3d Ahat=hat(a);
    p_ba += v_ba*h-Rdt2;
    p_bg += v_bg*h-Rdt2*Ahat*R_bg;
    v_ba -= Rdt;
    v_bg -= Rdt*Ahat*R_bg;
    R_bg = Ri.transpose()*R_bg-Jr*h;
    Eigen::Matrix<double,9,9> A=Eigen::Matrix<double,9,9>::Identity();
    Eigen::Matrix<double,9,6> B=Eigen::Matrix<double,9,6>::Zero();
    A.block<3,3>(0,0)=Ri.transpose();
    A.block<3,3>(3,0)=-Rdt2*Ahat;
    A.block<3,3>(3,6)=Eigen::Matrix3d::Identity()*h;
    A.block<3,3>(6,0)=-Rdt*Ahat;
    B.block<3,3>(0,0)=Jr*h;
    B.block<3,3>(3,3)=Rdt2;
    B.block<3,3>(6,3)=Rdt;
    Eigen::Matrix<double,6,6> Q=Eigen::Matrix<double,6,6>::Zero();
    Q.block<3,3>(0,0)=Eigen::Matrix3d::Identity()*gyr_cov;
    Q.block<3,3>(3,3)=Eigen::Matrix3d::Identity()*acc_cov;
    covariance=A*covariance*A.transpose()+B*Q*B.transpose();
    p_delta += v_delta*h+Rdt2*a;
    v_delta += Rdt*a;
    R_delta *= Ri;
  }

  Eigen::Matrix<double,9,1> residual(const BAState &si, const InertialState &xi,
                                     const BAState &sj, const InertialState &xj,
                                     const Eigen::Vector3d &gravity,
                                     Eigen::Matrix<double,9,15> *Ji,
                                     Eigen::Matrix<double,9,15> *Jj,
                                     Eigen::Matrix<double,9,2> *Jg) const {
    const Eigen::Vector3d dbg=xi.bg-bg_hat, dba=xi.ba-ba_hat;
    const Eigen::Matrix3d Rc=R_delta*Exp(R_bg*dbg);
    const Eigen::Vector3d tc=p_delta+p_bg*dbg+p_ba*dba;
    const Eigen::Vector3d vc=v_delta+v_bg*dbg+v_ba*dba;
    const Eigen::Matrix3d rr=Rc.transpose()*si.R.transpose()*sj.R;
    const Eigen::Vector3d ev=si.R.transpose()*(xj.v-xi.v-dt*gravity);
    const Eigen::Vector3d ep=si.R.transpose()*(sj.p-si.p-xi.v*dt-0.5*dt*dt*gravity);
    Eigen::Matrix<double,9,1> r; r << Log(rr), ep-tc, ev-vc;
    if (Ji || Jj) {
      if(Ji) Ji->setZero(); if(Jj) Jj->setZero();
      const Eigen::Matrix3d Jr_inv=jr_inv_local(rr);
      if(Ji) {
        Ji->block<3,3>(0,0)=-Jr_inv*sj.R.transpose()*si.R;
        Ji->block<3,3>(0,9)=-Jr_inv*rr.transpose()*jr_local(R_bg*dbg)*R_bg;
        Ji->block<3,3>(3,0)=hat(ep); Ji->block<3,3>(3,3)=-si.R.transpose();
        Ji->block<3,3>(3,6)=-si.R.transpose()*dt;
        Ji->block<3,3>(3,9)=-p_bg; Ji->block<3,3>(3,12)=-p_ba;
        Ji->block<3,3>(6,0)=hat(ev); Ji->block<3,3>(6,6)=-si.R.transpose();
        Ji->block<3,3>(6,9)=-v_bg; Ji->block<3,3>(6,12)=-v_ba;
      }
      if(Jj) {
        Jj->block<3,3>(0,0)=Jr_inv;
        Jj->block<3,3>(3,3)=si.R.transpose();
        Jj->block<3,3>(6,6)=si.R.transpose();
      }
    }
    if(Jg) {
      Jg->setZero();
      Eigen::Matrix<double,3,2> B;
      const Eigen::Vector3d u=gravity.normalized();
      Eigen::Vector3d e1=u.cross(std::abs(u.z())<0.9?Eigen::Vector3d::UnitZ():Eigen::Vector3d::UnitX()).normalized();
      Eigen::Vector3d e2=u.cross(e1);
      B<<e1,e2;
      const double gm=gravity.norm();
      Jg->block<3,2>(3,0)=si.R.transpose()*(-0.5*dt*dt*gm)*B;
      Jg->block<3,2>(6,0)=si.R.transpose()*(-dt*gm)*B;
    }
    return r;
  }
};

inline void gravity_basis(const Eigen::Vector3d &up, Eigen::Matrix<double,3,2> &B) {
  const Eigen::Vector3d u=up.normalized();
  const Eigen::Vector3d a=std::abs(u.z())<0.9?Eigen::Vector3d::UnitZ():Eigen::Vector3d::UnitX();
  B.col(0)=u.cross(a).normalized(); B.col(1)=u.cross(B.col(0));
}

}
