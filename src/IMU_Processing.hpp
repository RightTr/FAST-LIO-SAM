#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <cassert>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <common_lib.h>
#include <ros_utils.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include "simple_EKF/simple_ekf.hpp"
#include "preprocess.h"
#include "posebuffer.h"
#include "map_optimization.h"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void set_grav_align(bool enabled);

  void Process(const MeasureGroup &meas, SimpleEKF &kf_state, PointCloudXYZI::Ptr pcl_un_);

  void PBufferPop(Pose &pose);

  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;
  int lidar_type;
  PoseBuffer pbuffer;

 private:
  double gap_handler(double dt);
  void IMU_init(const MeasureGroup &meas, SimpleEKF &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, SimpleEKF &kf_state, PointCloudXYZI &pcl_in_out);

  bool   grav_align_ = false;

  PointCloudXYZI::Ptr cur_pcl_un_;
  ImuMsgConstPtr last_imu_;
  deque<ImuMsgConstPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time_;
  double imu_gap_ = 0.5;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new ImuMsg());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  pbuffer.Clear();
  last_imu_.reset(new ImuMsg());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::set_grav_align(bool enabled)
{
  grav_align_ = enabled;
}

double ImuProcess::gap_handler(double dt)
{
  if (dt < 0.0)
  {
    return -1.0;
  }

  if (dt > imu_gap_)
  {
    return -1.0;
  }

  return dt;
}

void ImuProcess::PBufferPop(Pose &pose)
{
  pose = pbuffer.Pop();
}

void ImuProcess::IMU_init(const MeasureGroup &meas, SimpleEKF &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++;
  }
  SimpleEKF::State init_state = kf_state.state();
  const double mean_acc_norm = mean_acc.norm();
  V3D init_gravity;
  if (grav_align_ )
  {
    // Use the averaged gravity direction to initialize roll/pitch; yaw remains unconstrained.
    const Eigen::Vector3d gravity_meas = mean_acc / mean_acc_norm;
    const Eigen::Quaterniond q_init = Eigen::Quaterniond::FromTwoVectors(gravity_meas,
                                                                         Eigen::Vector3d::UnitZ());
    init_state.rot = q_init;
    init_gravity = Eigen::Vector3d(0.0, 0.0, -G_m_s2);
  }
  else init_gravity = -mean_acc / mean_acc.norm() * G_m_s2;

  setGravityUp((-init_gravity).normalized());
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  SimpleEKF::Covariance init_P = SimpleEKF::Covariance::Identity();
  init_P.block<3, 3>(9, 9) =
      Eigen::Matrix3d::Identity() * 1e-4;
  init_P.block<3, 3>(12, 12) =
      Eigen::Matrix3d::Identity() * 1e-3;
  kf_state.initialize(init_state, init_P, init_gravity);
  last_imu_ = meas.imu.back();

}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, SimpleEKF &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double imu_beg_time = get_ros_time_sec(v_imu.front()->header.stamp);
  const double imu_end_time = get_ros_time_sec(v_imu.back()->header.stamp);

  double pcl_beg_time = meas.lidar_beg_time;
  double pcl_end_time = meas.lidar_end_time;

    if (lidar_type == MARSIM) {
        pcl_beg_time = last_lidar_end_time_;
        pcl_end_time = meas.lidar_beg_time;
    }

    /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU state ***/
  SimpleEKF::State imu_state = kf_state.state();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    const double head_time = get_ros_time_sec(head->header.stamp);
    const double tail_time = get_ros_time_sec(tail->header.stamp);

    if (tail_time < last_lidar_end_time_) continue;

    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head_time - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    if(head_time < last_lidar_end_time_)
    {
      dt = tail_time - last_lidar_end_time_;
      // dt = tail_time - pcl_beg_time;
    }
    else
    {
      dt = tail_time - head_time;
    }

    dt = gap_handler(dt);
    if (dt < 0.0)
    {
      continue;
    }

    SimpleEKF::Noise noise;
    noise.gyr = cov_gyr;
    noise.acc = cov_acc;
    noise.bg = cov_bias_gyr;
    noise.ba = cov_bias_acc;
    kf_state.predict(dt, acc_avr, angvel_avr, noise);
    /* save the poses at each IMU measurements */
    imu_state = kf_state.state();
    const V3D grav_vec = kf_state.gravity();
    const Eigen::Quaterniond imu_q(imu_state.rot);
    Pose pose(imu_state.pos.x(), imu_state.pos.y(), imu_state.pos.z(),
              imu_q.x(), imu_q.y(), imu_q.z(), imu_q.w(), tail_time);
    pose._vx = imu_state.vel.x();
    pose._vy = imu_state.vel.y();
    pose._vz = imu_state.vel.z();
    pose._bgx = imu_state.bg.x();
    pose._bgy = imu_state.bg.y();
    pose._bgz = imu_state.bg.z();
    pose._bax = imu_state.ba.x();
    pose._bay = imu_state.ba.y();
    pose._baz = imu_state.ba.z();
    pose._gravx = grav_vec[0];
    pose._gravy = grav_vec[1];
    pose._gravz = grav_vec[2];
    const Eigen::Quaterniond extrinsic_q(Lidar_R_wrt_IMU);
    pose._exqx = extrinsic_q.x();
    pose._exqy = extrinsic_q.y();
    pose._exqz = extrinsic_q.z();
    pose._exqw = extrinsic_q.w();
    pose._extx = Lidar_T_wrt_IMU.x();
    pose._exty = Lidar_T_wrt_IMU.y();
    pose._extz = Lidar_T_wrt_IMU.z();
    pbuffer.Push(pose);
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += grav_vec[i];
    }
    const double offs_t = tail_time - last_lidar_end_time_;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  dt = gap_handler(dt);
  if (dt < 0.0)
  {
    last_imu_ = meas.imu.back();
    last_lidar_end_time_ = pcl_end_time;
    return;
  }
  SimpleEKF::Noise noise;
  noise.gyr = cov_gyr;
  noise.acc = cov_acc;
  noise.bg = cov_bias_gyr;
  noise.ba = cov_bias_acc;
  kf_state.predict(dt, acc_avr, angvel_avr, noise);
  imu_state = kf_state.state();
  const V3D grav_vec = kf_state.gravity();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;

  if(lidar_type != MARSIM){
      auto it_pcl = pcl_out.points.end() - 1;
      for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
      {
          auto head = it_kp - 1;
          auto tail = it_kp;
          R_imu<<MAT_FROM_ARRAY(head->rot);
          // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
          vel_imu<<VEC_FROM_ARRAY(head->vel);
          pos_imu<<VEC_FROM_ARRAY(head->pos);
          acc_imu<<VEC_FROM_ARRAY(tail->acc);
          angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

          for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
          {
              dt = it_pcl->curvature / double(1000) - head->offset_time;

              /* Transform to the 'end' frame, using only the rotation
               * Note: Compensation direction is INVERSE of Frame's moving direction
               * So if we want to compensate a point at timestamp-i to the frame-e
               * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
              M3D R_i(R_imu * Exp(angvel_avr, dt));

              V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
              V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
              V3D P_compensate = Lidar_R_wrt_IMU.transpose() * (imu_state.rot.transpose() * (R_i * (Lidar_R_wrt_IMU * P_i + Lidar_T_wrt_IMU) + T_ei) - Lidar_T_wrt_IMU);// not accurate!

              // save Undistorted points and their rotation
              it_pcl->x = P_compensate(0);
              it_pcl->y = P_compensate(1);
              it_pcl->z = P_compensate(2);

              if (it_pcl == pcl_out.points.begin()) break;
          }
      }
  }
}

void ImuProcess::Process(const MeasureGroup &meas, SimpleEKF &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  assert(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    SimpleEKF::State imu_state = kf_state.state();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;

      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
