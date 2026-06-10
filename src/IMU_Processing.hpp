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

#include "use-ikfom.hpp"
#include "preprocess.h"
#include "posebuffer.h"

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
  void Reset(double start_timestamp, const ImuMsgConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void set_use_zupt(bool enabled);
  void set_grav_align(bool enabled);
  void set_zupt_thresholds(double acc_norm_threshold, double gyro_threshold);
  void set_zupt_adaptive_params(double r_min, double r_max, double conf_min,
                                 double inflate_pos, double inflate_rot, int inflate_start);

  double compute_static_confidence(const MeasureGroup &meas);
  double get_static_confidence() const { return static_confidence_; }

  Eigen::Matrix<double, 12, 12> Q;
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

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
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);
  void zupt_update(esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state, double confidence);

  bool   use_zupt = false;
  bool   grav_align_ = false;
  double zupt_acc_var_threshold;
  double zupt_gyro_var_threshold;

  double static_confidence_      = 0.0;
  int    static_step_count_      = 0;
  double zupt_vel_ema_           = 0.0;
  double zupt_r_min_             = 1e-5;
  double zupt_r_max_             = 1.0;
  double zupt_confidence_min_    = 0.05;
  double cov_inflate_pos_        = 1e-7;
  double cov_inflate_rot_        = 1e-8;
  int    static_inflation_start_ = 200;

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
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  Q = process_noise_cov();
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

void ImuProcess::set_use_zupt(bool enabled) { 
  use_zupt = enabled; 
}

void ImuProcess::set_grav_align(bool enabled)
{
  grav_align_ = enabled;
}

void ImuProcess::set_zupt_thresholds(double acc_var_threshold, double gyro_var_threshold)
{
  zupt_acc_var_threshold = acc_var_threshold;
  zupt_gyro_var_threshold = gyro_var_threshold;
}

void ImuProcess::set_zupt_adaptive_params(double r_min, double r_max, double conf_min,
                                           double inflate_pos, double inflate_rot, int inflate_start)
{
  zupt_r_min_             = r_min;
  zupt_r_max_             = r_max;
  zupt_confidence_min_    = conf_min;
  cov_inflate_pos_        = inflate_pos;
  cov_inflate_rot_        = inflate_rot;
  static_inflation_start_ = inflate_start;
}

// confidence = exp(-3 * max_normalized_variance), in [0,1]
// 1 = definitely static, 0 = definitely moving
double ImuProcess::compute_static_confidence(const MeasureGroup &meas)
{
  if (meas.imu.empty()) { static_confidence_ = 0.0; return 0.0; }

  V3D acc_sum = Zero3d, gyro_sum = Zero3d;
  for (const auto &imu : meas.imu)
  {
    acc_sum  += V3D(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);
    gyro_sum += V3D(imu->angular_velocity.x,    imu->angular_velocity.y,    imu->angular_velocity.z);
  }
  const double inv_n  = 1.0 / static_cast<double>(meas.imu.size());
  const V3D acc_mean  = acc_sum  * inv_n;
  const V3D gyro_mean = gyro_sum * inv_n;

  V3D acc_var = Zero3d, gyro_var = Zero3d;
  for (const auto &imu : meas.imu)
  {
    V3D da = V3D(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z) - acc_mean;
    V3D dg = V3D(imu->angular_velocity.x,    imu->angular_velocity.y,    imu->angular_velocity.z)    - gyro_mean;
    acc_var  += da.cwiseProduct(da);
    gyro_var += dg.cwiseProduct(dg);
  }
  acc_var  *= inv_n;
  gyro_var *= inv_n;

  // compute the maximum variance ratio among all axes and both acc and gyro
  double max_ratio = 0.0;
  for (int i = 0; i < 3; i++)
  {
    max_ratio = std::max(max_ratio, acc_var[i]  / zupt_acc_var_threshold);
    max_ratio = std::max(max_ratio, gyro_var[i] / zupt_gyro_var_threshold);
  }
   
  // static_confidence_ -> 1 when variance is much smaller than threshold, very likely static
  // static confidence_ -> 0 when variance is much larger than threshold, very likely moving
  static_confidence_ = std::exp(-3.0 * max_ratio);
  return static_confidence_;
}

void ImuProcess::PBufferPop(Pose &pose)
{
  pose = pbuffer.Pop();
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
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
  state_ikfom init_state = kf_state.get_x();
  const double mean_acc_norm = mean_acc.norm();
  if (grav_align_ )
  {
    // Use the averaged gravity direction to initialize roll/pitch; yaw remains unconstrained.
    const Eigen::Vector3d gravity_meas = mean_acc / mean_acc_norm;
    const Eigen::Quaterniond q_init = Eigen::Quaterniond::FromTwoVectors(gravity_meas,
                                                                         Eigen::Vector3d::UnitZ());
    init_state.rot = q_init;
    init_state.grav = S2(Eigen::Vector3d(0.0, 0.0, -G_m_s2));
  }
  else init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = 0.00001; 
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();

}

void ImuProcess::zupt_update(esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state, double confidence)
{
  const state_ikfom& s = kf_state.get_x();

  // Residual adaptation: EMA of velocity norm attenuates confidence when
  // the system moves slowly but continuously (low IMU variance, nonzero vel)
  const double vel_norm    = s.vel.norm();
  zupt_vel_ema_            = 0.9 * zupt_vel_ema_ + 0.1 * vel_norm;
  
  // confidence -> 1 when static, 0 when moving; also attenuated when slow movement is detected
  const double eff_conf    = confidence * std::exp(-zupt_vel_ema_ / 0.05);

  // Dynamic ZUPT weight: R = zupt_r_min / eff_conf, capped at zupt_r_max
  const double r_scalar    = std::min(zupt_r_min_ / std::max(eff_conf, 1e-4), zupt_r_max_);

  esekfom::dyn_share_datastruct<double> ekfom_data;
  ekfom_data.h_x = MatrixXd::Zero(3, state_ikfom::DOF);
  ekfom_data.h.resize(3);
  ekfom_data.h_x.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
  ekfom_data.h.block<3, 1>(0, 0)    = -s.vel;
  ekfom_data.R = MatrixXd::Identity(3, 3) * r_scalar;

  auto P = kf_state.get_P();

  // Covariance inflation: prevent lock-in after long static periods
  static_step_count_++;
  if (static_step_count_ > static_inflation_start_)
  {
    P.block<3, 3>(0, 0) += Eigen::Matrix3d::Identity() * cov_inflate_pos_;
    P.block<3, 3>(3, 3) += Eigen::Matrix3d::Identity() * cov_inflate_rot_;
    kf_state.change_P(P);
  }

  // EKF update with pseudo-measurement: vel = 0
  const MatrixXd K = P * ekfom_data.h_x.transpose() *
                     (ekfom_data.h_x * P * ekfom_data.h_x.transpose() + ekfom_data.R).inverse();
  const Matrix<double, state_ikfom::DOF, state_ikfom::DOF> I =
    Matrix<double, state_ikfom::DOF, state_ikfom::DOF>::Identity();

  state_ikfom x = kf_state.get_x();
  x.boxplus(Matrix<double, state_ikfom::DOF, 1>(K * ekfom_data.h));
  kf_state.change_x(x);

  const Matrix<double, state_ikfom::DOF, state_ikfom::DOF> KH = K * ekfom_data.h_x;
  Matrix<double, state_ikfom::DOF, state_ikfom::DOF> P_new =
    (I - KH) * P * (I - KH).transpose() + K * ekfom_data.R * K.transpose();
  kf_state.change_P(P_new);
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** Compute static confidence for this LiDAR frame (caches result in static_confidence_) ***/
  const double frame_confidence = use_zupt ? compute_static_confidence(meas) : 0.0;
  const bool   is_frame_static  = frame_confidence >= zupt_confidence_min_;
  if (!is_frame_static) static_step_count_ = 0;

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
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    const double head_time = get_ros_time_sec(head->header.stamp);
    const double tail_time = get_ros_time_sec(tail->header.stamp);

    if (head_time < last_lidar_end_time_) continue;

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

    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);
    if (is_frame_static) zupt_update(kf_state, frame_confidence);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    Pose pose(imu_state.pos.x(), imu_state.pos.y(), imu_state.pos.z(),
              imu_state.rot.x(), imu_state.rot.y(), imu_state.rot.z(), imu_state.rot.w(),
              tail_time);
    pose._vx = imu_state.vel.x();
    pose._vy = imu_state.vel.y();
    pose._vz = imu_state.vel.z();
    pose._bgx = imu_state.bg.x();
    pose._bgy = imu_state.bg.y();
    pose._bgz = imu_state.bg.z();
    pose._bax = imu_state.ba.x();
    pose._bay = imu_state.ba.y();
    pose._baz = imu_state.ba.z();
    pose._gravx = imu_state.grav[0];
    pose._gravy = imu_state.grav[1];
    pose._gravz = imu_state.grav[2];
    pose._exqx = imu_state.offset_R_L_I.x();
    pose._exqy = imu_state.offset_R_L_I.y();
    pose._exqz = imu_state.offset_R_L_I.z();
    pose._exqw = imu_state.offset_R_L_I.w();
    pose._extx = imu_state.offset_T_L_I.x();
    pose._exty = imu_state.offset_T_L_I.y();
    pose._extz = imu_state.offset_T_L_I.z();
    pbuffer.Push(pose);
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    const double offs_t = tail_time - last_lidar_end_time_;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  if (is_frame_static) zupt_update(kf_state, frame_confidence);

  imu_state = kf_state.get_x();
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
              V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!

              // save Undistorted points and their rotation
              it_pcl->x = P_compensate(0);
              it_pcl->y = P_compensate(1);
              it_pcl->z = P_compensate(2);

              if (it_pcl == pcl_out.points.begin()) break;
          }
      }
  }
}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
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

    state_ikfom imu_state = kf_state.get_x();
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
