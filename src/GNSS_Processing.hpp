#ifndef GNSS_PROCESSING_HPP
#define GNSS_PROCESSING_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ros_utils.h"

#ifdef USE_ROS1
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/NavSatFix.h>
using GnssFixMsg = sensor_msgs::NavSatFix;
using GnssFixMsgConstPtr = sensor_msgs::NavSatFix::ConstPtr;
using GnssOdomMsg = nav_msgs::Odometry;
using GnssOdomMsgConstPtr = nav_msgs::Odometry::ConstPtr;
#elif defined(USE_ROS2)
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
using GnssFixMsg = sensor_msgs::msg::NavSatFix;
using GnssFixMsgConstPtr = sensor_msgs::msg::NavSatFix::ConstSharedPtr;
using GnssOdomMsg = nav_msgs::msg::Odometry;
using GnssOdomMsgConstPtr = nav_msgs::msg::Odometry::ConstSharedPtr;
#endif

struct PosData
{
  double t = -1.0;
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
};

struct YawData
{
  double t = -1.0;
  double yaw = 0.0;
};

class GnssProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GnssProcess() { Reset(); }

  GnssProcess(double origin_lat, double origin_lon, double origin_alt)
  {
    Reset();
    InitOriginPosition(origin_lat, origin_lon, origin_alt);
  }

  void Reset()
  {
    std::lock_guard<std::mutex> lock(mtx_);

    origin_ready_ = false;
    origin_lat_ = 0.0;
    origin_lon_ = 0.0;
    origin_alt_ = 0.0;
    origin_ecef_.setZero();
    origin_rot_.setIdentity();

    cur_lat_ = 0.0;
    cur_lon_ = 0.0;
    cur_alt_ = 0.0;
    cur_ecef_.setZero();
    cur_enu_.setZero();
    cur_cov_.setZero();
    cur_fix_status_ = -1;
    cur_service_ = 0;
    cur_valid_ = false;
    cur_stamp_ = 0.0;

    pos_buf_.clear();
    yaw_buf_.clear();
    has_pos_ = false;
    has_yaw_ = false;
    last_pos_ = PosData();
    last_yaw_ = YawData();

    lever_.setZero();
    off_ = 0.0;
  }

  void InitOriginPosition(double latitude, double longitude, double altitude)
  {
    origin_lat_ = latitude;
    origin_lon_ = longitude;
    origin_alt_ = altitude;
    origin_ecef_ = GeodeticToECEF(latitude, longitude, altitude);
    origin_rot_ = EnuRotation(origin_lat_, origin_lon_);
    origin_ready_ = true;
  }

  bool InitOriginPosition(const GnssFixMsg &fix)
  {
    if (!IsFixUsable(fix)) {
      return false;
    }
    InitOriginPosition(fix.latitude, fix.longitude, fix.altitude);
    return true;
  }

  void setLever(const Eigen::Vector3d &lever)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    lever_ = lever;
  }

  void setOffset(double off)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    off_ = off;
  }

  void pushFix(const GnssFixMsgConstPtr &msg)
  {
    if (!msg) {
      return;
    }

    const GnssFixMsg &fix = *msg;
    if (!IsFixUsable(fix)) {
      return;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    if (!origin_ready_) {
      InitOriginPosition(fix);
    }
    if (!UpdateXYZ(fix)) {
      return;
    }

    PosData data;
    data.t = get_ros_time_sec(fix.header.stamp);
    data.p = cur_enu_;
    data.cov = cur_cov_;

    pos_buf_.push_back(data);
    while (pos_buf_.size() > 2000) {
      pos_buf_.pop_front();
    }

    last_pos_ = data;
    has_pos_ = true;
  }

  void pushYaw(const GnssOdomMsgConstPtr &msg)
  {
    if (!msg) {
      return;
    }

    const auto &q = msg->pose.pose.orientation;
    const double raw = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                                  1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    if (!std::isfinite(raw)) {
      return;
    }

    YawData data;
    data.t = get_ros_time_sec(msg->header.stamp);
    data.yaw = WrapYaw(M_PI * 0.5 - raw - off_);

    std::lock_guard<std::mutex> lock(mtx_);
    yaw_buf_.push_back(data);
    while (yaw_buf_.size() > 2000) {
      yaw_buf_.pop_front();
    }

    last_yaw_ = data;
    has_yaw_ = true;
  }

  bool peekOldestPos(PosData &out) const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (pos_buf_.empty())
      return false;
    out = pos_buf_.front();
    return true;
  }

  bool popOldestPos(PosData &out)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (pos_buf_.empty())
      return false;
    out = pos_buf_.front();
    pos_buf_.pop_front();
    return true;
  }

  bool peekOldestYaw(YawData &out) const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (yaw_buf_.empty())
      return false;
    out = yaw_buf_.front();
    return true;
  }

  bool popOldestYaw(YawData &out)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (yaw_buf_.empty())
      return false;
    out = yaw_buf_.front();
    yaw_buf_.pop_front();
    return true;
  }

  bool pickInitPair(PosData &pos_out, YawData &yaw_out)
  {
    std::lock_guard<std::mutex> lock(mtx_);

    while (!pos_buf_.empty() && !yaw_buf_.empty()) {
      const double pos_t = pos_buf_.front().t;
      const double yaw_t = yaw_buf_.front().t;

      if (std::abs(pos_t - yaw_t) <= 0.1) {
        pos_out = pos_buf_.front();
        yaw_out = yaw_buf_.front();
        pos_buf_.pop_front();
        yaw_buf_.pop_front();
        return true;
      }

      if (pos_t < yaw_t) {
        pos_buf_.pop_front();
      } else {
        yaw_buf_.pop_front();
      }
    }

    return false;
  }

  bool latestPos(PosData &out) const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!has_pos_) {
      return false;
    }
    out = last_pos_;
    return true;
  }

  bool latestYaw(YawData &out) const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!has_yaw_) {
      return false;
    }
    out = last_yaw_;
    return true;
  }

  Eigen::Vector3d lever() const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    return lever_;
  }

  bool UpdateXYZ(double latitude, double longitude, double altitude)
  {
    if (!origin_ready_) {
      return false;
    }

    cur_lat_ = latitude;
    cur_lon_ = longitude;
    cur_alt_ = altitude;
    cur_ecef_ = GeodeticToECEF(latitude, longitude, altitude);
    cur_enu_ = ECEFToENU(cur_ecef_);
    cur_valid_ = true;
    return true;
  }

  bool UpdateXYZ(const GnssFixMsg &fix)
  {
    if (!IsFixUsable(fix)) {
      cur_valid_ = false;
      return false;
    }

    if (!origin_ready_) {
      InitOriginPosition(fix);
    }

    cur_stamp_ = get_ros_time_sec(fix.header.stamp);
    cur_fix_status_ = fix.status.status;
    cur_service_ = fix.status.service;
    cur_cov_ = covarianceFromMsg(fix);
    return UpdateXYZ(fix.latitude, fix.longitude, fix.altitude);
  }

  bool UpdateXYZ(const GnssFixMsg &fix, Eigen::Vector3d &enu_out)
  {
    if (!UpdateXYZ(fix)) {
      return false;
    }
    enu_out = cur_enu_;
    return true;
  }

  GnssOdomMsg ToOdometry(const std::string &frame_id = "map",
                         const std::string &child_frame_id = "gnss") const
  {
    GnssOdomMsg msg;
    msg.header.stamp = get_ros_time(cur_stamp_);
    msg.header.frame_id = frame_id;
    msg.child_frame_id = child_frame_id;
    msg.pose.pose.position.x = cur_enu_.x();
    msg.pose.pose.position.y = cur_enu_.y();
    msg.pose.pose.position.z = cur_enu_.z();
    msg.pose.pose.orientation.x = 0.0;
    msg.pose.pose.orientation.y = 0.0;
    msg.pose.pose.orientation.z = 0.0;
    msg.pose.pose.orientation.w = 1.0;

    msg.pose.covariance.fill(0.0);
    msg.pose.covariance[0] = cur_cov_(0, 0);
    msg.pose.covariance[7] = cur_cov_(1, 1);
    msg.pose.covariance[14] = cur_cov_(2, 2);
    return msg;
  }

  bool origin_ready() const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    return origin_ready_;
  }

  bool valid() const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    return cur_valid_;
  }

  double stamp_sec() const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    return cur_stamp_;
  }

  Eigen::Vector3d CurrentENU() const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    return cur_enu_;
  }

  Eigen::Matrix3d covariance() const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    return cur_cov_;
  }

 private:
  static constexpr double kWgs84A = 6378137.0;
  static constexpr double kWgs84F = 1.0 / 298.257223563;
  static constexpr double kWgs84E2 = kWgs84F * (2.0 - kWgs84F);
  static constexpr double kPi = 3.14159265358979323846;

  static double WrapYaw(double yaw)
  {
    while (yaw > M_PI) {
      yaw -= 2.0 * M_PI;
    }
    while (yaw < -M_PI) {
      yaw += 2.0 * M_PI;
    }
    return yaw;
  }

  static bool IsFixUsable(const GnssFixMsg &fix)
  {
    const bool has_valid_status = fix.status.status >= 0;
    const bool has_finite_lla = std::isfinite(fix.latitude) &&
                                std::isfinite(fix.longitude) &&
                                std::isfinite(fix.altitude);
    return has_valid_status && has_finite_lla;
  }

  static Eigen::Matrix3d EnuRotation(double latitude_deg, double longitude_deg)
  {
    const double lat = DegToRad(latitude_deg);
    const double lon = DegToRad(longitude_deg);
    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);
    const double sin_lon = std::sin(lon);
    const double cos_lon = std::cos(lon);

    Eigen::Matrix3d rot;
    rot << -sin_lon,            cos_lon,           0.0,
           -sin_lat * cos_lon,  -sin_lat * sin_lon, cos_lat,
            cos_lat * cos_lon,   cos_lat * sin_lon, sin_lat;
    return rot;
  }

  static Eigen::Vector3d GeodeticToECEF(double latitude_deg,
                                        double longitude_deg,
                                        double altitude)
  {
    const double lat = DegToRad(latitude_deg);
    const double lon = DegToRad(longitude_deg);
    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);
    const double sin_lon = std::sin(lon);
    const double cos_lon = std::cos(lon);
    const double n = kWgs84A / std::sqrt(1.0 - kWgs84E2 * sin_lat * sin_lat);

    Eigen::Vector3d ecef;
    ecef.x() = (n + altitude) * cos_lat * cos_lon;
    ecef.y() = (n + altitude) * cos_lat * sin_lon;
    ecef.z() = (n * (1.0 - kWgs84E2) + altitude) * sin_lat;
    return ecef;
  }

  static void ECEFToGeodetic(const Eigen::Vector3d &ecef,
                             double &latitude_deg,
                             double &longitude_deg,
                             double &altitude)
  {
    const double x = ecef.x();
    const double y = ecef.y();
    const double z = ecef.z();
    const double b = kWgs84A * (1.0 - kWgs84F);
    const double ep = std::sqrt((kWgs84A * kWgs84A - b * b) / (b * b));
    const double p = std::sqrt(x * x + y * y);
    const double th = std::atan2(kWgs84A * z, b * p);
    const double lon = std::atan2(y, x);
    const double sin_th = std::sin(th);
    const double cos_th = std::cos(th);
    const double lat = std::atan2(z + ep * ep * b * sin_th * sin_th * sin_th,
                                  p - kWgs84E2 * kWgs84A * cos_th * cos_th * cos_th);
    const double sin_lat = std::sin(lat);
    const double n = kWgs84A / std::sqrt(1.0 - kWgs84E2 * sin_lat * sin_lat);
    altitude = p / std::cos(lat) - n;
    latitude_deg = RadToDeg(lat);
    longitude_deg = RadToDeg(lon);
  }

  Eigen::Vector3d ECEFToENU(const Eigen::Vector3d &ecef) const
  {
    return origin_rot_ * (ecef - origin_ecef_);
  }

  static double DegToRad(double deg)
  {
    return deg * kPi / 180.0;
  }

  static double RadToDeg(double rad)
  {
    return rad * 180.0 / kPi;
  }

  static Eigen::Matrix3d covarianceFromMsg(const GnssFixMsg &fix)
  {
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

    if (fix.position_covariance_type == GnssFixMsg::COVARIANCE_TYPE_APPROXIMATED ||
        fix.position_covariance_type == GnssFixMsg::COVARIANCE_TYPE_DIAGONAL_KNOWN ||
        fix.position_covariance_type == GnssFixMsg::COVARIANCE_TYPE_KNOWN) {
      cov(0, 0) = fix.position_covariance[0];
      cov(0, 1) = fix.position_covariance[1];
      cov(0, 2) = fix.position_covariance[2];
      cov(1, 0) = fix.position_covariance[3];
      cov(1, 1) = fix.position_covariance[4];
      cov(1, 2) = fix.position_covariance[5];
      cov(2, 0) = fix.position_covariance[6];
      cov(2, 1) = fix.position_covariance[7];
      cov(2, 2) = fix.position_covariance[8];
    } else {
      cov.diagonal().setConstant(10000.0);
    }
    return cov;
  }

  mutable std::mutex mtx_;

  std::deque<PosData> pos_buf_;
  std::deque<YawData> yaw_buf_;

  PosData last_pos_;
  YawData last_yaw_;
  bool has_pos_ = false;
  bool has_yaw_ = false;

  Eigen::Vector3d lever_ = Eigen::Vector3d::Zero();
  double off_ = 0.0;

  bool origin_ready_ = false;
  double origin_lat_ = 0.0;
  double origin_lon_ = 0.0;
  double origin_alt_ = 0.0;
  Eigen::Vector3d origin_ecef_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d origin_rot_ = Eigen::Matrix3d::Identity();

  double cur_lat_ = 0.0;
  double cur_lon_ = 0.0;
  double cur_alt_ = 0.0;
  Eigen::Vector3d cur_ecef_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d cur_enu_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d cur_cov_ = Eigen::Matrix3d::Zero();
  int cur_fix_status_ = -1;
  uint16_t cur_service_ = 0;
  bool cur_valid_ = false;
  double cur_stamp_ = 0.0;
};

extern std::shared_ptr<GnssProcess> p_gnss;

#endif  // GNSS_PROCESSING_HPP
