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
  bool pushFix(const GnssFixMsgConstPtr &msg, PosData &data)
  {
    if (!msg) {
      return false;
    }

    const GnssFixMsg &fix = *msg;
    if (fix.status.status < 0 ||
        !std::isfinite(fix.latitude) ||
        !std::isfinite(fix.longitude) ||
        !std::isfinite(fix.altitude))
    {
      return false;
    }

    std::lock_guard<std::mutex> lock(mtx_);

    if (!origin_ready_) {
      origin_ecef_ = GeodeticToECEF(fix.latitude, fix.longitude, fix.altitude);
      origin_rot_ = EnuRotation(fix.latitude, fix.longitude);
      origin_ready_ = true;
    }

    const Eigen::Vector3d ecef = GeodeticToECEF(fix.latitude, fix.longitude, fix.altitude);
    data.t = get_ros_time_sec(fix.header.stamp);
    data.p = origin_rot_ * (ecef - origin_ecef_);
    data.cov = covarianceFromMsg(fix);

    if (!data.p.allFinite() ||
        !data.cov.allFinite() ||
        (data.cov.diagonal().array() <= 0.0).any())
    {
      return false;
    }

    pos_buf_.push_back(data);
    return true;
  }

  void pushYaw(const GnssOdomMsgConstPtr &msg)
  {
    if (!msg) {
      return;
    }

    const auto &q = msg->pose.pose.orientation;
    const double raw = std::atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z));

    YawData data;
    data.t = get_ros_time_sec(msg->header.stamp);
    const double yaw = M_PI * 0.5 - raw - off_;
    data.yaw = std::atan2(std::sin(yaw), std::cos(yaw));

    std::lock_guard<std::mutex> lock(mtx_);
    yaw_buf_.push_back(data);
  }

  bool syncPos(double time, PosData &out)
  {
    std::lock_guard<std::mutex> lock(mtx_);

    bool found = false;
    while (!pos_buf_.empty() && pos_buf_.front().t <= time) {
      out = pos_buf_.front();
      pos_buf_.pop_front();
      found = true;
    }

    return found;
  }

  bool syncYaw(double time, YawData &out)
  {
    std::lock_guard<std::mutex> lock(mtx_);

    bool found = false;
    while (!yaw_buf_.empty() && yaw_buf_.front().t <= time) {
      out = yaw_buf_.front();
      yaw_buf_.pop_front();
      found = true;
    }

    return found;
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

  Eigen::Vector3d lever() const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    return lever_;
  }

 private:
  static constexpr double kWgs84A = 6378137.0;
  static constexpr double kWgs84F = 1.0 / 298.257223563;
  static constexpr double kWgs84E2 = kWgs84F * (2.0 - kWgs84F);

  static Eigen::Matrix3d EnuRotation(double latitude_deg, double longitude_deg)
  {
    const double lat = latitude_deg * M_PI / 180.0;
    const double lon = longitude_deg * M_PI / 180.0;
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
    const double lat = latitude_deg * M_PI / 180.0;
    const double lon = longitude_deg * M_PI / 180.0;
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

  Eigen::Vector3d ECEFToENU(const Eigen::Vector3d &ecef) const
  {
    return origin_rot_ * (ecef - origin_ecef_);
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

  Eigen::Vector3d lever_ = Eigen::Vector3d::Zero();
  double off_ = 0.0;

  bool origin_ready_ = false;
  Eigen::Vector3d origin_ecef_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d origin_rot_ = Eigen::Matrix3d::Identity();
};

extern std::shared_ptr<GnssProcess> p_gnss;

#endif  // GNSS_PROCESSING_HPP
