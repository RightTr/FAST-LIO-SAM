#ifndef GNSS_PROCESSING_HPP
#define GNSS_PROCESSING_HPP

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "common_utils.h"
#include "utility.h"
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
    if (fix.status.status < 0) {
      return false;
    }
    const double t = get_ros_time_sec(fix.header.stamp);

    std::lock_guard<std::mutex> lock(mtx_);
    if (!origin_ready_) {
      origin_ecef_ = GeodeticToECEF(fix.latitude, fix.longitude, fix.altitude);
      origin_rot_ = EnuRotation(fix.latitude, fix.longitude);
      origin_ready_ = true;
    }

    const Eigen::Vector3d ecef = GeodeticToECEF(fix.latitude, fix.longitude, fix.altitude);
    data.t = t;
    data.p = origin_rot_ * (ecef - origin_ecef_);
    data.cov = covarianceFromMsg(fix);

    if ((data.cov.diagonal().array() <= 0.0).any())
    {
      return false;
    }

    const auto pos_it = std::lower_bound(
        pos_buf_.begin(), pos_buf_.end(), t,
        [](const PosData &item, double stamp) { return item.t < stamp; });
    pos_buf_.insert(pos_it, data);
    if (!has_latest_pos_ || t > latest_pos_.t) {
      latest_pos_ = data;
      has_latest_pos_ = true;
    }
    return true;
  }

  bool pushYaw(const GnssOdomMsgConstPtr &msg)
  {
    if (!msg) {
      return false;
    }

    const auto &q = msg->pose.pose.orientation;
    const double t = get_ros_time_sec(msg->header.stamp);
    const double raw = quaternionToRPY(q.w, q.x, q.y, q.z).z();

    YawData data;
    data.t = t;
    data.yaw = normalizeYaw(M_PI * 0.5 - raw - off_);

    std::lock_guard<std::mutex> lock(mtx_);
    const auto yaw_it = std::lower_bound(
        yaw_buf_.begin(), yaw_buf_.end(), t,
        [](const YawData &item, double stamp) { return item.t < stamp; });
    yaw_buf_.insert(yaw_it, data);
    if (!has_latest_yaw_ || t > latest_yaw_.t) {
      latest_yaw_ = data;
      has_latest_yaw_ = true;
    }
    return true;
  }

  bool latest(PosData &pos, YawData &yaw) const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!has_latest_pos_ || !has_latest_yaw_) {
      return false;
    }

    pos = latest_pos_;
    yaw = latest_yaw_;
    return true;
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

  bool frontPos(PosData &pos) const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (pos_buf_.empty()) {
      return false;
    }
    pos = pos_buf_.front();
    return true;
  }

  bool popPos(double expected_stamp)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!pos_buf_.empty() && pos_buf_.front().t == expected_stamp) {
      pos_buf_.pop_front();
      return true;
    }
    return false;
  }

  bool frontYaw(YawData &yaw) const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (yaw_buf_.empty()) {
      return false;
    }
    yaw = yaw_buf_.front();
    return true;
  }

  bool popYaw(double expected_stamp)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!yaw_buf_.empty() && yaw_buf_.front().t == expected_stamp) {
      yaw_buf_.pop_front();
      return true;
    }
    return false;
  }

  bool pickInitPair(PosData &pos_out, YawData &yaw_out)
  {
    std::lock_guard<std::mutex> lock(mtx_);

    std::size_t pos_idx = 0;
    std::size_t yaw_idx = 0;
    while (pos_idx < pos_buf_.size() && yaw_idx < yaw_buf_.size()) {
      const double pos_t = pos_buf_[pos_idx].t;
      const double yaw_t = yaw_buf_[yaw_idx].t;

      if (std::abs(pos_t - yaw_t) <= 0.1) {
        pos_out = pos_buf_[pos_idx];
        yaw_out = yaw_buf_[yaw_idx];
        pos_buf_.erase(pos_buf_.begin() + static_cast<std::ptrdiff_t>(pos_idx));
        yaw_buf_.erase(yaw_buf_.begin() + static_cast<std::ptrdiff_t>(yaw_idx));
        return true;
      }

      if (pos_t < yaw_t) {
        ++pos_idx;
      } else {
        ++yaw_idx;
      }
    }

    return false;
  }

 public:
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

  PosData latest_pos_;
  YawData latest_yaw_;
  bool has_latest_pos_ = false;
  bool has_latest_yaw_ = false;

  Eigen::Vector3d lever_ = Eigen::Vector3d::Zero();
  double off_ = 0.0;

  bool origin_ready_ = false;
  Eigen::Vector3d origin_ecef_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d origin_rot_ = Eigen::Matrix3d::Identity();
};

extern std::shared_ptr<GnssProcess> p_gnss;

#endif  // GNSS_PROCESSING_HPP
