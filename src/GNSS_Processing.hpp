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

template <typename Allocator>
inline int find_gnss_key(const std::vector<PointTypePose, Allocator> &keyposes,
                         double stamp)
{
  if (keyposes.empty() || stamp > keyposes.back().time)
    return -2;

  int key = -1;
  double dt = 0.0;
  findNearestByTime(
      keyposes, stamp, key, dt,
      [](const PointTypePose &pose) { return pose.time; });
  return dt <= gnss_dt ? key : -1;
}

class GnssProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool gnss_fix_cbk(const GnssFixMsgConstPtr &msg, PosData &data)
  {
    const GnssFixMsg &fix = *msg;
    if (fix.status.status < 0) {
      return false;
    }
    const double t = get_ros_time_sec(fix.header.stamp);

    std::lock_guard<std::mutex> lock(mtx_gnss);
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
        pos_buf.begin(), pos_buf.end(), t,
        [](const PosData &item, double stamp) { return item.t < stamp; });
    pos_buf.insert(pos_it, data);
    if (t > latest_pos.t) {
      latest_pos = data;
    }
    return true;
  }

  bool gnss_yaw_cbk(const GnssOdomMsgConstPtr &msg)
  {
    const auto &q = msg->pose.pose.orientation;
    const double t = get_ros_time_sec(msg->header.stamp);
    const double raw = quaternionToRPY(q.w, q.x, q.y, q.z).z();

    YawData data;
    data.t = t;
    data.yaw = normalizeYaw(M_PI * 0.5 - raw - heading_offset);

    std::lock_guard<std::mutex> lock(mtx_gnss);
    const auto yaw_it = std::lower_bound(
        yaw_buf.begin(), yaw_buf.end(), t,
        [](const YawData &item, double stamp) { return item.t < stamp; });
    yaw_buf.insert(yaw_it, data);
    if (t > latest_yaw.t) {
      latest_yaw = data;
    }
    return true;
  }

  bool sync_gnss_init(PosData &pos_out, YawData &yaw_out)
  {
    std::lock_guard<std::mutex> lock(mtx_gnss);

    while (!pos_buf.empty() && !yaw_buf.empty()) {
      const double dt = pos_buf.front().t - yaw_buf.front().t;

      if (std::abs(dt) <= 0.1) {
        pos_out = pos_buf.front();
        yaw_out = yaw_buf.front();
        pos_buf.pop_front();
        yaw_buf.pop_front();
        return true;
      }

      if (dt < 0.0) {
        pos_buf.pop_front();
      } else {
        yaw_buf.pop_front();
      }
    }

    return false;
  }

  std::deque<PosData> pos_buf;
  std::deque<YawData> yaw_buf;
  std::mutex mtx_gnss;

  PosData latest_pos;
  YawData latest_yaw;

  Eigen::Vector3d lever = Eigen::Vector3d::Zero();
  double heading_offset = 0.0;

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

  bool origin_ready_ = false;
  Eigen::Vector3d origin_ecef_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d origin_rot_ = Eigen::Matrix3d::Identity();
};

extern std::shared_ptr<GnssProcess> p_gnss;

#endif  // GNSS_PROCESSING_HPP
