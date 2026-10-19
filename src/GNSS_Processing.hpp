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

struct GnssPositionData
{
  double time = -1.0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
};

struct GnssYawData
{
  double time = -1.0;
  double yaw = 0.0;
};

template <typename Allocator>
inline int findGnssKey(const std::vector<PointTypePose, Allocator> &keyPoses,
                       double stamp)
{
  if (keyPoses.empty() || stamp > keyPoses.back().time)
    return -2;

  int keyIndex = -1;
  double timeDiff = 0.0;
  findNearestByTime(
      keyPoses, stamp, keyIndex, timeDiff,
      [](const PointTypePose &pose) { return pose.time; });
  return timeDiff <= gnss_dt ? keyIndex : -1;
}

class GnssProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool processGnssFix(const GnssFixMsgConstPtr &msg, GnssPositionData &data)
  {
    const GnssFixMsg &fix = *msg;
    if (fix.status.status < 0) {
      return false;
    }
    const double stamp = get_ros_time_sec(fix.header.stamp);

    std::lock_guard<std::mutex> lock(gnssMutex);
    if (!originReady_) {
      originEcef_ = geodeticToEcef(fix.latitude, fix.longitude, fix.altitude);
      originRotation_ = enuRotation(fix.latitude, fix.longitude);
      originReady_ = true;
    }

    const Eigen::Vector3d ecef = geodeticToEcef(fix.latitude, fix.longitude, fix.altitude);
    data.time = stamp;
    data.position = originRotation_ * (ecef - originEcef_);
    data.covariance = covarianceFromMsg(fix);

    if ((data.covariance.diagonal().array() <= 0.0).any())
    {
      return false;
    }

    const auto posIt = std::lower_bound(
        posBuffer.begin(), posBuffer.end(), stamp,
        [](const GnssPositionData &item, double stamp) { return item.time < stamp; });
    posBuffer.insert(posIt, data);
    if (stamp > latestPos.time) {
      latestPos = data;
    }
    return true;
  }

  bool processGnssYaw(const GnssOdomMsgConstPtr &msg)
  {
    const auto &q = msg->pose.pose.orientation;
    const double stamp = get_ros_time_sec(msg->header.stamp);
    const double rawYaw = quaternionToRPY(q.w, q.x, q.y, q.z).z();

    GnssYawData data;
    data.time = stamp;
    data.yaw = normalizeYaw(M_PI * 0.5 - rawYaw - headingOffset);

    std::lock_guard<std::mutex> lock(gnssMutex);
    const auto yawIt = std::lower_bound(
        yawBuffer.begin(), yawBuffer.end(), stamp,
        [](const GnssYawData &item, double stamp) { return item.time < stamp; });
    yawBuffer.insert(yawIt, data);
    if (stamp > latestYaw.time) {
      latestYaw = data;
    }
    return true;
  }

  bool syncGnssInit(GnssPositionData &posOut, GnssYawData &yawOut)
  {
    std::lock_guard<std::mutex> lock(gnssMutex);

    while (!posBuffer.empty() && !yawBuffer.empty()) {
      const double timeDiff = posBuffer.front().time - yawBuffer.front().time;

      if (std::abs(timeDiff) <= 0.1) {
        posOut = posBuffer.front();
        yawOut = yawBuffer.front();
        posBuffer.pop_front();
        yawBuffer.pop_front();
        return true;
      }

      if (timeDiff < 0.0) {
        posBuffer.pop_front();
      } else {
        yawBuffer.pop_front();
      }
    }

    return false;
  }

  std::deque<GnssPositionData> posBuffer;
  std::deque<GnssYawData> yawBuffer;
  std::mutex gnssMutex;

  GnssPositionData latestPos;
  GnssYawData latestYaw;

  Eigen::Vector3d leverArm = Eigen::Vector3d::Zero();
  double headingOffset = 0.0;

 private:
  static constexpr double kWgs84A = 6378137.0;
  static constexpr double kWgs84F = 1.0 / 298.257223563;
  static constexpr double kWgs84E2 = kWgs84F * (2.0 - kWgs84F);

  static Eigen::Matrix3d enuRotation(double latitudeDeg, double longitudeDeg)
  {
    const double lat = latitudeDeg * M_PI / 180.0;
    const double lon = longitudeDeg * M_PI / 180.0;
    const double sinLat = std::sin(lat);
    const double cosLat = std::cos(lat);
    const double sinLon = std::sin(lon);
    const double cosLon = std::cos(lon);

    Eigen::Matrix3d rot;
    rot << -sinLon,          cosLon,         0.0,
           -sinLat * cosLon, -sinLat * sinLon, cosLat,
            cosLat * cosLon,  cosLat * sinLon, sinLat;
    return rot;
  }

  static Eigen::Vector3d geodeticToEcef(double latitudeDeg,
                                        double longitudeDeg,
                                        double altitude)
  {
    const double lat = latitudeDeg * M_PI / 180.0;
    const double lon = longitudeDeg * M_PI / 180.0;
    const double sinLat = std::sin(lat);
    const double cosLat = std::cos(lat);
    const double sinLon = std::sin(lon);
    const double cosLon = std::cos(lon);
    const double radius = kWgs84A / std::sqrt(1.0 - kWgs84E2 * sinLat * sinLat);

    Eigen::Vector3d ecef;
    ecef.x() = (radius + altitude) * cosLat * cosLon;
    ecef.y() = (radius + altitude) * cosLat * sinLon;
    ecef.z() = (radius * (1.0 - kWgs84E2) + altitude) * sinLat;
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

  bool originReady_ = false;
  Eigen::Vector3d originEcef_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d originRotation_ = Eigen::Matrix3d::Identity();
};

extern std::shared_ptr<GnssProcess> p_gnss;

#endif  // GNSS_PROCESSING_HPP
