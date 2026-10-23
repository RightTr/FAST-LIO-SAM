#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <iterator>
#include <atomic>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Geometry>
#include <pcl_conversions/pcl_conversions.h>

#include "ros/ros_utils.h"
#include "ros/ros_parameters.h"
#include "global_graph.h"
#include "ros/ros_manager.h"
#include <pcl/common/transforms.h>

inline std::string format_unix_time(double t)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9) << t;
    return oss.str();
}

inline int64_t nowTimeUs()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline double normalizeYaw(double yaw, double offset = 0.0)
{
    yaw += offset;
    while (yaw > M_PI)
        yaw -= 2.0 * M_PI;
    while (yaw < -M_PI)
        yaw += 2.0 * M_PI;
    return yaw;
}

inline Eigen::Vector3d quaternionToRPY(double qw, double qx, double qy, double qz)
{
    const double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    const double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    const double roll = std::atan2(sinr_cosp, cosr_cosp);

    const double sinp = 2.0 * (qw * qy - qz * qx);
    const double pitch = (std::abs(sinp) >= 1.0)
        ? std::copysign(M_PI / 2.0, sinp)
        : std::asin(sinp);

    const double siny_cosp = 2.0 * (qw * qz + qx * qy);
    const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    const double yaw = std::atan2(siny_cosp, cosy_cosp);

    return Eigen::Vector3d(roll, pitch, yaw);
}

inline bool time_in_window(double source_stamp_sec,
                           double target_stamp_sec,
                           double window_sec)
{
    return source_stamp_sec >= target_stamp_sec - window_sec / 2 &&
           source_stamp_sec <= target_stamp_sec + window_sec / 2;
}

template <typename Container, typename TimeOf>
inline bool findNearestByTime(const Container &items,
                              double stamp,
                              int &index,
                              double &dt,
                              TimeOf time_of)
{
    if (items.empty())
        return false;

    auto next = std::lower_bound(
        items.begin(), items.end(), stamp,
        [&time_of](const auto &item, double t) {
            return time_of(item) < t;
        });

    if (next == items.end())
        --next;

    auto nearest = next;
    if (next != items.begin())
    {
        auto prev = std::prev(next);
        if (std::abs(stamp - time_of(*prev)) <= std::abs(stamp - time_of(*next)))
            nearest = prev;
    }

    index = std::distance(items.begin(), nearest);
    dt = std::abs(stamp - time_of(*nearest));
    return true;
}

inline double rad(double deg)
{
    return deg * M_PI / 180.0;
}

inline double clampDot(double v)
{
    return std::max(-1.0, std::min(1.0, v));
}

inline bool repairTimestamp(double raw_stamp, double expected_dt, double& stamp_offset,
                            double& last_raw_stamp, double& last_corrected_stamp,
                            double& corrected_stamp)
{
    corrected_stamp = raw_stamp + stamp_offset;

    if (last_corrected_stamp < 0.0)
    {
        last_corrected_stamp = corrected_stamp;
        last_raw_stamp = raw_stamp;
        return true;
    }

    if (raw_stamp < last_raw_stamp)
    {
        stamp_offset += last_corrected_stamp + expected_dt - corrected_stamp;
        corrected_stamp = raw_stamp + stamp_offset;
    }
    else if (stamp_offset != 0.0 && raw_stamp >= last_corrected_stamp)
    {
        stamp_offset = 0.0;
        corrected_stamp = raw_stamp;
    }

    if (corrected_stamp <= last_corrected_stamp)
    {
        return false;
    }

    last_corrected_stamp = corrected_stamp;
    last_raw_stamp = raw_stamp;
    return true;
}

inline Eigen::Vector3d transformPoint(const Eigen::Vector3d &p_old,
                                      const Eigen::Matrix3d &R_new_old,
                                      const Eigen::Vector3d &t_new_old)
{
    return R_new_old * p_old + t_new_old;
}

inline void composeMapPose(const Eigen::Matrix3d &R_odom_body,
                           const Eigen::Vector3d &t_odom_body,
                           Eigen::Matrix3d &R_map_body,
                           Eigen::Vector3d &t_map_body)
{
    Eigen::Matrix3d R_map_odom;
    Eigen::Vector3d t_map_odom;
    ros_manager.getMapOdom(R_map_odom, t_map_odom);
    R_map_body = R_map_odom * R_odom_body;
    t_map_body = R_map_odom * t_odom_body + t_map_odom;
}

inline void composeOdomPose(const Eigen::Matrix3d &R_map_body,
                            const Eigen::Vector3d &t_map_body,
                            Eigen::Matrix3d &R_odom_body,
                            Eigen::Vector3d &t_odom_body)
{
    Eigen::Matrix3d R_map_odom;
    Eigen::Vector3d t_map_odom;
    ros_manager.getMapOdom(R_map_odom, t_map_odom);
    const Eigen::Matrix3d R_odom_map = R_map_odom.transpose();
    R_odom_body = R_odom_map * R_map_body;
    t_odom_body = R_odom_map * (t_map_body - t_map_odom);
}

inline void transformPoseStamped(PoseStampedMsg &pose,
                                 const Eigen::Matrix3d &R_new_old,
                                 const Eigen::Vector3d &t_new_old)
{
    const Eigen::Vector3d t_old(
        pose.pose.position.x,
        pose.pose.position.y,
        pose.pose.position.z);
    const Eigen::Quaterniond q_old(
        pose.pose.orientation.w,
        pose.pose.orientation.x,
        pose.pose.orientation.y,
        pose.pose.orientation.z);
    const Eigen::Vector3d t_new = transformPoint(t_old, R_new_old, t_new_old);
    Eigen::Quaterniond q_new(R_new_old * q_old.toRotationMatrix());
    q_new.normalize();

    pose.pose.position.x = t_new.x();
    pose.pose.position.y = t_new.y();
    pose.pose.position.z = t_new.z();
    pose.pose.orientation.x = q_new.x();
    pose.pose.orientation.y = q_new.y();
    pose.pose.orientation.z = q_new.z();
    pose.pose.orientation.w = q_new.w();
}

inline void transformPath(PathMsg &path_msg,
                          const Eigen::Matrix3d &R_new_old,
                          const Eigen::Vector3d &t_new_old)
{
    for (auto &pose : path_msg.poses)
    {
        transformPoseStamped(pose, R_new_old, t_new_old);
    }
}

template<typename PointT>
inline bool isFinitePoint(const PointT &pt)
{
    return std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z);
}

extern std::atomic<bool> flg_exit;

inline Eigen::Vector3d standardize(const Eigen::Vector3d &v)
{
    return flip_en ? (IMU_FLIP_R * v) : v;
}

inline Eigen::Matrix3d standardize(const Eigen::Matrix3d &R)
{
    return flip_en ? (IMU_FLIP_R * R * IMU_FLIP_R.transpose()) : R;
}

template<typename PointT>
inline auto standardize(PointT &point)
    -> decltype((void) point.x, (void) point.y, (void) point.z, void())
{
    if (!flip_en)
        return;

    const Eigen::Vector3d p_std =
        standardize(Eigen::Vector3d(point.x, point.y, point.z));
    point.x = p_std.x();
    point.y = p_std.y();
    point.z = p_std.z();
}

template<typename CloudT>
inline auto standardize(CloudT &cloud)
    -> decltype((void) cloud.points, void())
{
    if (!flip_en)
        return;

    for (auto &point : cloud.points)
        standardize(point);
}

template<typename T>
inline void publishCloud(Pcl2Publisher &publisher, const T &cloud,
                         TimeType stamp, const std::string &frame)
{
    PointCloud2Msg message;
    pcl::toROSMsg(*cloud, message);
    message.header.stamp = stamp;
    message.header.frame_id = frame;
    if (ros_subscription_count(publisher) != 0)
        ros_manager.publish(publisher, message);
}

struct State
{
    double _x{0.0}, _y{0.0}, _z{0.0};
    double _qx{0.0}, _qy{0.0}, _qz{0.0}, _qw{1.0};
    double _timestamp{0.0};
    double _vx{0.0}, _vy{0.0}, _vz{0.0};
    double _bgx{0.0}, _bgy{0.0}, _bgz{0.0};
    double _bax{0.0}, _bay{0.0}, _baz{0.0};
    double _gravx{0.0}, _gravy{0.0}, _gravz{0.0};
    double _exqx{0.0}, _exqy{0.0}, _exqz{0.0}, _exqw{1.0};
    double _extx{0.0}, _exty{0.0}, _extz{0.0};

    State(double x, double y, double z, double qx, double qy, double qz,
         double qw, double timestamp)
        : _x(x), _y(y), _z(z), _qx(qx), _qy(qy), _qz(qz), _qw(qw),
          _timestamp(timestamp) {}
    State() = default;
};
