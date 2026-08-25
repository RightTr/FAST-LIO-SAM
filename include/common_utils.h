#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <string>

#include "map_optimization.h"
#include <pcl/common/transforms.h>

#include "utility.h"

namespace fs = std::filesystem;

inline bool create_directory(const std::string& path)
{
    if (fs::exists(path)) {
        return fs::is_directory(path);
    }
    return fs::create_directories(path);
}

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

inline bool time_in_window(double source_stamp_sec,
                           double target_stamp_sec,
                           double window_sec)
{
    return source_stamp_sec >= target_stamp_sec - window_sec / 2 &&
           source_stamp_sec <= target_stamp_sec + window_sec / 2;
}

inline double rad(double deg)
{
    return deg * M_PI / 180.0;
}

inline double clampDot(double v)
{
    return std::max(-1.0, std::min(1.0, v));
}

inline bool repairTimestamp(const double raw_ts, const double expected_dt,
    double &off, double &last_raw_ts,
    double &last_ts, double &cur_ts)
{
    cur_ts = raw_ts + off;

    if (last_ts < 0.0)
    {
        last_ts = cur_ts;
        last_raw_ts = raw_ts;
        return true;
    }

    if (raw_ts < last_raw_ts)
    {
        double rollback = last_raw_ts - raw_ts;

        double correction =
            (last_ts + expected_dt) - cur_ts;

        off += correction;
        cur_ts = raw_ts + off;
    }
    else if (off != 0.0 && raw_ts >= last_ts)
    {
        off = 0.0;
        cur_ts = raw_ts;
    }

    if (cur_ts <= last_ts)
    {
        return false;
    }

    last_ts = cur_ts;
    last_raw_ts = raw_ts;
    return true;
}

inline Eigen::Vector3d transformPoint(const Eigen::Vector3d &p_old,
                                      const Eigen::Matrix3d &R_new_old,
                                      const Eigen::Vector3d &t_new_old)
{
    return R_new_old * p_old + t_new_old;
}

extern Eigen::Matrix3d R_map_odom;
extern Eigen::Vector3d t_map_odom;
extern Eigen::Matrix3d R_enu_map;
extern Eigen::Vector3d t_enu_map;

inline void composeMapPose(const Eigen::Matrix3d &R_odom_body,
                           const Eigen::Vector3d &t_odom_body,
                           Eigen::Matrix3d &R_map_body,
                           Eigen::Vector3d &t_map_body)
{
    R_map_body = R_map_odom * R_odom_body;
    t_map_body = R_map_odom * t_odom_body + t_map_odom;
}

inline void composeOdomPose(const Eigen::Matrix3d &R_map_body,
                            const Eigen::Vector3d &t_map_body,
                            Eigen::Matrix3d &R_odom_body,
                            Eigen::Vector3d &t_odom_body)
{
    const Eigen::Matrix3d R_odom_map = R_map_odom.transpose();
    R_odom_body = R_odom_map * R_map_body;
    t_odom_body = R_odom_map * (t_map_body - t_map_odom);
}

void setMapOdom(const Eigen::Matrix3d &R_map_odom_,
                const Eigen::Vector3d &t_map_odom_);
void publishEnuToMapTf(const TimeType& stamp);
void publishMapToOdomTf(const TimeType& stamp);

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

inline std::string result_dir = std::string(ROOT_DIR) + "RESULTS/";
inline std::string scan_frames_dir = result_dir + "/SCAN_FRAMES/";
inline std::string imu_states_dir = result_dir + "/IMU_STATES/";
inline std::string keyframe_frames_dir = result_dir + "/KEY_FRAMES/";
inline std::string ikdtree_output_dir = result_dir + "/TREE_CLOUDS/";

extern std::ofstream scan_frame_pose_file;
extern std::ofstream imu_pose_file;
extern std::ofstream keyframe_pose_file;

inline void prepareResultDirs()
{
    if (scan_frame_save_en)
    {
        if (!create_directory(scan_frames_dir + "scans/"))
        {
            ROS_PRINT_ERROR("Failed to create scan frame save directories, disable scan_frame_save_en");
            scan_frame_save_en = false;
        }
        else if (!create_directory(scan_frames_dir + "scans_tstamp/"))
        {
            ROS_PRINT_ERROR("Failed to create tstamp scan frame save directories, disable scan_frame_save_en");
            scan_frame_save_en = false;
        }
        else
        {
            const std::string scan_frame_pose_path = scan_frames_dir + "scan_pose.txt";
            scan_frame_pose_file.open(scan_frame_pose_path.c_str(), std::ios::out | std::ios::app);
            scan_frame_pose_file << std::fixed << std::setprecision(9);
        }
    }

    if (feat_accum_save_en)
    {
        if (!create_directory(result_dir + "/PCD/"))
        {
            ROS_PRINT_ERROR("Failed to create accumulated feature cloud directory, disable feat_accum_save_en");
            feat_accum_save_en = false;
        }
    }

    if (imu_state_save_en)
    {
        if (!create_directory(imu_states_dir))
        {
            ROS_PRINT_ERROR("Failed to create IMU state save directory, disable imu_state_save_en");
            imu_state_save_en = false;
        }
        else
        {
            const std::string imu_state_path = imu_states_dir + "imu_state.txt";
            imu_pose_file.open(imu_state_path.c_str(), std::ios::out);
            imu_pose_file << std::fixed << std::setprecision(9);
            imu_pose_file << "# imu_states.txt columns:\n";
            imu_pose_file << "# timestamp[s] pos_x[m] pos_y[m] pos_z[m] qx qy qz qw "
                             "vel_x[m/s] vel_y[m/s] vel_z[m/s] "
                             "bg_x[rad/s] bg_y[rad/s] bg_z[rad/s] "
                             "ba_x[m/s^2] ba_y[m/s^2] ba_z[m/s^2] "
                             "grav_x[m/s^2] grav_y[m/s^2] grav_z[m/s^2] "
                             "ext_qx ext_qy ext_qz ext_qw ext_tx[m] ext_ty[m] ext_tz[m]\n";
        }
    }

    if (ikdtree_output_save_en)
    {
        if (!create_directory(ikdtree_output_dir))
        {
            ROS_PRINT_ERROR("Failed to create ikdtree cloud save directory, disable ikdtree_output_save_en");
            ikdtree_output_save_en = false;
        }
    }

    if (keyframe_export_en || keyframe_global_pcd_en)
    {
        if (!create_directory(keyframe_frames_dir + "scans/"))
        {
            ROS_PRINT_ERROR("Failed to create keyframe save directories, disable keyframe export");
            keyframe_export_en = false;
            keyframe_global_pcd_en = false;
        }
        else if (keyframe_export_en)
        {
            const std::string keyframe_pose_path = keyframe_frames_dir + "keyframe_pose.txt";
            keyframe_pose_file.open(keyframe_pose_path.c_str(), std::ios::out | std::ios::app);
            keyframe_pose_file << std::fixed << std::setprecision(9);
        }
    }
}
