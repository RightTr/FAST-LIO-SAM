#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

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
