#include "debug/result_writer.h"

#include "common_utils.h"
#include "global_graph.h"
#include "map/ikdtree/ikdtree_public.h"

#include <pcl/io/pcd_io.h>

#include <filesystem>
#include <iomanip>
#include <iostream>

namespace
{
bool createDirectory(const std::string &path)
{
    namespace fs = std::filesystem;
    return fs::exists(path) ? fs::is_directory(path)
                            : fs::create_directories(path);
}
}

ResultWriter::ResultWriter()
    : resultDir_(std::string(ROOT_DIR) + "RESULTS/"),
      accumulatedCloudDir_(std::string(ROOT_DIR) + "PCD/"),
      scanFramesDir_(resultDir_ + "SCAN_FRAMES/"),
      imuStatesDir_(resultDir_ + "IMU_STATES/"),
      keyframeFramesDir_(resultDir_ + "KEY_FRAMES/"),
      ikdtreeOutputDir_(resultDir_ + "TREE_CLOUDS/")
{}

void ResultWriter::prepareDirectories()
{
    if (scan_frame_save_en)
    {
        if (!createDirectory(scanFramesDir_ + "scans/"))
        {
            ROS_PRINT_ERROR("Failed to create scan frame save directories, disable scan_frame_save_en");
            scan_frame_save_en = false;
        }
        else if (!createDirectory(scanFramesDir_ + "scans_tstamp/"))
        {
            ROS_PRINT_ERROR("Failed to create tstamp scan frame save directories, disable scan_frame_save_en");
            scan_frame_save_en = false;
        }
        else
        {
            scanFramePoseFile_.open(scanFramesDir_ + "scan_pose.txt",
                                    std::ios::out | std::ios::app);
            scanFramePoseFile_ << std::fixed << std::setprecision(9);
        }
    }

    if (feat_accum_save_en && !createDirectory(accumulatedCloudDir_))
    {
        ROS_PRINT_ERROR("Failed to create accumulated feature cloud directory, disable feat_accum_save_en");
        feat_accum_save_en = false;
    }

    if (imu_state_save_en)
    {
        if (!createDirectory(imuStatesDir_))
        {
            ROS_PRINT_ERROR("Failed to create IMU state save directory, disable imu_state_save_en");
            imu_state_save_en = false;
        }
        else
        {
            imuPoseFile_.open(imuStatesDir_ + "imu_state.txt", std::ios::out);
            imuPoseFile_ << std::fixed << std::setprecision(9);
            imuPoseFile_ << "# imu_states.txt columns:\n";
            imuPoseFile_ << "# timestamp[s] pos_x[m] pos_y[m] pos_z[m] qx qy qz qw "
                            "vel_x[m/s] vel_y[m/s] vel_z[m/s] "
                            "bg_x[rad/s] bg_y[rad/s] bg_z[rad/s] "
                            "ba_x[m/s^2] ba_y[m/s^2] ba_z[m/s^2] "
                            "grav_x[m/s^2] grav_y[m/s^2] grav_z[m/s^2] "
                            "ext_qx ext_qy ext_qz ext_qw ext_tx[m] ext_ty[m] ext_tz[m]\n";
        }
    }

    if (ikdtree_output_save_en && !createDirectory(ikdtreeOutputDir_))
    {
        ROS_PRINT_ERROR("Failed to create ikdtree cloud save directory, disable ikdtree_output_save_en");
        ikdtree_output_save_en = false;
    }

    if (keyframe_export_en || keyframe_global_pcd_en)
    {
        if (!createDirectory(keyframeFramesDir_ + "scans/"))
        {
            ROS_PRINT_ERROR("Failed to create keyframe save directories, disable keyframe export");
            keyframe_export_en = false;
            keyframe_global_pcd_en = false;
        }
        else if (keyframe_export_en)
        {
            keyframePoseFile_.open(keyframeFramesDir_ + "keyframe_pose.txt",
                                   std::ios::out | std::ios::app);
            keyframePoseFile_ << std::fixed << std::setprecision(9);
        }
    }
}

void ResultWriter::accumulateFeatureCloud(const PointCloudXYZI &cloud,
                                          double timestamp,
                                          int saveInterval)
{
    accumulatedCloud_ += cloud;
    ++scansSinceSave_;
    if (accumulatedCloud_.empty() || saveInterval <= 0 ||
        scansSinceSave_ < saveInterval)
        return;

    saveAccumulatedCloud(accumulatedCloud_, timestamp);
    accumulatedCloud_.clear();
    scansSinceSave_ = 0;
}

void ResultWriter::flushAccumulatedCloud(double timestamp)
{
    if (!accumulatedCloud_.empty())
        saveAccumulatedCloud(accumulatedCloud_, timestamp);
    accumulatedCloud_.clear();
    scansSinceSave_ = 0;
}

void ResultWriter::saveAccumulatedCloud(const PointCloudXYZI &cloud,
                                       double timestamp)
{
    if (cloud.empty())
        return;

    const std::string fileName = format_unix_time(timestamp) + ".pcd";
    const std::string path = accumulatedCloudDir_ + fileName;
    pcl::PCDWriter writer;
    std::cout << "current accumulated feature cloud saved to /PCD/"
              << fileName << std::endl;
    writer.writeBinary(path, cloud);
}

void ResultWriter::saveKeyframePoses(const std::vector<PointTypePose> &poses)
{
    if (!keyframePoseFile_.is_open())
        return;

    for (const auto &pose : poses)
    {
        keyframePoseFile_ << pose.time << " " << pose.x << " " << pose.y << " " << pose.z << " "
                          << pose.roll << " " << pose.pitch << " " << pose.yaw << "\n";
    }
    keyframePoseFile_.close();
}

bool ResultWriter::saveGlobalCloud(GlobalGraph *graph)
{
    const std::string path = keyframeFramesDir_ + "global.pcd";
    std::cout << "current global keyframe map saved to " << path << std::endl;
    if (graph && graph->SaveGlobalCloud(path))
        return true;

    ROS_PRINT_WARN("failed to save global keyframe map to %s", path.c_str());
    return false;
}

bool ResultWriter::saveTree(KD_TREE_PUBLIC<PointType> &tree)
{
    if (tree.Root_Node == nullptr)
        return false;

    PointVector points;
    tree.flatten(tree.Root_Node, points, NOT_RECORD);
    if (points.empty())
        return false;

    PointCloudXYZI cloud;
    cloud.points = std::move(points);
    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    pcl::PCDWriter writer;
    const std::string cloudPath = ikdtreeOutputDir_ + "prior_cloud.pcd";
    const std::string treePath = ikdtreeOutputDir_ + "prior_tree.bin";
    writer.writeBinary(cloudPath, cloud);

    if (tree.SaveIkdtree(treePath))
    {
        ROS_PRINT_INFO("saved final ikdtree to %s", treePath.c_str());
        return true;
    }

    ROS_PRINT_WARN("failed to save final ikdtree to %s", treePath.c_str());
    return false;
}

void ResultWriter::saveKeyframeCloud(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, double timestamp)
{
    pcl::PointCloud<pcl::PointXYZINormal> exportCloud;
    if (cloud)
    {
        for (const auto &point : cloud->points)
        {
            pcl::PointXYZINormal exported;
            exported.x = point.x;
            exported.y = point.y;
            exported.z = point.z;
            exported.intensity = point.intensity;
            exportCloud.push_back(exported);
        }
    }

    const std::string stamp = format_unix_time(timestamp);
    pcl::PCDWriter writer;
    writer.writeBinary(keyframeFramesDir_ + "scans/" + stamp + ".pcd", exportCloud);
}

void ResultWriter::saveScanFrame(const PointCloudXYZI &cloud, double timestamp,
                                const Eigen::Matrix3d &R_world_imu,
                                const Eigen::Vector3d &p_world_imu,
                                const Eigen::Matrix3d &R_imu_lidar,
                                const Eigen::Vector3d &t_imu_lidar)
{
    const std::string stamp = format_unix_time(timestamp);
    pcl::PointCloud<pcl::PointXYZ> scan_xyz;
    pcl::PointCloud<pcl::PointXYZI> scan_xyz_tstamp;
    scan_xyz.reserve(cloud.size());
    scan_xyz_tstamp.reserve(cloud.size());

    for (const auto &point : cloud.points)
    {
        pcl::PointXYZ xyz;
        xyz.x = point.x;
        xyz.y = point.y;
        xyz.z = point.z;
        scan_xyz.push_back(xyz);

        pcl::PointXYZI xyz_tstamp;
        xyz_tstamp.x = point.x;
        xyz_tstamp.y = point.y;
        xyz_tstamp.z = point.z;
        xyz_tstamp.intensity = point.curvature;
        scan_xyz_tstamp.push_back(xyz_tstamp);
    }

    pcl::PCDWriter writer;
    writer.writeBinary(scanFramesDir_ + "scans/" + stamp + ".pcd", scan_xyz);
    writer.writeBinary(scanFramesDir_ + "scans_tstamp/" + stamp + ".pcd",
                       scan_xyz_tstamp);

    const Eigen::Vector3d p_world_lidar =
        p_world_imu + R_world_imu * t_imu_lidar;
    const Eigen::Quaterniond q_world_lidar(R_world_imu * R_imu_lidar);
    scanFramePoseFile_ << timestamp << " "
        << p_world_lidar.x() << " " << p_world_lidar.y() << " "
        << p_world_lidar.z() << " " << q_world_lidar.coeffs()[0] << " "
        << q_world_lidar.coeffs()[1] << " " << q_world_lidar.coeffs()[2]
        << " " << q_world_lidar.coeffs()[3] << "\n";
}

void ResultWriter::saveImuPose(const State &state)
{
    if (!imu_state_save_en || !imuPoseFile_.is_open())
        return;

    imuPoseFile_ << state._timestamp << " "
        << state._x << " " << state._y << " " << state._z << " "
        << state._qx << " " << state._qy << " " << state._qz << " "
        << state._qw << " " << state._vx << " " << state._vy << " "
        << state._vz << " " << state._bgx << " " << state._bgy << " "
        << state._bgz << " " << state._bax << " " << state._bay << " "
        << state._baz << " " << state._gravx << " " << state._gravy << " "
        << state._gravz << " " << state._exqx << " " << state._exqy << " "
        << state._exqz << " " << state._exqw << " " << state._extx << " "
        << state._exty << " " << state._extz << "\n";
    imuPoseFile_.flush();
}
