#include "debug/result_writer.h"

#include "common_utils.h"
#include "global_graph.h"
#include "ikd-Tree/ikdtree_public.h"

#include <pcl/io/pcd_io.h>

#include <iostream>

ResultWriter::ResultWriter()
    : accumulatedCloudDir_(std::string(ROOT_DIR) + "PCD/"),
      keyframeScanDir_(keyframe_frames_dir + "scans/"),
      globalKeyframePath_(keyframe_frames_dir + "global.pcd"),
      ikdtreeCloudPath_(ikdtree_output_dir + "prior_cloud.pcd"),
      ikdtreePath_(ikdtree_output_dir + "prior_tree.bin")
{}

ResultWriter &ResultWriter::instance()
{
    static ResultWriter writer;
    return writer;
}

void ResultWriter::saveAccumulatedCloud(const PointCloudXYZI &cloud,
                                        double timestamp) const
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

void ResultWriter::saveKeyframePoses(const std::vector<PointTypePose> &poses,
                                     std::ofstream &file) const
{
    if (!file.is_open())
        return;

    for (const auto &pose : poses)
    {
        file << pose.time << " " << pose.x << " " << pose.y << " " << pose.z << " "
             << pose.roll << " " << pose.pitch << " " << pose.yaw << "\n";
    }
    file.close();
}

bool ResultWriter::saveGlobalCloud(GlobalGraph *graph) const
{
    std::cout << "current global keyframe map saved to " << globalKeyframePath_ << std::endl;
    if (graph && graph->SaveGlobalCloud(globalKeyframePath_))
        return true;

    ROS_PRINT_WARN("failed to save global keyframe map to %s", globalKeyframePath_.c_str());
    return false;
}

bool ResultWriter::saveTree(KD_TREE_PUBLIC<PointType> &tree) const
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
    writer.writeBinary(ikdtreeCloudPath_, cloud);

    if (tree.SaveIkdtree(ikdtreePath_))
    {
        ROS_PRINT_INFO("saved final ikdtree to %s", ikdtreePath_.c_str());
        return true;
    }

    ROS_PRINT_WARN("failed to save final ikdtree to %s", ikdtreePath_.c_str());
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
    writer.writeBinary(keyframeScanDir_ + stamp + ".pcd", exportCloud);
}
