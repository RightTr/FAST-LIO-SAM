#ifndef RESULT_WRITER_H
#define RESULT_WRITER_H

#include "common_lib.h"
#include "global_graph.h"

#include <fstream>
#include <string>
#include <vector>

template<typename PointT>
class KD_TREE_PUBLIC;
struct State;

class ResultWriter
{
public:
    ResultWriter();
    void prepareDirectories();
    void accumulateFeatureCloud(const PointCloudXYZI &cloud, double timestamp,
                                int saveInterval);
    void flushAccumulatedCloud(double timestamp);
    void saveAccumulatedCloud(const PointCloudXYZI &cloud, double timestamp);
    void saveKeyframePoses(const std::vector<PointTypePose> &poses);
    bool saveGlobalCloud(GlobalGraph *graph);
    bool saveTree(KD_TREE_PUBLIC<PointType> &tree);
    void saveKeyframeCloud(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, double timestamp);
    void saveImuPose(const State &state);
    void saveScanFrame(const PointCloudXYZI &cloud, double timestamp,
                       const Eigen::Matrix3d &R_world_imu,
                       const Eigen::Vector3d &p_world_imu,
                       const Eigen::Matrix3d &R_imu_lidar,
                       const Eigen::Vector3d &t_imu_lidar);

private:
    std::string resultDir_;
    std::string accumulatedCloudDir_;
    std::string scanFramesDir_;
    std::string imuStatesDir_;
    std::string keyframeFramesDir_;
    std::string ikdtreeOutputDir_;
    std::ofstream scanFramePoseFile_;
    std::ofstream imuPoseFile_;
    std::ofstream keyframePoseFile_;
    PointCloudXYZI accumulatedCloud_;
    int scansSinceSave_ = 0;
};

#endif
