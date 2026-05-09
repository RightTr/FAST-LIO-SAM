#ifndef UTILITY_H
#define UTILITY_H

#include <string>
#include <deque>
#include <vector>
#include <atomic>
#include <type_traits>

#include <Eigen/Geometry>
#include <pcl_conversions/pcl_conversions.h>
#include "ros_utils.h"

// CPU Params
extern int numberOfCores;

// Surrounding map
extern float surroundingkeyframeAddingDistThreshold; 
extern float surroundingkeyframeAddingAngleThreshold; 
extern float surroundingKeyframeDensity;
extern float surroundingKeyframeSearchRadius;

// Loop closure
extern bool  loopClosureEnableFlag;
extern float loopClosureFrequency;
extern int   surroundingKeyframeSize;
extern float historyKeyframeSearchRadius;
extern float historyKeyframeSearchTimeDiff;
extern int   historyKeyframeSearchNum;
extern float historyKeyframeFitnessScore;

// global map visualization radius
extern float globalMapVisualizationSearchRadius;
extern float globalMapVisualizationPoseDensity;
extern float globalMapVisualizationLeafSize;

extern float mappingICPSize;

extern int ikdtreeSearchNeighborNum;

extern bool keyframe_export_en;
extern bool keyframe_global_pcd_en;

extern std::string map_frame;
extern std::string odom_frame;
extern std::string base_frame;
extern std::string high_freq_base_frame;
extern bool flip_en;
extern const Eigen::Matrix3d IMU_FLIP_R;

extern std::atomic<bool> flg_exit;

void read_liosam_params();

inline Eigen::Vector3d standardize(const Eigen::Vector3d &v)
{
    return flip_en ? (IMU_FLIP_R * v) : v;
}

inline Eigen::Matrix3d standardize(const Eigen::Matrix3d &R)
{
    return flip_en ? (IMU_FLIP_R * R * IMU_FLIP_R.transpose()) : R;
}

template<typename PointT>
inline auto standardize(PointT &point) -> decltype((void) point.x, (void) point.y, (void) point.z, void())
{
    if (!flip_en)
        return;

    const Eigen::Vector3d p_std = standardize(Eigen::Vector3d(point.x, point.y, point.z));
    point.x = p_std.x();
    point.y = p_std.y();
    point.z = p_std.z();
}

template<typename CloudT>
inline auto standardize(CloudT &cloud) -> decltype((void) cloud.points, void())
{
    if (!flip_en)
        return;

    for (auto &point : cloud.points)
    {
        standardize(point);
    }
}

template<typename T>
void publishCloud(Pcl2Publisher &thisPub, const T &thisCloud, TimeType thisStamp, const std::string &thisFrame)
{
    PointCloud2Msg tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (ros_subscription_count(thisPub) != 0)
        ros_publish(thisPub, tempCloud);
}

#endif
