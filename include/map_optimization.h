#ifndef MAP_OPTIMIZATION_H
#define MAP_OPTIMIZATION_H

#include <vector>

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

typedef pcl::PointXYZI PointTypeIndex;

extern float transformTobeMapped[6];
extern Eigen::Vector3d translationLidarToIMU;
extern Eigen::Matrix3d rotationLidarToIMU;

extern std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> featCloudKeyFrames;
extern pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

void MapOptimizationInit();

void saveKeyFramesAndFactor(pcl::PointCloud<pcl::PointXYZINormal>::Ptr feats_undistort);

void correctPoses();

void publishSamMsg();

void loopClosureThread();

void setLaserCurTime(double lidar_end_time);

void visualizeGlobalMapThread();

pcl::PointCloud<PointTypeIndex>::Ptr transformPointCloud(pcl::PointCloud<PointTypeIndex>::Ptr cloudIn, PointTypePose* transformIn);

#endif