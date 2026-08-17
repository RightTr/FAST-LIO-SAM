#ifndef MAP_OPTIMIZATION_H
#define MAP_OPTIMIZATION_H

#include <vector>
#include <mutex>
#include <Eigen/Core>
#include <pcl/common/transforms.h>

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

inline Eigen::Vector3d poseTranslation(const PointTypePose &pose)
{
    return Eigen::Vector3d(pose.x, pose.y, pose.z);
}

inline Eigen::Matrix3d poseRotation(const PointTypePose &pose)
{
    return pcl::getTransformation(0.0, 0.0, 0.0, pose.roll, pose.pitch, pose.yaw).rotation().cast<double>();
}

extern float transformTobeMapped[6];
extern Eigen::Vector3d translationLidarToIMU;
extern Eigen::Matrix3d rotationLidarToIMU;

extern std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> featCloudKeyFrames;
extern pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

void MapOptimizationInit();

void setGravityUp(const Eigen::Vector3d &gravity_up);
Eigen::Vector3d getGravityUp();

extern std::recursive_mutex mtxLoop;

void saveKeyFramesAndFactor(pcl::PointCloud<pcl::PointXYZINormal>::Ptr feats_undistort);
void addLoopFactor();
void poseGraphUpdate();

bool isKeyFrame();

void correctPoses();

void publishSamMsg();
void visualizeLoopClosure();
void publishGlobalMap();

void shutdownMapOptimization();

void loopClosureThread();

void gnssMatchingThread();

void structureMatchingThread();

void setLaserCurTime(double lidar_end_time);

void visualizeGlobalMapThread();

pcl::PointCloud<PointTypeIndex>::Ptr transformPointCloud(pcl::PointCloud<PointTypeIndex>::Ptr cloudIn, const PointTypePose *transformIn);

#endif
