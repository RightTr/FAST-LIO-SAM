// Modified from LIO-SAM: MapOptimization.cpp

#include "utility.h"
#include "common_utils.h"
#include "GNSS_Processing.hpp"
#include "gnssYaw_factor.h"
#include "s-graph/factor/floor_factor.h"
#include "s-graph/plane.h"
#include "s-graph/floorMap.h"
#include "map_optimization.h"
#include "ros_utils.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <limits>
#include <deque>
#include <tuple>
#include <utility>
#include <unordered_set>
#include <atomic>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/OrientedPlane3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/OrientedPlane3Factor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace gtsam;
using namespace std;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

// gtsam
NonlinearFactorGraph gtSAMgraph;
Values initialEstimate;
ISAM2 *isam;
Values isamCurrentEstimate;
Eigen::MatrixXd poseCovariance;

Pcl2Publisher pubKeyPoses;
PathPublisher pubPath;

Pcl2Publisher pubLaserCloudGlobal;

Pcl2Publisher pubRecentKeyFrame;
MarkerArrayPublisher pubLoopConstraintEdge;
MarkerArrayPublisher pubKeyFrameYawMarkers;
#ifdef USE_ROS1
std::shared_ptr<tf::TransformBroadcaster> samTfBroadcaster;
#elif defined(USE_ROS2)
std::shared_ptr<tf2_ros::TransformBroadcaster> samTfBroadcaster;
#endif

TimeType timeLaserInfoStamp;

pcl::PointCloud<PointTypeIndex>::Ptr cloudKeyPoses3D; // Store keyframe poses and indexes 
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
pcl::PointCloud<PointTypePose>::Ptr cloudKeyOdomPoses6D;

double timeLaserInfoCur;

float transformTobeMapped[6];
Eigen::Vector3d translationLidarToIMU;
Eigen::Matrix3d rotationLidarToIMU;

PathMsg globalPath;
Plane plane;
FloorMap floorMap;

gtsam::noiseModel::Diagonal::shared_ptr planeNoise;
gtsam::SharedNoiseModel floorNormalPriorNoise;
gtsam::SharedNoiseModel floorPlaneNoise;
std::mutex gravityAxisMutex;
Eigen::Vector3d gravityUpAxis = Eigen::Vector3d::UnitZ();

std::mutex mtxKeyframe;
std::mutex mtxLoopFactor;
std::mutex mtxGnssFactor;
std::mutex mtxSceneBatch;
std::atomic<int> floorKey{-1};
std::atomic<int> loopKey{-1};
std::atomic<bool> sceneDone{false};
std::atomic<bool> loopDone{false};

bool graphUpdate = false;
bool loopIsClosed = false;
std::deque<SceneBatch> sceneQueue;
std::unordered_set<int> loopUsedKeys;
std::vector<std::pair<int, int>> loopEdges;
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;

std::deque<std::tuple<int, Eigen::Vector3d, Eigen::Matrix3d>> gnssPosFactorQueue;
std::deque<std::pair<int, double>> gnssYawFactorQueue;
double last_pos_t = -1.0;
double last_yaw_t = -1.0;

std::unordered_set<int> pos_keys;
std::unordered_set<int> yaw_keys;

void correctPoses();
void addSceneFactor();
void performSceneMatching();
void sceneMatchingThread();

void setGravityUp(const Eigen::Vector3d &gravity_up)
{
    std::lock_guard<std::mutex> lock(gravityAxisMutex);
    gravityUpAxis = gravity_up.normalized();
}

Eigen::Vector3d getGravityUp()
{
    std::lock_guard<std::mutex> lock(gravityAxisMutex);
    return gravityUpAxis.normalized();
}

pcl::VoxelGrid<PointTypeIndex> downSizeFilterICP;

vector<pcl::PointCloud<PointTypeIndex>::Ptr> featCloudKeyFrames;

Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
{ 
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
}

gtsam::Pose3 trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

float pointDistance(PointTypeIndex p1, PointTypeIndex p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

PointTypePose trans2PointTypePose(float transformIn[])
{
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll  = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw   = transformIn[2];
    return thisPose6D;
}

void setLaserCurTime(double lidar_end_time)
{
    timeLaserInfoCur = lidar_end_time;
}

pcl::PointCloud<PointTypeIndex>::Ptr transformPointCloud(pcl::PointCloud<PointTypeIndex>::Ptr cloudIn, const PointTypePose *transformIn)
{
    pcl::PointCloud<PointTypeIndex>::Ptr cloudOut(new pcl::PointCloud<PointTypeIndex>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}

void allocateMemory()
{
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointTypeIndex>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    cloudKeyOdomPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    for (int i = 0; i < 6; ++i){
        transformTobeMapped[i] = 0;
    }
}

void MapOptimizationInit()
{
    sceneDone.store(!groundEnableFlag && !floorEnableFlag);
    loopDone.store(!loopClosureEnableFlag);

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);

    gtsam::Vector3 plane_sigmas;
    plane_sigmas << (M_PI / 180.0) * 3.0, (M_PI / 180.0) * 3.0, 0.08;
    planeNoise = gtsam::noiseModel::Diagonal::Sigmas(plane_sigmas);
    floorNormalPriorNoise = gtsam::noiseModel::Isotropic::Sigma(
        2, (M_PI / 180.0) * 0.2);
    floorPlaneNoise = gtsam::noiseModel::Isotropic::Sigma(
        2, (M_PI / 180.0) * 0.5);
    setGravityUp(Eigen::Vector3d::UnitZ());

    init_ros_node();
    
    pubKeyPoses = create_publisher<PointCloud2Msg>("lio_sam/trajectory", 1);
    pubPath = create_publisher<PathMsg>("lio_sam/mapping/path", 1);
    pubLaserCloudGlobal = create_publisher<PointCloud2Msg>("lio_sam/mapping/cloud_global", 1);
    pubRecentKeyFrame = create_publisher<PointCloud2Msg>("lio_sam/mapping/cloud_recent_keyframe", 1);
    pubLoopConstraintEdge = create_publisher<MarkerArrayMsg>("lio_sam/loop_closure_constraints", 1);
    pubKeyFrameYawMarkers = create_publisher<MarkerArrayMsg>("lio_sam/mapping/keyframe_yaw", 1);
#ifdef USE_ROS1
    samTfBroadcaster = std::make_shared<tf::TransformBroadcaster>();
#elif defined(USE_ROS2)
    samTfBroadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(get_ros_node());
#endif

    downSizeFilterICP.setLeafSize(mappingICPSize, mappingICPSize, mappingICPSize);

    allocateMemory();
}

bool isKeyFrame()
{
    if (cloudKeyOdomPoses6D->points.empty())
        return true;

    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyOdomPoses6D->back());
    Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

    if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
        abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
        return false;
    
    return true;
}  

int findLoopKey(const std::vector<PointTypeIndex> &poses3D,
                const std::vector<PointTypePose> &poses6D,
                pcl::KdTreeFLANN<PointTypeIndex> &poseTree,
                int loopKeyCur)
{
    double current_time = 0.0;
    if (poses3D.empty() ||
        loopKeyCur < 0)
        return -1;

    if (loopKeyCur >= static_cast<int>(poses3D.size()) ||
        loopKeyCur >= static_cast<int>(poses6D.size()))
        return -1;

    if (loopUsedKeys.count(loopKeyCur) != 0)
        return -1;

    current_time = poses6D[loopKeyCur].time;
    std::vector<int> indices;
    std::vector<float> sqDists;
    if (poseTree.radiusSearch(
            poses3D[loopKeyCur],
            historyKeyframeSearchRadius,
            indices,
            sqDists) == 0)
        return -1;

    int loopKeyPre = -1;
    float nearestSqDis = std::numeric_limits<float>::max();
    for (size_t i = 0; i < indices.size(); ++i)
    {
        const int id = indices[i];
        if (id < 0 || id >= loopKeyCur)
            continue;
        if (loopUsedKeys.count(id) != 0)
            continue;

        const double nearTime = poses6D[id].time;
        if (current_time - nearTime <= historyKeyframeSearchTimeDiff)
            continue;

        if (floorEnableFlag && !floorMap.sameFloor(loopKeyCur, id))
            continue;

        const float sqDis = sqDists[i];
        if (sqDis < nearestSqDis)
        {
            nearestSqDis = sqDis;
            loopKeyPre = id;
        }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return -1;

    return loopKeyPre;
}

void performLoopClosure(int loopKeyCur,
                        const std::vector<PointTypeIndex> &poses3D,
                        const std::vector<PointTypePose> &poses6D,
                        const std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> &clouds,
                        pcl::KdTreeFLANN<PointTypeIndex> &poseTree)
{
    if (poses3D.empty())
        return;

    const int loopKeyPre = findLoopKey(poses3D, poses6D, poseTree, loopKeyCur);
    if (loopKeyPre < 0)
        return;

    // extract cloud
    pcl::PointCloud<PointTypeIndex>::Ptr curCloud;
    PointTypePose curPose;
    const int cloudSize = static_cast<int>(std::min(clouds.size(), poses6D.size()));
    if (loopKeyCur >= cloudSize ||
        loopKeyPre >= cloudSize)
        return;

    curCloud = clouds[loopKeyCur];
    curPose = poses6D[loopKeyCur];

    FloorRange loopPreRange;
    int historyBegin = std::max(0, loopKeyPre - historyKeyframeSearchNum);
    int historyEnd = std::min(loopKeyPre + historyKeyframeSearchNum, cloudSize - 1);

    if (floorEnableFlag)
    {
        if (!floorMap.getFloorRange(loopKeyPre, loopPreRange))
        {
            ROS_PRINT_INFO(
                "[LOOP] no floor range pre=%d cur=%d",
                loopKeyPre,
                loopKeyCur);
            return;
        }

        historyBegin = std::max(loopKeyPre - historyKeyframeSearchNum, loopPreRange.begin);
        const int rangeEnd = loopPreRange.end >= 0 ? loopPreRange.end : cloudSize - 1;
        historyEnd = std::min(loopKeyPre + historyKeyframeSearchNum, rangeEnd);
    }

    if (!curCloud || curCloud->empty())
        return;

    pcl::PointCloud<PointTypeIndex>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointTypeIndex>());
    *cureKeyframeCloud += *transformPointCloud(curCloud, &curPose);
    for (int keyNear = historyBegin; keyNear <= historyEnd; ++keyNear)
    {
        *prevKeyframeCloud += *transformPointCloud(clouds[keyNear], &poses6D[keyNear]);
    }

    pcl::PointCloud<PointTypeIndex>::Ptr cureKeyframeCloudDS(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr prevKeyframeCloudDS(new pcl::PointCloud<PointTypeIndex>());
    downSizeFilterICP.setInputCloud(cureKeyframeCloud);
    downSizeFilterICP.filter(*cureKeyframeCloudDS);
    downSizeFilterICP.setInputCloud(prevKeyframeCloud);
    downSizeFilterICP.filter(*prevKeyframeCloudDS);

    if (cureKeyframeCloudDS->size() < 300 ||
        prevKeyframeCloudDS->size() < 1000)
        return;

    ROS_PRINT_INFO(
        "[LOOP] ICP start cur=%d pre=%d src=%zu target=%zu",
        loopKeyCur,
        loopKeyPre,
        cureKeyframeCloudDS->size(),
        prevKeyframeCloudDS->size());

    // ICP Settings
    static pcl::IterativeClosestPoint<PointTypeIndex, PointTypeIndex> icp;
    icp.setMaxCorrespondenceDistance(2.0);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(cureKeyframeCloudDS);
    icp.setInputTarget(prevKeyframeCloudDS);
    pcl::PointCloud<PointTypeIndex>::Ptr unused_result(new pcl::PointCloud<PointTypeIndex>());
    icp.align(*unused_result);

    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
    {
        ROS_PRINT_INFO(
            "[LOOP] ICP reject cur=%d pre=%d score=%.3f",
            loopKeyCur,
            loopKeyPre,
            icp.getFitnessScore());
        return;
    }

    // publish corrected cloud
    // if (pubIcpKeyFrames.getNumSubscribers() != 0)
    // {
    //     pcl::PointCloud<PointTypeIndex>::Ptr closed_cloud(new pcl::PointCloud<PointTypeIndex>());
    //     pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
    //     publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    // }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    // transform from world origin to wrong pose
    Eigen::Affine3f tWrong = pclPointToAffine3f(curPose);
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    PointTypePose prevPose;
    if (loopKeyPre < 0 ||
        loopKeyPre >= static_cast<int>(poses6D.size()))
        return;
    prevPose = poses6D[loopKeyPre];
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(prevPose);
    gtsam::Vector Vector6(6);
    const double noiseScore = std::max(
        static_cast<double>(icp.getFitnessScore()) * loopWeight, 1e-4);
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint and reserve both endpoints atomically.
    {
        std::lock_guard<std::mutex> lock(mtxLoopFactor);
        loopIndexQueue.emplace_back(loopKeyCur, loopKeyPre);
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);

        loopUsedKeys.insert(loopKeyCur);
        loopUsedKeys.insert(loopKeyPre);
    }

    ROS_PRINT_INFO(
        "[LOOP] ICP pass cur=%d pre=%d score=%.3f",
        loopKeyCur,
        loopKeyPre,
        icp.getFitnessScore());
}

void addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        noiseModel::Diagonal::shared_ptr priorNoise = // Set fixed pose0
            noiseModel::Diagonal::Variances(
                (Vector(6) <<
                1e-4, 1e-4, 1e-4,
                1e-4, 1e-4, 1e-4).finished());
        const gtsam::Pose3 poseTo = [&]() {
            const gtsam::Pose3 poseOdom = trans2gtsamPose(transformTobeMapped);
            const Eigen::Matrix3d R_map_body = R_map_odom * poseOdom.rotation().matrix();
            const Eigen::Vector3d t_map_body = R_map_odom *
                Eigen::Vector3d(poseOdom.translation().x(),
                                poseOdom.translation().y(),
                                poseOdom.translation().z()) + t_map_odom;
            return gtsam::Pose3(gtsam::Rot3(R_map_body),
                                gtsam::Point3(t_map_body.x(), t_map_body.y(), t_map_body.z()));
        }();
        gtSAMgraph.add(PriorFactor<Pose3>(0, poseTo, priorNoise));
        initialEstimate.insert(0, poseTo);
    }else{
        noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        const gtsam::Pose3 poseFromFront = pclPointTogtsamPose3(cloudKeyOdomPoses6D->points.back());
        const gtsam::Pose3 poseToFront = trans2gtsamPose(transformTobeMapped);
        const gtsam::Pose3 odomDelta = poseFromFront.between(poseToFront);
        const gtsam::Pose3 poseFromOpt = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
        const gtsam::Pose3 poseToOpt = poseFromOpt.compose(odomDelta);
        gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), odomDelta, odometryNoise));
        initialEstimate.insert(cloudKeyPoses3D->size(), poseToOpt);
    }
}

void addLoopFactor()
{
    if (!loopClosureEnableFlag)
        return;

    std::vector<std::pair<int, int>> indexQueue;
    std::vector<gtsam::Pose3> poseQueue;
    std::vector<gtsam::noiseModel::Diagonal::shared_ptr> noiseQueue;

    {
        std::lock_guard<std::mutex> lock(mtxLoopFactor);
        if (loopIndexQueue.empty())
            return;

        indexQueue.swap(loopIndexQueue);
        poseQueue.swap(loopPoseQueue);
        noiseQueue.swap(loopNoiseQueue);
    }

    for (int i = 0; i < (int)indexQueue.size(); ++i)
    {
        const int indexFrom = indexQueue[i].first;
        const int indexTo = indexQueue[i].second;

        const gtsam::Pose3 poseBetween = poseQueue[i];
        const gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = noiseQueue[i];
        gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    {
        std::lock_guard<std::mutex> lock(mtxLoopFactor);
        for (const auto &edge : indexQueue)
        {
            loopEdges.emplace_back(edge.first, edge.second);
        }
    }

    loopIsClosed = true;
    graphUpdate = true;
}

void addGNSSFactor();
void addGNSSYawFactor();

void poseGraphUpdate()
{
    addGNSSFactor();
    addGNSSYawFactor();
    addSceneFactor();
    addLoopFactor();
    if (gtSAMgraph.empty())
        return;

    isam->update(gtSAMgraph, initialEstimate);
    for (int i = 0; i < 5; ++i)
        isam->update();

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    graphUpdate = true;
    correctPoses();
}

void addSceneFactor()
{
    if (!groundEnableFlag && !floorEnableFlag)
        return;

    std::deque<SceneBatch> batches;

    {
        std::lock_guard<std::mutex> lock(mtxSceneBatch);
        if (sceneQueue.empty())
            return;

        batches.swap(sceneQueue);
    }

    while (!batches.empty())
    {
        SceneBatch batch = std::move(batches.front());
        batches.pop_front();

        const gtsam::Unit3 floor_up(getGravityUp());

        for (const auto &plane : batch.planes)
        {
            const gtsam::Key pkey = gtsam::Symbol('p', plane.id);

            if (!initialEstimate.exists(pkey) &&
                !isamCurrentEstimate.exists(pkey))
            {
                initialEstimate.insert(
                    pkey,
                    gtsam::OrientedPlane3(plane.plane));
            }

            if (floorEnableFlag)
            {
                const gtsam::Key fkey = gtsam::Symbol('f', plane.floor);

                if (!initialEstimate.exists(fkey) &&
                    !isamCurrentEstimate.exists(fkey))
                {
                    initialEstimate.insert(fkey, floor_up);
                    gtSAMgraph.add(
                        gtsam::PriorFactor<gtsam::Unit3>(
                            fkey,
                            floor_up,
                            floorNormalPriorNoise));
                }

                gtSAMgraph.add(
                    boost::shared_ptr<FloorFactor>(
                        new FloorFactor(
                            fkey,
                            pkey,
                            floorPlaneNoise)));
            }
        }

        if (groundEnableFlag)
        {
            for (const auto &factor : batch.factors)
            {
                gtSAMgraph.add(
                    gtsam::OrientedPlane3Factor(
                        factor.obs,
                        planeNoise,
                        static_cast<gtsam::Key>(factor.key),
                        gtsam::Symbol('p', factor.id)));
            }
        }
        graphUpdate = true;
    }
}

void sceneMatchingThread()
{
    if (!groundEnableFlag && !floorEnableFlag)
        return;

    RateType rate(20);
    while (ros_ok() && !flg_exit)
    {
        rate.sleep();
        performSceneMatching();
    }
}

void performSceneMatching()
{
    if (!groundEnableFlag && !floorEnableFlag)
        return;

    static int nextKey = 0;
    int key = -1;
    pcl::PointCloud<PointTypeIndex>::Ptr cloud;
    PointTypePose pose;

        {
            std::lock_guard<std::mutex> lock(mtxKeyframe);
        if (cloudKeyPoses3D == nullptr ||
            cloudKeyOdomPoses6D == nullptr ||
            nextKey >= static_cast<int>(featCloudKeyFrames.size()) ||
            nextKey >= static_cast<int>(cloudKeyOdomPoses6D->points.size()))
        {
            sceneDone.store(true);
            return;
        }

        const int numKeyFrame = static_cast<int>(std::min(
            featCloudKeyFrames.size(),
            cloudKeyOdomPoses6D->points.size()));
        if (nextKey >= numKeyFrame)
        {
            sceneDone.store(true);
            return;
        }

        key = nextKey;
        cloud = featCloudKeyFrames[key];
        pose = cloudKeyOdomPoses6D->points[key];
    }

    sceneDone.store(false);

    plane.update(key, cloud, pose);

    std::vector<PlaneObs> planes;
    const bool isGroundKey = key >= 3 && key % 3 == 0;
    if (isGroundKey)
        plane.extract(planes);

    SceneBatch batch;
    if (floorEnableFlag)
        floorMap.update(key, pose);

    if (isGroundKey && (groundEnableFlag || floorEnableFlag))
    {
        if (floorMap.updateGround(planes, plane.keys(), plane.poses(), batch))
        {
            std::lock_guard<std::mutex> batchLock(mtxSceneBatch);
            sceneQueue.push_back(std::move(batch));
        }
    }
    floorKey.store(key);
    ++nextKey;
    sceneDone.store(nextKey >= static_cast<int>(std::min(
        featCloudKeyFrames.size(),
        cloudKeyOdomPoses6D->points.size())));
}

bool findNearestKeyframeByTime(const std::vector<PointTypePose> &keyposes,
                                      double stamp,
                                      int &key_out)
{
    key_out = -1;
    if (keyposes.empty())
        return false;

    const double first_time = keyposes.front().time;
    const double last_time = keyposes.back().time;
    constexpr double gnss_keyframe_tol = 0.12;

    auto it = std::lower_bound(
        keyposes.begin(),
        keyposes.end(),
        stamp,
        [](const PointTypePose &pose, double t)
        {
            return pose.time < t;
        });

    size_t best_idx = 0;
    double best_dt = std::numeric_limits<double>::infinity();

    if (it == keyposes.begin())
    {
        best_idx = 0;
        best_dt = std::abs(keyposes[0].time - stamp);
    }
    else if (it == keyposes.end())
    {
        best_idx = keyposes.size() - 1;
        best_dt = std::abs(keyposes.back().time - stamp);
    }
    else
    {
        const size_t upper_idx = static_cast<size_t>(std::distance(keyposes.begin(), it));
        const size_t lower_idx = upper_idx - 1;
        const double lower_dt = std::abs(keyposes[lower_idx].time - stamp);
        const double upper_dt = std::abs(keyposes[upper_idx].time - stamp);
        if (lower_dt <= upper_dt)
        {
            best_idx = lower_idx;
            best_dt = lower_dt;
        }
        else
        {
            best_idx = upper_idx;
            best_dt = upper_dt;
        }
    }

    key_out = static_cast<int>(best_idx);
    return best_dt <= gnss_keyframe_tol && stamp <= last_time + gnss_keyframe_tol;
}

void addGNSSFactor()
{
    decltype(gnssPosFactorQueue) posQueue;

    {
        std::lock_guard<std::mutex> lock(mtxGnssFactor);
        if (gnssPosFactorQueue.empty())
            return;
        posQueue.swap(gnssPosFactorQueue);
    }

    while (!posQueue.empty())
    {
        const auto &[key, pos, cov] = posQueue.front();

        const double gnss_x = pos.x();
        const double gnss_y = pos.y();
        const double gnss_z = pos.z();

        const double cov_x = cov(0, 0);
        const double cov_y = cov(1, 1);
        const double cov_z = cov(2, 2);

        gtsam::Vector sigma(3);
        sigma << std::sqrt(cov_x),
                 std::sqrt(cov_y),
                 std::sqrt(cov_z);

        gtSAMgraph.add(
            gtsam::GPSFactor(
                key,
                gtsam::Point3(gnss_x, gnss_y, gnss_z),
                gtsam::noiseModel::Diagonal::Sigmas(sigma)));

        posQueue.pop_front();
    }

    graphUpdate = true;
}

void addGNSSYawFactor()
{
    decltype(gnssYawFactorQueue) yawQueue;

    {
        std::lock_guard<std::mutex> lock(mtxGnssFactor);
        if (gnssYawFactorQueue.empty())
            return;
        yawQueue.swap(gnssYawFactorQueue);
    }

    while (!yawQueue.empty())
    {
        const auto &[key, yaw] = yawQueue.front();

        const double yaw_sigma = std::max(gnss_yaw_factor_sigma, 1e-4);
        const auto yawNoise = gtsam::noiseModel::Isotropic::Sigma(1, yaw_sigma);

        gtSAMgraph.add(
            boost::shared_ptr<GnssYawFactor>(
                new GnssYawFactor(key, yaw, yawNoise)));

        yawQueue.pop_front();
    }

    graphUpdate = true;
}

void processGnssPos(const std::vector<PointTypePose> &keyposes)
{
    if (!p_gnss)
        return;

    if (keyposes.size() < 5)
        return;

    const Eigen::Vector3d key_start(
        keyposes.front().x,
        keyposes.front().y,
        keyposes.front().z);
    const Eigen::Vector3d key_end(
        keyposes.back().x,
        keyposes.back().y,
        keyposes.back().z);

    if ((key_end - key_start).norm() < gpsFactorMinDis)
    {
        return;
    }

    static Eigen::Vector3d last_fpos = Eigen::Vector3d::Zero();
    static bool has_fpos = false;

    while (true)
    {
        PosData pos;
        if (!p_gnss->peekOldestPos(pos))
            return;

        int key = -1;
        if (!findNearestKeyframeByTime(keyposes, pos.t, key))
        {
            if (!keyposes.empty() && pos.t <= keyposes.back().time)
            {
                PosData dropped;
                p_gnss->popOldestPos(dropped);
                continue;
            }

            return;
        }
        if (key < 0)
            return;
        if (pos_keys.count(key) != 0 || pos.t <= last_pos_t)
        {
            PosData dropped;
            p_gnss->popOldestPos(dropped);
            continue;
        }

        const PointTypePose &pose = keyposes[key];
        const gtsam::Pose3 key_pose = pclPointTogtsamPose3(pose);
        const Eigen::Matrix3d R_map_imu = key_pose.rotation().matrix();
        Eigen::Vector3d gnss_pos = pos.p - R_map_imu * p_gnss->lever();
        Eigen::Matrix3d gnss_cov = pos.cov;
        if (!useGnssElevation)
        {
            gnss_pos.z() = pose.z;
            gnss_cov(2, 2) = 0.01;
        }

        if (!gnss_pos.allFinite())
        {
            PosData dropped;
            p_gnss->popOldestPos(dropped);
            continue;
        }

        if (!gnss_cov.allFinite() ||
            gnss_cov(0, 0) <= 0.0 ||
            gnss_cov(1, 1) <= 0.0 ||
            gnss_cov(2, 2) <= 0.0)
        {
            PosData dropped;
            p_gnss->popOldestPos(dropped);
            continue;
        }

        if (has_fpos &&
            (gnss_pos - last_fpos).norm() < gpsFactorMinDis)
        {
            PosData dropped;
            p_gnss->popOldestPos(dropped);
            continue;
        }

        if (!p_gnss->popOldestPos(pos))
            return;

        pos_keys.insert(key);
        last_pos_t = pos.t;
        last_fpos = gnss_pos;
        has_fpos = true;

        {
            std::lock_guard<std::mutex> lock(mtxGnssFactor);
            gnssPosFactorQueue.emplace_back(
                key,
                gnss_pos,
                gnss_cov);
        }
    }
}

void processGnssYaw(const std::vector<PointTypePose> &keyposes)
{
    if (!p_gnss)
        return;

    while (true)
    {
        YawData yaw;
        if (!p_gnss->peekOldestYaw(yaw))
            return;

        if (!useGnssYawFactor)
        {
            YawData dropped;
            p_gnss->popOldestYaw(dropped);
            continue;
        }

        int key = -1;
        if (!findNearestKeyframeByTime(keyposes, yaw.t, key))
        {
            if (!keyposes.empty() && yaw.t <= keyposes.back().time)
            {
                YawData dropped;
                p_gnss->popOldestYaw(dropped);
                continue;
            }

            return;
        }
        if (key < 0)
            return;
        if (yaw_keys.count(key) != 0 || yaw.t <= last_yaw_t)
        {
            YawData dropped;
            p_gnss->popOldestYaw(dropped);
            continue;
        }

        if (std::abs(normalizeYaw(yaw.yaw - keyposes[key].yaw)) > 60.0 * M_PI / 180.0)
        {
            YawData dropped;
            p_gnss->popOldestYaw(dropped);
            continue;
        }

        if (!std::isfinite(yaw.yaw))
        {
            YawData dropped;
            p_gnss->popOldestYaw(dropped);
            continue;
        }

        if (!p_gnss->popOldestYaw(yaw))
            return;

        yaw_keys.insert(key);
        last_yaw_t = yaw.t;

        {
            std::lock_guard<std::mutex> lock(mtxGnssFactor);
            gnssYawFactorQueue.emplace_back(
                key,
                yaw.yaw);
        }
    }
}

void performGnssMatching()
{
    if (!gnssEnableFlag || !gnss_aligned.load() || !p_gnss)
        return;

    std::vector<PointTypePose> keyposes;
    {
        std::lock_guard<std::mutex> lock(mtxKeyframe);
        if (cloudKeyPoses6D == nullptr || cloudKeyPoses6D->points.empty())
            return;
        keyposes.assign(cloudKeyPoses6D->points.begin(), cloudKeyPoses6D->points.end());
    }

    processGnssPos(keyposes);
    processGnssYaw(keyposes);
}

void updatePath(const PointTypePose& pose_in)
{
    PoseStampedMsg pose_stamped;
    pose_stamped.header.stamp = get_ros_time(pose_in.time);
    pose_stamped.header.frame_id = map_frame;
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    pose_stamped.pose.orientation = quaternion_from_rpy(pose_in.roll, pose_in.pitch, pose_in.yaw);

    globalPath.poses.push_back(pose_stamped);
}

void saveKeyFramesAndFactor(pcl::PointCloud<pcl::PointXYZINormal>::Ptr feats_undistort)
{
    const PointTypePose OdomPose = trans2PointTypePose(transformTobeMapped);
    const gtsam::Key latestPoseKey = static_cast<gtsam::Key>(cloudKeyPoses3D->size());

    // odom factor
    addOdomFactor();
    addGNSSFactor();

    addGNSSYawFactor();
    addSceneFactor();
    // loop factor
    addLoopFactor();
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    if (loopIsClosed)
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();
    }

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    graphUpdate = true;

    //save key poses
    PointTypeIndex thisPose3D;
    PointTypePose thisPose6D;
    const Pose3 latestEstimate = isamCurrentEstimate.at<Pose3>(latestPoseKey);

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
    thisPose6D.roll  = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw   = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserInfoCur;

    pcl::PointCloud<PointTypeIndex>::Ptr featCloudKeyFrame(new pcl::PointCloud<PointTypeIndex>());
    PointTypeIndex point;
    for (const auto &pt : feats_undistort->points)
    {
        Eigen::Vector3d pointBodyLidar(pt.x, pt.y, pt.z);
        Eigen::Vector3d pointBodyImu(rotationLidarToIMU * pointBodyLidar + translationLidarToIMU);

        point.x = pointBodyImu(0);
        point.y = pointBodyImu(1);
        point.z = pointBodyImu(2);
        point.intensity = pt.intensity;
        featCloudKeyFrame->push_back(point);
    }

    {
        std::lock_guard<std::mutex> lock(mtxKeyframe);

        cloudKeyPoses3D->push_back(thisPose3D);
        cloudKeyPoses6D->push_back(thisPose6D);
        featCloudKeyFrames.push_back(featCloudKeyFrame);
        cloudKeyOdomPoses6D->push_back(OdomPose);
    }

    updatePath(thisPose6D);
    poseCovariance = isam->marginalCovariance(latestPoseKey);

    if (keyframe_export_en)
    {
        const std::string stamp_str = format_unix_time(thisPose6D.time);
        const std::string pcd_path = keyframe_frames_dir + "scans/" + stamp_str + ".pcd";
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(pcd_path, *feats_undistort);
    }
}

void shutdownMapOptimization()
{
    pubKeyPoses.reset();
    pubPath.reset();
    pubLaserCloudGlobal.reset();
    pubRecentKeyFrame.reset();
    pubLoopConstraintEdge.reset();
    pubKeyFrameYawMarkers.reset();
#ifdef USE_ROS1
    samTfBroadcaster.reset();
#elif defined(USE_ROS2)
    samTfBroadcaster.reset();
#endif

    if (isam != nullptr)
    {
        delete isam;
        isam = nullptr;
    }
}

void correctPoses()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (graphUpdate)
    {
        {
            std::lock_guard<std::mutex> lock(mtxKeyframe);

            // clear path
            globalPath.poses.clear();
            // update key poses
            const int numPoses = static_cast<int>(cloudKeyPoses6D->points.size());
            for (int i = 0; i < numPoses; ++i)
            {
                if (!isamCurrentEstimate.exists(i))
                    continue;

                const Pose3 new_pose = isamCurrentEstimate.at<Pose3>(i);

                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = new_pose.rotation().roll();
                cloudKeyPoses6D->points[i].pitch = new_pose.rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = new_pose.rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }
        }
    }

    if (!cloudKeyOdomPoses6D->points.empty() &&
        cloudKeyOdomPoses6D->points.size() == cloudKeyPoses6D->points.size())
    {
        const size_t ref = cloudKeyPoses6D->points.size() - 1;
        const gtsam::Pose3 pose_map_ref = pclPointTogtsamPose3(cloudKeyPoses6D->points[ref]);
        const gtsam::Pose3 pose_odom_ref = pclPointTogtsamPose3(cloudKeyOdomPoses6D->points[ref]);
        const Eigen::Matrix3d R_map_odom =
            pose_map_ref.rotation().matrix() * pose_odom_ref.rotation().matrix().transpose();
        const Eigen::Vector3d t_map_odom =
            Eigen::Vector3d(pose_map_ref.translation().x(),
                             pose_map_ref.translation().y(),
                             pose_map_ref.translation().z()) -
            R_map_odom * Eigen::Vector3d(pose_odom_ref.translation().x(),
                                             pose_odom_ref.translation().y(),
                                             pose_odom_ref.translation().z());
        setMapOdom(R_map_odom, t_map_odom);
        publishMapToOdomTf(get_ros_time(timeLaserInfoCur));
    }

    graphUpdate = false;
    loopIsClosed = false;
}

void publishSamMsg()
{
    std::vector<PointTypeIndex> poses3D;
    std::vector<PointTypePose> poses6D;
    std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> clouds;
    {
        std::lock_guard<std::mutex> lock(mtxKeyframe);
        if (cloudKeyPoses3D == nullptr || cloudKeyPoses6D == nullptr || cloudKeyOdomPoses6D == nullptr)
            return;
        if (cloudKeyPoses3D->points.empty())
            return;
        poses3D.assign(cloudKeyPoses3D->points.begin(), cloudKeyPoses3D->points.end());
        poses6D.assign(cloudKeyPoses6D->points.begin(), cloudKeyPoses6D->points.end());
        clouds = featCloudKeyFrames;
    }

    if (poses3D.empty() || poses6D.empty() || clouds.empty())
        return;

    pcl::PointCloud<PointTypeIndex>::Ptr keyPosesCloud(new pcl::PointCloud<PointTypeIndex>());
    keyPosesCloud->points.reserve(poses3D.size());
    for (size_t i = 0; i < poses3D.size(); ++i)
    {
        PointTypeIndex pt;
        pt.x = poses3D[i].x;
        pt.y = poses3D[i].y;
        pt.z = poses3D[i].z;
        pt.intensity = poses3D[i].intensity;
        keyPosesCloud->push_back(pt);
    }

    // publish key poses
    publishCloud(pubKeyPoses, keyPosesCloud, timeLaserInfoStamp, map_frame);
    if (ros_subscription_count(pubPath) != 0)
    {
        globalPath.header.stamp = timeLaserInfoStamp;
        globalPath.header.frame_id = map_frame;
        ros_publish(pubPath, globalPath);
    }

    if (ros_subscription_count(pubKeyFrameYawMarkers) != 0)
    {
        MarkerArrayMsg markerArray;

        MarkerMsg markerClear;
        markerClear.header.frame_id = map_frame;
        markerClear.header.stamp = timeLaserInfoStamp;
        markerClear.action = MarkerMsg::DELETEALL;
        markerArray.markers.push_back(markerClear);

        markerArray.markers.reserve(markerArray.markers.size() + poses6D.size());
        for (size_t i = 0; i < poses6D.size(); ++i)
        {
            const auto &pose = poses6D[i];
            MarkerMsg marker;
            marker.header.frame_id = map_frame;
            marker.header.stamp = timeLaserInfoStamp;
            marker.ns = "keyframe_yaw";
            marker.id = static_cast<int>(i);
            marker.action = MarkerMsg::ADD;
            marker.type = MarkerMsg::ARROW;
            marker.pose.position.x = pose.x;
            marker.pose.position.y = pose.y;
            marker.pose.position.z = pose.z;
            marker.pose.orientation = quaternion_from_rpy(0.0, 0.0, pose.yaw);
            marker.scale.x = 0.8;
            marker.scale.y = 0.12;
            marker.scale.z = 0.12;
            marker.color.r = 1.0f;
            marker.color.g = 1.0f;
            marker.color.b = 0.0f;
            marker.color.a = 0.9f;
            markerArray.markers.push_back(marker);
        }

        ros_publish(pubKeyFrameYawMarkers, markerArray);
    }

    if (ros_subscription_count(pubRecentKeyFrame) != 0)
    {
        pcl::PointCloud<PointTypeIndex>::Ptr cloudOut(new pcl::PointCloud<PointTypeIndex>());
        PointTypePose thisPose6D = poses6D.back();
        *cloudOut += *transformPointCloud(clouds.back(),  &thisPose6D);
        publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, map_frame);
    }
}

void visualizeLoopClosure()
{
    std::vector<std::pair<int, int>> loop_edges;
    std::vector<PointTypePose> keyposes;
    {
        std::lock_guard<std::mutex> lock(mtxLoopFactor);
        if (loopEdges.empty())
            return;
        loop_edges = loopEdges;
    }
    {
        std::lock_guard<std::mutex> lock(mtxKeyframe);
        if (cloudKeyPoses6D == nullptr || cloudKeyPoses6D->points.empty())
            return;
        keyposes.assign(cloudKeyPoses6D->points.begin(), cloudKeyPoses6D->points.end());
    }

    MarkerArrayMsg markerArray;
    // loop nodes
    MarkerMsg markerNode;
    markerNode.header.frame_id = map_frame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = MarkerMsg::ADD;
    markerNode.type = MarkerMsg::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.1; markerNode.scale.y = 0.1; markerNode.scale.z = 0.1; 
    markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
    markerNode.color.a = 1;
    // loop edges
    MarkerMsg markerEdge;
    markerEdge.header.frame_id = map_frame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = MarkerMsg::ADD;
    markerEdge.type = MarkerMsg::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    for (const auto &edge : loop_edges)
    {
        int key_cur = edge.first;
        int key_pre = edge.second;
        if (key_cur < 0 || key_cur >= static_cast<int>(keyposes.size()) ||
            key_pre < 0 || key_pre >= static_cast<int>(keyposes.size()))
            continue;
        PointMsg p;
        p.x = keyposes[key_cur].x;
        p.y = keyposes[key_cur].y;
        p.z = keyposes[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = keyposes[key_pre].x;
        p.y = keyposes[key_pre].y;
        p.z = keyposes[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    ros_publish(pubLoopConstraintEdge, markerArray);
}

void loopClosureThread()
{
    if (!loopClosureEnableFlag)
        return;

    constexpr int kLoopBatchSize = 5;
    RateType rate(loopClosureFrequency);
    int nextKey = 1;

    while (ros_ok() && !flg_exit)
    {
        rate.sleep();

        std::vector<PointTypeIndex> poses3D;
        std::vector<PointTypePose> poses6D;
        std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> clouds;
        {
            std::lock_guard<std::mutex> lock(mtxKeyframe);
            if (cloudKeyPoses3D == nullptr || cloudKeyPoses6D == nullptr || cloudKeyOdomPoses6D == nullptr)
            {
                loopDone.store(true);
                continue;
            }
            if (cloudKeyPoses3D->points.size() < 6 ||
                cloudKeyPoses6D->points.size() < 6)
            {
                loopDone.store(true);
                continue;
            }

            poses3D.assign(cloudKeyPoses3D->points.begin(), cloudKeyPoses3D->points.end());
            poses6D.assign(cloudKeyPoses6D->points.begin(), cloudKeyPoses6D->points.end());
            clouds = featCloudKeyFrames;
        }

        if (poses3D.size() < 6)
            continue;

        int readyKey = static_cast<int>(poses3D.size()) - 5;
        if (floorEnableFlag)
            readyKey = std::min(readyKey, floorKey.load());

        if (nextKey % 25 == 0 || nextKey == 1 || nextKey >= readyKey)
        {
            ROS_PRINT_INFO(
                "[LOOP] next=%d ready=%d latest=%d floor=%d",
                nextKey,
                readyKey,
                static_cast<int>(poses3D.size()) - 1,
                floorKey.load());
        }

        if (nextKey > readyKey)
        {
            loopDone.store(true);
            continue;
        }

        loopDone.store(false);

        pcl::PointCloud<PointTypeIndex>::Ptr poseCloud(new pcl::PointCloud<PointTypeIndex>());
        poseCloud->points.insert(
            poseCloud->points.end(),
            poses3D.begin(),
            poses3D.end());
        poseCloud->width = static_cast<uint32_t>(poseCloud->points.size());
        poseCloud->height = 1;
        poseCloud->is_dense = true;

        pcl::KdTreeFLANN<PointTypeIndex> poseTree;
        poseTree.setInputCloud(poseCloud);

        int processedCount = 0;
        while (nextKey <= readyKey && processedCount < kLoopBatchSize)
        {
            performLoopClosure(nextKey, poses3D, poses6D, clouds, poseTree);
            loopKey.store(nextKey);
            ++nextKey;
            ++processedCount;
        }

        loopDone.store(nextKey > readyKey);
    }
}

void gnssMatchingThread()
{
    if (!gnssEnableFlag)
        return;

    RateType rate(10);

    while (ros_ok() && !flg_exit)
    {
        rate.sleep();
        if (!gnss_aligned.load())
            continue;
        performGnssMatching();
        if (gnssPathVis && p_gnss)
        {
            PosData pos;
            YawData yaw;
            if (p_gnss->latestPos(pos) && p_gnss->latestYaw(yaw))
            {
                TransformStampedMsg tf_msg;
                tf_msg.header.stamp = get_ros_time(std::max(pos.t, yaw.t));
                tf_msg.header.frame_id = map_frame;
                tf_msg.child_frame_id = "gnss_link";
                tf_msg.transform.translation.x = pos.p.x();
                tf_msg.transform.translation.y = pos.p.y();
                tf_msg.transform.translation.z = pos.p.z();
                tf_msg.transform.rotation = quaternion_from_rpy(0.0, 0.0, yaw.yaw);

            if (samTfBroadcaster)
                samTfBroadcaster->sendTransform(tf_msg);
            }
        }
    }
}

void publishGlobalMap() {
    if (ros_subscription_count(pubLaserCloudGlobal) == 0)
        return;

    pcl::PointCloud<PointTypeIndex>::Ptr keyPoses3D(new pcl::PointCloud<PointTypeIndex>());
    std::vector<PointTypePose> keyPoses6D;
    std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> keyClouds;
    {
        std::lock_guard<std::mutex> lock(mtxKeyframe);
        if (cloudKeyPoses3D == nullptr || cloudKeyPoses6D == nullptr || cloudKeyOdomPoses6D == nullptr)
            return;
        if (cloudKeyPoses3D->points.empty())
            return;
        *keyPoses3D = *cloudKeyPoses3D;
        keyPoses6D.assign(cloudKeyPoses6D->points.begin(), cloudKeyPoses6D->points.end());
        keyClouds = featCloudKeyFrames;
    }

    if (keyPoses3D->empty() || keyPoses6D.empty() || keyClouds.empty())
        return;

    const PointTypeIndex &currentPose = keyPoses3D->back();
    pcl::PointCloud<PointTypeIndex>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointTypeIndex>());

    for (const auto &pose : keyPoses3D->points)
    {
        if (pointDistance(pose, currentPose) > globalMapVisualizationSearchRadius)
            continue;
        globalMapKeyPoses->push_back(pose);
    }

    pcl::VoxelGrid<PointTypeIndex> downSizeFilterGlobalMapKeyPoses; // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

    for (auto &pt : globalMapKeyPosesDS->points)
    {
        float minDist = std::numeric_limits<float>::max();
        int key = -1;
        for (const auto &pose : globalMapKeyPoses->points)
        {
            const float dist = pointDistance(pt, pose);
            if (dist < minDist)
            {
                minDist = dist;
                key = static_cast<int>(pose.intensity);
            }
        }

        if (key < 0 ||
            key >= static_cast<int>(keyPoses6D.size()) ||
            key >= static_cast<int>(keyClouds.size()))
            continue;

        pt.intensity = static_cast<float>(key);
        *globalMapKeyFrames += *transformPointCloud(keyClouds[key], &keyPoses6D[key]);
    }

    pcl::VoxelGrid<PointTypeIndex> downSizeFilterGlobalMapKeyFrames; // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(pubLaserCloudGlobal, globalMapKeyFramesDS, timeLaserInfoStamp, map_frame);
}

void visualizeGlobalMapThread()
{
    RateType rate(0.2);
    while (ros_ok() && !flg_exit){
        rate.sleep();
        publishGlobalMap();
        visualizeLoopClosure();
    }
}
