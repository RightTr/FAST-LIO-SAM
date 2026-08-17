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

#include "ikd-Tree/ikdtree_public.h"

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

std::mutex mtxSceneBatch;
SceneBatch sceneBatch;
bool sceneReady = false;
gtsam::noiseModel::Diagonal::shared_ptr planeNoise;
gtsam::SharedNoiseModel floorNormalPriorNoise;
gtsam::SharedNoiseModel floorPlaneNoise;
std::mutex gravityAxisMutex;
Eigen::Vector3d gravityUpAxis = Eigen::Vector3d::UnitZ();

std::recursive_mutex mtxLoop;
std::mutex mtxGnssFactor;
bool planeDirty = false;
bool planeResetPending = false;
int planeResetKey = -1;
bool poseTreeDirty = false;

bool graphUpdate = false;
bool loopIsClosed = false;
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
void ReconstructIkdTree();
void performSceneMatching();

void setGravityUp(const Eigen::Vector3d &gravity_up)
{
    std::lock_guard<std::mutex> lock(gravityAxisMutex);
    if (gravity_up.allFinite() && gravity_up.norm() > 1e-6)
        gravityUpAxis = gravity_up.normalized();
    else
        gravityUpAxis = Eigen::Vector3d::UnitZ();
}

Eigen::Vector3d getGravityUp()
{
    std::lock_guard<std::mutex> lock(gravityAxisMutex);
    if (!gravityUpAxis.allFinite() || gravityUpAxis.norm() <= 1e-6)
        return Eigen::Vector3d::UnitZ();
    return gravityUpAxis.normalized();
}

pcl::VoxelGrid<PointTypeIndex> downSizeFilterICP;

vector<pcl::PointCloud<PointTypeIndex>::Ptr> featCloudKeyFrames;

KD_TREE_PUBLIC<PointTypeIndex>::Ptr ikdtreeHistoryKeyPoses;

KD_TREE_PUBLIC<PointTypeIndex>::PointVector initPoses3D;

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
    
    #pragma omp parallel for num_threads(numberOfCores)
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

    ikdtreeHistoryKeyPoses.reset(new KD_TREE_PUBLIC<PointTypeIndex>());

    for (int i = 0; i < 6; ++i){
        transformTobeMapped[i] = 0;
    }
}

void MapOptimizationInit()
{
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

bool detectLoopClosureDistance(int loopKeyCur,
                               const std::vector<FloorRange> &floorRanges,
                               int *closestID)
{
    std::lock_guard<std::recursive_mutex> lock(mtxLoop);

    PointTypeIndex current_pose;
    double current_time = 0.0;
    KD_TREE_PUBLIC<PointTypeIndex>::PointVector nearPoses;
    if (cloudKeyPoses3D->empty() ||
        ikdtreeHistoryKeyPoses->Root_Node == nullptr)
        return false;

    if (poseTreeDirty)
    {
        poseTreeDirty = false;
        ReconstructIkdTree();
    }

    if (loopKeyCur < 0)
        return false;

    if (loopKeyCur >= static_cast<int>(cloudKeyPoses3D->points.size()) ||
        loopKeyCur >= static_cast<int>(cloudKeyPoses6D->points.size()))
        return false;

    current_pose = cloudKeyPoses3D->points[loopKeyCur];
    current_time = cloudKeyPoses6D->points[loopKeyCur].time;
    nearPoses.clear();
    ikdtreeHistoryKeyPoses->Radius_Search(
        current_pose,
        historyKeyframeSearchRadius,
        nearPoses);

    int loopKeyPre = -1;
    float nearestSqDis = std::numeric_limits<float>::max();
    for (const auto &near_pose : nearPoses)
    {
        const int id = static_cast<int>(near_pose.intensity);
        if (id < 0 || id >= loopKeyCur)
            continue;
        if (id >= static_cast<int>(cloudKeyPoses6D->points.size()))
            continue;
        if (loopUsedKeys.count(id) != 0)
            continue;

        const double nearTime = cloudKeyPoses6D->points[id].time;
        if (current_time - nearTime <= historyKeyframeSearchTimeDiff)
            continue;

        bool inFloorRange = false;
        for (const auto &range : floorRanges)
        {
            if (id < range.begin)
                continue;
            if (range.end >= 0 && id > range.end)
                continue;
            inFloorRange = true;
            break;
        }

        if (!inFloorRange)
            continue;

        const float dx = near_pose.x - current_pose.x;
        const float dy = near_pose.y - current_pose.y;
        const float dz = near_pose.z - current_pose.z;
        const float sqDis = dx * dx + dy * dy + dz * dz;
        if (sqDis < nearestSqDis)
        {
            nearestSqDis = sqDis;
            loopKeyPre = id;
        }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;

    *closestID = loopKeyPre;
    return true;
}

void performLoopClosure(int loopKeyCur)
{
    std::lock_guard<std::recursive_mutex> poseLock(mtxLoop);

    if (cloudKeyPoses3D->points.empty())
        return;

    std::vector<FloorRange> floorRanges;
    FloorRange currentRange;
    if (!floorMap.getFloorRanges(loopKeyCur, floorRanges, currentRange))
        return;
    (void)currentRange;

    // find keys
    int loopKeyPre;
    if (detectLoopClosureDistance(loopKeyCur, floorRanges, &loopKeyPre) == false)
        return;

    // extract cloud
    pcl::PointCloud<PointTypeIndex>::Ptr curCloud;
    std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> prevKeyframeClouds;
    std::vector<PointTypePose> prevKeyposes;
    PointTypePose curPose;
    if (loopKeyCur >= static_cast<int>(featCloudKeyFrames.size()) ||
        loopKeyCur >= static_cast<int>(cloudKeyPoses6D->points.size()))
        return;

    curCloud = featCloudKeyFrames[loopKeyCur];
    curPose = cloudKeyPoses6D->points[loopKeyCur];

    const int cloudSize = static_cast<int>(
        std::min(featCloudKeyFrames.size(), cloudKeyPoses6D->points.size()));
    FloorRange loopPreRange;
    for (const auto &range : floorRanges)
    {
        if (loopKeyPre < range.begin)
            continue;
        if (range.end >= 0 && loopKeyPre > range.end)
            continue;

        loopPreRange = range;
        break;
    }

    const int rangeEnd = loopPreRange.end >= 0 ? loopPreRange.end : cloudSize - 1;
    const int historyBegin = std::max(loopKeyPre - historyKeyframeSearchNum, loopPreRange.begin);
    const int historyEnd = std::min(loopKeyPre + historyKeyframeSearchNum, rangeEnd);
    for (int keyNear = historyBegin; keyNear <= historyEnd; ++keyNear)
    {
        prevKeyframeClouds.push_back(featCloudKeyFrames[keyNear]);
        prevKeyposes.push_back(cloudKeyPoses6D->points[keyNear]);
    }

    if (!curCloud || curCloud->empty())
        return;

    pcl::PointCloud<PointTypeIndex>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointTypeIndex>());
    *cureKeyframeCloud += *transformPointCloud(curCloud, &curPose);
    for (size_t i = 0; i < prevKeyframeClouds.size(); ++i)
    {
        *prevKeyframeCloud += *transformPointCloud(prevKeyframeClouds[i], &prevKeyposes[i]);
    }

    pcl::PointCloud<PointTypeIndex>::Ptr cureKeyframeCloudDS(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr prevKeyframeCloudDS(new pcl::PointCloud<PointTypeIndex>());
    downSizeFilterICP.setInputCloud(cureKeyframeCloud);
    downSizeFilterICP.filter(*cureKeyframeCloudDS);
    downSizeFilterICP.setInputCloud(prevKeyframeCloud);
    downSizeFilterICP.filter(*prevKeyframeCloudDS);

    if (cureKeyframeCloudDS->size() < 300)
        return;

    if (prevKeyframeCloudDS->size() < 1000)
        return;

    // ICP Settings
    static pcl::IterativeClosestPoint<PointTypeIndex, PointTypeIndex> icp;
    icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
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
        return;

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
        loopKeyPre >= static_cast<int>(cloudKeyPoses6D->points.size()))
        return;
    prevPose = cloudKeyPoses6D->points[loopKeyPre];
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(prevPose);
    gtsam::Vector Vector6(6);
    const double noiseScore = std::max(
        static_cast<double>(icp.getFitnessScore()) * loopWeight, 1e-4);
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint and reserve both endpoints atomically.
    {
        loopUsedKeys.insert(loopKeyPre);
        loopIndexQueue.emplace_back(loopKeyCur, loopKeyPre);
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
    }
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
    std::vector<std::pair<int, int>> indexQueue;
    std::vector<gtsam::Pose3> poseQueue;
    std::vector<gtsam::noiseModel::Diagonal::shared_ptr> noiseQueue;

    {
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

        loopEdges.emplace_back(indexFrom, indexTo);
    }

    loopIsClosed = true;
    graphUpdate = true;
}

void poseGraphUpdate()
{
    std::lock_guard<std::recursive_mutex> poseLock(mtxLoop);

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

bool addSceneFactor()
{
    SceneBatch batch;
    {
        std::lock_guard<std::mutex> lock(mtxSceneBatch);
        if (!sceneReady)
            return false;

        batch = std::move(sceneBatch);
        sceneBatch = SceneBatch();
        sceneReady = false;
    }

    const gtsam::Unit3 floor_up(getGravityUp());

    for (const auto &plane : batch.planes)
    {
        const gtsam::Key fkey = gtsam::Symbol('f', plane.floor);
        const gtsam::Key pkey = gtsam::Symbol('p', plane.id);

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

        if (!initialEstimate.exists(pkey) &&
            !isamCurrentEstimate.exists(pkey))
        {
            initialEstimate.insert(
                pkey,
                gtsam::OrientedPlane3(plane.plane));
        }

        gtSAMgraph.add(
            boost::shared_ptr<FloorFactor>(
                new FloorFactor(
                    fkey,
                    pkey,
                    floorPlaneNoise)));
    }

    for (const auto &factor : batch.factors)
    {
        gtSAMgraph.add(
            gtsam::OrientedPlane3Factor(
                factor.obs,
                planeNoise,
                static_cast<gtsam::Key>(factor.key),
                gtsam::Symbol('p', factor.id)));
    }

    graphUpdate = true;
    return true;
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
        std::lock_guard<std::recursive_mutex> lock(mtxLoop);
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
    bool floorChanged = false;

    // odom factor
    addOdomFactor();

    addSceneFactor();

    addGNSSFactor();

    addGNSSYawFactor();
    // loop factor
    addLoopFactor();

    // cout << "****************************************************" << endl;
    // gtSAMgraph.print("GTSAM Graph:\n");

    // update iSAM
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

    //save key poses
    PointTypeIndex thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<Pose3>(latestPoseKey);
    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");

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

    {
        std::lock_guard<std::recursive_mutex> lock(mtxLoop);

        cloudKeyPoses3D->push_back(thisPose3D);
        cloudKeyPoses6D->push_back(thisPose6D);
        floorChanged = floorMap.update(
            static_cast<int>(cloudKeyPoses6D->points.size()) - 1,
            OdomPose);
        if (floorChanged)
        {
            planeResetPending = true;
            planeResetKey = static_cast<int>(cloudKeyPoses6D->points.size()) - 1;
        }
        updatePath(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        poseCovariance = isam->marginalCovariance(latestPoseKey);

        pcl::PointCloud<PointTypeIndex>::Ptr featCloudKeyFrame(new pcl::PointCloud<PointTypeIndex>());
        PointTypeIndex point;
        for (const auto &pt : feats_undistort->points) {
            Eigen::Vector3d pointBodyLidar(pt.x, pt.y, pt.z);
            Eigen::Vector3d pointBodyImu(rotationLidarToIMU * pointBodyLidar + translationLidarToIMU);

            point.x = pointBodyImu(0);
            point.y = pointBodyImu(1);
            point.z = pointBodyImu(2);
            point.intensity = pt.intensity;
            featCloudKeyFrame->push_back(point);
        }

        featCloudKeyFrames.push_back(featCloudKeyFrame);
        cloudKeyOdomPoses6D->push_back(OdomPose);

        if (keyframe_export_en)
        {
            const std::string stamp_str = format_unix_time(thisPose6D.time);
            const std::string pcd_path = keyframe_frames_dir + "scans/" + stamp_str + ".pcd";
            pcl::PCDWriter pcd_writer;
            pcd_writer.writeBinary(pcd_path, *feats_undistort);
        }

        const bool treeNeedsRebuild = poseTreeDirty;
        poseTreeDirty = false;
        if (treeNeedsRebuild) {
            ReconstructIkdTree();
        } else if (ikdtreeHistoryKeyPoses->Root_Node == nullptr) {
            initPoses3D.push_back(thisPose3D);
            if (cloudKeyPoses3D->points.size() >= 10)
                ikdtreeHistoryKeyPoses->Build(initPoses3D);
        } else {
            ikdtreeHistoryKeyPoses->Add_Point(thisPose3D);
        }
    }

    if (floorChanged)
    {
        std::lock_guard<std::mutex> batchLock(mtxSceneBatch);
        sceneBatch = SceneBatch();
        sceneReady = false;
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

    if (isam != nullptr)
    {
        delete isam;
        isam = nullptr;
    }
}

static void rebuildPlane(
    int lastProcessedKey,
    const std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> &clouds,
    const std::vector<PointTypePose> &poses)
{
    if (lastProcessedKey < 0)
        return;

    plane.reset();

    const int begin = std::max(0, lastProcessedKey - Plane::kWindowSize + 1);
    for (int key = begin; key <= lastProcessedKey; ++key)
    {
        if (key < 0 || key >= static_cast<int>(clouds.size()))
            continue;
        if (!clouds[key])
            continue;

        plane.update(key, clouds[key], poses[key]);
    }
}

void performSceneMatching()
{
    if (!groundEnableFlag)
        return;

    static int lastProcessedKey = -1;

    {
        std::lock_guard<std::mutex> lock(mtxSceneBatch);
        if (sceneReady)
            return;
    }

    std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> clouds;
    std::vector<PointTypePose> poses;
    bool resetPlane = false;
    int resetKey = -1;
    bool rebuild = false;
    {
        std::lock_guard<std::recursive_mutex> lock(mtxLoop);
        if (!cloudKeyPoses6D)
            return;

        if (planeResetPending)
        {
            resetPlane = true;
            resetKey = planeResetKey;
            planeResetPending = false;
        }

        const int numKeyFrame = static_cast<int>(
            std::min(featCloudKeyFrames.size(), cloudKeyPoses6D->points.size()));
        if (numKeyFrame <= 0 || lastProcessedKey >= numKeyFrame - 1)
        {
            if (!resetPlane)
                return;
        }

        if (resetPlane)
        {
            plane.reset();
            lastProcessedKey = resetKey - 1;
            planeDirty = false;
        }

        if (numKeyFrame <= 0)
            return;

        rebuild = planeDirty && !resetPlane;
        if (rebuild)
            planeDirty = false;

        clouds.assign(
            featCloudKeyFrames.begin(),
            featCloudKeyFrames.begin() + numKeyFrame);
        poses.assign(
            cloudKeyPoses6D->points.begin(),
            cloudKeyPoses6D->points.begin() + numKeyFrame);
    }

    if (rebuild)
        rebuildPlane(lastProcessedKey, clouds, poses);

    for (int key = lastProcessedKey + 1; key < static_cast<int>(clouds.size()); ++key)
    {
        plane.update(key, clouds[key], poses[key]);

        std::vector<PlaneObs> ground_plane_obs;
        plane.extract(ground_plane_obs);

        SceneBatch batch;
        const bool valid = floorMap.updateGround(
            ground_plane_obs,
            plane.keys(),
            plane.poses(),
            batch);

        if (valid)
        {
            {
                std::lock_guard<std::recursive_mutex> lock(mtxLoop);
                if (planeDirty)
                    return;
            }

            std::lock_guard<std::mutex> batchLock(mtxSceneBatch);
            if (sceneReady)
                return;

            sceneBatch = std::move(batch);
            sceneReady = true;
            lastProcessedKey = key;
            return;
        }

        lastProcessedKey = key;
    }
}

void structureMatchingThread()
{
    if (!sceneEnableFlag || !groundEnableFlag)
        return;

    RateType rate(20);
    while (ros_ok() && !flg_exit)
    {
        rate.sleep();
        performSceneMatching();
    }
}

void ReconstructIkdTree()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (ikdtreeHistoryKeyPoses->Root_Node != nullptr)
        ikdtreeHistoryKeyPoses->delete_tree_nodes(&ikdtreeHistoryKeyPoses->Root_Node);

    KD_TREE_PUBLIC<PointTypeIndex>::PointVector pose_points;
    pose_points.reserve(cloudKeyPoses3D->points.size());
    for (const auto &pose : cloudKeyPoses3D->points)
    {
        pose_points.push_back(pose);
    }

    ikdtreeHistoryKeyPoses->Build(pose_points);
}

void correctPoses()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (graphUpdate)
    {
        bool clearSceneQueue = false;
        {
            std::lock_guard<std::recursive_mutex> lock(mtxLoop);

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

            poseTreeDirty = true;
            if (loopIsClosed)
            {
                planeDirty = true;
                clearSceneQueue = true;
            }
        }

        if (clearSceneQueue)
        {
            std::lock_guard<std::mutex> batchLock(mtxSceneBatch);
            sceneBatch = SceneBatch();
            sceneReady = false;
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
    if (cloudKeyPoses3D->points.empty())
        return;
    // publish key poses
    publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, map_frame);
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

        markerArray.markers.reserve(markerArray.markers.size() + cloudKeyPoses6D->points.size());
        for (size_t i = 0; i < cloudKeyPoses6D->points.size(); ++i)
        {
            const auto &pose = cloudKeyPoses6D->points[i];
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
        PointTypePose thisPose6D = cloudKeyPoses6D->back();
        *cloudOut += *transformPointCloud(featCloudKeyFrames.back(),  &thisPose6D);
        publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, map_frame);
    }
}

void visualizeLoopClosure()
{
    std::vector<std::pair<int, int>> loop_edges;
    pcl::PointCloud<PointTypePose>::VectorType keyposes;
    {
        std::lock_guard<std::recursive_mutex> lock(mtxLoop);
        if (loopEdges.empty() || cloudKeyPoses6D->points.empty())
            return;
        loop_edges = loopEdges;
        keyposes = cloudKeyPoses6D->points;
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

    RateType rate(loopClosureFrequency);
    int lastKey = -1;

    while (ros_ok() && !flg_exit)
    {
        rate.sleep();

        int key = -1;
        {
            std::lock_guard<std::recursive_mutex> lock(mtxLoop);
            if (cloudKeyPoses3D->size() < 6)
                continue;

            key = static_cast<int>(cloudKeyPoses3D->size()) - 5;
        }

        if (key == lastKey)
            continue;

        lastKey = key;
        performLoopClosure(key);
        visualizeLoopClosure();
    }
}

void gnssMatchingThread()
{
    if (!gnssEnableFlag)
        return;

    RateType rate(50);

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

            #ifdef USE_ROS1
                static tf::TransformBroadcaster br;
            #elif defined(USE_ROS2)
                static tf2_ros::TransformBroadcaster br(get_ros_node());
            #endif
                br.sendTransform(tf_msg);
            }
        }
    }
}

void publishGlobalMap() {
    if (ros_subscription_count(pubLaserCloudGlobal) == 0)
        return;

    if (cloudKeyPoses3D->points.empty())
        return;

    pcl::PointCloud<PointTypeIndex>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointTypeIndex>());

    // ikd-tree to find near key frames to visualize
    KD_TREE_PUBLIC<PointTypeIndex>::PointVector globalMapSearchPoses3D;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    {
        std::lock_guard<std::recursive_mutex> lock(mtxLoop);
        if (poseTreeDirty)
        {
            poseTreeDirty = false;
            ReconstructIkdTree();
        }
        ikdtreeHistoryKeyPoses->Radius_Search(
            cloudKeyPoses3D->back(),
            globalMapVisualizationSearchRadius,
            globalMapSearchPoses3D);
    }

    for (int i = 0; i < (int)globalMapSearchPoses3D.size(); ++i)
        globalMapKeyPoses->push_back(cloudKeyPoses3D->points[globalMapSearchPoses3D[i].intensity]); // index stored in intensity field
    // downsample near selected key frames
    pcl::VoxelGrid<PointTypeIndex> downSizeFilterGlobalMapKeyPoses; // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    for(auto& pt : globalMapKeyPosesDS->points)
    {
        ikdtreeHistoryKeyPoses->Nearest_Search(pt, 1, globalMapSearchPoses3D, pointSearchSqDisGlobalMap);
        pt.intensity = cloudKeyPoses3D->points[globalMapSearchPoses3D[0].intensity].intensity;
    }

    // extract visualized and downsampled key frames
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
        if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
            continue;
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
        *globalMapKeyFrames += *transformPointCloud(featCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
    }
    // downsample visualized points
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
    }
}
