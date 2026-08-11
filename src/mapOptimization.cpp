// Modified from LIO-SAM: MapOptimization.cpp

#include "utility.h"
#include "common_utils.h"
#include "GNSS_Processing.hpp"
#include "gnssYaw_factor.h"
#include "s-graph/groundMap.h"
#include "s-graph/planeMap.h"
#include "map_optimization.h"
#include "ros_utils.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <limits>
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

#include "ikd-Tree/ikdtree_public.h"

using namespace gtsam;
using namespace std;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

extern std::string root_dir;

// gtsam
NonlinearFactorGraph gtSAMgraph;
Values initialEstimate;
ISAM2 *isam;
Values isamCurrentEstimate;
Eigen::MatrixXd poseCovariance;

Pcl2Publisher pubKeyPoses;
PathPublisher pubPath;

Pcl2Publisher pubLaserCloudGlobal;
Pcl2Publisher pubLaserCloudLocal;
Pcl2Publisher pubPlanes;

Pcl2Publisher pubHistoryKeyFrames;
Pcl2Publisher pubIcpKeyFrames;
Pcl2Publisher pubRecentKeyFrame;
Pcl2Publisher pubCloudRegisteredRaw;
MarkerArrayPublisher pubLoopConstraintEdge;
MarkerArrayPublisher pubKeyFrameYawMarkers;

TimeType timeLaserInfoStamp;

pcl::PointCloud<PointTypeIndex>::Ptr cloudKeyPoses3D; // Store keyframe poses and indexes 
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
pcl::PointCloud<PointTypePose>::Ptr cloudKeyOdomPoses6D;
pcl::PointCloud<PointTypeIndex>::Ptr copy_cloudKeyPoses3D;
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

double timeLaserInfoCur;

float transformTobeMapped[6];
Eigen::Vector3d translationLidarToIMU;
Eigen::Matrix3d rotationLidarToIMU;

bool isDegenerate = false;

PathMsg globalPath;
PlaneMap planeMap;
GroundMap groundMap;

PlaneBatch planeBatch;
gtsam::noiseModel::Diagonal::shared_ptr planeNoise;
std::mutex mtxPlane;

std::mutex mtx;
std::mutex mtxLoopInfo;
std::mutex mtxGnssFactor;
std::atomic<bool> poseDirty{false};

bool graphUpdate = false;
bool loopIsClosed = false;
map<int, int> loopIndexContainer; // from new to old
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
static void performStructureMatching();
static void AddPlaneFactor();

pcl::VoxelGrid<PointTypeIndex> downSizeFilterICP;

vector<pcl::PointCloud<PointTypeIndex>::Ptr> featCloudKeyFrames;

KD_TREE_PUBLIC<PointTypeIndex>::Ptr ikdtreeHistoryKeyPoses;

KD_TREE_PUBLIC<PointTypeIndex>::PointVector initPoses3D;

map<int, pair<pcl::PointCloud<PointTypeIndex>, pcl::PointCloud<PointTypeIndex>>> laserCloudMapContainer;

Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
{ 
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

Eigen::Affine3f trans2Affine3f(float transformIn[])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
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

float pointDistance(PointTypeIndex p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

float pointDistance(PointTypeIndex p1, PointTypeIndex p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

float rotationDistance(const gtsam::Pose3& poseFrom, const gtsam::Pose3& poseTo)
{
    const Eigen::Matrix3d deltaR = poseFrom.between(poseTo).rotation().matrix();
    double cosTheta = (deltaR.trace() - 1.0) * 0.5;
    cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
    return static_cast<float>(std::acos(cosTheta));
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

pcl::PointCloud<PointTypeIndex>::Ptr transformPointCloud(pcl::PointCloud<PointTypeIndex>::Ptr cloudIn, PointTypePose* transformIn)
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
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointTypeIndex>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

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

    init_ros_node();
    
    pubKeyPoses = create_publisher<PointCloud2Msg>("lio_sam/trajectory", 1);
    pubPath = create_publisher<PathMsg>("lio_sam/mapping/path", 1);
    pubLaserCloudGlobal = create_publisher<PointCloud2Msg>("lio_sam/mapping/cloud_global", 1);
    pubPlanes = create_publisher<PointCloud2Msg>("lio_sam/mapping/planes", 1);
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

void loopFindNearKeyframes(pcl::PointCloud<PointTypeIndex>::Ptr& nearKeyframes, const int& key, const int& searchNum)
{
    // extract near keyframes
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize )
            continue;
        *nearKeyframes += *transformPointCloud(featCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframes
    pcl::PointCloud<PointTypeIndex>::Ptr cloud_temp(new pcl::PointCloud<PointTypeIndex>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}

bool detectLoopClosureDistance(int *latestID, int *closestID)
{
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;

    // check loop constraint added before
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;

    if (ikdtreeHistoryKeyPoses->Root_Node == nullptr)
        return false;

    // Find the closest history key frame in XY only.
    const auto &current_pose = copy_cloudKeyPoses3D->back();
    const double max_dist_sq = historyKeyframeSearchRadius * historyKeyframeSearchRadius;
    double best_dist_sq = std::numeric_limits<double>::infinity();
    for (int id = 0; id < static_cast<int>(copy_cloudKeyPoses3D->size()); ++id)
    {
        if (id == loopKeyCur)
            continue;

        const auto &candidate_pose = copy_cloudKeyPoses3D->points[id];
        const double dx = candidate_pose.x - current_pose.x;
        const double dy = candidate_pose.y - current_pose.y;
        const double dist_sq = dx * dx + dy * dy;
        if (dist_sq > max_dist_sq || dist_sq >= best_dist_sq)
            continue;

        if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
        {
            const gtsam::Pose3 poseCur = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyCur]);
            const gtsam::Pose3 posePre = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[id]);
            if (rotationDistance(poseCur, posePre) > historyKeyframeSearchAngleThreshold)
                continue;

            loopKeyPre = id;
            best_dist_sq = dist_sq;
        }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
}

void performLoopClosure()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // find keys
    int loopKeyCur;
    int loopKeyPre;
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false) return;

    // extract cloud
    pcl::PointCloud<PointTypeIndex>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointTypeIndex>());
    pcl::PointCloud<PointTypeIndex>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointTypeIndex>());
    {
        // cloud near latest keyframe 
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
        // cloud near previous loop keyframe
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            return;
        // if (pubHistoryKeyFrames.getNumSubscribers() != 0)
        //     publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
    }

    // ICP Settings
    static pcl::IterativeClosestPoint<PointTypeIndex, PointTypeIndex> icp;
    icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
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
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    const double noiseScore = std::max(
        static_cast<double>(icp.getFitnessScore()) * loopWeight, 1e-4);
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    // add loop constriant
    loopIndexContainer[loopKeyCur] = loopKeyPre;
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
    if (loopIndexQueue.empty())
        return;

    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
        int indexFrom = loopIndexQueue[i].first;
        int indexTo = loopIndexQueue[i].second;
        gtsam::Pose3 poseBetween = loopPoseQueue[i];
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
        gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    graphUpdate = true;
    loopIsClosed = true;
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
        std::lock_guard<std::mutex> lock(mtx);
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
    // loop factor
    addLoopFactor();
    
    AddPlaneFactor();

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
        std::lock_guard<std::mutex> lock(mtx);

        cloudKeyPoses3D->push_back(thisPose3D);
        cloudKeyPoses6D->push_back(thisPose6D);
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

        if (ikdtreeHistoryKeyPoses->Root_Node == nullptr) {
            initPoses3D.push_back(thisPose3D);
            if (cloudKeyPoses3D->points.size() >= 10)
                ikdtreeHistoryKeyPoses->Build(initPoses3D);
        } else {
            ikdtreeHistoryKeyPoses->Add_Point(thisPose3D);
        }
    }

}

void AddPlaneFactor()
{
    PlaneBatch batch;

    {
        std::lock_guard<std::mutex> lock(mtxPlane);
        if (!planeBatch.valid)
            return;
        batch = std::move(planeBatch);
        planeBatch = PlaneBatch();
    }

    if (!planeNoise)
        return;

    for (const auto &plane : batch.init)
    {
        const gtsam::Key pkey = gtsam::Symbol('p', plane.id);
        initialEstimate.insert(pkey, gtsam::OrientedPlane3(plane.plane));
    }

    for (const auto &factor : batch.factors)
    {
        const gtsam::Key pose_key = static_cast<gtsam::Key>(factor.key);
        const gtsam::Key pkey = gtsam::Symbol('p', factor.id);
        gtSAMgraph.add(gtsam::OrientedPlane3Factor(
            factor.obs, planeNoise,
            pose_key, pkey));
    }

    graphUpdate = true;
}

void performStructureMatching()
{
    if (!groundEnableFlag)
        return;

    static int lastProcessedKey = -1;

    int numKeyFrame = 0;
    std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> clouds;
    std::vector<PointTypePose> poses;

    {
        std::lock_guard<std::mutex> lock(mtx);

        if (!cloudKeyPoses6D)
            return;

        numKeyFrame = static_cast<int>(
            std::min(featCloudKeyFrames.size(), cloudKeyPoses6D->points.size()));

        clouds.assign(featCloudKeyFrames.begin(),
                      featCloudKeyFrames.begin() + numKeyFrame);
        poses.assign(cloudKeyPoses6D->points.begin(),
                     cloudKeyPoses6D->points.begin() + numKeyFrame);
    }

    if (poseDirty.exchange(false) && lastProcessedKey >= 0)
    {
        planeMap.reset();

        const int begin = std::max(0, lastProcessedKey - windowSize + 1);

        for (int key = begin; key <= lastProcessedKey; ++key)
        {
            if (!clouds[key])
                continue;

            std::vector<PlaneObs> unused;
            planeMap.update(key, clouds[key], poses[key], unused, nullptr);
        }
    }

    for (int key = lastProcessedKey + 1; key < numKeyFrame; ++key)
    {
        if (!clouds[key])
        {
            lastProcessedKey = key;
            continue;
        }

        std::vector<PlaneObs> plane_obs;
        pcl::PointCloud<PointTypeIndex>::Ptr plane_cloud;

        plane_cloud = std::make_shared<pcl::PointCloud<PointTypeIndex>>();

        planeMap.update(key, clouds[key], poses[key], plane_obs, plane_cloud);
        lastProcessedKey = key;

        publishCloud(pubPlanes, plane_cloud, timeLaserInfoStamp, map_frame);

        PlaneBatch batch;
        if (!groundMap.update(plane_obs, planeMap.keys(), planeMap.poses(), batch))
            continue;

        std::lock_guard<std::mutex> lock(mtxPlane);

        if (!planeBatch.valid)
        {
            planeBatch = std::move(batch);
            continue;
        }

        planeBatch.init.insert(
            planeBatch.init.end(),
            std::make_move_iterator(batch.init.begin()),
            std::make_move_iterator(batch.init.end()));

        planeBatch.factors.insert(
            planeBatch.factors.end(),
            std::make_move_iterator(batch.factors.begin()),
            std::make_move_iterator(batch.factors.end()));

        planeBatch.valid = !planeBatch.init.empty() || !planeBatch.factors.empty();
    }
}

void structureMatchingThread()
{
    ROS_PRINT_INFO("...... Structure Matching Thread Start......");

    RateType rate(20);
    while (ros_ok() && !flg_exit)
    {
        rate.sleep();
        performStructureMatching();
    }
}

void ReconstructIkdTree()
{
    if (ikdtreeHistoryKeyPoses->Root_Node == nullptr)
        return;
    if (cloudKeyPoses3D->points.empty())
        return;

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
        std::lock_guard<std::mutex> lock(mtx);

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

        if (loopIsClosed)
        {
            ReconstructIkdTree();
            poseDirty.store(true, std::memory_order_release);
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
    if (loopIndexContainer.empty())
        return;
    
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

    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        PointMsg p;
        p.x = copy_cloudKeyPoses6D->points[key_cur].x;
        p.y = copy_cloudKeyPoses6D->points[key_cur].y;
        p.z = copy_cloudKeyPoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = copy_cloudKeyPoses6D->points[key_pre].x;
        p.y = copy_cloudKeyPoses6D->points[key_pre].y;
        p.z = copy_cloudKeyPoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    ros_publish(pubLoopConstraintEdge, markerArray);
}

void loopClosureThread()
{
    if (loopClosureEnableFlag == false)
        return;

    ROS_PRINT_INFO("...... Loop Closure Thread Start......");

    RateType rate(loopClosureFrequency);
    while (ros_ok() && !flg_exit)
    {
        rate.sleep();
        performLoopClosure();
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
    mtx.lock();
    ikdtreeHistoryKeyPoses->Radius_Search(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, globalMapSearchPoses3D);
    mtx.unlock();

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
