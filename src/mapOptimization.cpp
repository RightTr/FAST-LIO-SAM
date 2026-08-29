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
std::atomic<bool> sceneDone{false};
std::atomic<bool> loopDone{false};

bool graphUpdate = false;
bool loopIsClosed = false;
std::deque<SceneBatch> sceneQueue;
std::deque<gtsam::NonlinearFactor::shared_ptr> gnssFactorQueue;
int gnssPosKey = 0;
int gnssYawKey = 0;
std::unordered_set<int> loopUsedKeys;
std::vector<std::pair<int, int>> loopEdges;
std::deque<LoopFactor> loopQueue;
Eigen::Vector3d last_fpos = Eigen::Vector3d::Zero();
bool has_fpos = false;

static double yawToMap(double yaw)
{
    const Eigen::Vector3d heading_enu(std::cos(yaw), std::sin(yaw), 0.0);
    const Eigen::Vector3d heading_map = R_enu_map.transpose() * heading_enu;
    return normalizeYaw(std::atan2(heading_map.y(), heading_map.x()));
}

void correctPoses();
void addSceneFactor();
void addGNSSFactor();
void performGnssMatching();
void sceneMatchingThread();
void gnssMatchingThread();

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
    return transformPointCloud(
        cloudIn,
        pcl::getTransformation(
            transformIn->x,
            transformIn->y,
            transformIn->z,
            transformIn->roll,
            transformIn->pitch,
            transformIn->yaw));
}

pcl::PointCloud<PointTypeIndex>::Ptr transformPointCloud(pcl::PointCloud<PointTypeIndex>::Ptr cloudIn, const Eigen::Affine3f &transCur)
{
    pcl::PointCloud<PointTypeIndex>::Ptr cloudOut(new pcl::PointCloud<PointTypeIndex>());

    const int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);
    
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
    sceneDone.store(!groundEnableFlag);
    loopDone.store(!loopClosureEnableFlag);
    gnss_aligned.store(false);
    R_enu_map = Eigen::Matrix3d::Identity();
    t_enu_map = Eigen::Vector3d::Zero();
    last_fpos.setZero();
    has_fpos = false;
    gnssPosKey = 0;
    gnssYawKey = 0;
    {
        std::lock_guard<std::mutex> lock(mtxGnssFactor);
        gnssFactorQueue.clear();
    }

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    parameters.factorization = ISAM2Params::QR;
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

void performLoopClosure(int loopKeyCur,
                        const std::vector<PointTypePose> &poses6D,
                        const std::vector<PointTypePose> &odomPoses6D,
                        const std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> &clouds)
{
    if (loopUsedKeys.count(loopKeyCur) != 0)
        return;

    std::vector<FloorRange> floorRanges{{0, loopKeyCur}};
    if (sceneEnableFlag && !floorMap.getRanges(loopKeyCur, floorRanges))
        return;

    pcl::PointCloud<PointTypeIndex>::Ptr poseCloud(
        new pcl::PointCloud<PointTypeIndex>());
    for (const auto &range : floorRanges)
    {
        const int begin = std::max(0, range.begin);
        const int end = std::min(
            range.end < 0 ? loopKeyCur : range.end,
            loopKeyCur);

        for (int key = begin; key <= end; ++key)
        {
            PointTypeIndex pt;
            pt.x = poses6D[key].x;
            pt.y = poses6D[key].y;
            pt.z = 0.0f;
            pt.intensity = static_cast<float>(key);
            poseCloud->push_back(pt);
        }
    }

    if (poseCloud->size() < 6)
        return;

    poseCloud->width = static_cast<uint32_t>(poseCloud->points.size());
    poseCloud->height = 1;

    pcl::KdTreeFLANN<PointTypeIndex> poseTree;
    poseTree.setInputCloud(poseCloud);

    std::vector<int> indices;
    std::vector<float> sqDists;
    if (poseTree.radiusSearch(
            poseCloud->points.back(),
            historyKeyframeSearchRadius,
            indices,
            sqDists) == 0)
        return;

    const pcl::PointCloud<PointTypeIndex>::Ptr &curCloud = clouds[loopKeyCur];
    if (!curCloud || curCloud->empty())
        return;

    const PointTypePose &curOdomPose = odomPoses6D[loopKeyCur];
    pcl::PointCloud<PointTypeIndex>::Ptr cureKeyframeCloud(
        new pcl::PointCloud<PointTypeIndex>());
    *cureKeyframeCloud = *curCloud;

    pcl::PointCloud<PointTypeIndex>::Ptr cureKeyframeCloudDS(
        new pcl::PointCloud<PointTypeIndex>());
    downSizeFilterICP.setInputCloud(cureKeyframeCloud);
    downSizeFilterICP.filter(*cureKeyframeCloudDS);
    if (cureKeyframeCloudDS->size() < 300)
        return;

    const double current_time = poses6D[loopKeyCur].time;
    int preCount = 0;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        const int id = static_cast<int>(poseCloud->points[indices[i]].intensity);
        if (id >= loopKeyCur)
            continue;
        if (loopUsedKeys.count(id) != 0)
            continue;
        if (current_time - poses6D[id].time <= historyKeyframeSearchTimeDiff)
            continue;
        if (preCount++ >= loopPreNum)
            break;

        int historyBegin = std::max(0, id - historyKeyframeSearchNum);
        int historyEnd = std::min(id + historyKeyframeSearchNum, loopKeyCur - 1);

        for (const auto &range : floorRanges)
        {
            if (id >= range.begin &&
                (range.end < 0 || id <= range.end))
            {
                historyBegin = std::max(historyBegin, range.begin);
                if (range.end >= 0)
                    historyEnd = std::min(historyEnd, range.end);
                break;
            }
        }

        if (historyBegin > historyEnd)
            continue;

        pcl::PointCloud<PointTypeIndex>::Ptr prevKeyframeCloud(
            new pcl::PointCloud<PointTypeIndex>());
        const Eigen::Affine3f T_odom_pre = pclPointToAffine3f(odomPoses6D[id]);
        const Eigen::Affine3f T_pre_odom = T_odom_pre.inverse();
        for (int keyNear = historyBegin; keyNear <= historyEnd; ++keyNear)
        {
            if (!clouds[keyNear] || clouds[keyNear]->empty())
                continue;
            const Eigen::Affine3f T_odom_k = pclPointToAffine3f(odomPoses6D[keyNear]);
            const Eigen::Affine3f T_pre_k = T_pre_odom * T_odom_k;
            *prevKeyframeCloud += *transformPointCloud(clouds[keyNear], T_pre_k);
        }
        pcl::PointCloud<PointTypeIndex>::Ptr prevKeyframeCloudDS(
            new pcl::PointCloud<PointTypeIndex>());
        downSizeFilterICP.setInputCloud(prevKeyframeCloud);
        downSizeFilterICP.filter(*prevKeyframeCloudDS);

        if (prevKeyframeCloudDS->size() < 1000)
            continue;

        ROS_PRINT_INFO(
            "[LOOP] ICP start cur=%d pre=%d src=%zu target=%zu",
            loopKeyCur,
            id,
            cureKeyframeCloudDS->size(),
            prevKeyframeCloudDS->size());

        static pcl::IterativeClosestPoint<PointTypeIndex, PointTypeIndex> icp;
        icp.setMaxCorrespondenceDistance(icpMaxCorrDistance);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        icp.setInputSource(cureKeyframeCloudDS);
        icp.setInputTarget(prevKeyframeCloudDS);
        pcl::PointCloud<PointTypeIndex> aligned;
        const Eigen::Affine3f T_odom_cur = pclPointToAffine3f(curOdomPose);
        const Eigen::Affine3f T_pre_cur_guess = T_pre_odom * T_odom_cur;
        icp.align(aligned, T_pre_cur_guess.matrix());

        const double score = icp.getFitnessScore();
        if (!icp.hasConverged() ||
            score > historyKeyframeFitnessScore)
        {
            ROS_PRINT_INFO(
                "[LOOP] ICP reject cur=%d pre=%d score=%.3f",
                loopKeyCur,
                id,
                score);
            continue;
        }

        const Eigen::Affine3f T_pre_cur(icp.getFinalTransformation());
        const Eigen::Affine3f T_cur_pre = T_pre_cur.inverse();
        float pre_x, pre_y, pre_z, pre_roll, pre_pitch, pre_yaw;
        pcl::getTranslationAndEulerAngles(T_pre_cur, pre_x, pre_y, pre_z, pre_roll, pre_pitch, pre_yaw);

        float cur_x, cur_y, cur_z, cur_roll, cur_pitch, cur_yaw;
        pcl::getTranslationAndEulerAngles(T_cur_pre, cur_x, cur_y, cur_z, cur_roll, cur_pitch, cur_yaw);
        gtsam::Vector Vector6(6);
        const double noiseScore = std::max(
            static_cast<double>(score) * loopWeight, 1e-4);
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        loopUsedKeys.insert(loopKeyCur);
        loopUsedKeys.insert(id);

        {
            std::lock_guard<std::mutex> lock(mtxLoopFactor);
            loopQueue.push_back(LoopFactor{
                loopKeyCur,
                id,
                Pose3(
                    Rot3::RzRyRx(cur_roll, cur_pitch, cur_yaw),
                    Point3(cur_x, cur_y, cur_z)),
                constraintNoise});
        }

        ROS_PRINT_INFO(
            "[LOOP] ICP pass cur=%d pre=%d score=%.3f",
            loopKeyCur,
            id,
            score);
        break;
    }
}

void addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        noiseModel::Diagonal::shared_ptr priorNoise = 
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
        noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6).finished());
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

    std::deque<LoopFactor> factors;

    {
        std::lock_guard<std::mutex> lock(mtxLoopFactor);
        if (loopQueue.empty())
            return;

        factors.swap(loopQueue);
    }

    std::vector<std::pair<int, int>> addedEdges;
    addedEdges.reserve(factors.size());

    for (const auto &factor : factors)
    {
        gtSAMgraph.add(BetweenFactor<Pose3>(factor.from, factor.to, factor.pose, factor.noise));
        addedEdges.emplace_back(factor.from, factor.to);
    }

    {
        std::lock_guard<std::mutex> lock(mtxLoopFactor);
        for (const auto &edge : addedEdges)
        {
            loopEdges.emplace_back(edge.first, edge.second);
        }
    }

    loopIsClosed = true;
    graphUpdate = true;
}

void poseGraphUpdate()
{
    addGNSSFactor();
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
    if (!groundEnableFlag)
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

        for (const auto &factor : batch.factors)
        {
            const gtsam::Key pkey = gtsam::Symbol('p', factor.id);
            if (!initialEstimate.exists(pkey) &&
                !isamCurrentEstimate.exists(pkey))
            {
                if (batch.planes.empty() ||
                    !isamCurrentEstimate.exists(static_cast<gtsam::Key>(factor.key)))
                    continue;

                Eigen::Vector4d body_plane = factor.obs;
                const double body_norm = body_plane.head<3>().norm();
                if (!body_plane.allFinite() || body_norm <= 1e-6)
                {
                    continue;
                }
                body_plane /= body_norm;

                const gtsam::Pose3 pose_map =
                    isamCurrentEstimate.at<Pose3>(
                        static_cast<gtsam::Key>(factor.key));
                const Eigen::Matrix3d R = pose_map.rotation().matrix();
                const Eigen::Vector3d t(
                    pose_map.translation().x(),
                    pose_map.translation().y(),
                    pose_map.translation().z());

                Eigen::Vector3d normal = R * body_plane.head<3>();
                double d = body_plane[3] - normal.dot(t);
                Eigen::Vector4d map_plane;
                map_plane << normal.x(), normal.y(), normal.z(), d;

                const Eigen::Vector3d up = getGravityUp();
                if (map_plane.head<3>().dot(up) < 0.0)
                    map_plane = -map_plane;

                initialEstimate.insert(pkey, gtsam::OrientedPlane3(map_plane));

                const int floor_id = sceneEnableFlag ? batch.planes.front().floor : 0;
                const gtsam::Key fkey = gtsam::Symbol('f', floor_id);

                if (!initialEstimate.exists(fkey) &&
                    !isamCurrentEstimate.exists(fkey))
                {
                    initialEstimate.insert(fkey, floor_up);
                    gtSAMgraph.add(
                        gtsam::PriorFactor<gtsam::Unit3>(fkey, floor_up, floorNormalPriorNoise));
                }

                gtSAMgraph.add(
                    boost::shared_ptr<FloorFactor>(
                        new FloorFactor(fkey, pkey, floorPlaneNoise)));

                ROS_PRINT_INFO(
                    "[GROUND GRAPH] NEW p%d floor=%d key=%d body=(%.3f %.3f %.3f %.3f) map=(%.3f %.3f %.3f %.3f)",
                    factor.id, floor_id, factor.key,
                    body_plane[0], body_plane[1], body_plane[2], body_plane[3],
                    map_plane[0], map_plane[1], map_plane[2], map_plane[3]);
            }

            gtSAMgraph.add(
                gtsam::OrientedPlane3Factor(
                    factor.obs, planeNoise, static_cast<gtsam::Key>(factor.key), pkey));
        }
        graphUpdate = true;
    }
}

void sceneMatchingThread()
{
    if (!groundEnableFlag)
        return;

    RateType rate(20);
    int nextKey = 0;

    while (ros_ok() && !flg_exit)
    {
        rate.sleep();

        pcl::PointCloud<PointTypeIndex>::Ptr cloud;
        PointTypePose pose;
        int readyKeyNum = 0;

        {
            std::lock_guard<std::mutex> lock(mtxKeyframe);
            readyKeyNum = static_cast<int>(featCloudKeyFrames.size());
            if (nextKey >= readyKeyNum)
            {
                sceneDone.store(true);
                continue;
            }

            cloud = featCloudKeyFrames[nextKey];
            pose = cloudKeyOdomPoses6D->points[nextKey];
        }

        sceneDone.store(false);

        plane.update(nextKey, cloud, pose);

        if (nextKey && nextKey % groundInterval == 0)
        {
            std::vector<PlaneObs> planes;
            plane.extract(planes);

            SceneBatch batch;
            if (floorMap.updateGround(
                    planes,
                    nextKey,
                    pose,
                    batch))
            {
                std::lock_guard<std::mutex> batchLock(mtxSceneBatch);
                sceneQueue.push_back(std::move(batch));
            }
        }

        ++nextKey;
        sceneDone.store(nextKey >= readyKeyNum);
    }

    sceneDone.store(true);
}

void performGnssMatching()
{
    if (!gnssEnableFlag || !gnss_aligned.load() || !p_gnss)
        return;

    std::vector<PointTypePose> keyposes;

    {
        std::lock_guard<std::mutex> lock(mtxKeyframe);
        if (!cloudKeyPoses6D || cloudKeyPoses6D->points.empty())
            return;

        keyposes.assign(cloudKeyPoses6D->points.begin(), cloudKeyPoses6D->points.end());
    }

    while (gnssPosKey < static_cast<int>(keyposes.size()))
    {
        PosData pos;
        if (!p_gnss->matchPos(keyposes[gnssPosKey].time, pos))
            break;

        const int key = gnssPosKey++;
        const PointTypePose &pose = keyposes[key];
        const Eigen::Matrix3d R = poseRotation(pose);
        const Eigen::Vector3d p_map_ant = R_enu_map.transpose() * (pos.p - t_enu_map);
        Eigen::Vector3d gnss_pos = p_map_ant - R * p_gnss->lever();
        Eigen::Matrix3d gnss_cov = pos.cov;
        if (!useGnssElevation)
        {
            gnss_pos.z() = pose.z;
            gnss_cov(2, 2) = 100.0;
        }

        if (!has_fpos || (gnss_pos - last_fpos).norm() >= gpsFactorMinDis)
        {
            gtsam::Vector3 sigma;
            sigma << std::sqrt(gnss_cov(0, 0)),
                     std::sqrt(gnss_cov(1, 1)),
                     std::sqrt(gnss_cov(2, 2));
            const gtsam::NonlinearFactor::shared_ptr factor(
                new gtsam::GPSFactor(
                    key,
                    gtsam::Point3(gnss_pos.x(), gnss_pos.y(), gnss_pos.z()),
                    gtsam::noiseModel::Diagonal::Sigmas(sigma)));
            {
                std::lock_guard<std::mutex> lock(mtxGnssFactor);
                gnssFactorQueue.push_back(factor);
            }

            last_fpos = gnss_pos;
            has_fpos = true;
        }
    }

    if (!useGnssYawFactor)
        return;

    while (gnssYawKey < static_cast<int>(keyposes.size()))
    {
        YawData yaw;
        if (!p_gnss->matchYaw(keyposes[gnssYawKey].time, yaw))
            break;

        const int key = gnssYawKey++;
        const PointTypePose &pose = keyposes[key];
        const double yaw_map = yawToMap(yaw.yaw);
        const double yaw_error = std::abs(normalizeYaw(yaw_map - pose.yaw));
        if (yaw_error <= 60.0 * M_PI / 180.0)
        {
            const double yaw_sigma = std::max(gnss_yaw_factor_sigma, 1e-4);
            const auto yawNoise = gtsam::noiseModel::Isotropic::Sigma(1, yaw_sigma);
            const gtsam::NonlinearFactor::shared_ptr factor(
                new GnssYawFactor(key, yaw_map, yawNoise));
            {
                std::lock_guard<std::mutex> lock(mtxGnssFactor);
                gnssFactorQueue.push_back(factor);
            }
        }
    }
}

void addGNSSFactor()
{
    if (!gnssEnableFlag)
        return;

    std::deque<gtsam::NonlinearFactor::shared_ptr> factors;

    {
        std::lock_guard<std::mutex> lock(mtxGnssFactor);
        if (gnssFactorQueue.empty())
            return;

        factors.swap(gnssFactorQueue);
    }

    for (const auto &factor : factors)
        gtSAMgraph.add(factor);
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
    }
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

    if (sceneEnableFlag)
    {
        floorMap.update(static_cast<int>(latestPoseKey), OdomPose);
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
        *cloudOut += *transformPointCloud(clouds.back(), &thisPose6D);
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

    RateType rate(loopClosureFrequency);
    int nextKey = 1;

    while (ros_ok() && !flg_exit)
    {
        rate.sleep();

        std::vector<PointTypePose> poses6D;
        std::vector<PointTypePose> odomPoses6D;
        std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> clouds;
        {
            std::lock_guard<std::mutex> lock(mtxKeyframe);
            if (cloudKeyPoses6D == nullptr ||
                cloudKeyOdomPoses6D == nullptr ||
                cloudKeyPoses6D->points.size() < 6 ||
                cloudKeyOdomPoses6D->points.size() < 6 ||
                featCloudKeyFrames.size() < 6)
            {
                loopDone.store(true);
                continue;
            }

            poses6D.assign(cloudKeyPoses6D->points.begin(), cloudKeyPoses6D->points.end());
            odomPoses6D.assign(cloudKeyOdomPoses6D->points.begin(), cloudKeyOdomPoses6D->points.end());
            clouds = featCloudKeyFrames;
        }

        const int readyKey = static_cast<int>(poses6D.size()) - 1;

        if (nextKey > readyKey)
        {
            loopDone.store(true);
            continue;
        }

        loopDone.store(false);

        if (loopCurNum <= 0)
        {
            loopDone.store(true);
            continue;
        }

        nextKey = std::max(nextKey, readyKey - loopCurNum + 1);
        ROS_PRINT_INFO(
            "[LOOP] range next=%d ready=%d latest=%zu",
            nextKey,
            readyKey,
            poses6D.size() - 1);

        for (int processed = 0; processed < loopCurNum && nextKey <= readyKey; )
        {
            performLoopClosure(nextKey, poses6D, odomPoses6D, clouds);

            ++nextKey;
            ++processed;
        }

        loopDone.store(nextKey > readyKey);
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
