// Modified from LIO-SAM: MapOptimization.cpp

#include "utility.h"
#include "common_utils.h"
#include "GNSS_Processing.hpp"
#include "factor/gnssYaw_factor.h"
#include "factor/gnssXY_factor.h"
#include "s-graph/floorMap.h"
#include "scan_context/Scancontext.h"
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
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/AttitudeFactor.h>
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
PathPublisher pubGlobalPath;

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
std::vector<PointTypePose> cloudKeyBAPoses6D;

PathMsg globalPath;
FloorMap floorMap;
SCManager sc_manager;
std::mutex gravityAxisMutex;
Eigen::Vector3d gravityUpAxis = Eigen::Vector3d::UnitZ();
Eigen::Vector3d gravityUpMapAxis = Eigen::Vector3d::UnitZ();
bool gravityUpMapInitialized = false;
size_t gravityFactorCount = 0;

std::mutex mtxKeyframe;
std::mutex mtxLoopFactor;
std::atomic<bool> loopDone{false};

bool graphUpdate = false;
bool loopIsClosed = false;
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
}

void MapOptimizationInit()
{
    cloudKeyBAPoses6D.clear();
    loopDone.store(!loopClosureEnableFlag);
    gnss_aligned.store(false);
    R_enu_map = Eigen::Matrix3d::Identity();
    t_enu_map = Eigen::Vector3d::Zero();
    last_fpos.setZero();
    has_fpos = false;
    gravityUpMapAxis = Eigen::Vector3d::UnitZ();
    gravityUpMapInitialized = false;
    gravityFactorCount = 0;

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    parameters.factorization = ISAM2Params::QR;
    isam = new ISAM2(parameters);

    setGravityUp(Eigen::Vector3d::UnitZ());

    init_ros_node();
    
    pubKeyPoses = create_publisher<PointCloud2Msg>("lio_sam/trajectory", 1);
    pubPath = create_publisher<PathMsg>("lio_sam/mapping/path", 1);
    pubGlobalPath = create_publisher<PathMsg>("/global_path", 1);
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

std::vector<std::pair<int, int>> getLoopSearchRanges(
    int loopKeyCur,
    const std::vector<FloorRange> &floorRanges)
{
    std::vector<std::pair<int, int>> ranges;
    ranges.reserve(floorRanges.size());

    for (const auto &range : floorRanges)
    {
        const int begin = std::max(0, range.begin);
        const int end = std::min(
            range.end < 0 ? loopKeyCur - 1 : range.end,
            loopKeyCur - 1);

        if (begin <= end)
            ranges.emplace_back(begin, end);
    }

    return ranges;
}

std::vector<std::pair<int, float>> detectLoopKeys(
    int loopKeyCur,
    const std::vector<PointTypePose> &poses6D,
    const std::vector<FloorRange> &floorRanges)
{
    const auto searchRanges = getLoopSearchRanges(loopKeyCur, floorRanges);

    pcl::PointCloud<PointTypeIndex>::Ptr poseCloud(
        new pcl::PointCloud<PointTypeIndex>());
    for (const auto &[begin, end] : searchRanges)
    {
        for (int key = begin; key <= end; ++key)
        {
            PointTypeIndex pt;
            pt.x = poses6D[key].x;
            pt.y = poses6D[key].y;
            pt.z = 0.0f;
            pt.intensity = key;
            poseCloud->push_back(pt);
        }
    }

    const float no_sc_yaw = std::numeric_limits<float>::quiet_NaN();
    std::vector<std::pair<int, float>> loopKeys;

    if (!poseCloud->empty())
    {
        poseCloud->width = static_cast<uint32_t>(poseCloud->points.size());
        poseCloud->height = 1;

        pcl::KdTreeFLANN<PointTypeIndex> poseTree;
        poseTree.setInputCloud(poseCloud);

        std::vector<int> indices;
        std::vector<float> sqDists;
        PointTypeIndex queryPoint;
        queryPoint.x = poses6D[loopKeyCur].x;
        queryPoint.y = poses6D[loopKeyCur].y;
        queryPoint.z = 0.0f;
        poseTree.radiusSearch(
            queryPoint,
            historyKeyframeSearchRadius,
            indices,
            sqDists);

        loopKeys.reserve(indices.size() + 1);
        for (const int index : indices)
        {
            loopKeys.emplace_back(
                static_cast<int>(poseCloud->points[index].intensity),
                no_sc_yaw);
        }
    }

    if (loopClosureEnableFlag && scanContextEnableFlag)
    {
        std::pair<int, float> sc_result{-1, 0.0f};
        {
            std::lock_guard<std::mutex> lock(mtxKeyframe);
            sc_result = sc_manager.detectLoopClosureID(
                loopKeyCur, searchRanges);
        }

        if (sc_result.first >= 0)
        {
            const bool duplicate = std::any_of(
                loopKeys.begin(), loopKeys.end(),
                [&](const std::pair<int, float> &loopKey) {
                    return loopKey.first == sc_result.first;
                });
            if (!duplicate)
                loopKeys.push_back(sc_result);
        }
    }

    return loopKeys;
}

void performLoopICP(
    int loopKeyCur,
    const std::vector<std::pair<int, float>> &loopKeys,
    const std::vector<FloorRange> &floorRanges,
    const std::vector<PointTypePose> &poses6D,
    const std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> &clouds)
{
    const pcl::PointCloud<PointTypeIndex>::Ptr &curCloud = clouds[loopKeyCur];

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
    int count = 0;
    for (const auto &loopKey : loopKeys)
    {
        const int loopKeyPre = loopKey.first;
        const bool fromScanContext = std::isfinite(loopKey.second);
        if (loopUsedKeys.count(loopKeyPre) != 0)
            continue;
        if (current_time - poses6D[loopKeyPre].time <= historyKeyframeSearchTimeDiff)
            continue;

        const auto floor_range = std::find_if(
            floorRanges.begin(), floorRanges.end(),
            [&](const FloorRange &range) {
                return loopKeyPre >= range.begin &&
                       (range.end < 0 || loopKeyPre <= range.end);
            });
        if (floor_range == floorRanges.end())
            continue;

        if (!fromScanContext)
        {
            if (count >= loopPreNum)
                continue;
            ++count;
        }

        int historyBegin = std::max(0, loopKeyPre - historyKeyframeSearchNum);
        int historyEnd = std::min(
            loopKeyPre + historyKeyframeSearchNum, loopKeyCur - 1);
        historyBegin = std::max(historyBegin, floor_range->begin);
        if (floor_range->end >= 0)
            historyEnd = std::min(historyEnd, floor_range->end);

        pcl::PointCloud<PointTypeIndex>::Ptr prevKeyframeCloud(
            new pcl::PointCloud<PointTypeIndex>());
        const Eigen::Affine3f T_map_pre = pclPointToAffine3f(poses6D[loopKeyPre]);
        const Eigen::Affine3f T_pre_map = T_map_pre.inverse();
        for (int keyNear = historyBegin; keyNear <= historyEnd; ++keyNear)
        {
            const Eigen::Affine3f T_map_k = pclPointToAffine3f(poses6D[keyNear]);
            const Eigen::Affine3f T_pre_k = T_pre_map * T_map_k;
            *prevKeyframeCloud += *transformPointCloud(clouds[keyNear], T_pre_k);
        }
        pcl::PointCloud<PointTypeIndex>::Ptr prevKeyframeCloudDS(
            new pcl::PointCloud<PointTypeIndex>());
        downSizeFilterICP.setInputCloud(prevKeyframeCloud);
        downSizeFilterICP.filter(*prevKeyframeCloudDS);

        if (prevKeyframeCloudDS->size() < 1000)
            continue;

        // ROS_PRINT_INFO(
        //     "[LOOP] ICP start cur=%d pre=%d src=%zu target=%zu",
        //     loopKeyCur,
        //     loopKeyPre,
        //     cureKeyframeCloudDS->size(),
        //     prevKeyframeCloudDS->size());

        pcl::IterativeClosestPoint<PointTypeIndex, PointTypeIndex> icp;
        icp.setMaxCorrespondenceDistance(icpMaxCorrDistance);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        icp.setInputSource(cureKeyframeCloudDS);
        icp.setInputTarget(prevKeyframeCloudDS);
        pcl::PointCloud<PointTypeIndex> aligned;
        const Eigen::Affine3f T_map_cur = pclPointToAffine3f(poses6D[loopKeyCur]);
        const Eigen::Affine3f T_pre_cur_guess = T_pre_map * T_map_cur;
        icp.align(aligned, T_pre_cur_guess.matrix());

        const double score = icp.getFitnessScore();
        if (!icp.hasConverged() ||
            score > historyKeyframeFitnessScore)
        {
            // ROS_PRINT_INFO(
            //     "[LOOP] ICP reject cur=%d pre=%d score=%.3f",
            //     loopKeyCur,
            //     loopKeyPre,
            //     score);
            continue;
        }

        const Eigen::Affine3f T_pre_cur(icp.getFinalTransformation());
        const Eigen::Affine3f T_cur_pre = T_pre_cur.inverse();
        float pre_x, pre_y, pre_z, pre_roll, pre_pitch, pre_yaw;
        pcl::getTranslationAndEulerAngles(T_pre_cur, pre_x, pre_y, pre_z, pre_roll, pre_pitch, pre_yaw);

        float cur_x, cur_y, cur_z, cur_roll, cur_pitch, cur_yaw;
        pcl::getTranslationAndEulerAngles(T_cur_pre, cur_x, cur_y, cur_z, cur_roll, cur_pitch, cur_yaw);
        gtsam::Vector6 loopVar;
        loopVar << loopRotVar, loopRotVar, loopRotVar,
                   loopPosVar, loopPosVar, loopPosVar;
        noiseModel::Diagonal::shared_ptr constraintNoise =
            noiseModel::Diagonal::Variances(loopVar);

        loopUsedKeys.insert(loopKeyCur);
        loopUsedKeys.insert(loopKeyPre);

        {
            std::lock_guard<std::mutex> lock(mtxLoopFactor);
            loopQueue.push_back(LoopFactor{
                loopKeyCur,
                loopKeyPre,
                Pose3(
                    Rot3::RzRyRx(cur_roll, cur_pitch, cur_yaw),
                    Point3(cur_x, cur_y, cur_z)),
                constraintNoise});
        }

        break;
    }
}

void performLoopClosure(
    int loopKeyCur,
    const std::vector<PointTypePose> &poses6D,
    const std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> &clouds)
{
    if (loopUsedKeys.count(loopKeyCur) != 0)
        return;

    std::vector<FloorRange> floorRanges;
    if (sceneEnableFlag)
    {
        if (!floorMap.getRanges(loopKeyCur, floorRanges))
            return;
    }
    else
    {
        floorRanges = {{0, -1}};
    }

    const auto loopKeys = detectLoopKeys(loopKeyCur, poses6D, floorRanges);
    if (!loopKeys.empty())
        performLoopICP(loopKeyCur, loopKeys, floorRanges, poses6D, clouds);
}

void addOdomFactor(const PointTypePose &odomPose,
                  const PointTypePose &baPose)
{
    if (cloudKeyPoses3D->points.empty())
    {
        gtsam::Vector6 priorVariance;
        priorVariance << priorVar, priorVar, priorVar,
                         priorVar, priorVar, priorVar;
        noiseModel::Diagonal::shared_ptr priorNoise = 
            noiseModel::Diagonal::Variances(priorVariance);
        const gtsam::Pose3 poseTo = [&]() {
            const gtsam::Pose3 poseOdom = pclPointTogtsamPose3(baPose);
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
        const gtsam::Pose3 poseFromFront = pclPointTogtsamPose3(cloudKeyBAPoses6D.back());
        const gtsam::Pose3 poseToFront = pclPointTogtsamPose3(baPose);
        const gtsam::Pose3 odomDelta = poseFromFront.between(poseToFront);
        const gtsam::Pose3 poseFromOpt = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
        const gtsam::Pose3 poseToOpt = poseFromOpt.compose(odomDelta);
        gtsam::Vector6 odomVar;
        odomVar << odomRotVar, odomRotVar, odomRotVar,
                   odomPosVar, odomPosVar, odomPosVar;
        const auto odometryNoise = noiseModel::Diagonal::Variances(odomVar);
        gtSAMgraph.add(BetweenFactor<Pose3>(
            cloudKeyPoses3D->size()-1,
            cloudKeyPoses3D->size(),
            odomDelta,
            odometryNoise));
        initialEstimate.insert(cloudKeyPoses3D->size(), poseToOpt);
    }
}

void initGravityRef(const Eigen::Vector3d &gravityUpOdom)
{
    if (gravityUpMapInitialized)
        return;
    gravityUpMapAxis = (R_map_odom * gravityUpOdom).normalized();
    gravityUpMapInitialized = true;
}

void addGravityFactor(gtsam::Key key,
                      const Eigen::Vector3d &gravity_body,
                      const Eigen::Vector3d &gravity_up_odom,
                      double gravity_sigma)
{
    if (!gravityFactorEnable)
        return;

    std::fprintf(stderr, "GravityFactor key=%lu sigma=%.6f body=[%.3f %.3f %.3f] odom=[%.3f %.3f %.3f]\n",
                 static_cast<unsigned long>(key), gravity_sigma,
                 gravity_body.x(), gravity_body.y(), gravity_body.z(),
                 gravity_up_odom.x(), gravity_up_odom.y(), gravity_up_odom.z());

    std::fprintf(stderr, "GravityFactor key=%lu sigma=%.6f body=[%.3f %.3f %.3f] odom=[%.3f %.3f %.3f]\n",
                 static_cast<unsigned long>(key), gravity_sigma,
                 gravity_body.x(), gravity_body.y(), gravity_body.z(),
                 gravity_up_odom.x(), gravity_up_odom.y(), gravity_up_odom.z());

    initGravityRef(gravity_up_odom);
    const gtsam::Unit3 worldUp(gravityUpMapAxis);
    const gtsam::Unit3 bodyUp(gravity_body);
    const auto noise = gtsam::noiseModel::Isotropic::Sigma(2, gravity_sigma);
    gtSAMgraph.add(boost::shared_ptr<gtsam::Pose3AttitudeFactor>(
        new gtsam::Pose3AttitudeFactor(key, worldUp, noise, bodyUp)));
    ++gravityFactorCount;
    if (gravityFactorCount % 20 == 0)
        ROS_PRINT_INFO("GTSAM gravity factors added: %zu", gravityFactorCount);
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

void performGnssMatching()
{
    if (!gnssEnableFlag || !gnss_aligned.load())
        return;

    if (!cloudKeyPoses6D || cloudKeyPoses6D->empty())
        return;

    const auto &keyposes = cloudKeyPoses6D->points;

    while (true)
    {
        PosData pos;
        int key = -1;
        {
            std::lock_guard<std::mutex> lock(p_gnss->mtx_gnss);
            if (p_gnss->pos_buf.empty()) break;

            pos = p_gnss->pos_buf.front();
            key = find_gnss_key(keyposes, pos.t);
            if (key == -2) break;

            p_gnss->pos_buf.pop_front();
        }

        if (key < 0) continue;

        const PointTypePose &pose = keyposes[key];
        const Eigen::Matrix3d R = poseRotation(pose);
        const Eigen::Vector3d p_map_ant = R_enu_map.transpose() * (pos.p - t_enu_map);
        Eigen::Vector3d gnss_pos = p_map_ant - R * p_gnss->lever;
        const Eigen::Vector3d positionDelta = gnss_pos - last_fpos;
        const double positionDistance = useGnssElevation
            ? positionDelta.norm() : positionDelta.head<2>().norm();
        if (!has_fpos || positionDistance >= gpsFactorMinDis)
        {
            if (useGnssElevation)
            {
                gtsam::Vector3 gnssVar;
                gnssVar << gnssPosXYVar,
                           gnssPosXYVar,
                           gnssPosZVar;
                gtSAMgraph.add(gtsam::GPSFactor(key,
                        gtsam::Point3(gnss_pos.x(), gnss_pos.y(), gnss_pos.z()),
                        gtsam::noiseModel::Diagonal::Variances(gnssVar)));
            }
            else
            {
                const gtsam::Vector2 gnssVar = gtsam::Vector2::Constant(gnssPosXYVar);
                gtSAMgraph.add(boost::shared_ptr<GnssXYFactor>(
                    new GnssXYFactor(key, gnss_pos.head<2>(),
                        gtsam::noiseModel::Diagonal::Variances(gnssVar))));
            }

            last_fpos = gnss_pos;
            has_fpos = true;
        }
    }

    if (!useGnssYawFactor)
        return;

    while (true)
    {
        YawData yaw;
        int key = -1;
        {
            std::lock_guard<std::mutex> lock(p_gnss->mtx_gnss);
            if (p_gnss->yaw_buf.empty()) break;

            yaw = p_gnss->yaw_buf.front();
            key = find_gnss_key(keyposes, yaw.t);
            if (key == -2) break;

            p_gnss->yaw_buf.pop_front();
        }

        if (key < 0) continue;

        const PointTypePose &pose = keyposes[key];
        const double yaw_map = yawToMap(yaw.yaw);
        const double yaw_error = std::abs(normalizeYaw(yaw_map - pose.yaw));
        if (yaw_error <= 60.0 * M_PI / 180.0)
        {
            const auto yawNoise = gtsam::noiseModel::Isotropic::Sigma(
                1, std::sqrt(gnssYawVar));
            gtSAMgraph.add(boost::shared_ptr<GnssYawFactor>(
                    new GnssYawFactor(key, yaw_map, yawNoise)));
        }
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

void KeyframeToGlobal(
    const PointTypePose &odomPose,
    const PointTypePose &baPose,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &frameCloud,
    const Eigen::Vector3d &gravityBody,
    const Eigen::Vector3d &gravityUpOdom,
    double gravitySigma)
{
    const gtsam::Key latestPoseKey=static_cast<gtsam::Key>(cloudKeyPoses3D->size());
    addOdomFactor(odomPose,baPose);
    addGravityFactor(latestPoseKey, gravityBody, gravityUpOdom, gravitySigma);
    addLoopFactor();
    isam->update(gtSAMgraph,initialEstimate);
    isam->update();
    if(loopIsClosed) for(int i=0;i<5;++i) isam->update();
    gtSAMgraph.resize(0); initialEstimate.clear();
    isamCurrentEstimate=isam->calculateEstimate(); graphUpdate=true;

    const Pose3 estimate=isamCurrentEstimate.at<Pose3>(latestPoseKey);
    PointTypeIndex pose3; pose3.x=estimate.translation().x(); pose3.y=estimate.translation().y(); pose3.z=estimate.translation().z();
    pose3.intensity=cloudKeyPoses3D->size();
    PointTypePose pose6; pose6.x=pose3.x; pose6.y=pose3.y; pose6.z=pose3.z; pose6.intensity=pose3.intensity;
    pose6.roll=estimate.rotation().roll(); pose6.pitch=estimate.rotation().pitch(); pose6.yaw=estimate.rotation().yaw(); pose6.time=odomPose.time;

    pcl::PointCloud<PointTypeIndex>::Ptr cloud(new pcl::PointCloud<PointTypeIndex>());
    cloud->reserve(frameCloud ? frameCloud->size() : 0);
    if(frameCloud) for(const auto &pt: frameCloud->points) { PointTypeIndex q; q.x=pt.x;q.y=pt.y;q.z=pt.z;q.intensity=pt.intensity;cloud->push_back(q); }
    {
        std::lock_guard<std::mutex> lock(mtxKeyframe);
        cloudKeyPoses3D->push_back(pose3); cloudKeyPoses6D->push_back(pose6);
        cloudKeyOdomPoses6D->push_back(odomPose); cloudKeyBAPoses6D.push_back(baPose);
        featCloudKeyFrames.push_back(cloud);
        if(loopClosureEnableFlag && scanContextEnableFlag) sc_manager.makeAndSaveScancontextAndKeys(*cloud);
    }
    if (loopClosureEnableFlag) loopDone.store(false);
    if(sceneEnableFlag) floorMap.update(static_cast<int>(latestPoseKey),odomPose);
    updatePath(pose6);
    poseCovariance=isam->marginalCovariance(latestPoseKey);

    if(keyframe_export_en) {
        pcl::PointCloud<pcl::PointXYZINormal> exportCloud;
        if(frameCloud) for(const auto &pt:frameCloud->points){pcl::PointXYZINormal q;q.x=pt.x;q.y=pt.y;q.z=pt.z;q.intensity=pt.intensity;exportCloud.push_back(q);}
        const std::string stamp=format_unix_time(pose6.time);
        pcl::PCDWriter writer; writer.writeBinary(keyframe_frames_dir+"scans/"+stamp+".pcd",exportCloud);
    }
}

void shutdownMapOptimization()
{
#ifdef USE_ROS1
    pubKeyPoses.shutdown();
    pubPath.shutdown();
    pubGlobalPath.shutdown();
    pubLaserCloudGlobal.shutdown();
    pubRecentKeyFrame.shutdown();
    pubLoopConstraintEdge.shutdown();
    pubKeyFrameYawMarkers.shutdown();
#elif defined(USE_ROS2)
    pubKeyPoses.reset();
    pubPath.reset();
    pubGlobalPath.reset();
    pubLaserCloudGlobal.reset();
    pubRecentKeyFrame.reset();
    pubLoopConstraintEdge.reset();
    pubKeyFrameYawMarkers.reset();
#endif
    samTfBroadcaster.reset();

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
    if (ros_subscription_count(pubGlobalPath) != 0)
    {
        globalPath.header.stamp = timeLaserInfoStamp;
        globalPath.header.frame_id = map_frame;
        ros_publish(pubGlobalPath, globalPath);
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
        std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> clouds;
        {
            std::lock_guard<std::mutex> lock(mtxKeyframe);
            if (cloudKeyPoses6D == nullptr ||
                cloudKeyPoses6D->points.size() < 6 ||
                featCloudKeyFrames.size() < 6)
            {
                loopDone.store(true);
                continue;
            }

            poses6D.assign(cloudKeyPoses6D->points.begin(), cloudKeyPoses6D->points.end());
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
        // ROS_PRINT_INFO(
        //     "[LOOP] range next=%d ready=%d latest=%zu",
        //     nextKey,
        //     readyKey,
        //     poses6D.size() - 1);

        for (int processed = 0; processed < loopCurNum && nextKey <= readyKey; )
        {
            performLoopClosure(nextKey, poses6D, clouds);

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

        pt.intensity = key;
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
