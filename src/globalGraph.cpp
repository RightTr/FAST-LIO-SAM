// Global graph, GNSS, and loop processing.

#include "common_utils.h"
#include "GNSS_Processing.hpp"
#include "factor/gnssYaw_factor.h"
#include "factor/gnssXY_factor.h"
#include "s-graph/floorMap.h"
#include "scan_context/Scancontext.h"
#include "lidarBA.h"
#include "global_graph.h"
#include "debug/result_writer.h"
#include "ros/ros_utils.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <limits>
#include <deque>
#include <tuple>
#include <utility>
#include <unordered_set>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <thread>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
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

std::mutex GlobalGraph::gravityAxisMutex;
Eigen::Vector3d GlobalGraph::gravityUpAxis = Eigen::Vector3d::UnitZ();

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

pcl::PointCloud<PointTypeIndex>::Ptr transformPointCloud(
    pcl::PointCloud<PointTypeIndex>::Ptr cloudIn,
    const Eigen::Affine3f &transCur);

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

void GlobalGraph::allocateMemory()
{
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointTypeIndex>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    cloudKeyOdomPoses6D.reset(new pcl::PointCloud<PointTypePose>());
}

void GlobalGraph::Initialize()
{
    floorMap = std::make_unique<FloorMap>();
    cloudKeyBAPoses6D.clear();
    loopDone.store(!loopClosureEnableFlag);
    gnss_aligned.store(false);
    ros_manager.setEnuMap(
        Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    last_fpos.setZero();
    has_fpos = false;

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    parameters.factorization = ISAM2Params::QR;
    isam.reset(new ISAM2(parameters));

    GlobalGraph::SetGravityUp(Eigen::Vector3d::UnitZ());

    init_ros_node();
    
    downSizeFilterICP.setLeafSize(mappingICPSize, mappingICPSize, mappingICPSize);

    allocateMemory();
}

double GlobalGraph::yawToMap(double yaw) const
{
    const Eigen::Vector3d heading_enu(std::cos(yaw), std::sin(yaw), 0.0);
    Eigen::Matrix3d R_enu_map;
    Eigen::Vector3d t_enu_map;
    ros_manager.getEnuMap(R_enu_map, t_enu_map);
    const Eigen::Vector3d heading_map = R_enu_map.transpose() * heading_enu;
    return normalizeYaw(std::atan2(heading_map.y(), heading_map.x()));
}

std::vector<std::pair<int, int>> GlobalGraph::getLoopSearchRanges(
    int loopKeyCur,
    const std::vector<FloorRange> &floorRanges) const
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

std::vector<std::pair<int, float>> GlobalGraph::detectLoopKeys(
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

void GlobalGraph::performLoopICP(
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

void GlobalGraph::performLoopClosure(
    int loopKeyCur,
    const std::vector<PointTypePose> &poses6D,
    const std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> &clouds)
{
    if (loopUsedKeys.count(loopKeyCur) != 0)
        return;

    std::vector<FloorRange> floorRanges;
    if (sceneEnableFlag)
    {
        if (!floorMap->getRanges(loopKeyCur, floorRanges))
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

void GlobalGraph::addOdomFactor(const PointTypePose &odomPose,
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
            Eigen::Matrix3d R_map_odom;
            Eigen::Vector3d t_map_odom;
            ros_manager.getMapOdom(
                R_map_odom, t_map_odom);
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

void GlobalGraph::addLoopFactor()
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

void GlobalGraph::poseGraphUpdate()
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

void GlobalGraph::performGnssMatching()
{
    if (!gnssEnableFlag || !gnss_aligned.load())
        return;

    if (!cloudKeyPoses6D || cloudKeyPoses6D->empty())
        return;

    const auto &keyPoses6D = cloudKeyPoses6D->points;

    while (true)
    {
        GnssPositionData gnssPos;
        int keyIndex = -1;
        {
            std::lock_guard<std::mutex> lock(p_gnss->gnssMutex);
            if (p_gnss->posBuffer.empty()) break;

            gnssPos = p_gnss->posBuffer.front();
            keyIndex = findGnssKey(keyPoses6D, gnssPos.time);
            if (keyIndex == -2) break;

            p_gnss->posBuffer.pop_front();
        }

        if (keyIndex < 0) continue;

        const PointTypePose &keyPose = keyPoses6D[keyIndex];
        const Eigen::Matrix3d R_map_body = poseRotation(keyPose);
        Eigen::Matrix3d R_enu_map;
        Eigen::Vector3d t_enu_map;
        ros_manager.getEnuMap(
            R_enu_map, t_enu_map);
        const Eigen::Vector3d p_map_ant = R_enu_map.transpose() *
            (gnssPos.position - t_enu_map);
        Eigen::Vector3d gnssPosition = p_map_ant - R_map_body * p_gnss->leverArm;
        const Eigen::Vector3d positionDelta = gnssPosition - last_fpos;
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
                gtSAMgraph.add(gtsam::GPSFactor(keyIndex,
                        gtsam::Point3(gnssPosition.x(), gnssPosition.y(), gnssPosition.z()),
                        gtsam::noiseModel::Diagonal::Variances(gnssVar)));
                gnssGraphUpdated = true;
            }
            else
            {
                const gtsam::Vector2 gnssVar = gtsam::Vector2::Constant(gnssPosXYVar);
                gtSAMgraph.add(boost::shared_ptr<GnssXYFactor>(
                    new GnssXYFactor(keyIndex, gnssPosition.head<2>(),
                        gtsam::noiseModel::Diagonal::Variances(gnssVar))));
                gnssGraphUpdated = true;
            }

            last_fpos = gnssPosition;
            has_fpos = true;
        }
    }

    if (!useGnssYawFactor)
        return;

    while (true)
    {
        GnssYawData gnssYaw;
        int keyIndex = -1;
        {
            std::lock_guard<std::mutex> lock(p_gnss->gnssMutex);
            if (p_gnss->yawBuffer.empty()) break;

            gnssYaw = p_gnss->yawBuffer.front();
            keyIndex = findGnssKey(keyPoses6D, gnssYaw.time);
            if (keyIndex == -2) break;

            p_gnss->yawBuffer.pop_front();
        }

        if (keyIndex < 0) continue;

        const PointTypePose &keyPose = keyPoses6D[keyIndex];
        const double yawMap = yawToMap(gnssYaw.yaw);
        const double yawError = std::abs(normalizeYaw(yawMap - keyPose.yaw));
        if (yawError <= 60.0 * M_PI / 180.0)
        {
            const auto yawNoise = gtsam::noiseModel::Isotropic::Sigma(
                1, std::sqrt(gnssYawVar));
            gtSAMgraph.add(boost::shared_ptr<GnssYawFactor>(
                    new GnssYawFactor(keyIndex, yawMap, yawNoise)));
            gnssGraphUpdated = true;
        }
    }
}

void GlobalGraph::updatePath(const PointTypePose& pose_in)
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

namespace
{
PointTypePose poseFromIsometry(const Eigen::Isometry3d &transform, double timestamp);
}

void GlobalGraph::KeyframeToGlobal(const BAFramePtr &frame)
{
    const PointTypePose odomPose = poseFromIsometry(frame->odom_pose, frame->timestamp);
    const PointTypePose baPose = poseFromIsometry(frame->optimized_pose, frame->timestamp);
    const gtsam::Key latestPoseKey=static_cast<gtsam::Key>(cloudKeyPoses3D->size());
    addOdomFactor(odomPose,baPose);
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

    pcl::PointCloud<PointTypeIndex>::Ptr cloud = frame->cloud;
    if (!cloud) cloud.reset(new pcl::PointCloud<PointTypeIndex>());
    {
        std::lock_guard<std::mutex> lock(mtxKeyframe);
        cloudKeyPoses3D->push_back(pose3); cloudKeyPoses6D->push_back(pose6);
        cloudKeyOdomPoses6D->push_back(odomPose); cloudKeyBAPoses6D.push_back(baPose);
        featCloudKeyFrames.push_back(cloud);
        if(loopClosureEnableFlag && scanContextEnableFlag) sc_manager.makeAndSaveScancontextAndKeys(*cloud);
    }
    if (loopClosureEnableFlag) loopDone.store(false);
    if(sceneEnableFlag) floorMap->update(static_cast<int>(latestPoseKey),odomPose);
    updatePath(pose6);
    poseCovariance=isam->marginalCovariance(latestPoseKey);

    if (keyframe_export_en)
    {
        ResultWriter resultWriter;
        resultWriter.saveKeyframeCloud(frame->cloud, pose6.time);
    }
}

void GlobalGraph::Cleanup()
{
    isam.reset();
}

void GlobalGraph::correctPoses()
{
    bool hasReference = false;
    PointTypePose mapReference;
    PointTypePose odomReference;
    if (graphUpdate)
    {
        {
            std::lock_guard<std::mutex> lock(mtxKeyframe);
            if (cloudKeyPoses3D == nullptr || cloudKeyPoses6D == nullptr ||
                cloudKeyOdomPoses6D == nullptr || cloudKeyPoses3D->points.empty())
                return;

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

            if (!cloudKeyOdomPoses6D->points.empty() &&
                cloudKeyOdomPoses6D->points.size() == cloudKeyPoses6D->points.size())
            {
                mapReference = cloudKeyPoses6D->points.back();
                odomReference = cloudKeyOdomPoses6D->points.back();
                hasReference = true;
            }
        }
    }

    if (graphUpdate && hasReference)
    {
        const gtsam::Pose3 pose_map_ref = pclPointTogtsamPose3(mapReference);
        const gtsam::Pose3 pose_odom_ref = pclPointTogtsamPose3(odomReference);
        const Eigen::Matrix3d R_map_odom =
            pose_map_ref.rotation().matrix() * pose_odom_ref.rotation().matrix().transpose();
        const Eigen::Vector3d t_map_odom =
            Eigen::Vector3d(pose_map_ref.translation().x(),
                             pose_map_ref.translation().y(),
                             pose_map_ref.translation().z()) -
            R_map_odom * Eigen::Vector3d(pose_odom_ref.translation().x(),
                                             pose_odom_ref.translation().y(),
                                             pose_odom_ref.translation().z());
        ros_manager.setMapOdom(R_map_odom, t_map_odom);
    }

    graphUpdate = false;
    loopIsClosed = false;
    gnssGraphUpdated = false;
}

void GlobalGraph::publishSamMsg()
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

    const auto stamp = get_ros_time(poses6D.back().time);
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
    auto &ros = ros_manager;
    publishCloud(ros.pubKeyPoses, keyPosesCloud, stamp, map_frame);
    if (ros_subscription_count(ros.pubGraphPath) != 0)
    {
        globalPath.header.stamp = stamp;
        globalPath.header.frame_id = map_frame;
        ros.publish(ros.pubGraphPath, globalPath);
    }
    if (ros_subscription_count(ros.pubGlobalPath) != 0)
    {
        globalPath.header.stamp = stamp;
        globalPath.header.frame_id = map_frame;
        ros.publish(ros.pubGlobalPath, globalPath);
    }

    if (ros_subscription_count(ros.pubKeyFrameYawMarkers) != 0)
    {
        MarkerArrayMsg markerArray;

        MarkerMsg markerClear;
        markerClear.header.frame_id = map_frame;
        markerClear.header.stamp = stamp;
        markerClear.action = MarkerMsg::DELETEALL;
        markerArray.markers.push_back(markerClear);

        markerArray.markers.reserve(markerArray.markers.size() + poses6D.size());
        for (size_t i = 0; i < poses6D.size(); ++i)
        {
            const auto &pose = poses6D[i];
            MarkerMsg marker;
            marker.header.frame_id = map_frame;
            marker.header.stamp = stamp;
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

        ros.publish(ros.pubKeyFrameYawMarkers, markerArray);
    }

    if (ros_subscription_count(ros.pubRecentKeyFrame) != 0)
    {
        pcl::PointCloud<PointTypeIndex>::Ptr cloudOut(new pcl::PointCloud<PointTypeIndex>());
        PointTypePose thisPose6D = poses6D.back();
        *cloudOut += *transformPointCloud(clouds.back(), &thisPose6D);
        publishCloud(ros.pubRecentKeyFrame, cloudOut, stamp, map_frame);
    }
}

void GlobalGraph::visualizeLoopClosure()
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

    const auto stamp = get_ros_time(keyposes.back().time);
    MarkerArrayMsg markerArray;
    // loop nodes
    MarkerMsg markerNode;
    markerNode.header.frame_id = map_frame;
    markerNode.header.stamp = stamp;
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
    markerEdge.header.stamp = stamp;
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
    ros_manager.publish(
        ros_manager.pubLoopConstraintEdge, markerArray);
}

namespace
{
PointTypePose poseFromIsometry(const Eigen::Isometry3d &transform, double timestamp)
{
    PointTypePose pose;
    const Eigen::Vector3d rpy = transform.linear().eulerAngles(2, 1, 0);
    pose.x = transform.translation().x();
    pose.y = transform.translation().y();
    pose.z = transform.translation().z();
    pose.roll = rpy(2);
    pose.pitch = rpy(1);
    pose.yaw = rpy(0);
    pose.time = timestamp;
    return pose;
}
} // namespace

bool GlobalGraph::processPendingLoopChecks(int &nextKey)
{
    if (!loopClosureEnableFlag || loopCurNum <= 0)
    {
        loopDone.store(true);
        return false;
    }

    std::vector<PointTypePose> poses6D;
    std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> clouds;
    {
        std::lock_guard<std::mutex> lock(mtxKeyframe);
        if (cloudKeyPoses6D == nullptr || cloudKeyPoses6D->points.size() < 6 ||
            featCloudKeyFrames.size() < 6)
        {
            loopDone.store(true);
            return false;
        }
        poses6D.assign(cloudKeyPoses6D->points.begin(), cloudKeyPoses6D->points.end());
        clouds = featCloudKeyFrames;
    }

    const int readyKey = static_cast<int>(poses6D.size()) - 1;
    if (nextKey > readyKey)
    {
        loopDone.store(true);
        return false;
    }

    loopDone.store(false);
    for (int i = 0; i < loopCurNum && nextKey <= readyKey; ++i, ++nextKey)
        performLoopClosure(nextKey, poses6D, clouds);
    loopDone.store(nextKey > readyKey);
    return true;
}

bool GlobalGraph::InsertKeyFrame(const BAFramePtr &frame)
{
    if (!frame || !frame->keyframe || !frame->finalized.load(std::memory_order_acquire))
        return false;
    {
        std::lock_guard<std::mutex> lock(mMutexNewKFs);
        if (mbFinishRequested)
            return false;
        mlNewKeyFrames.push_back(frame);
    }
    mNewKfCV.notify_one();
    return true;
}

void GlobalGraph::Shutdown()
{
    {
        std::lock_guard<std::mutex> lock(mMutexNewKFs);
        mbFinishRequested = true;
    }
    mNewKfCV.notify_all();
    mVisualizationCV.notify_all();
    if (mWorker.joinable())
        mWorker.join();
    if (mVisualizationWorker.joinable())
        mVisualizationWorker.join();
}

std::vector<PointTypePose> GlobalGraph::GetKeyframePoses() const
{
    std::lock_guard<std::mutex> lock(mtxKeyframe);
    if (!cloudKeyPoses6D)
        return {};
    return std::vector<PointTypePose>(
        cloudKeyPoses6D->points.begin(), cloudKeyPoses6D->points.end());
}

bool GlobalGraph::SaveGlobalCloud(const std::string &filename) const
{
    std::vector<PointTypePose> poses;
    std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> clouds;
    {
        std::lock_guard<std::mutex> lock(mtxKeyframe);
        if (!cloudKeyPoses6D || featCloudKeyFrames.empty())
            return false;
        poses.assign(cloudKeyPoses6D->points.begin(), cloudKeyPoses6D->points.end());
        clouds = featCloudKeyFrames;
    }

    pcl::PointCloud<PointTypeIndex> globalCloud;
    const size_t count = std::min(poses.size(), clouds.size());
    for (size_t i = 0; i < count; ++i)
    {
        if (clouds[i])
            globalCloud += *transformPointCloud(clouds[i], &poses[i]);
    }
    pcl::PCDWriter writer;
    return writer.writeBinary(filename, globalCloud) == 0;
}

void GlobalGraph::Run()
{
    int nextLoopKey = 1;
    auto nextLoopTime = std::chrono::steady_clock::now();
    const double loopPeriodSec = loopClosureFrequency > 0.0
        ? 1.0 / loopClosureFrequency : 1.0;
    size_t processedKeyframes = 0;
    auto globalGraphWindowStart = std::chrono::steady_clock::now();
    size_t globalGraphOperations = 0;
    double globalGraphProcessingMs = 0.0;

    for (;;)
    {
        BAFramePtr frame;
        size_t batchCount = 0;
        while (batchCount < 8)
        {
            {
                std::unique_lock<std::mutex> lock(mMutexNewKFs);
                if (batchCount == 0)
                    mNewKfCV.wait_for(lock, std::chrono::milliseconds(30), [this] {
                        return mbFinishRequested || !mlNewKeyFrames.empty();
                    });
                if (mlNewKeyFrames.empty())
                    break;
                frame = std::move(mlNewKeyFrames.front());
                mlNewKeyFrames.pop_front();
            }
            if (frame->keyframe)
            {
                const auto operationStart = std::chrono::steady_clock::now();
                KeyframeToGlobal(frame);
                correctPoses();
                publishSamMsg();
                globalGraphProcessingMs += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - operationStart).count();
                ++globalGraphOperations;
            }
            ++batchCount;
            ++processedKeyframes;
        }

        performGnssMatching();

        const auto now = std::chrono::steady_clock::now();
        bool stopping;
        {
            std::lock_guard<std::mutex> lock(mMutexNewKFs);
            stopping = mbFinishRequested && mlNewKeyFrames.empty();
        }
        if (loopClosureEnableFlag &&
            (now >= nextLoopTime || stopping) && loopCurNum > 0)
        {
            const auto loopStart = std::chrono::steady_clock::now();
            processPendingLoopChecks(nextLoopKey);
            globalGraphProcessingMs += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - loopStart).count();
            ++globalGraphOperations;
            nextLoopTime = now + std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(loopPeriodSec));
        }

        bool hasQueuedLoopFactors = false;
        {
            std::lock_guard<std::mutex> lock(mtxLoopFactor);
            hasQueuedLoopFactors = !loopQueue.empty();
        }
        if (!gtSAMgraph.empty() || hasQueuedLoopFactors || gnssGraphUpdated)
        {
            const auto operationStart = std::chrono::steady_clock::now();
            poseGraphUpdate();
            publishSamMsg();
            globalGraphProcessingMs += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - operationStart).count();
            ++globalGraphOperations;
        }

        if (loopClosureEnableFlag && loopCurNum <= 0)
            loopDone.store(true);
        if (globalGraphOperations >= 20)
        {
            ROS_PRINT_INFO("GlobalGraph average processing: %.2f ms over %zu operations (%zu keyframes)",
                           globalGraphProcessingMs / globalGraphOperations,
                           globalGraphOperations, processedKeyframes);
            globalGraphProcessingMs = 0.0;
            globalGraphOperations = 0;
        }
        if (processedKeyframes >= 100)
        {
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - globalGraphWindowStart).count();
            ROS_PRINT_INFO("GlobalGraph throughput: %zu keyframes in %.1f ms", processedKeyframes, elapsed_ms);
            processedKeyframes = 0;
            globalGraphWindowStart = std::chrono::steady_clock::now();
        }

        {
            std::lock_guard<std::mutex> lock(mMutexNewKFs);
            stopping = mbFinishRequested;
        }
        const bool loopsDrained = !loopClosureEnableFlag ||
                                  loopCurNum <= 0 || loopDone.load();
        if (stopping && loopsDrained)
        {
            performGnssMatching();
            if (!gtSAMgraph.empty() || gnssGraphUpdated)
            {
                poseGraphUpdate();
                publishSamMsg();
            }
            correctPoses();
            publishSamMsg();
            visualizeLoopClosure();
            publishGlobalMap();
            break;
        }
    }
}

void GlobalGraph::VisualizationRun()
{
    std::unique_lock<std::mutex> lock(mMutexNewKFs);
    while (!mbFinishRequested)
    {
        if (mVisualizationCV.wait_for(lock, std::chrono::seconds(5), [this] {
                return mbFinishRequested;
            }))
            break;
        lock.unlock();
        publishGlobalMap();
        visualizeLoopClosure();
        lock.lock();
    }
}

GlobalGraph::GlobalGraph()
{
    Initialize();
    mWorker = std::thread(&GlobalGraph::Run, this);
    mVisualizationWorker = std::thread(&GlobalGraph::VisualizationRun, this);
}

GlobalGraph::~GlobalGraph()
{
    Shutdown();
    Cleanup();
}

void GlobalGraph::SetGravityUp(const Eigen::Vector3d &gravity_up)
{
    std::lock_guard<std::mutex> lock(gravityAxisMutex);
    gravityUpAxis = gravity_up.normalized();
}

Eigen::Vector3d GlobalGraph::GetGravityUp()
{
    std::lock_guard<std::mutex> lock(gravityAxisMutex);
    return gravityUpAxis.normalized();
}

void GlobalGraph::publishGlobalMap() {
    auto &ros = ros_manager;
    if (ros_subscription_count(ros.pubLaserCloudGlobal) == 0)
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

    const auto stamp = get_ros_time(keyPoses6D.back().time);
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
    publishCloud(ros.pubLaserCloudGlobal, globalMapKeyFramesDS, stamp, map_frame);
}

