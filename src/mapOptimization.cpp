// Modified from LIO-SAM: MapOptimization.cpp

#include "utility.h"
#include "common_utils.h"
#include "map_optimization.h"
#include "ros_utils.h"

#include <algorithm>
#include <fstream>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
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
Values optimizedEstimate;
ISAM2 *isam;
Values isamCurrentEstimate;
Eigen::MatrixXd poseCovariance;

Pcl2Publisher pubKeyPoses;
PathPublisher pubPath;

Pcl2Publisher pubLaserCloudGlobal;
Pcl2Publisher pubLaserCloudLocal;

Pcl2Publisher pubHistoryKeyFrames;
Pcl2Publisher pubIcpKeyFrames;
Pcl2Publisher pubRecentKeyFrame;
Pcl2Publisher pubCloudRegisteredRaw;
MarkerArrayPublisher pubLoopConstraintEdge;

TimeType timeLaserInfoStamp;

pcl::PointCloud<PointTypeIndex>::Ptr cloudKeyPoses3D; // Store keyframe poses and indexes 
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
pcl::PointCloud<PointTypeIndex>::Ptr copy_cloudKeyPoses3D;
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

double timeLaserInfoCur;

float transformTobeMapped[6];
Eigen::Vector3d translationLidarToIMU;
Eigen::Matrix3d rotationLidarToIMU;

bool isDegenerate = false;

PathMsg globalPath;

std::mutex mtx;
std::mutex mtxLoopInfo;

bool aLoopIsClosed = false;
map<int, int> loopIndexContainer; // from new to old
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;

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

    init_ros_node();
    
    pubKeyPoses = create_publisher<PointCloud2Msg>("lio_sam/trajectory", 1);
    pubPath = create_publisher<PathMsg>("lio_sam/mapping/path", 1);
    pubLaserCloudGlobal = create_publisher<PointCloud2Msg>("lio_sam/mapping/cloud_global", 1);
    pubRecentKeyFrame = create_publisher<PointCloud2Msg>("lio_sam/mapping/cloud_recent_keyframe", 1);
    pubLoopConstraintEdge = create_publisher<MarkerArrayMsg>("lio_sam/loop_closure_constraints", 1);

    downSizeFilterICP.setLeafSize(mappingICPSize, mappingICPSize, mappingICPSize);

    allocateMemory();
}

bool saveFrame()
{
    if (cloudKeyPoses3D->points.empty())
        return true;

    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
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

    // find the closest history key frame
    KD_TREE_PUBLIC<PointTypeIndex>::PointVector pointSearchPoses3D;
    std::vector<float> pointSearchSqDisLoop;
    
    ikdtreeHistoryKeyPoses->Nearest_Search(copy_cloudKeyPoses3D->back(), ikdtreeSearchNeighborNum, pointSearchPoses3D, pointSearchSqDisLoop, historyKeyframeSearchRadius);
    for (int i = 0; i < (int)pointSearchPoses3D.size(); ++i)
    {
        int id = pointSearchPoses3D[i].intensity; // index stored in intensity field
        if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
        {
            const gtsam::Pose3 poseCur = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyCur]);
            const gtsam::Pose3 posePre = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[id]);
            if (rotationDistance(poseCur, posePre) > historyKeyframeSearchAngleThreshold)
                continue;

            loopKeyPre = id;
            break;
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
    float noiseScore = icp.getFitnessScore();
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
        // TODO: use IKFoM covariance as prior covariance
        noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    }else{
        noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
        gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
        gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
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
    aLoopIsClosed = true;
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
    if (saveFrame() == false)
        return;

    // odom factor
    addOdomFactor();

    // loop factor
    addLoopFactor();

    // cout << "****************************************************" << endl;
    // gtSAMgraph.print("GTSAM Graph:\n");

    // update iSAM
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    if (aLoopIsClosed == true)
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
    latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
    thisPose6D.roll  = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw   = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserInfoCur;
    cloudKeyPoses6D->push_back(thisPose6D);

    updatePath(thisPose6D);

    // cout << "****************************************************" << endl;
    // cout << "Pose covariance:" << endl;
    // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

    // save updated transform
    transformTobeMapped[0] = latestEstimate.rotation().roll();
    transformTobeMapped[1] = latestEstimate.rotation().pitch();
    transformTobeMapped[2] = latestEstimate.rotation().yaw();
    transformTobeMapped[3] = latestEstimate.translation().x();
    transformTobeMapped[4] = latestEstimate.translation().y();
    transformTobeMapped[5] = latestEstimate.translation().z();

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

    if (keyframe_export_en)
    {
        const std::string keyframe_frames_dir = root_dir + "/KEY_FRAMES/";
        const std::string stamp_str = format_unix_time(thisPose6D.time);
        const std::string pcd_path = keyframe_frames_dir + "scans/" + stamp_str + ".pcd";
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(pcd_path, *feats_undistort);
    }

    if (ikdtreeHistoryKeyPoses->Root_Node == nullptr) {
        initPoses3D.push_back(thisPose3D);
        if (cloudKeyPoses3D->points.size() < 10)
            return;
        ikdtreeHistoryKeyPoses->Build(initPoses3D);
    } else {
        ikdtreeHistoryKeyPoses->Add_Point(thisPose3D);
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

    if (aLoopIsClosed == true)
    {
        // clear path
        globalPath.poses.clear();
        // update key poses
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

            updatePath(cloudKeyPoses6D->points[i]);
        }

        ReconstructIkdTree();
        aLoopIsClosed = false;
    }
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

    if (ros_subscription_count(pubRecentKeyFrame) != 0)
    {
        pcl::PointCloud<PointTypeIndex>::Ptr cloudOut(new pcl::PointCloud<PointTypeIndex>());
        PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
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
