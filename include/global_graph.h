#ifndef GLOBAL_GRAPH_H
#define GLOBAL_GRAPH_H

#include "lidarBA.h"
#include "ros/ros_utils.h"

#include <Eigen/Core>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>

struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw) (double, time, time))

using PointTypePose = PointXYZIRPYT;
using PointTypeIndex = pcl::PointXYZI;

inline Eigen::Vector3d poseTranslation(const PointTypePose &pose)
{
    return Eigen::Vector3d(pose.x, pose.y, pose.z);
}

inline Eigen::Matrix3d poseRotation(const PointTypePose &pose)
{
    return pcl::getTransformation(0.0, 0.0, 0.0, pose.roll, pose.pitch, pose.yaw)
        .rotation().cast<double>();
}

#include "scan_context/Scancontext.h"

class FloorMap;
struct FloorRange;

class GlobalGraph
{
public:
    GlobalGraph();
    ~GlobalGraph();

    GlobalGraph(const GlobalGraph &) = delete;
    GlobalGraph &operator=(const GlobalGraph &) = delete;

    bool InsertKeyFrame(const BAFramePtr &frame);
    void Shutdown();
    std::vector<PointTypePose> GetKeyframePoses() const;
    bool SaveGlobalCloud(const std::string &filename) const;

    static void SetGravityUp(const Eigen::Vector3d &gravity_up);
    static Eigen::Vector3d GetGravityUp();

private:
    struct LoopFactor
    {
        int from = -1;
        int to = -1;
        gtsam::Pose3 pose;
        gtsam::noiseModel::Diagonal::shared_ptr noise;
    };

    void Initialize();
    void Cleanup();
    void Run();
    void LoopRun();
    void VisualizationRun();
    void allocateMemory();
    double yawToMap(double yaw) const;
    std::vector<std::pair<int, int>> getLoopSearchRanges(
        int loopKeyCur, const std::vector<FloorRange> &floorRanges) const;
    std::vector<std::pair<int, float>> detectLoopKeys(
        int loopKeyCur, const std::vector<PointTypePose> &poses6D,
        const std::vector<FloorRange> &floorRanges);
    void performLoopICP(int loopKeyCur,
        const std::vector<std::pair<int, float>> &loopKeys,
        const std::vector<FloorRange> &floorRanges,
        const std::vector<PointTypePose> &poses6D,
        const std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> &clouds);
    void performLoopClosure(int loopKeyCur,
        const std::vector<PointTypePose> &poses6D,
        const std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> &clouds);
    void addOdomFactor(const PointTypePose &odomPose, const PointTypePose &baPose);
    std::vector<std::pair<int, int>> addLoopFactor();
    void publishLoopEdges(const std::vector<std::pair<int, int>> &edges);
    bool poseGraphUpdate();
    void performGnssMatching();
    void updatePath(const PointTypePose &pose);
    std::vector<std::pair<int, int>> KeyframeToGlobal(const BAFramePtr &frame);
    void correctPoses();
    void publishSamMsg();
    void visualizeLoopClosure();
    void publishLoopProgress(int key, const PointTypePose &pose);
    bool processLoopChecks(int &nextKey);
    void publishGlobalMap();

    gtsam::NonlinearFactorGraph gtSAMgraph;
    gtsam::Values initialEstimate;
    std::unique_ptr<gtsam::ISAM2> isam;
    gtsam::Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    pcl::PointCloud<PointTypeIndex>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D, cloudKeyOdomPoses6D;
    std::vector<PointTypePose> cloudKeyBAPoses6D;
    std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> featCloudKeyFrames;
    PathMsg globalPath;
    std::unique_ptr<FloorMap> floorMap;
    SCManager sc_manager;
    pcl::VoxelGrid<PointTypeIndex> downSizeFilterICP;

    mutable std::mutex mtxKeyframe;
    std::mutex mtxFloorMap;
    std::mutex mtxLoopFactor;
    std::atomic<int> latestLoopKey_{-1};
    std::atomic<bool> loopInputClosed_{false};
    std::atomic<bool> loopFinished_{false};
    bool graphUpdate = false, loopIsClosed = false, gnssGraphUpdated = false;
    std::unordered_set<int> loopUsedKeys;
    std::vector<std::pair<int, int>> loopEdges;
    std::deque<LoopFactor> loopQueue;
    Eigen::Vector3d lastGnssPosition = Eigen::Vector3d::Zero();
    bool hasGnssPosition = false;

    mutable std::mutex mMutexNewKFs;
    std::condition_variable mNewKfCV;
    std::mutex mLoopMutex;
    std::condition_variable mLoopCV;
    std::condition_variable mVisualizationCV;
    std::deque<BAFramePtr> mlNewKeyFrames;
    bool mbFinishRequested = false;
    std::thread mWorker, mLoopWorker, mVisualizationWorker;

    static std::mutex gravityAxisMutex;
    static Eigen::Vector3d gravityUpAxis;
};

#endif
