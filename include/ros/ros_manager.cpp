#include "ros/ros_manager.h"

#include "common_utils.h"

RosManager &RosManager::instance()
{
    static RosManager manager;
    return manager;
}

RosManager &ros_manager = RosManager::instance();

void RosManager::updateMapOdom(const Eigen::Matrix3d &R_map_body,
                              const Eigen::Vector3d &t_map_body,
                              const Eigen::Matrix3d &R_odom_body,
                              const Eigen::Vector3d &t_odom_body)
{
    std::lock_guard<std::mutex> lock(mapOdomMutex_);
    R_map_odom_ = R_map_body * R_odom_body.transpose();
    t_map_odom_ = t_map_body - R_map_odom_ * t_odom_body;
    mapOdomUpdated_.store(true, std::memory_order_release);
}

void RosManager::setMapOdom(const Eigen::Matrix3d &rotation,
                           const Eigen::Vector3d &translation)
{
    std::lock_guard<std::mutex> lock(mapOdomMutex_);
    R_map_odom_ = rotation;
    t_map_odom_ = translation;
    mapOdomUpdated_.store(true, std::memory_order_release);
}

void RosManager::getMapOdom(Eigen::Matrix3d &rotation,
                           Eigen::Vector3d &translation) const
{
    std::lock_guard<std::mutex> lock(mapOdomMutex_);
    rotation = R_map_odom_;
    translation = t_map_odom_;
}

bool RosManager::tryTakeMapOdomUpdate(Eigen::Matrix3d &rotation,
                                     Eigen::Vector3d &translation)
{
    if (!mapOdomUpdated_.load(std::memory_order_acquire))
        return false;

    std::unique_lock<std::mutex> lock(mapOdomMutex_, std::try_to_lock);
    if (!lock.owns_lock() ||
        !mapOdomUpdated_.load(std::memory_order_acquire))
        return false;

    rotation = R_map_odom_;
    translation = t_map_odom_;
    mapOdomUpdated_.store(false, std::memory_order_release);
    return true;
}

void RosManager::setEnuMap(const Eigen::Matrix3d &rotation,
                          const Eigen::Vector3d &translation)
{
    std::lock_guard<std::mutex> lock(enuMapMutex_);
    R_enu_map_ = rotation;
    t_enu_map_ = translation;
}

void RosManager::getEnuMap(Eigen::Matrix3d &rotation,
                          Eigen::Vector3d &translation) const
{
    std::lock_guard<std::mutex> lock(enuMapMutex_);
    rotation = R_enu_map_;
    translation = t_enu_map_;
}

void RosManager::initPublishers(bool enableLocalBA, bool enableGlobalGraph,
                                bool enableGnssPath)
{
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef USE_ROS1
    int bestQos = 0;
    int reliableQos = 0;
#elif defined(USE_ROS2)
    auto bestQos = rclcpp::QoS(10).best_effort();
    auto reliableQos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
#endif
    pubLaserCloudFull = create_publisher<PointCloud2Msg>("/cloud_registered", 5);
    pubLaserCloudFullBody = create_publisher<PointCloud2Msg>("/cloud_registered_body", 5);
    pubLaserCloudEffect = create_publisher<PointCloud2Msg>("/cloud_effected", 5);
    pubLaserCloudMap = create_publisher<PointCloud2Msg>("/Laser_map", 1);
    pubLaserCloudBAMap = create_publisher<PointCloud2Msg>("/Laser_map_ba", 1);
    pubLaserCloudPriorMap = create_publisher<PointCloud2Msg>("/Laser_map_prior", 1);
    pubGeoMapDropped = create_publisher<PointCloud2Msg>("/geo_map/dropped", 100000);
    pubOdom = create_publisher_qos<OdometryMsg>("/Odometry", reliableQos);
    pubPath = create_publisher_qos<PathMsg>("/path", reliableQos);
    pubOdomPath = create_publisher_qos<PathMsg>("/odom_path", reliableQos);
    pubOdomHighFreq = create_publisher_qos<OdometryMsg>("/OdometryHighFreq", bestQos);
    pubCornFeature = create_publisher<PointCloud2Msg>("/corn_feature", 100000);
    pubSurfFeature = create_publisher<PointCloud2Msg>("/surf_feature", 100000);
    if (enableLocalBA)
    {
        pubLidarBAPath = create_publisher_qos<PathMsg>("/ba_path", reliableQos);
        pubBAFrameMarkers = create_publisher_qos<MarkerArrayMsg>("/ba_frames", reliableQos);
    }
    if (enableGnssPath)
        pubGnssPath = create_publisher_qos<PathMsg>("/gnss_path", reliableQos);
    if (enableGlobalGraph)
    {
        pubKeyPoses = create_publisher<PointCloud2Msg>("lio_sam/trajectory", 1);
        pubGraphPath = create_publisher<PathMsg>("lio_sam/mapping/path", 1);
        pubGlobalPath = create_publisher<PathMsg>("/global_path", 1);
        pubLaserCloudGlobal = create_publisher<PointCloud2Msg>("lio_sam/mapping/cloud_global", 1);
        pubRecentKeyFrame = create_publisher<PointCloud2Msg>("lio_sam/mapping/cloud_recent_keyframe", 1);
        pubLoopConstraintEdge = create_publisher<MarkerArrayMsg>("lio_sam/loop_closure_constraints", 1);
        pubKeyFrameYawMarkers = create_publisher<MarkerArrayMsg>("lio_sam/mapping/keyframe_yaw", 1);
    }
#ifdef USE_ROS1
    broadcaster_ = std::make_shared<tf::TransformBroadcaster>();
#elif defined(USE_ROS2)
    broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(get_ros_node());
#endif
}

void RosManager::reset()
{
    {
        std::lock_guard<std::mutex> lock(subscriberMutex_);
        subscribers_.clear();
#ifdef USE_ROS2
        callbackGroups_.clear();
#endif
    }
    std::lock_guard<std::mutex> lock(mutex_);
    broadcaster_.reset();
#ifdef USE_ROS1
    pubLaserCloudFull.shutdown();
    pubLaserCloudFullBody.shutdown();
    pubLaserCloudEffect.shutdown();
    pubLaserCloudMap.shutdown();
    pubLaserCloudBAMap.shutdown();
    pubLaserCloudPriorMap.shutdown();
    pubGeoMapDropped.shutdown();
    pubCornFeature.shutdown();
    pubSurfFeature.shutdown();
    pubOdom.shutdown();
    pubOdomHighFreq.shutdown();
    pubPath.shutdown();
    pubOdomPath.shutdown();
    pubLidarBAPath.shutdown();
    pubGnssPath.shutdown();
    pubBAFrameMarkers.shutdown();
    pubKeyPoses.shutdown();
    pubLaserCloudGlobal.shutdown();
    pubRecentKeyFrame.shutdown();
    pubGraphPath.shutdown();
    pubGlobalPath.shutdown();
    pubLoopConstraintEdge.shutdown();
    pubKeyFrameYawMarkers.shutdown();
#elif defined(USE_ROS2)
    pubLaserCloudFull.reset();
    pubLaserCloudFullBody.reset();
    pubLaserCloudEffect.reset();
    pubLaserCloudMap.reset();
    pubLaserCloudBAMap.reset();
    pubLaserCloudPriorMap.reset();
    pubGeoMapDropped.reset();
    pubCornFeature.reset();
    pubSurfFeature.reset();
    pubOdom.reset();
    pubOdomHighFreq.reset();
    pubPath.reset();
    pubOdomPath.reset();
    pubLidarBAPath.reset();
    pubGnssPath.reset();
    pubBAFrameMarkers.reset();
    pubKeyPoses.reset();
    pubLaserCloudGlobal.reset();
    pubRecentKeyFrame.reset();
    pubGraphPath.reset();
    pubGlobalPath.reset();
    pubLoopConstraintEdge.reset();
    pubKeyFrameYawMarkers.reset();
#endif
}

void RosManager::sendTf(const TransformStampedMsg &transform)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (broadcaster_)
        broadcaster_->sendTransform(transform);
}

void RosManager::publishEnuToMapTf(const TimeType &stamp)
{
    if (!gnss_aligned.load())
        return;

    TransformStampedMsg transform;
    transform.header.stamp = stamp;
    transform.header.frame_id = "enu";
    transform.child_frame_id = map_frame;

    Eigen::Matrix3d R_enu_map;
    Eigen::Vector3d t_enu_map;
    getEnuMap(R_enu_map, t_enu_map);
    transform.transform.translation.x = t_enu_map.x();
    transform.transform.translation.y = t_enu_map.y();
    transform.transform.translation.z = t_enu_map.z();

    Eigen::Quaterniond q_enu_map(R_enu_map);
    q_enu_map.normalize();
    transform.transform.rotation.x = q_enu_map.x();
    transform.transform.rotation.y = q_enu_map.y();
    transform.transform.rotation.z = q_enu_map.z();
    transform.transform.rotation.w = q_enu_map.w();
    sendTf(transform);
}
