#ifndef ROS_MANAGER_H
#define ROS_MANAGER_H

#include "ros/ros_utils.h"

#include <Eigen/Core>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

class RosManager {
public:
    static RosManager &instance();

    struct SubscriberOptions
    {
        bool avia = false;
        std::string lidarTopic, imuTopic, relocTopic, gnssTopic, gnssHeadingTopic;
        std::function<void(const LivoxCustomMsgConstPtr &)> livoxCallback;
        std::function<void(const Pcl2MsgConstPtr &)> cloudCallback;
        std::function<void(const PoseStampedMsgConstPtr &)> relocCallback;
        std::function<void(const ImuMsgConstPtr &)> imuCallback;
        std::function<void(const GnssFixMsgConstPtr &)> gnssCallback;
        std::function<void(const OdometryMsgConstPtr &)> headingCallback;
    };

    void initPublishers(bool enableLocalBA, bool enableGlobalGraph,
                        bool enableGnssPath);

    void initSubscribers(const SubscriberOptions &options)
    {
#ifdef USE_ROS1
        if (options.avia)
            keepSubscriber(create_subscriber<LivoxMsg>(options.lidarTopic, 200000,
                                                       options.livoxCallback));
        else
            keepSubscriber(create_subscriber<PointCloud2Msg>(options.lidarTopic, 200000,
                                                             options.cloudCallback));
        keepSubscriber(create_subscriber<ImuMsg>(options.imuTopic, 200000,
                                                options.imuCallback));
#elif defined(USE_ROS2)
        auto node = get_ros_node();
        auto lidarGroup = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        auto imuGroup = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        {
            std::lock_guard<std::mutex> lock(subscriberMutex_);
            callbackGroups_.push_back(lidarGroup);
            callbackGroups_.push_back(imuGroup);
        }
        if (options.avia)
            keepSubscriber(create_sensor_subscriber<LivoxMsg>(
                options.lidarTopic, 1000, options.livoxCallback, lidarGroup));
        else
            keepSubscriber(create_sensor_subscriber<PointCloud2Msg>(
                options.lidarTopic, 1000, options.cloudCallback, lidarGroup));
        keepSubscriber(create_sensor_subscriber<ImuMsg>(
            options.imuTopic, 80000, options.imuCallback, imuGroup));
#endif
        keepSubscriber(create_subscriber<PoseStampedMsg>(options.relocTopic, 10,
                                                         options.relocCallback));
        keepSubscriber(create_subscriber<GnssFixMsg>(options.gnssTopic, 10,
                                                     options.gnssCallback));
        keepSubscriber(create_subscriber<OdometryMsg>(
            options.gnssHeadingTopic, 10, options.headingCallback));
    }
    void reset();
    template<typename Subscriber>
    void keepSubscriber(Subscriber &&subscriber)
    {
        std::lock_guard<std::mutex> lock(subscriberMutex_);
        subscribers_.push_back(keep_subscriber(std::forward<Subscriber>(subscriber)));
    }
    void sendTf(const TransformStampedMsg &transform);
    void publishEnuToMapTf(const TimeType &stamp);

    template<typename Publisher, typename Message>
    void publish(Publisher publisher, const Message &message) const
    {
        ros_publish(publisher, message);
    }

    void updateMapOdom(const Eigen::Matrix3d &R_map_body,
                       const Eigen::Vector3d &t_map_body,
                       const Eigen::Matrix3d &R_odom_body,
                       const Eigen::Vector3d &t_odom_body);
    void setMapOdom(const Eigen::Matrix3d &rotation,
                    const Eigen::Vector3d &translation);
    void getMapOdom(Eigen::Matrix3d &rotation,
                    Eigen::Vector3d &translation) const;
    bool tryTakeMapOdomUpdate(Eigen::Matrix3d &rotation,
                              Eigen::Vector3d &translation);

    void setEnuMap(const Eigen::Matrix3d &rotation,
                   const Eigen::Vector3d &translation);
    void getEnuMap(Eigen::Matrix3d &rotation,
                   Eigen::Vector3d &translation) const;
    void initializeEnuMap(const Eigen::Vector3d &gnssPosition,
                          double gnssYaw,
                          const Eigen::Matrix3d &R_map_body,
                          const Eigen::Vector3d &t_map_body,
                          const Eigen::Vector3d &leverArm,
                          const TimeType &stamp);

    Pcl2Publisher pubLaserCloudFull, pubLaserCloudFullBody;
    Pcl2Publisher pubLaserCloudEffect, pubLaserCloudMap, pubLaserCloudBAMap;
    Pcl2Publisher pubLaserCloudPriorMap;
    Pcl2Publisher pubGeoMapDropped, pubCornFeature, pubSurfFeature;
    OdomPublisher pubOdom, pubOdomHighFreq;
    PathPublisher pubPath, pubOdomPath, pubLidarBAPath, pubGnssPath;
    MarkerArrayPublisher pubBAFrameMarkers;
    Pcl2Publisher pubKeyPoses, pubLaserCloudGlobal, pubRecentKeyFrame;
    PathPublisher pubGraphPath, pubGlobalPath;
    MarkerArrayPublisher pubLoopConstraintEdge, pubKeyFrameYawMarkers;

private:
    RosManager() = default;
    std::mutex mutex_;
    std::mutex subscriberMutex_;
    std::vector<std::shared_ptr<void>> subscribers_;
#ifdef USE_ROS2
    std::vector<rclcpp::CallbackGroup::SharedPtr> callbackGroups_;
#endif
    mutable std::mutex mapOdomMutex_;
    Eigen::Matrix3d R_map_odom_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_map_odom_ = Eigen::Vector3d::Zero();
    std::atomic<bool> mapOdomUpdated_{true};
    mutable std::mutex enuMapMutex_;
    Eigen::Matrix3d R_enu_map_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_enu_map_ = Eigen::Vector3d::Zero();
#ifdef USE_ROS1
    std::shared_ptr<tf::TransformBroadcaster> broadcaster_;
#elif defined(USE_ROS2)
    std::shared_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
#endif
};

extern RosManager &ros_manager;

#endif
