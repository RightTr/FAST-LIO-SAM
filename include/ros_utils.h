#ifndef FAST_LIO_SAM_ROS_UTILS_H
#define FAST_LIO_SAM_ROS_UTILS_H

#include <cmath>
#include <cstddef>
#include <cstdarg>
#include <cstdio>
#include <memory>

#ifdef USE_ROS1
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <fast_lio_sam/Pose6D.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_broadcaster.h>

// Message type aliases and publisher/subscriber types for ROS1
using PointCloud2Msg = sensor_msgs::PointCloud2;
using PathMsg = nav_msgs::Path;
using OdometryMsg = nav_msgs::Odometry;
using OdometryMsgConstPtr = nav_msgs::Odometry::ConstPtr;
using OdomMsg = nav_msgs::Odometry;
using Pose6D = fast_lio_sam::Pose6D;
using MarkerMsg = visualization_msgs::Marker;
using MarkerArrayMsg = visualization_msgs::MarkerArray;
using PointMsg = geometry_msgs::Point;
using PoseStampedMsg = geometry_msgs::PoseStamped;
using ImuMsg = sensor_msgs::Imu;
using Float64Msg = std_msgs::Float64;
using Vector3StampedMsg = geometry_msgs::Vector3Stamped;
using PathPublisher = ros::Publisher;
using OdomPublisher = ros::Publisher;
using Pcl2Publisher = ros::Publisher;
using MarkerArrayPublisher = ros::Publisher;
using OdomSubscriber = ros::Subscriber;
using TimeType = ros::Time;
using RateType = ros::Rate;
using Pcl2Msg = sensor_msgs::PointCloud2;
using QuaternionMsg = geometry_msgs::Quaternion;
using PoseStampedMsgConstPtr = geometry_msgs::PoseStamped::ConstPtr;
using ImuMsgConstPtr = sensor_msgs::Imu::ConstPtr;
using ImuMsgPtr = sensor_msgs::Imu::Ptr;
using Float64MsgConstPtr = std_msgs::Float64::ConstPtr;
using Vector3StampedMsgConstPtr = geometry_msgs::Vector3Stamped::ConstPtr;
using LivoxCustomMsgConstPtr = livox_ros_driver2::CustomMsg::ConstPtr;
using LivoxCustomMsg = livox_ros_driver2::CustomMsg;
using Pcl2MsgConstPtr = sensor_msgs::PointCloud2::ConstPtr;
using LivoxMsg = PointCloud2Msg;  // ROS1 Livox driver outputs PointCloud2
using TransformStampedMsg = geometry_msgs::TransformStamped;

#elif defined(USE_ROS2)
#include <rclcpp/rclcpp.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <fast_lio_sam/msg/pose6_d.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>

// Message type aliases and publisher/subscriber types for ROS2
using PointCloud2Msg = sensor_msgs::msg::PointCloud2;
using PathMsg = nav_msgs::msg::Path;
using OdometryMsg = nav_msgs::msg::Odometry;
using OdometryMsgConstPtr = nav_msgs::msg::Odometry::ConstSharedPtr;
using OdomMsg = nav_msgs::msg::Odometry;
using Pose6D = fast_lio_sam::msg::Pose6D;
using MarkerMsg = visualization_msgs::msg::Marker;
using MarkerArrayMsg = visualization_msgs::msg::MarkerArray;
using PointMsg = geometry_msgs::msg::Point;
using PoseStampedMsg = geometry_msgs::msg::PoseStamped;
using ImuMsg = sensor_msgs::msg::Imu;
using Float64Msg = std_msgs::msg::Float64;
using Vector3StampedMsg = geometry_msgs::msg::Vector3Stamped;
using PathPublisher = rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr;
using OdomPublisher = rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr;
using Pcl2Publisher = rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr;
using MarkerArrayPublisher = rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr;
using OdomSubscriber = rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr;
using TimeType = rclcpp::Time;
using RateType = rclcpp::Rate;
using Pcl2Msg = sensor_msgs::msg::PointCloud2;
using QuaternionMsg = geometry_msgs::msg::Quaternion;
using PoseStampedMsgConstPtr = geometry_msgs::msg::PoseStamped::ConstPtr;
using ImuMsgConstPtr = sensor_msgs::msg::Imu::ConstPtr;
using ImuMsgPtr = sensor_msgs::msg::Imu::Ptr;
using Float64MsgConstPtr = std_msgs::msg::Float64::ConstSharedPtr;
using Vector3StampedMsgConstPtr = geometry_msgs::msg::Vector3Stamped::ConstSharedPtr;
using LivoxCustomMsgConstPtr = livox_ros_driver2::msg::CustomMsg::ConstPtr;
using LivoxCustomMsg = livox_ros_driver2::msg::CustomMsg;
using Pcl2MsgConstPtr = sensor_msgs::msg::PointCloud2::ConstPtr;
using LivoxMsg = livox_ros_driver2::msg::CustomMsg;  // ROS2 Livox driver outputs CustomMsg
using TransformStampedMsg = geometry_msgs::msg::TransformStamped;

#endif

inline bool ros_ok() {
#ifdef USE_ROS1
    return ros::ok();
#elif defined(USE_ROS2)
    return rclcpp::ok();
#else
    return false;
#endif
}

inline void ros_shutdown() {
#ifdef USE_ROS1
    ros::shutdown();
#elif defined(USE_ROS2)
    rclcpp::shutdown();
#endif
}

inline void ROS_PRINT_WARN(const char *fmt, ...) {
    char msg[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
#ifdef USE_ROS1
    ROS_WARN("%s", msg);
#elif defined(USE_ROS2)
    RCLCPP_WARN(rclcpp::get_logger("fast_lio_sam"), "%s", msg);
#endif
}

inline void ROS_PRINT_ERROR(const char *fmt, ...) {
    char msg[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
#ifdef USE_ROS1
    ROS_ERROR("%s", msg);
#elif defined(USE_ROS2)
    RCLCPP_ERROR(rclcpp::get_logger("fast_lio_sam"), "%s", msg);
#endif
}

inline void ROS_PRINT_INFO(const char *fmt, ...) {
    char msg[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
#ifdef USE_ROS1
    ROS_INFO("%s", msg);
#elif defined(USE_ROS2)
    RCLCPP_INFO(rclcpp::get_logger("fast_lio_sam"), "%s", msg);
#endif
}

#ifdef USE_ROS1
inline void spin_once() {
    ros::spinOnce();
}
#elif defined(USE_ROS2)
inline void spin_once(const rclcpp::Node::SharedPtr &node) {
    rclcpp::spin_some(node);
}
#endif

inline QuaternionMsg quaternion_from_rpy(double roll, double pitch, double yaw) {
#ifdef USE_ROS1
    return tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
#elif defined(USE_ROS2)
    tf2::Quaternion tf_q;
    tf_q.setRPY(roll, pitch, yaw);
    QuaternionMsg q;
    q.x = tf_q.x();
    q.y = tf_q.y();
    q.z = tf_q.z();
    q.w = tf_q.w();
    return q;
#endif
}

#ifdef USE_ROS1
inline ros::Time get_ros_time(double stamp_sec)
{
    return ros::Time().fromSec(stamp_sec);
}
inline double get_ros_time_sec(const ros::Time &stamp)
{
    return stamp.toSec();
}
#elif defined(USE_ROS2)
inline rclcpp::Time get_ros_time(double stamp_sec)
{
    const int32_t sec = static_cast<int32_t>(std::floor(stamp_sec));
    const uint32_t nanosec = static_cast<uint32_t>((stamp_sec - sec) * 1e9);
    return rclcpp::Time(sec, nanosec);
}
inline double get_ros_time_sec(const builtin_interfaces::msg::Time &stamp)
{
    return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}
#endif

#ifdef USE_ROS1
inline ros::Time get_ros_now()
{
    return ros::Time::now();
}
#elif defined(USE_ROS2)
inline rclcpp::Time get_ros_now(const rclcpp::Node::SharedPtr &node = nullptr)
{
    // Prefer node clock so use_sim_time is respected when enabled.
    if (node) {
        return node->get_clock()->now();
    }
    return rclcpp::Clock().now();
}

#endif

// Generic publish wrapper for both ROS1 and ROS2
#ifdef USE_ROS1
template<typename PubType, typename MsgType>
inline void ros_publish(PubType &pub, const MsgType &msg) {
    if (pub.getNumSubscribers() == 0)
        return;
    pub.publish(msg);
}
#elif defined(USE_ROS2)
template<typename PubType, typename MsgType>
inline void ros_publish(const PubType &pub, const MsgType &msg) {
    if (!pub || pub->get_subscription_count() == 0)
        return;
    pub->publish(msg);
}
#endif

// Unified subscriber count query for ROS1/ROS2 publishers.
#ifdef USE_ROS1
inline std::size_t ros_subscription_count(const ros::Publisher &pub) {
    return pub.getNumSubscribers();
}
#elif defined(USE_ROS2)
template<typename PubT>
inline std::size_t ros_subscription_count(const std::shared_ptr<PubT> &pub) {
    return pub ? pub->get_subscription_count() : 0;
}
#endif

// Global ROS1 NodeHandle and ROS2 Node for parameter loading and pub/sub management
#ifdef USE_ROS1

inline ros::NodeHandle*& get_ros_nh() {
    static ros::NodeHandle *instance = nullptr;
    return instance;
}

inline void init_ros_node(const void *node = nullptr) {
    auto &g_ros_nh = get_ros_nh();
    if (!g_ros_nh) {
        g_ros_nh = new ros::NodeHandle();
    }
}

template<typename T>
inline void rosparam_get(const std::string &param_name, T &param_value, const T &default_value) {
    init_ros_node();
    get_ros_nh()->param<T>(param_name, param_value, default_value);
}

// Subscriber creation for ROS1
template<typename T, typename Callback>
inline ros::Subscriber create_subscriber(const std::string& topic, uint32_t queue_size, Callback cb) {
    return get_ros_nh()->subscribe(topic, queue_size, cb);
}

// Publisher creation for ROS1
template<typename T>
inline ros::Publisher create_publisher(const std::string& topic, uint32_t queue_size) {
    return get_ros_nh()->advertise<T>(topic, queue_size);
}

// Publisher creation for ROS1 with QoS (ignores QoS parameter, ROS1 has no QoS support)
template<typename T, typename QosType>
inline ros::Publisher create_publisher_qos(const std::string& topic, const QosType& qos) {
    // ROS1 doesn't support QoS, so we ignore the qos parameter
    // and use a default queue size of 50 for critical topics like odometry
    return get_ros_nh()->advertise<T>(topic, 50);
}

#elif defined(USE_ROS2)

inline rclcpp::Node::SharedPtr& get_ros_node() {
    static rclcpp::Node::SharedPtr instance = nullptr;
    return instance;
}

inline void init_ros_node(const rclcpp::Node::SharedPtr &node = nullptr) {
    auto &g_ros_node = get_ros_node();
    if (!g_ros_node) {
        if (node) {
            g_ros_node = node;
        } else {
            g_ros_node = rclcpp::Node::make_shared("fast_lio_sam");
        }
    }
}

inline void spin_once() {
    auto &g_ros_node = get_ros_node();
    if (g_ros_node) {
        rclcpp::spin_some(g_ros_node);
    }
}

template<typename T>
inline void rosparam_get(const std::string &param_name, T &param_value, const T &default_value) {
    get_ros_node()->declare_parameter<T>(param_name, default_value);
    param_value = get_ros_node()->get_parameter(param_name).get_value<T>();
}

// ROS2 has no float parameter type (only double), so specialize float to go through double
template<>
inline void rosparam_get<float>(const std::string &param_name, float &param_value, const float &default_value) {
    get_ros_node()->declare_parameter<double>(param_name, static_cast<double>(default_value));
    param_value = static_cast<float>(get_ros_node()->get_parameter(param_name).get_value<double>());
}

// Subscriber creation for ROS2
template<typename T, typename Callback>
inline typename rclcpp::Subscription<T>::SharedPtr create_subscriber(const std::string& topic, uint32_t queue_size, Callback cb) {
    return get_ros_node()->create_subscription<T>(topic, rclcpp::QoS(rclcpp::KeepLast(queue_size)), cb);
}

// Publisher creation for ROS2
template<typename T>
inline typename rclcpp::Publisher<T>::SharedPtr create_publisher(const std::string& topic, uint32_t queue_size) {
    return get_ros_node()->create_publisher<T>(topic, rclcpp::QoS(rclcpp::KeepLast(queue_size)));
}

// Publisher creation for ROS2 with custom QoS
template<typename T>
inline typename rclcpp::Publisher<T>::SharedPtr create_publisher_qos(const std::string& topic, const rclcpp::QoS& qos) {
    return get_ros_node()->create_publisher<T>(topic, qos);
}

// Subscriber creation for ROS2 with custom QoS
template<typename T, typename Callback>
inline typename rclcpp::Subscription<T>::SharedPtr create_subscriber_qos(const std::string& topic, const rclcpp::QoS& qos, Callback cb) {
    return get_ros_node()->create_subscription<T>(topic, qos, cb);
}
#endif

#endif
