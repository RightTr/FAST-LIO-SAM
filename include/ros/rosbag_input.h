#ifndef ROSBAG_INPUT_H
#define ROSBAG_INPUT_H

#include "ros/ros_utils.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_ROS2
#include <unordered_map>

#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>

class RosbagInput {
public:
    RosbagInput(const std::string &path, const std::string &cloudTopic,
                const std::string &imuTopic, const std::string &gnssTopic,
                const std::string &gnssHeadingTopic, bool livox,
                std::function<void(const Pcl2MsgConstPtr &)> cloudCallback,
                std::function<void(const LivoxCustomMsgConstPtr &)> livoxCallback,
                std::function<void(const ImuMsgConstPtr &)> imuCallback,
                std::function<void(const GnssFixMsgConstPtr &)> gnssCallback,
                std::function<void(const OdometryMsgConstPtr &)> headingCallback)
        : cloudTopic_(cloudTopic), imuTopic_(imuTopic), livox_(livox),
          cloudCallback_(std::move(cloudCallback)),
          livoxCallback_(std::move(livoxCallback)), imuCallback_(std::move(imuCallback)),
          gnssTopic_(gnssTopic), gnssHeadingTopic_(gnssHeadingTopic),
          gnssCallback_(std::move(gnssCallback)), headingCallback_(std::move(headingCallback)) {
        reader_.open(path);
        for (const auto &topic : reader_.get_all_topics_and_types())
            types_[topic.name] = topic.type;
        const std::string expectedCloud = livox_ ? "livox_ros_driver2/msg/CustomMsg"
                                                 : "sensor_msgs/msg/PointCloud2";
        const auto cloudType = types_.find(cloudTopic_);
        const auto imuType = types_.find(imuTopic_);
        if (cloudType == types_.end() || cloudType->second != expectedCloud ||
            imuType == types_.end() || imuType->second != "sensor_msgs/msg/Imu")
            throw std::runtime_error("rosbag LiDAR/IMU topic missing or message type mismatch: " +
                                     cloudTopic_ + ", " + imuTopic_);
    }

    // Read until one relevant message is delivered. Processing in the caller
    // keeps the input queues bounded without a second reader thread.
    bool feedNext() {
        while (reader_.has_next()) {
            auto record = reader_.read_next();
            if (record->topic_name != cloudTopic_ && record->topic_name != imuTopic_ &&
                (!gnssCallback_ || record->topic_name != gnssTopic_) &&
                (!headingCallback_ || record->topic_name != gnssHeadingTopic_))
                continue;
            rclcpp::SerializedMessage serialized(*record->serialized_data);
            if (gnssCallback_ && record->topic_name == gnssTopic_) {
                auto msg = std::make_shared<GnssFixMsg>();
                rclcpp::Serialization<GnssFixMsg>().deserialize_message(&serialized, msg.get());
                gnssCallback_(msg);
            } else if (headingCallback_ && record->topic_name == gnssHeadingTopic_) {
                auto msg = std::make_shared<OdometryMsg>();
                rclcpp::Serialization<OdometryMsg>().deserialize_message(&serialized, msg.get());
                headingCallback_(msg);
            } else if (record->topic_name == imuTopic_) {
                auto msg = std::make_shared<ImuMsg>();
                rclcpp::Serialization<ImuMsg>().deserialize_message(&serialized, msg.get());
                imuCallback_(msg);
            } else if (livox_) {
                auto msg = std::make_shared<LivoxCustomMsg>();
                rclcpp::Serialization<LivoxCustomMsg>().deserialize_message(&serialized, msg.get());
                livoxCallback_(msg);
            } else {
                auto msg = std::make_shared<Pcl2Msg>();
                rclcpp::Serialization<Pcl2Msg>().deserialize_message(&serialized, msg.get());
                cloudCallback_(msg);
            }
            return true;
        }
        return false;
    }

private:
    rosbag2_cpp::Reader reader_;
    std::string cloudTopic_, imuTopic_, gnssTopic_, gnssHeadingTopic_;
    bool livox_;
    std::unordered_map<std::string, std::string> types_;
    std::function<void(const Pcl2MsgConstPtr &)> cloudCallback_;
    std::function<void(const LivoxCustomMsgConstPtr &)> livoxCallback_;
    std::function<void(const ImuMsgConstPtr &)> imuCallback_;
    std::function<void(const GnssFixMsgConstPtr &)> gnssCallback_;
    std::function<void(const OdometryMsgConstPtr &)> headingCallback_;
};
#elif defined(USE_ROS1)
#include <rosbag/bag.h>
#include <rosbag/view.h>

class RosbagInput {
public:
    RosbagInput(const std::string &path, const std::string &cloudTopic,
                const std::string &imuTopic, const std::string &gnssTopic,
                const std::string &gnssHeadingTopic, bool livox,
                std::function<void(const Pcl2MsgConstPtr &)> cloudCallback,
                std::function<void(const LivoxCustomMsgConstPtr &)> livoxCallback,
                std::function<void(const ImuMsgConstPtr &)> imuCallback,
                std::function<void(const GnssFixMsgConstPtr &)> gnssCallback,
                std::function<void(const OdometryMsgConstPtr &)> headingCallback)
        : cloudTopic_(cloudTopic), imuTopic_(imuTopic), livox_(livox),
          cloudCallback_(std::move(cloudCallback)),
          livoxCallback_(std::move(livoxCallback)), imuCallback_(std::move(imuCallback)),
          gnssTopic_(gnssTopic), gnssHeadingTopic_(gnssHeadingTopic),
          gnssCallback_(std::move(gnssCallback)), headingCallback_(std::move(headingCallback)) {
        bag_.open(path, rosbag::bagmode::Read);
        std::vector<std::string> topics{cloudTopic_, imuTopic_};
        if (gnssCallback_) topics.push_back(gnssTopic_);
        if (headingCallback_) topics.push_back(gnssHeadingTopic_);
        view_.addQuery(bag_, rosbag::TopicQuery(topics));
        const std::string expectedCloud = livox_ ? "livox_ros_driver2/CustomMsg"
                                                 : "sensor_msgs/PointCloud2";
        bool foundCloud = false, foundImu = false;
        for (const auto *connection : view_.getConnections()) {
            if (connection->topic == cloudTopic_ && connection->datatype == expectedCloud)
                foundCloud = true;
            if (connection->topic == imuTopic_ && connection->datatype == "sensor_msgs/Imu")
                foundImu = true;
        }
        if (!foundCloud || !foundImu)
            throw std::runtime_error("rosbag LiDAR/IMU topic missing or message type mismatch: " +
                                     cloudTopic_ + ", " + imuTopic_);
        next_ = view_.begin();
    }

    bool feedNext() {
        while (next_ != view_.end()) {
        const rosbag::MessageInstance record = *next_++;
        const std::string topic = record.getTopic();
        if (topic != cloudTopic_ && topic != imuTopic_ &&
            (!gnssCallback_ || topic != gnssTopic_) &&
            (!headingCallback_ || topic != gnssHeadingTopic_))
            continue;
        if (gnssCallback_ && record.getTopic() == gnssTopic_) {
            auto msg = record.instantiate<GnssFixMsg>();
            if (!msg) throw std::runtime_error("Cannot decode GNSS message");
            gnssCallback_(msg);
        } else if (headingCallback_ && record.getTopic() == gnssHeadingTopic_) {
            auto msg = record.instantiate<OdometryMsg>();
            if (!msg) throw std::runtime_error("Cannot decode GNSS heading message");
            headingCallback_(msg);
        } else if (record.getTopic() == imuTopic_) {
            auto msg = record.instantiate<ImuMsg>();
            if (!msg) throw std::runtime_error("Cannot decode IMU message");
            imuCallback_(msg);
        } else if (livox_) {
            auto msg = record.instantiate<LivoxCustomMsg>();
            if (!msg) throw std::runtime_error("Cannot decode Livox message");
            livoxCallback_(msg);
        } else {
            auto msg = record.instantiate<Pcl2Msg>();
            if (!msg) throw std::runtime_error("Cannot decode point cloud message");
            cloudCallback_(msg);
        }
        return true;
        }
        return false;
    }

private:
    rosbag::Bag bag_;
    rosbag::View view_;
    rosbag::View::iterator next_;
    std::string cloudTopic_, imuTopic_, gnssTopic_, gnssHeadingTopic_;
    bool livox_;
    std::function<void(const Pcl2MsgConstPtr &)> cloudCallback_;
    std::function<void(const LivoxCustomMsgConstPtr &)> livoxCallback_;
    std::function<void(const ImuMsgConstPtr &)> imuCallback_;
    std::function<void(const GnssFixMsgConstPtr &)> gnssCallback_;
    std::function<void(const OdometryMsgConstPtr &)> headingCallback_;
};
#endif
#endif
