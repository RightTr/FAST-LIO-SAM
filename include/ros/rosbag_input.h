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
                const std::string &imuTopic, bool livox,
                std::function<void(const Pcl2MsgConstPtr &)> cloudCallback,
                std::function<void(const LivoxCustomMsgConstPtr &)> livoxCallback,
                std::function<void(const ImuMsgConstPtr &)> imuCallback)
        : cloudTopic_(cloudTopic), imuTopic_(imuTopic), livox_(livox),
          cloudCallback_(std::move(cloudCallback)),
          livoxCallback_(std::move(livoxCallback)), imuCallback_(std::move(imuCallback)) {
        reader_.open(path);
        for (const auto &topic : reader_.get_all_topics_and_types())
            types_[topic.name] = topic.type;
        const std::string expectedCloud = livox_ ? "livox_ros_driver2/msg/CustomMsg"
                                                 : "sensor_msgs/msg/PointCloud2";
        if (types_[cloudTopic_] != expectedCloud ||
            types_[imuTopic_] != "sensor_msgs/msg/Imu")
            throw std::runtime_error("rosbag LiDAR/IMU topic missing or message type mismatch: " +
                                     cloudTopic_ + ", " + imuTopic_);
    }

    // Read until one relevant message is delivered. Processing in the caller
    // keeps the input queues bounded without a second reader thread.
    bool feedNext() {
        while (reader_.has_next()) {
            auto record = reader_.read_next();
            if (record->topic_name != cloudTopic_ && record->topic_name != imuTopic_)
                continue;
            rclcpp::SerializedMessage serialized(*record->serialized_data);
            if (record->topic_name == imuTopic_) {
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
    std::string cloudTopic_, imuTopic_;
    bool livox_;
    std::unordered_map<std::string, std::string> types_;
    std::function<void(const Pcl2MsgConstPtr &)> cloudCallback_;
    std::function<void(const LivoxCustomMsgConstPtr &)> livoxCallback_;
    std::function<void(const ImuMsgConstPtr &)> imuCallback_;
};
#elif defined(USE_ROS1)
#include <rosbag/bag.h>
#include <rosbag/view.h>

class RosbagInput {
public:
    RosbagInput(const std::string &path, const std::string &cloudTopic,
                const std::string &imuTopic, bool livox,
                std::function<void(const Pcl2MsgConstPtr &)> cloudCallback,
                std::function<void(const LivoxCustomMsgConstPtr &)> livoxCallback,
                std::function<void(const ImuMsgConstPtr &)> imuCallback)
        : cloudTopic_(cloudTopic), imuTopic_(imuTopic), livox_(livox),
          cloudCallback_(std::move(cloudCallback)),
          livoxCallback_(std::move(livoxCallback)), imuCallback_(std::move(imuCallback)) {
        bag_.open(path, rosbag::bagmode::Read);
        view_.addQuery(bag_, rosbag::TopicQuery(
            std::vector<std::string>{cloudTopic_, imuTopic_}));
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
        if (next_ == view_.end())
            return false;
        const rosbag::MessageInstance record = *next_++;
        if (record.getTopic() == imuTopic_) {
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

private:
    rosbag::Bag bag_;
    rosbag::View view_;
    rosbag::View::iterator next_;
    std::string cloudTopic_, imuTopic_;
    bool livox_;
    std::function<void(const Pcl2MsgConstPtr &)> cloudCallback_;
    std::function<void(const LivoxCustomMsgConstPtr &)> livoxCallback_;
    std::function<void(const ImuMsgConstPtr &)> imuCallback_;
};
#endif
#endif
