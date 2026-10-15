#include "s-graph/floorMap.h"

#include "common_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace
{
constexpr int kSlopeWindow = 6;
const double kStairAngle = std::atan(0.1);  // 4.6°
constexpr double kSlopeDistance = 0.50;
constexpr double kStairMinDz = 1.5;
constexpr double kLandingDistance = 1.8;
} // namespace

double FloorMap::trajectoryAngle() const
{
    if (recent_poses_.size() < kSlopeWindow)
        return 0.0;

    const size_t count = kSlopeWindow;
    const size_t begin = recent_poses_.size() - count;
    const Eigen::Vector3d up = getGravityUp();

    std::array<double, kSlopeWindow> dist{};
    for (size_t i = 1; i < count; ++i)
    {
        const Eigen::Vector3d diff =
            poseTranslation(recent_poses_[begin + i]) -
            poseTranslation(recent_poses_[begin + i - 1]);
        const Eigen::Vector3d horizontal = diff - up * diff.dot(up);
        dist[i] = dist[i - 1] + horizontal.norm();
    }

    const double horizontal_distance = dist.back();
    if (horizontal_distance < kSlopeDistance)
        return 0.0;

    double x_mean = 0.0;
    double z_mean = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        x_mean += dist[i];
        z_mean += up.dot(poseTranslation(recent_poses_[begin + i]));
    }
    x_mean /= static_cast<double>(count);
    z_mean /= static_cast<double>(count);

    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        const double x = dist[i];
        const double z = up.dot(poseTranslation(recent_poses_[begin + i]));
        num += (x - x_mean) * (z - z_mean);
        den += (x - x_mean) * (x - x_mean);
    }

    if (den <= 1e-12)
        return 0.0;

    const double slope = num / den;
    return std::atan(slope);
}

void FloorMap::update(int key,
                      const PointTypePose &pose)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (key < 0)
        return;

    recent_poses_.push_back(pose);
    if (recent_poses_.size() > kSlopeWindow)
        recent_poses_.pop_front();

    if (floors_.empty())
    {
        floors_.push_back(Floor{});
        floors_.back().ranges.push_back(FloorRange{key, -1});
        current_floor_id_ = 0;
        transition_dir_ = 0;
        transition_start_z_ = 0.0;
        flat_distance_ = 0.0;
    }

    Floor &current_floor = floors_[static_cast<size_t>(current_floor_id_)];

    const double angle = trajectoryAngle();
    const Eigen::Vector3d gravity_up = getGravityUp();

    if (transition_dir_ == 0)
    {
        if (std::abs(angle) <= kStairAngle)
            return;

        flat_distance_ = 0.0;
        transition_dir_ = angle > 0.0 ? 1 : -1;
        transition_start_z_ = gravity_up.dot(poseTranslation(pose));

        current_floor.ranges.back().end = key - 1;
        return;
    }

    if (std::abs(angle) > kStairAngle)
    {
        flat_distance_ = 0.0;
        return;
    }

    if (recent_poses_.size() >= 2)
    {
        const Eigen::Vector3d step =
            poseTranslation(recent_poses_.back()) -
            poseTranslation(recent_poses_[recent_poses_.size() - 2]);
        const Eigen::Vector3d horizontal = step - gravity_up * step.dot(gravity_up);
        flat_distance_ += horizontal.norm();
    }

    if (flat_distance_ < kLandingDistance)
        return;

    const int old_floor = current_floor_id_;
    const double current_z = gravity_up.dot(poseTranslation(pose));
    const double dz = current_z - transition_start_z_;

    int target_floor = old_floor;

    if (static_cast<double>(transition_dir_) * dz >= kStairMinDz)
    {
        if (transition_dir_ > 0)
            target_floor = floors_[static_cast<size_t>(old_floor)].up;
        else
            target_floor = floors_[static_cast<size_t>(old_floor)].down;

        if (target_floor < 0)
        {
            Floor floor;
            floors_.push_back(std::move(floor));
            target_floor = static_cast<int>(floors_.size()) - 1;
            if (transition_dir_ > 0)
            {
                floors_[static_cast<size_t>(old_floor)].up = target_floor;
                floors_[static_cast<size_t>(target_floor)].down = old_floor;
            }
            else
            {
                floors_[static_cast<size_t>(old_floor)].down = target_floor;
                floors_[static_cast<size_t>(target_floor)].up = old_floor;
            }
        }
    }

    current_floor_id_ = target_floor;
    floors_[static_cast<size_t>(current_floor_id_)].ranges.push_back(FloorRange{key, -1});
    transition_dir_ = 0;
    transition_start_z_ = 0.0;
    flat_distance_ = 0.0;
}

bool FloorMap::getRanges(int key,
                         std::vector<FloorRange> &ranges) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    ranges.clear();

    if (key < 0)
        return false;

    for (const auto &floor : floors_)
    {
        for (const auto &floor_range : floor.ranges)
        {
            if (key >= floor_range.begin &&
                (floor_range.end < 0 || key <= floor_range.end))
            {
                ranges = floor.ranges;
                return true;
            }
        }
    }

    return false;
}
