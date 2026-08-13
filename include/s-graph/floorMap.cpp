#include "s-graph/floorMap.h"

#include "common_utils.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace
{
constexpr int kFloorSlopeWindow = 6;
constexpr double kStairStartSlope = 0.20;
constexpr double kStairEndSlope = 0.10;
constexpr double kStairMaxSlope = 1.00;
constexpr double kMinHorizontalDistance = 0.50;
}

void FloorMap::setEnabled(bool enabled)
{
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}

void FloorMap::update(const std::deque<int> &keys,
                      const std::deque<PointTypePose> &poses)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (keys.empty() || poses.empty())
        return;

    const int cur_key = keys.back();
    if (!enabled_)
    {
        if (floors_.empty())
        {
            Floor floor;
            floor.height = poses.back().z;
            floors_.push_back(std::move(floor));
        }

        current_floor_id_ = 0;
        for (int key : keys)
            key_floor_[key] = 0;
        last_state_key_ = cur_key;
        transition_ = 0;
        return;
    }

    if (floors_.empty())
    {
        Floor floor;
        floor.height = poses.back().z;
        floors_.push_back(std::move(floor));
        current_floor_id_ = 0;
        ROS_PRINT_INFO("Floor init: floor=0 height=%.3f", poses.back().z);
    }

    if (cur_key == last_state_key_)
        return;

    const double slope = trajectorySlope(poses);
    const double height = poses.back().z;
    const int old_floor = current_floor_id_;

    auto createLinkedFloor = [&](int direction) -> int
    {
        Floor floor;
        floor.height = height;
        floors_.push_back(std::move(floor));

        const int new_floor = static_cast<int>(floors_.size()) - 1;
        if (direction > 0)
        {
            floors_[static_cast<size_t>(old_floor)].up = new_floor;
            floors_[static_cast<size_t>(new_floor)].down = old_floor;
        }
        else
        {
            floors_[static_cast<size_t>(old_floor)].down = new_floor;
            floors_[static_cast<size_t>(new_floor)].up = old_floor;
        }

        ROS_PRINT_INFO("Floor init: floor=%d height=%.3f", new_floor, height);
        return new_floor;
    };

    if (transition_ == 0)
    {
        if (slope > kStairStartSlope &&
            slope < kStairMaxSlope)
        {
            transition_ = 1;

            const size_t count = std::min<size_t>(kFloorSlopeWindow, keys.size());
            const size_t begin = keys.size() - count;
            for (size_t i = begin; i < keys.size(); ++i)
                key_floor_[keys[i]] = -1;

            ROS_PRINT_INFO("Floor transition start: floor=%d direction=UP key=%d slope=%.3f",
                           current_floor_id_,
                           cur_key,
                           slope);
            key_floor_[cur_key] = -1;
            last_state_key_ = cur_key;
            return;
        }
        if (slope < -kStairStartSlope &&
            slope > -kStairMaxSlope)
        {
            transition_ = -1;

            const size_t count = std::min<size_t>(kFloorSlopeWindow, keys.size());
            const size_t begin = keys.size() - count;
            for (size_t i = begin; i < keys.size(); ++i)
                key_floor_[keys[i]] = -1;

            ROS_PRINT_INFO("Floor transition start: floor=%d direction=DOWN key=%d slope=%.3f",
                           current_floor_id_,
                           cur_key,
                           slope);
            key_floor_[cur_key] = -1;
            last_state_key_ = cur_key;
            return;
        }
        key_floor_[cur_key] = current_floor_id_;
        last_state_key_ = cur_key;
        return;
    }

    if (transition_ == 1)
    {
        if (slope < kStairEndSlope)
        {
            transition_ = 0;
            int floor_id = floors_[static_cast<size_t>(old_floor)].up;
            if (floor_id < 0)
                floor_id = createLinkedFloor(+1);

            ROS_PRINT_INFO("Floor transition end: UP floor=%d -> floor=%d key=%d height=%.3f",
                           old_floor,
                           floor_id,
                           cur_key,
                           height);

            current_floor_id_ = floor_id;
            key_floor_[cur_key] = floor_id;
            last_state_key_ = cur_key;
            return;
        }

        key_floor_[cur_key] = -1;
        last_state_key_ = cur_key;
        return;
    }

    if (transition_ == -1)
    {
        if (slope > -kStairEndSlope)
        {
            transition_ = 0;
            int floor_id = floors_[static_cast<size_t>(old_floor)].down;
            if (floor_id < 0)
                floor_id = createLinkedFloor(-1);

            ROS_PRINT_INFO("Floor transition end: DOWN floor=%d -> floor=%d key=%d height=%.3f",
                           old_floor,
                           floor_id,
                           cur_key,
                           height);

            current_floor_id_ = floor_id;
            key_floor_[cur_key] = floor_id;
            last_state_key_ = cur_key;
            return;
        }

        key_floor_[cur_key] = -1;
        last_state_key_ = cur_key;
        return;
    }

    key_floor_[cur_key] = current_floor_id_;
    last_state_key_ = cur_key;
}

double FloorMap::trajectorySlope(const std::deque<PointTypePose> &poses) const
{
    if (poses.size() < kFloorSlopeWindow)
        return 0.0;

    const size_t count = kFloorSlopeWindow;

    const size_t begin = poses.size() - count;
    std::vector<double> dist(count, 0.0);
    for (size_t i = 1; i < count; ++i)
    {
        const auto &a = poses[begin + i - 1];
        const auto &b = poses[begin + i];
        const double dx = b.x - a.x;
        const double dy = b.y - a.y;
        dist[i] = dist[i - 1] + std::sqrt(dx * dx + dy * dy);
    }

    const double horizontal_distance = dist.back();
    if (horizontal_distance < kMinHorizontalDistance)
        return 0.0;

    double x_mean = 0.0;
    for (double x : dist)
        x_mean += x;
    x_mean /= static_cast<double>(count);

    double z_mean = 0.0;
    for (size_t i = 0; i < count; ++i)
        z_mean += poses[begin + i].z;
    z_mean /= static_cast<double>(count);

    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        const double x = dist[i];
        const double z = poses[begin + i].z;
        num += (x - x_mean) * (z - z_mean);
        den += (x - x_mean) * (x - x_mean);
    }

    if (den <= 1e-12)
        return 0.0;

    return num / den;
}

void FloorMap::groundKeys(const std::deque<int> &keys,
                          std::unordered_set<int> &allowed_keys) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    allowed_keys.clear();

    if (keys.empty())
        return;

    if (!enabled_)
    {
        for (int key : keys)
            allowed_keys.insert(key);
        return;
    }

    if (transition_ != 0 || current_floor_id_ < 0)
        return;

    const int min_key = last_state_key_ - (kFloorSlopeWindow - 1);
    for (int key : keys)
    {
        const auto it = key_floor_.find(key);
        if (it == key_floor_.end())
            continue;
        if (it->second != current_floor_id_)
            continue;
        if (key > min_key)
            continue;
        allowed_keys.insert(key);
    }
}

bool FloorMap::allowLoop(int key1, int key2) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_)
        return true;

    const auto a = key_floor_.find(key1);
    const auto b = key_floor_.find(key2);
    return a != key_floor_.end() &&
           b != key_floor_.end() &&
           a->second >= 0 &&
           a->second == b->second;
}

bool FloorMap::updateGround(const std::vector<PlaneObs> &planes,
                            const std::deque<int> &keys,
                            const std::deque<PointTypePose> &poses,
                            PlaneBatch &batch)
{
    std::lock_guard<std::mutex> lock(mutex_);
    batch = PlaneBatch();
    if (keys.empty() || poses.empty() || planes.empty())
        return false;

    if (current_floor_id_ < 0 ||
        current_floor_id_ >= static_cast<int>(floors_.size()))
        return false;

    Floor &floor = floors_[static_cast<size_t>(current_floor_id_)];
    if (!floor.ground.update(planes, keys, poses, batch))
        return false;

    if (floor.ground_id < 0)
        floor.ground_id = next_plane_id_++;

    for (auto &plane : batch.init)
        plane.id = floor.ground_id;
    for (auto &factor : batch.factors)
        factor.id = floor.ground_id;

    batch.valid = !batch.init.empty() || !batch.factors.empty();

    return batch.valid;
}
