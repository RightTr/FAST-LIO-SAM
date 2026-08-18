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
constexpr double kGroundRadius = 8.0;
constexpr double kStairMinDz = 1.5;
constexpr double kGroundDz = 0.5;
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

int FloorMap::floorIdForKey(int key) const
{
    if (key < 0)
        return -1;

    for (int i = 0; i < static_cast<int>(floors_.size()); ++i)
    {
        for (const auto &range : floors_[static_cast<size_t>(i)].ranges)
        {
            if (key < range.begin)
                continue;
            if (range.end >= 0 && key > range.end)
                continue;
            return i;
        }
    }

    return -1;
}

void FloorMap::update(int key,
                      const PointTypePose &pose)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (key < 0)
        return;

    if (key <= last_update_key_)
        return;
    last_update_key_ = key;

    recent_poses_.push_back(pose);
    if (recent_poses_.size() > kSlopeWindow)
        recent_poses_.pop_front();

    const double angle = trajectoryAngle();
    const Eigen::Vector3d gravity_up = getGravityUp();

    if (floors_.empty() || current_floor_id_ < 0 ||
        current_floor_id_ >= static_cast<int>(floors_.size()))
        return;

    Floor &current_floor = floors_[static_cast<size_t>(current_floor_id_)];

    if (!in_transition_)
    {
        landing_ = false;

        if (std::abs(angle) <= kStairAngle)
            return;

        in_transition_ = true;
        landing_ = false;
        transition_dir_ = angle > 0.0 ? 1 : -1;
        transition_start_z_ = gravity_up.dot(poseTranslation(pose));

        if (!current_floor.ranges.empty() &&
            current_floor.ranges.back().end < 0)
        {
            current_floor.ranges.back().end = key - 1;
        }

        ROS_PRINT_INFO(
            "[FLOOR] START key=%d floor=%d dir=%d angle=%.3f",
            key,
            current_floor_id_,
            transition_dir_,
            angle);
        return;
    }

    if (std::abs(angle) > kStairAngle)
    {
        landing_ = false;
        return;
    }

    if (!landing_)
    {
        landing_ = true;
        ROS_PRINT_INFO(
            "[FLOOR] LANDING key=%d floor=%d",
            key,
            current_floor_id_);
    }
}

bool FloorMap::getFloorRanges(int key,
                              std::vector<FloorRange> &ranges,
                              FloorRange &current_range) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    ranges.clear();
    current_range = FloorRange();

    if (key < 0)
        return false;

    for (const auto &floor : floors_)
    {
        for (const auto &range : floor.ranges)
        {
            if (key < range.begin)
                continue;
            if (range.end >= 0 && key > range.end)
                continue;

            ranges = floor.ranges;
            current_range = range;
            return true;
        }
    }

    return false;
}

bool FloorMap::sameFloor(int key_a, int key_b) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (key_a < 0 || key_b < 0)
        return false;
    if (key_a == key_b)
        return true;

    const int floor_a = floorIdForKey(key_a);
    const int floor_b = floorIdForKey(key_b);
    if (floor_a < 0 || floor_b < 0)
        return false;
    return floor_a == floor_b;
}

bool FloorMap::updateGround(const std::vector<PlaneObs> &planes,
                            const std::deque<int> &keys,
                            const std::deque<PointTypePose> &poses,
                            SceneBatch &batch)
{
    std::lock_guard<std::mutex> lock(mutex_);
    batch = SceneBatch();

    if (keys.empty() || poses.empty() || planes.empty())
        return false;

    PlaneObs candidate;
    if (!buildGroundCandidate(planes, keys, poses, candidate))
        return false;

    const Eigen::Vector3d gravity_up = getGravityUp();

    if (floors_.empty())
    {
        Floor floor;
        floor.ranges.push_back(FloorRange{keys.back(), -1});
        floors_.push_back(std::move(floor));
        current_floor_id_ = 0;
        in_transition_ = false;
        landing_ = false;
        transition_dir_ = 0;
        transition_start_z_ = 0.0;
        ROS_PRINT_INFO(
            "[FLOOR] CREATE id=0");
    }

    if (current_floor_id_ < 0 ||
        current_floor_id_ >= static_cast<int>(floors_.size()))
        return false;

    Floor &current_floor = floors_[static_cast<size_t>(current_floor_id_)];

    if (in_transition_)
    {
        if (!landing_)
            return false;

        const int old_floor = current_floor_id_;
        const double current_z = gravity_up.dot(poseTranslation(poses.back()));
        const double dz = current_z - transition_start_z_;

        int target_floor = old_floor;

        if (static_cast<double>(transition_dir_) * dz >= kStairMinDz)
        {
            if (transition_dir_ > 0)
            {
                target_floor = floors_[static_cast<size_t>(old_floor)].up;
            }
            else
            {
                target_floor = floors_[static_cast<size_t>(old_floor)].down;
            }

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
                ROS_PRINT_INFO(
                    "[FLOOR] CREATE id=%d from=%d dir=%d",
                    target_floor,
                    old_floor,
                    transition_dir_);
            }
            else
            {
                ROS_PRINT_INFO(
                    "[FLOOR] REUSE old=%d target=%d dir=%d",
                    old_floor,
                    target_floor,
                    transition_dir_);
            }
        }

        current_floor_id_ = target_floor;
        floors_[static_cast<size_t>(current_floor_id_)].ranges.push_back(
            FloorRange{keys.back(), -1});
        in_transition_ = false;
        landing_ = false;
        transition_dir_ = 0;
        transition_start_z_ = 0.0;
        last_update_key_ = keys.back();

        ROS_PRINT_INFO(
            "[FLOOR] END key=%d old=%d new=%d dz=%.2f",
            keys.back(),
            old_floor,
            target_floor,
            dz);
        return false;
    }

    Floor &floor = floors_[static_cast<size_t>(current_floor_id_)];

    Ground *best_ground = nullptr;
    double best_xy = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < floor.grounds.size(); ++i)
    {
        auto &ground = floor.grounds[i];
        const double dot = clampDot(
            ground.plane.head<3>().dot(candidate.plane.head<3>()));
        const double angle = std::acos(dot);
        if (angle > rad(groundAssociationAngleDeg))
            continue;

        const double xy = (candidate.center - ground.center).head<2>().norm();
        const double dz = std::abs(
            gravity_up.dot(candidate.center - ground.center));
        if (xy > kGroundRadius || dz > kGroundDz || xy >= best_xy)
            continue;

        best_xy = xy;
        best_ground = &ground;
    }

    Ground *active_ground = best_ground;
    const bool is_new_ground = (active_ground == nullptr);
    const int ground_id = is_new_ground ? next_plane_id_ : active_ground->id;
    const int key = keys.back();
    const int last_added_key = active_ground ? active_ground->last_key : -1;

    if (key <= last_added_key)
        return false;

    const auto it = candidate.obs.find(key);
    if (it == candidate.obs.end())
        return false;

    if (is_new_ground)
    {
        Ground new_ground;
        new_ground.id = ground_id;
        new_ground.plane = candidate.plane;
        new_ground.center = candidate.center;
        new_ground.last_key = key;
        floor.grounds.push_back(std::move(new_ground));
        next_plane_id_ = std::max(next_plane_id_, ground_id + 1);
    }
    else
    {
        active_ground->plane = candidate.plane;
        active_ground->center = candidate.center;
        active_ground->last_key = key;
    }

    batch.factors.emplace_back(key, ground_id, it->second);

    if (is_new_ground)
    {
        batch.planes.emplace_back(current_floor_id_, ground_id, candidate.plane);
    }

    ROS_PRINT_INFO(
        "[GROUND] READY floor=%d ground=%d key=%d new=%d",
        current_floor_id_,
        ground_id,
        key,
        is_new_ground ? 1 : 0);
    return true;
}
