#include "s-graph/floorMap.h"

#include "common_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace
{
constexpr int kFloorSlopeWindow = 6;
const double kStairStartAngle = std::atan(0.08);  // 4.6°
const double kStairEndAngle   = std::atan(0.04);  // 2.3°
const double kStairMaxAngle   = std::atan(1.00);  // 45°
constexpr double kMinHorizontalDistance = 0.50;
constexpr double kGroundLandmarkRadius = 8.0;
constexpr double kLandingConfirmDistance = 1.8;
constexpr double kTransitionCancelDistance = 4.0;
constexpr double kFloorHeightMin = 1.5;
constexpr double kFloorHeightMatchTolerance = 1.5;
} // namespace

double FloorMap::trajectoryAngle() const
{
    if (recent_poses_.size() < kFloorSlopeWindow)
        return 0.0;

    const size_t count = kFloorSlopeWindow;
    const size_t begin = recent_poses_.size() - count;
    const Eigen::Vector3d up = getGravityUp();

    std::array<double, kFloorSlopeWindow> dist{};
    for (size_t i = 1; i < count; ++i)
    {
        const Eigen::Vector3d diff =
            poseTranslation(recent_poses_[begin + i]) -
            poseTranslation(recent_poses_[begin + i - 1]);
        const Eigen::Vector3d horizontal = diff - up * diff.dot(up);
        dist[i] = dist[i - 1] + horizontal.norm();
    }

    const double horizontal_distance = dist.back();
    if (horizontal_distance < kMinHorizontalDistance)
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

bool FloorMap::update(int key,
                      const PointTypePose &pose)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (key < 0)
        return false;

    const Eigen::Vector3d up = getGravityUp();
    const double cur_z = up.dot(poseTranslation(pose));

    recent_poses_.push_back(pose);
    if (recent_poses_.size() > kFloorSlopeWindow)
        recent_poses_.pop_front();

    const double angle = trajectoryAngle();

    if (floors_.empty())
    {
        Floor floor;
        floor.height = cur_z;
        floor.ranges.push_back(FloorRange{key, -1});
        floors_.push_back(std::move(floor));
        current_floor_id_ = 0;
        ROS_PRINT_INFO(
            "[FLOOR] CREATE id=%d height=%.2f",
            current_floor_id_,
            floors_.front().height);
        return false;
    }

    if (current_floor_id_ < 0 ||
        current_floor_id_ >= static_cast<int>(floors_.size()))
        return false;

    if (!in_transition_)
    {
        if (std::abs(angle) > kStairStartAngle &&
            std::abs(angle) < kStairMaxAngle)
        {
            in_transition_ = true;
            transition_start_z_ = cur_z;
            landing_distance_ = 0.0;

            if (current_floor_id_ >= 0 &&
                current_floor_id_ < static_cast<int>(floors_.size()))
            {
                Floor &floor = floors_[static_cast<size_t>(current_floor_id_)];
                if (!floor.ranges.empty() && floor.ranges.back().end < 0)
                    floor.ranges.back().end = key - 1;
            }
        }

        if (in_transition_)
        {
            ROS_PRINT_INFO(
                "[FLOOR] START key=%d floor=%d angle=%.3f",
                key,
                current_floor_id_,
                angle);
        }

        return false;
    }

    if (std::abs(angle) < kStairEndAngle)
    {
        if (recent_poses_.size() >= 2)
        {
            const Eigen::Vector3d diff =
                poseTranslation(pose) -
                poseTranslation(recent_poses_[recent_poses_.size() - 2]);
            const Eigen::Vector3d horizontal = diff - up * diff.dot(up);
            landing_distance_ += horizontal.norm();
        }

        if (landing_distance_ < kLandingConfirmDistance)
            return false;

        const int old_floor = current_floor_id_;
        const double floor_dz = cur_z - transition_start_z_;
        const double target_height =
            floors_[static_cast<size_t>(old_floor)].height + floor_dz;

        if (std::abs(floor_dz) < kFloorHeightMin)
        {
            if (landing_distance_ < kTransitionCancelDistance)
                return false;

            in_transition_ = false;
            transition_start_z_ = 0.0;
            landing_distance_ = 0.0;

            Floor &floor = floors_[static_cast<size_t>(old_floor)];
            if (!floor.ranges.empty())
                floor.ranges.back().end = -1;

            ROS_PRINT_INFO(
                "[FLOOR] CANCEL old=%d stair_dh=%.2f",
                old_floor,
                floor_dz);
            return false;
        }

        int target_floor = -1;
        double best_diff = kFloorHeightMatchTolerance;
        for (int i = 0; i < static_cast<int>(floors_.size()); ++i)
        {
            if (i == old_floor)
                continue;

            const double diff = std::abs(
                floors_[static_cast<size_t>(i)].height - target_height);
            if (diff < best_diff)
            {
                best_diff = diff;
                target_floor = i;
            }
        }

        if (target_floor < 0)
        {
            Floor floor;
            floor.height = target_height;
            floors_.push_back(std::move(floor));
            target_floor = static_cast<int>(floors_.size()) - 1;
            ROS_PRINT_INFO(
                "[FLOOR] CREATE id=%d height=%.2f",
                target_floor,
                target_height);
        }

        const bool floor_changed = (target_floor != old_floor);
        current_floor_id_ = target_floor;
        floors_[static_cast<size_t>(current_floor_id_)].ranges.push_back(
            FloorRange{key, -1});

        ROS_PRINT_INFO(
            "[FLOOR] END old=%d new=%d dir=%s stair_dh=%.2f target_h=%.2f",
            old_floor,
            target_floor,
            floor_dz > 0.0 ? "UP" : "DOWN",
            floor_dz,
            target_height);

        in_transition_ = false;
        transition_start_z_ = 0.0;
        landing_distance_ = 0.0;
        return floor_changed;
    }

    if (std::abs(angle) > kStairStartAngle)
        landing_distance_ = 0.0;
    return false;
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

bool FloorMap::updateGround(const std::vector<PlaneObs> &planes,
                            const std::deque<int> &keys,
                            const std::deque<PointTypePose> &poses,
                            SceneBatch &batch)
{
    std::lock_guard<std::mutex> lock(mutex_);
    batch = SceneBatch();
    if (in_transition_)
        return false;

    if (keys.empty() || poses.empty() || planes.empty())
        return false;

    if (current_floor_id_ < 0 ||
        current_floor_id_ >= static_cast<int>(floors_.size()))
        return false;

    Floor &floor = floors_[static_cast<size_t>(current_floor_id_)];
    PlaneObs candidate;
    if (!buildGroundCandidate(planes, keys, poses, candidate))
        return false;

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
        if (xy > kGroundLandmarkRadius || xy >= best_xy)
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
