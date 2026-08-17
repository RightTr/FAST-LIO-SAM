#include "s-graph/floorMap.h"

#include "common_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace
{
constexpr int kFloorSlopeWindow = 6;
const double kStairStartAngle = std::atan(0.20);
const double kStairEndAngle = std::atan(0.10);
const double kStairMaxAngle = std::atan(1.00);
constexpr double kMinHorizontalDistance = 0.50;
constexpr double kGroundLandmarkRadius = 8.0;
constexpr double kLandingConfirmDistance = 1.8;
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

void FloorMap::update(int key,
                      const PointTypePose &pose)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (key < 0)
        return;

    if (key_floor_.size() <= static_cast<size_t>(key))
        key_floor_.resize(static_cast<size_t>(key) + 1, -1);

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
        floors_.push_back(std::move(floor));
        current_floor_id_ = 0;
        ROS_PRINT_INFO(
            "[FLOOR] CREATE id=%d height=%.2f",
            current_floor_id_,
            floors_.front().height);
    }

    if (current_floor_id_ < 0 ||
        current_floor_id_ >= static_cast<int>(floors_.size()))
        current_floor_id_ = 0;

    if (transition_ == 0)
    {
        if (std::abs(angle) > kStairStartAngle &&
            std::abs(angle) < kStairMaxAngle)
        {
            transition_ = angle > 0.0 ? 1 : -1;
        }

        if (transition_ != 0)
        {
            landing_distance_ = 0.0;
            key_floor_[static_cast<size_t>(key)] = -1;
            ROS_PRINT_INFO(
                "[FLOOR] START key=%d floor=%d dir=%s angle=%.3f",
                key,
                current_floor_id_,
                transition_ > 0 ? "UP" : "DOWN",
                angle);
            return;
        }

        key_floor_[static_cast<size_t>(key)] = current_floor_id_;
        return;
    }

    key_floor_[static_cast<size_t>(key)] = -1;

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
            return;

        const int old_floor = current_floor_id_;
        const double floor_dz = cur_z - floors_[static_cast<size_t>(old_floor)].height;

        if (std::abs(floor_dz) < kFloorHeightMatchTolerance ||
            transition_ * floor_dz <= 0.0)
        {
            const int direction = transition_;
            transition_ = 0;
            landing_distance_ = 0.0;

            key_floor_[static_cast<size_t>(key)] = old_floor;

            ROS_PRINT_INFO(
                "[FLOOR] CANCEL old=%d dir=%s floor_dz=%.2f",
                old_floor,
                direction > 0 ? "UP" : "DOWN",
                floor_dz);
            return;
        }

        int target_floor = -1;
        double best_diff = kFloorHeightMatchTolerance;
        for (int i = 0; i < static_cast<int>(floors_.size()); ++i)
        {
            if (i == old_floor)
                continue;

            const double diff = std::abs(
                floors_[static_cast<size_t>(i)].height - cur_z);
            if (diff < best_diff)
            {
                best_diff = diff;
                target_floor = i;
            }
        }

        if (target_floor < 0)
        {
            Floor floor;
            floor.height = cur_z;
            floors_.push_back(std::move(floor));
            target_floor = static_cast<int>(floors_.size()) - 1;
            ROS_PRINT_INFO(
                "[FLOOR] CREATE id=%d height=%.2f",
                target_floor,
                cur_z);
        }

        current_floor_id_ = target_floor;
        key_floor_[static_cast<size_t>(key)] = current_floor_id_;

        ROS_PRINT_INFO(
            "[FLOOR] END old=%d new=%d dir=%s floor_dz=%.2f",
            old_floor,
            target_floor,
            transition_ > 0 ? "UP" : "DOWN",
            floor_dz);

        transition_ = 0;
        landing_distance_ = 0.0;
        return;
    }

    if (std::abs(angle) > kStairStartAngle)
        landing_distance_ = 0.0;
}

bool FloorMap::sameFloor(int key1, int key2) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (key1 < 0 || key2 < 0 ||
        static_cast<size_t>(key1) >= key_floor_.size() ||
        static_cast<size_t>(key2) >= key_floor_.size())
        return false;
    const int floor1 = key_floor_[static_cast<size_t>(key1)];
    return floor1 >= 0 && floor1 == key_floor_[static_cast<size_t>(key2)];
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

    if (current_floor_id_ < 0 ||
        current_floor_id_ >= static_cast<int>(floors_.size()))
        return false;

    Floor &floor = floors_[static_cast<size_t>(current_floor_id_)];
    PlaneObs candidate;
    if (!buildGroundCandidate(planes, keys, poses, candidate))
        return false;

    Ground *best_ground = nullptr;
    double best_xy = std::numeric_limits<double>::infinity();
    for (auto &ground : floor.grounds)
    {
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

    Ground new_ground;
    Ground *active_ground = best_ground;
    const bool is_new_ground = (active_ground == nullptr);
    const int ground_id = is_new_ground ? next_plane_id_ : active_ground->id;
    const int last_added_key = active_ground ? active_ground->last_key : -1;
    int newest_key = last_added_key;

    if (is_new_ground)
    {
        new_ground.id = ground_id;
        new_ground.plane = candidate.plane;
        new_ground.center = candidate.center;
    }

    for (const auto &kv : candidate.obs)
    {
        const int key = kv.first;
        const Eigen::Vector4d &factor_obs = kv.second;
        if (key <= last_added_key)
            continue;

        batch.factors.emplace_back(key, ground_id, factor_obs);
        newest_key = std::max(newest_key, key);
    }

    if (batch.factors.empty())
        return false;

    if (is_new_ground)
    {
        batch.planes.emplace_back(current_floor_id_, ground_id, candidate.plane);
        floor.grounds.push_back(std::move(new_ground));
        floor.grounds.back().last_key = newest_key;
        next_plane_id_++;
    }
    else
    {
        active_ground->plane = candidate.plane;
        active_ground->center = candidate.center;
        active_ground->last_key = newest_key;
    }

    return true;
}
