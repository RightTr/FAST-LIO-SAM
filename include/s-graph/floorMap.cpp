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

double FloorMap::trajectoryAngle(const std::deque<PointTypePose> &poses) const
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
    double z_mean = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        x_mean += dist[i];
        z_mean += poses[begin + i].z;
    }
    x_mean /= static_cast<double>(count);
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

    const double slope = num / den;
    return std::atan(slope);
}

void FloorMap::update(const std::deque<int> &keys,
                      const std::deque<PointTypePose> &poses)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (keys.empty() || poses.empty())
        return;

    const int cur_key = keys.back();
    if (cur_key >= 0 && key_floor_.size() <= static_cast<size_t>(cur_key))
        key_floor_.resize(static_cast<size_t>(cur_key) + 1, -1);

    const PointTypePose &cur_pose = poses.back();
    const double cur_z = cur_pose.z;
    const double angle = trajectoryAngle(poses);

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
    {
        if (!floors_.empty())
            current_floor_id_ = 0;
    }

    if (current_floor_id_ < 0 ||
        current_floor_id_ >= static_cast<int>(floors_.size()))
        return;

    if (transition_ == 0)
    {
        if (angle > kStairStartAngle && angle < kStairMaxAngle)
        {
            transition_ = 1;
        }
        else if (angle < -kStairStartAngle && angle > -kStairMaxAngle)
        {
            transition_ = -1;
        }

        if (transition_ != 0)
        {
            landing_distance_ = 0.0;
            if (cur_key >= 0)
                key_floor_[static_cast<size_t>(cur_key)] = -1;
            ROS_PRINT_INFO(
                "[FLOOR] START key=%d floor=%d dir=%s angle=%.3f",
                cur_key,
                current_floor_id_,
                transition_ > 0 ? "UP" : "DOWN",
                angle);
            return;
        }

        if (cur_key >= 0)
            key_floor_[static_cast<size_t>(cur_key)] = current_floor_id_;
        return;
    }

    if (cur_key >= 0)
        key_floor_[static_cast<size_t>(cur_key)] = -1;

    if (std::abs(angle) < kStairEndAngle)
    {
        if (poses.size() >= 2)
        {
            const auto &last = poses[poses.size() - 2];
            const double dx = cur_pose.x - last.x;
            const double dy = cur_pose.y - last.y;
            landing_distance_ += std::sqrt(dx * dx + dy * dy);
        }

        if (landing_distance_ < kLandingConfirmDistance)
            return;

        const int old_floor = current_floor_id_;
        const double floor_dz = cur_z - floors_[static_cast<size_t>(old_floor)].height;

        if (std::abs(floor_dz) < kFloorHeightMatchTolerance ||
            transition_ * floor_dz <= 0.0)
        {
            transition_ = 0;
            landing_distance_ = 0.0;

            if (cur_key >= 0)
                key_floor_[static_cast<size_t>(cur_key)] = old_floor;

            ROS_PRINT_INFO(
                "[FLOOR] CANCEL old=%d dir=%s floor_dz=%.2f",
                old_floor,
                transition_ > 0 ? "UP" : "DOWN",
                floor_dz);
            return;
        }

        int target_floor = -1;
        double best_diff = kFloorHeightMatchTolerance;
        for (int i = 0; i < static_cast<int>(floors_.size()); ++i)
        {
            if (i == old_floor)
                continue;

            const double dh = floors_[static_cast<size_t>(i)].height -
                              floors_[static_cast<size_t>(old_floor)].height;
            if (transition_ * dh <= 0.0)
                continue;

            const double diff = std::abs(floors_[static_cast<size_t>(i)].height - cur_z);
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
        if (cur_key >= 0)
            key_floor_[static_cast<size_t>(cur_key)] = current_floor_id_;

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

void FloorMap::groundKeys(const std::deque<int> &keys,
                          std::unordered_set<int> &allowed_keys) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    allowed_keys.clear();

    if (keys.empty() || transition_ != 0 || current_floor_id_ < 0)
        return;

    const int confirmed_key = keys.back() - (kFloorSlopeWindow - 1);
    for (int key : keys)
    {
        if (key < 0 || static_cast<size_t>(key) >= key_floor_.size())
            continue;
        if (key > confirmed_key)
            continue;
        if (key_floor_[static_cast<size_t>(key)] != current_floor_id_)
            continue;
        allowed_keys.insert(key);
    }
}

bool FloorMap::allowLoop(int key1, int key2) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (key1 < 0 || key2 < 0)
        return false;
    if (static_cast<size_t>(key1) >= key_floor_.size() ||
        static_cast<size_t>(key2) >= key_floor_.size())
        return false;
    return key_floor_[static_cast<size_t>(key1)] >= 0 &&
           key_floor_[static_cast<size_t>(key1)] == key_floor_[static_cast<size_t>(key2)];
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
        const Eigen::Vector3d ground_n = ground.plane.head<3>();
        const Eigen::Vector3d candidate_n = candidate.plane.head<3>();
        const double ground_norm = ground_n.norm();
        const double candidate_norm = candidate_n.norm();
        if (ground_norm <= 1e-12 || candidate_norm <= 1e-12)
            continue;

        const Eigen::Vector3d gn = ground_n / ground_norm;
        const Eigen::Vector3d cn = candidate_n / candidate_norm;
        const double dot = std::max(-1.0, std::min(1.0, gn.dot(cn)));
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

    if (is_new_ground)
    {
        new_ground.id = ground_id;
        new_ground.plane = candidate.plane;
        new_ground.center = candidate.center;
    }

    int factor_key = -1;
    Eigen::Vector4d factor_obs = Eigen::Vector4d::Zero();
    for (const auto &kv : candidate.obs)
    {
        const int key = kv.first;
        if (key > factor_key)
        {
            factor_key = key;
            factor_obs = kv.second;
        }
    }

    if (factor_key < 0 || factor_key <= last_ground_key_)
        return false;

    batch.plane_factors.emplace_back(factor_key, ground_id, factor_obs);
    batch.floor_factors.emplace_back(current_floor_id_, factor_key, factor_obs);
    last_ground_key_ = factor_key;

    if (is_new_ground)
    {
        if (floor.grounds.empty())
            batch.floor_init.emplace_back(current_floor_id_, candidate.center.z());

        batch.plane_init.emplace_back(ground_id, candidate.plane, candidate.center);
        floor.grounds.push_back(std::move(new_ground));
        next_plane_id_++;
    }

    return !batch.plane_init.empty() ||
           !batch.plane_factors.empty() ||
           !batch.floor_init.empty() ||
           !batch.floor_factors.empty();
}
