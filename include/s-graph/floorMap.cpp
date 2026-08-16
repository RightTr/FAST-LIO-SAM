#include "s-graph/floorMap.h"

#include "common_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace
{
constexpr int kFloorSlopeWindow = 6;
constexpr double kStairStartSlope = 0.20;
constexpr double kStairEndSlope = 0.10;
constexpr double kStairMaxSlope = 1.00;
constexpr double kMinHorizontalDistance = 0.50;
constexpr double kGroundLandmarkRadius = 8.0;
constexpr double kLandingConfirmDistance = 1.8;
constexpr double kFloorHeightMatchTolerance = 1.5;
} // namespace

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

    return num / den;
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
    const double slope = trajectorySlope(poses);

    auto findFloorByLevel = [this](int level) -> int
    {
        for (int i = 0; i < static_cast<int>(floors_.size()); ++i)
        {
            if (floors_[static_cast<size_t>(i)].level == level)
                return i;
        }
        return -1;
    };

    auto createFloor = [this](int level, double height) -> int
    {
        Floor floor;
        floor.level = level;
        floor.height = height;
        floors_.push_back(std::move(floor));
        return static_cast<int>(floors_.size()) - 1;
    };

    if (floors_.empty())
    {
        current_floor_id_ = createFloor(0, cur_z);
        ROS_PRINT_INFO(
            "[FLOOR] CREATE id=%d level=%d height=%.2f",
            current_floor_id_,
            floors_.front().level,
            floors_.front().height);
    }

    if (current_floor_id_ < 0 ||
        current_floor_id_ >= static_cast<int>(floors_.size()))
    {
        current_floor_id_ = findFloorByLevel(0);
        if (current_floor_id_ < 0 && !floors_.empty())
            current_floor_id_ = 0;
    }

    if (current_floor_id_ < 0 ||
        current_floor_id_ >= static_cast<int>(floors_.size()))
        return;

    const int current_level = floors_[static_cast<size_t>(current_floor_id_)].level;

    if (transition_ == 0)
    {
        if (slope > kStairStartSlope && slope < kStairMaxSlope)
        {
            transition_ = 1;
            transition_start_z_ = cur_z;
            landing_distance_ = 0.0;
            if (cur_key >= 0)
                key_floor_[static_cast<size_t>(cur_key)] = -1;
            ROS_PRINT_INFO(
                "[FLOOR] START key=%d floor=%d level=%d dir=UP z=%.2f",
                cur_key,
                current_floor_id_,
                current_level,
                cur_z);
            return;
        }

        if (slope < -kStairStartSlope && slope > -kStairMaxSlope)
        {
            transition_ = -1;
            transition_start_z_ = cur_z;
            landing_distance_ = 0.0;
            if (cur_key >= 0)
                key_floor_[static_cast<size_t>(cur_key)] = -1;
            ROS_PRINT_INFO(
                "[FLOOR] START key=%d floor=%d level=%d dir=DOWN z=%.2f",
                cur_key,
                current_floor_id_,
                current_level,
                cur_z);
            return;
        }

        if (cur_key >= 0)
            key_floor_[static_cast<size_t>(cur_key)] = current_floor_id_;
        return;
    }

    if (std::abs(slope) < kStairEndSlope)
    {
        if (poses.size() >= 2)
        {
            const auto &last = poses[poses.size() - 2];
            const double dx = cur_pose.x - last.x;
            const double dy = cur_pose.y - last.y;
            landing_distance_ += std::sqrt(dx * dx + dy * dy);
        }

        if (cur_key >= 0)
            key_floor_[static_cast<size_t>(cur_key)] = -1;

        if (landing_distance_ < kLandingConfirmDistance)
            return;

        const int start_floor = current_floor_id_;
        const int start_level = floors_[static_cast<size_t>(start_floor)].level;
        const int direction = transition_;
        const double dz = cur_z - transition_start_z_;

        int target_floor = -1;
        double best_diff = kFloorHeightMatchTolerance;
        for (int i = 0; i < static_cast<int>(floors_.size()); ++i)
        {
            if (i == start_floor)
                continue;

            const int level = floors_[static_cast<size_t>(i)].level;
            if (direction > 0 && level <= start_level)
                continue;
            if (direction < 0 && level >= start_level)
                continue;

            const double diff = std::abs(floors_[static_cast<size_t>(i)].height - cur_z);
            if (diff > best_diff)
                continue;

            if (diff < best_diff)
            {
                best_diff = diff;
                target_floor = i;
                continue;
            }

            if (target_floor >= 0)
            {
                const int best_gap =
                    std::abs(floors_[static_cast<size_t>(target_floor)].level - start_level);
                const int gap = std::abs(level - start_level);
                if (gap < best_gap)
                    target_floor = i;
            }
        }

        bool matched_existing = (target_floor >= 0);
        int span = 1;
        int target_level = start_level + direction;

        if (target_floor < 0)
        {
            const double nominal = std::max(1e-3, floorHeight);
            span = std::max(
                1,
                static_cast<int>(std::lround(std::abs(dz) / nominal)));
            target_level = start_level + direction * span;

            target_floor = findFloorByLevel(target_level);
            if (target_floor < 0)
            {
                target_floor = createFloor(target_level, cur_z);
                ROS_PRINT_INFO(
                    "[FLOOR] CREATE id=%d level=%d height=%.2f",
                    target_floor,
                    target_level,
                    cur_z);
            }
            else
            {
                matched_existing = true;
            }
        }
        else
        {
            target_level = floors_[static_cast<size_t>(target_floor)].level;
            span = std::abs(target_level - start_level);
            if (span <= 0)
                span = 1;
        }

        if (target_floor < 0 ||
            target_floor >= static_cast<int>(floors_.size()))
            return;

        current_floor_id_ = target_floor;
        if (cur_key >= 0)
            key_floor_[static_cast<size_t>(cur_key)] = current_floor_id_;

        ROS_PRINT_INFO(
            "[FLOOR] END old=%d level=%d new=%d level=%d dir=%s dz=%.2f span=%d matched_existing=%d",
            start_floor,
            start_level,
            target_floor,
            floors_[static_cast<size_t>(target_floor)].level,
            direction > 0 ? "UP" : "DOWN",
            dz,
            span,
            static_cast<int>(matched_existing));

        transition_ = 0;
        transition_start_z_ = 0.0;
        landing_distance_ = 0.0;
        return;
    }

    landing_distance_ = 0.0;
    if (cur_key >= 0)
        key_floor_[static_cast<size_t>(cur_key)] = -1;
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

int FloorMap::floor(int key) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (key < 0 || static_cast<size_t>(key) >= key_floor_.size())
        return -1;

    return key_floor_[static_cast<size_t>(key)];
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
                            bool use_floor,
                            SceneBatch &batch)
{
    std::lock_guard<std::mutex> lock(mutex_);
    batch = SceneBatch();
    if (keys.empty() || poses.empty() || planes.empty())
        return false;

    if (floors_.empty())
    {
        Floor floor;
        floor.level = 0;
        floor.height = 0.0;
        floors_.push_back(std::move(floor));
        current_floor_id_ = 0;
    }

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

        if (!use_floor)
        {
            const double plane_dist =
                std::abs(gn.dot(candidate.center) + ground.plane[3] / ground_norm);
            if (plane_dist > groundAssociationDistance)
                continue;
        }

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
    if (use_floor)
        batch.floor_factors.emplace_back(current_floor_id_, factor_key, factor_obs);
    last_ground_key_ = factor_key;

    if (is_new_ground)
    {
        if (use_floor && floor.grounds.empty())
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
