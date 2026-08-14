#include "s-graph/floorMap.h"

#include "common_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

namespace
{
constexpr int kFloorSlopeWindow = 6;
constexpr double kStairStartSlope = 0.20;
constexpr double kStairEndSlope = 0.10;
constexpr double kStairMaxSlope = 1.00;
constexpr double kMinHorizontalDistance = 0.50;
constexpr double kGroundLandmarkRadius = 8.0;
}

void FloorMap::update(const std::deque<int> &keys,
                      const std::deque<PointTypePose> &poses)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (keys.empty() || poses.empty())
        return;

    auto ensureKeyFloorSize = [this](int key)
    {
        if (key < 0)
            return;
        const size_t next = static_cast<size_t>(key) + 1;
        if (key_floor_.size() < next)
            key_floor_.resize(next, -1);
    };

    const int cur_key = keys.back();
    ensureKeyFloorSize(cur_key);

    if (floors_.empty())
    {
        floors_.push_back(Floor());
        current_floor_id_ = 0;
    }
    if (current_floor_id_ < 0)
        current_floor_id_ = 0;

    const double slope = trajectorySlope(poses);
    const double height = poses.back().z;
    const int old_floor = current_floor_id_;

    if (transition_ == 0)
    {
        if (slope > kStairStartSlope &&
            slope < kStairMaxSlope)
        {
            transition_ = 1;
            key_floor_[static_cast<size_t>(cur_key)] = -1;
            return;
        }

        if (slope < -kStairStartSlope &&
            slope > -kStairMaxSlope)
        {
            transition_ = -1;
            key_floor_[static_cast<size_t>(cur_key)] = -1;
            return;
        }

        key_floor_[static_cast<size_t>(cur_key)] = current_floor_id_;
        return;
    }

    if (transition_ == 1)
    {
        if (slope < kStairEndSlope)
        {
            transition_ = 0;
            int floor_idx = floors_[static_cast<size_t>(old_floor)].up;
            if (floor_idx < 0)
            {
                floors_.push_back(Floor());
                floor_idx = static_cast<int>(floors_.size()) - 1;
                floors_[static_cast<size_t>(old_floor)].up = floor_idx;
                floors_[static_cast<size_t>(floor_idx)].down = old_floor;
            }

            current_floor_id_ = floor_idx;
            key_floor_[static_cast<size_t>(cur_key)] = floor_idx;
            return;
        }

        key_floor_[static_cast<size_t>(cur_key)] = -1;
        return;
    }

    if (transition_ == -1)
    {
        if (slope > -kStairEndSlope)
        {
            transition_ = 0;
            int floor_idx = floors_[static_cast<size_t>(old_floor)].down;
            if (floor_idx < 0)
            {
                floors_.push_back(Floor());
                floor_idx = static_cast<int>(floors_.size()) - 1;
                floors_[static_cast<size_t>(old_floor)].down = floor_idx;
                floors_[static_cast<size_t>(floor_idx)].up = old_floor;
            }

            current_floor_id_ = floor_idx;
            key_floor_[static_cast<size_t>(cur_key)] = floor_idx;
            return;
        }

        key_floor_[static_cast<size_t>(cur_key)] = -1;
        return;
    }

    key_floor_[static_cast<size_t>(cur_key)] = current_floor_id_;
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

    if (keys.empty() || transition_ != 0 || current_floor_id_ < 0)
        return;

    const int confirmed_key = keys.back() - (kFloorSlopeWindow - 1);
    for (int key : keys)
    {
        if (key < 0 || static_cast<size_t>(key) >= key_floor_.size())
            continue;
        if (key_floor_[static_cast<size_t>(key)] != current_floor_id_)
            continue;
        if (key > confirmed_key)
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
        floors_.push_back(Floor());
        current_floor_id_ = 0;
    }
    if (current_floor_id_ < 0)
        current_floor_id_ = 0;
    if (current_floor_id_ >= static_cast<int>(floors_.size()))
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
