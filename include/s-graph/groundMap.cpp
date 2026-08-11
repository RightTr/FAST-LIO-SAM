#include "s-graph/groundMap.h"

#include "common_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace
{
constexpr double kEps = 1e-12;

inline bool validPlane(const Eigen::Vector4d &plane)
{
    if (!plane.allFinite())
        return false;

    const double norm = plane.head<3>().norm();
    return std::isfinite(norm) && norm > 1e-6;
}

inline bool normalizePlane(Eigen::Vector4d &plane)
{
    if (!validPlane(plane))
        return false;

    const double norm = plane.head<3>().norm();
    plane /= norm;
    return plane.allFinite();
}

inline void canonicalizePlane(Eigen::Vector4d &plane)
{
    if (plane[2] < 0.0)
        plane = -plane;
}

inline double planeAngle(const Eigen::Vector3d &a,
                         const Eigen::Vector3d &b)
{
    if (a.norm() <= kEps || b.norm() <= kEps)
        return std::numeric_limits<double>::infinity();

    return std::acos(clampDot(a.normalized().dot(b.normalized())));
}

inline double planeDistanceToPoint(const Eigen::Vector3d &n,
                                   double d,
                                   const Eigen::Vector3d &p)
{
    return std::abs(n.dot(p) + d);
}

inline bool isGroundCandidate(const PlaneObs &plane)
{
    if (plane.n < groundMinPoints)
        return false;

    Eigen::Vector4d normalized = plane.plane;
    if (!normalizePlane(normalized))
        return false;

    if (std::acos(clampDot(normalized[2])) > rad(groundMaxSlopeDeg))
        return false;

    return true;
}

struct GroundCandidate
{
    Eigen::Vector3d n = Eigen::Vector3d::Zero();
    double d = 0.0;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    int n_pts = 0;
    std::unordered_map<int, Eigen::Vector4d> body;
};

inline void appendUniqueKey(std::vector<int> &keys, int key)
{
    if (std::find(keys.begin(), keys.end(), key) == keys.end())
        keys.push_back(key);
}

inline int chooseRepresentativeKey(const GroundCandidate &candidate,
                                   int window_start,
                                   int count)
{
    const int latest = window_start + count - 1;
    if (candidate.body.find(latest) != candidate.body.end())
        return latest;

    if (!candidate.body.empty())
        return candidate.body.begin()->first;

    return latest;
}

inline bool fitMergedGround(const std::vector<const PlaneObs *> &members,
                            const std::vector<PointTypePose> &poses,
                            int window_start,
                            const Eigen::Vector3d &seed_n,
                            GroundCandidate &out)
{
    if (members.empty())
        return false;

    Eigen::Vector3d sum_n = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_center = Eigen::Vector3d::Zero();
    double weight_sum = 0.0;
    int total_pts = 0;

    std::unordered_map<int, Eigen::Vector4d> body_sum;
    std::unordered_map<int, double> body_weight;

    for (const auto *plane : members)
    {
        Eigen::Vector4d map_plane = plane->plane;
        if (!normalizePlane(map_plane))
            continue;

        Eigen::Vector3d n = map_plane.head<3>();
        double d = map_plane[3];
        if (seed_n.dot(n) < 0.0)
        {
            n = -n;
            d = -d;
        }

        const double w = static_cast<double>(std::max(1, plane->n));
        sum_n += w * n;
        sum_center += w * plane->center;
        weight_sum += w;
        total_pts += plane->n;

        for (const auto &kv : plane->obs)
        {
            const int key = kv.first;
            const int local_id = key - window_start;
            if (local_id < 0 || local_id >= static_cast<int>(poses.size()))
                continue;

            Eigen::Vector4d obs = kv.second;
            if (!normalizePlane(obs))
                continue;

            const Eigen::Vector3d expected_body_n =
                poseRotation(poses[static_cast<size_t>(local_id)]).transpose() * seed_n;
            if (expected_body_n.dot(obs.head<3>()) < 0.0)
                obs = -obs;

            auto &sum_slot = body_sum[key];
            auto &weight_slot = body_weight[key];
            if (weight_slot == 0.0)
                sum_slot = Eigen::Vector4d::Zero();
            sum_slot += w * obs;
            weight_slot += w;
        }
    }

    if (weight_sum <= 0.0)
        return false;

    Eigen::Vector3d n = sum_n / weight_sum;
    if (!n.allFinite() || n.norm() <= kEps)
        return false;
    n.normalize();

    out = GroundCandidate();
    out.n = n;
    out.center = sum_center / weight_sum;
    if (!out.center.allFinite())
        return false;

    out.d = -out.n.dot(out.center);
    Eigen::Vector4d plane;
    plane << out.n.x(), out.n.y(), out.n.z(), out.d;
    canonicalizePlane(plane);
    out.n = plane.head<3>();
    out.d = plane[3];
    if (!validPlane(plane))
        return false;

    out.n_pts = total_pts;

    for (const auto &kv : body_sum)
    {
        const int key = kv.first;
        const double w = body_weight[key];
        if (w <= 0.0)
            continue;

        Eigen::Vector4d body = kv.second / w;
        if (!normalizePlane(body))
            continue;

        if (body.head<3>().norm() <= kEps)
            continue;

        out.body[key] = body;
    }

    return !out.body.empty();
}

inline bool buildStableGround(const std::vector<PlaneObs> &planes,
                              const std::vector<PointTypePose> &poses,
                              int window_start,
                              GroundCandidate &candidate)
{
    candidate = GroundCandidate();
    if (planes.empty() || poses.empty())
        return false;

    const Eigen::Vector3d cur_pos = poseTranslation(poses.back());

    std::vector<const PlaneObs *> seeds;
    seeds.reserve(planes.size());

    for (const auto &plane : planes)
    {
        if (!isGroundCandidate(plane))
            continue;

        Eigen::Vector4d map_plane = plane.plane;
        if (!normalizePlane(map_plane))
            continue;

        if (map_plane.head<3>().dot(cur_pos) + map_plane[3] <= 0.0)
            continue;

        seeds.push_back(&plane);
    }

    if (seeds.empty())
        return false;

    const PlaneObs *seed = *std::max_element(
        seeds.begin(),
        seeds.end(),
        [](const PlaneObs *a, const PlaneObs *b)
        {
            return a->n < b->n;
        });

    Eigen::Vector4d seed_plane = seed->plane;
    if (!normalizePlane(seed_plane))
        return false;

    const Eigen::Vector3d seed_n = seed_plane.head<3>();
    const double seed_d = seed_plane[3];

    std::vector<const PlaneObs *> members;
    members.push_back(seed);

    for (const auto *cand : seeds)
    {
        if (cand == seed)
            continue;

        Eigen::Vector4d cand_plane = cand->plane;
        if (!normalizePlane(cand_plane))
            continue;

        const double angle = planeAngle(seed_n, cand_plane.head<3>());
        if (!std::isfinite(angle) || angle > rad(groundAssociationAngleDeg))
            continue;

        if (planeDistanceToPoint(seed_n, seed_d, cand->center) >
            groundAssociationDistance)
        {
            continue;
        }

        members.push_back(cand);
    }

    return fitMergedGround(members, poses, window_start, seed_n, candidate);
}

inline bool matchGeometry(const Ground &ground,
                          const Eigen::Vector3d &n,
                          double d,
                          const Eigen::Vector3d &center,
                          double &score)
{
    if (!n.allFinite() || !std::isfinite(d) || !center.allFinite())
        return false;

    Eigen::Vector4d ground_plane = ground.plane;
    if (!normalizePlane(ground_plane))
        return false;

    const Eigen::Vector3d g_n = ground_plane.head<3>();
    const double g_d = ground_plane[3];

    const double angle = planeAngle(g_n, n);
    if (!std::isfinite(angle) || angle > rad(groundAssociationAngleDeg))
        return false;

    score = planeDistanceToPoint(g_n, g_d, center);
    return std::isfinite(score) && score <= groundAssociationDistance;
}
} // namespace

bool GroundMap::update(const std::vector<PlaneObs> &planes,
                       const std::vector<PointTypePose> &poses,
                       int window_start,
                       PlaneBatch &batch)
{
    batch = PlaneBatch();
    if (planes.empty() || poses.empty())
        return false;

    GroundCandidate candidate;
    if (!buildStableGround(planes, poses, window_start, candidate))
        return false;

    if (!candidate.n.allFinite() ||
        !std::isfinite(candidate.d) ||
        !candidate.center.allFinite() ||
        candidate.n.norm() <= kEps ||
        candidate.body.empty())
    {
        return false;
    }

    const int count = static_cast<int>(poses.size());

    int target_id = -1;
    int best_votes = -1;
    double best_score = std::numeric_limits<double>::infinity();

    if (current_id_ >= 0 &&
        current_id_ < static_cast<int>(grounds_.size()))
    {
        double score = std::numeric_limits<double>::infinity();
        if (matchGeometry(grounds_[static_cast<size_t>(current_id_)],
                          candidate.n,
                          candidate.d,
                          candidate.center,
                          score))
        {
            target_id = current_id_;
            best_score = score;
            best_votes = std::numeric_limits<int>::max();
        }
    }

    if (target_id < 0)
    {
        std::unordered_map<int, int> votes;
        for (const auto &kv : candidate.body)
        {
            const auto it = key_plane_.find(kv.first);
            if (it != key_plane_.end())
                ++votes[it->second];
        }

        for (const auto &vote : votes)
        {
            const int id = vote.first;
            const int vote_count = vote.second;
            if (id < 0 || id >= static_cast<int>(grounds_.size()))
                continue;

            double score = std::numeric_limits<double>::infinity();
            if (!matchGeometry(grounds_[static_cast<size_t>(id)],
                               candidate.n,
                               candidate.d,
                               candidate.center,
                               score))
            {
                continue;
            }

            if (vote_count > best_votes ||
                (vote_count == best_votes && score < best_score))
            {
                target_id = id;
                best_votes = vote_count;
                best_score = score;
            }
        }
    }

    if (target_id < 0)
    {
        for (size_t i = 0; i < grounds_.size(); ++i)
        {
            double score = std::numeric_limits<double>::infinity();
            if (!matchGeometry(grounds_[i],
                               candidate.n,
                               candidate.d,
                               candidate.center,
                               score))
            {
                continue;
            }

            if (score < best_score)
            {
                target_id = static_cast<int>(i);
                best_score = score;
            }
        }
    }

    const bool create_new = (target_id < 0);
    const int new_id = create_new ? next_id_ : -1;

    std::vector<PlaneFactor> factors;
    factors.reserve(candidate.body.size());
    std::vector<int> factor_keys;
    factor_keys.reserve(candidate.body.size());

    for (const auto &kv : candidate.body)
    {
        const int key = kv.first;

        if (key_plane_.find(key) != key_plane_.end())
            continue;

        const int local_id = key - window_start;
        if (local_id < 0 || local_id >= count)
            continue;

        Eigen::Vector4d obs = kv.second;
        if (!normalizePlane(obs) || !obs.allFinite())
            continue;
        if (obs.head<3>().norm() <= kEps)
            continue;

        PlaneFactor factor;
        factor.key = key;
        factor.id = create_new ? new_id : target_id;
        factor.obs = obs;
        factors.push_back(factor);
        factor_keys.push_back(key);
    }

    if (factors.empty())
        return false;

    Ground ground;
    Ground *target = nullptr;
    if (create_new)
    {
        ground.id = next_id_++;
        ground.plane << candidate.n.x(), candidate.n.y(), candidate.n.z(), candidate.d;
        ground.center = candidate.center;
        grounds_.push_back(ground);
        batch.init.push_back(ground);
        target = &grounds_.back();

        const int rep_key = chooseRepresentativeKey(candidate, window_start, count);
        ROS_PRINT_INFO(
            "Ground init: p%d from KF=%d normal=(%.3f %.3f %.3f %.3f)",
            target->id,
            rep_key,
            target->plane[0],
            target->plane[1],
            target->plane[2],
            target->plane[3]);
        current_id_ = target->id;
    }
    else
    {
        target = &grounds_[static_cast<size_t>(target_id)];
    }

    if (!target)
        return false;

    target->plane << candidate.n.x(), candidate.n.y(), candidate.n.z(), candidate.d;
    target->center = candidate.center;
    current_id_ = target->id;

    for (const auto &factor : factors)
    {
        key_plane_[factor.key] = target->id;
        appendUniqueKey(target->keys, factor.key);
        batch.factors.push_back(factor);

        const int local_id = factor.key - window_start;
        if (local_id >= 0 && local_id < count)
        {
            const Eigen::Vector3d pose_pos = poseTranslation(poses[static_cast<size_t>(local_id)]);
            const double plane_h = candidate.n.dot(pose_pos) + candidate.d;
            ROS_PRINT_INFO(
                "Ground KF=%d plane_h=%.3f normal=(%.3f %.3f %.3f)",
                factor.key,
                plane_h,
                candidate.n.x(),
                candidate.n.y(),
                candidate.n.z());
        }

        ROS_PRINT_INFO("Plane factor: pose=%d -> p%d", factor.key, target->id);
    }

    batch.valid = !batch.init.empty() || !batch.factors.empty();
    return batch.valid;
}
