#include "s-graph/ground.h"

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

struct GroundCandidate
{
    Eigen::Vector3d n = Eigen::Vector3d::Zero();
    double d = 0.0;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    std::unordered_map<int, Eigen::Vector4d> body;
};

inline bool fitMergedGround(const std::vector<const PlaneObs *> &members,
                            const std::deque<int> &keys,
                            const std::deque<PointTypePose> &poses,
                            const Eigen::Vector3d &seed_n,
                            GroundCandidate &out)
{
    if (members.empty())
        return false;

    Eigen::Vector3d sum_n = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_center = Eigen::Vector3d::Zero();
    double weight_sum = 0.0;
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
        for (const auto &kv : plane->obs)
        {
            const int key = kv.first;
            const int local_id = key - keys.front();
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
                              const std::deque<int> &keys,
                              const std::deque<PointTypePose> &poses,
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
        if (plane.type != PlaneType::GROUND)
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

    return fitMergedGround(members, keys, poses, seed_n, candidate);
}

inline bool matchGround(const PlaneLandmark &ground,
                        const GroundCandidate &candidate)
{
    if (!candidate.n.allFinite() ||
        !std::isfinite(candidate.d) ||
        !candidate.center.allFinite())
        return false;

    Eigen::Vector4d ground_plane = ground.plane;
    if (!normalizePlane(ground_plane))
        return false;

    const Eigen::Vector3d g_n = ground_plane.head<3>();
    const double g_d = ground_plane[3];

    const double angle = planeAngle(g_n, candidate.n);
    if (!std::isfinite(angle) || angle > rad(groundAssociationAngleDeg))
        return false;

    const double score = planeDistanceToPoint(g_n, g_d, candidate.center);
    return std::isfinite(score) && score <= groundAssociationDistance;
}
} // namespace

bool Ground::update(const std::vector<PlaneObs> &planes,
                    const std::deque<int> &keys,
                    const std::deque<PointTypePose> &poses,
                    PlaneBatch &batch)
{
    batch = PlaneBatch();
    if (planes.empty() || poses.empty() || keys.empty())
        return false;

    GroundCandidate candidate;
    if (!buildStableGround(planes, keys, poses, candidate))
        return false;

    if (candidate.body.empty())
        return false;

    if (ground_.id < 0)
    {
        ground_.id = 0;
        ground_.plane << candidate.n.x(), candidate.n.y(), candidate.n.z(), candidate.d;
        ground_.center = candidate.center;
        batch.init.push_back(ground_);

        const int rep_key = keys.empty() ? -1 : keys.back();
        ROS_PRINT_INFO("Ground init: p0 from KF=%d plane=(%.3f %.3f %.3f %.3f)",
                       rep_key,
                       ground_.plane[0],
                       ground_.plane[1],
                       ground_.plane[2],
                       ground_.plane[3]);
    }
    else
    {
        if (!matchGround(ground_, candidate))
        {
            return false;
        }
    }

    ground_.plane << candidate.n.x(), candidate.n.y(), candidate.n.z(), candidate.d;
    ground_.center = candidate.center;
    std::vector<PlaneFactor> factors;
    factors.reserve(candidate.body.size());

    for (const auto &kv : candidate.body)
    {
        const int key = kv.first;
        if (keys_.find(key) != keys_.end())
            continue;

        PlaneFactor factor;
        factor.key = key;
        factor.id = ground_.id;
        factor.obs = kv.second;
        factors.push_back(factor);
    }

    if (factors.empty())
        return false;

    for (const auto &factor : factors)
    {
        keys_.insert(factor.key);
        batch.factors.push_back(factor);
    }

    batch.valid = !batch.init.empty() || !batch.factors.empty();
    return batch.valid;
}
