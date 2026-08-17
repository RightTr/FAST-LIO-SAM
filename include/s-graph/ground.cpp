#include "s-graph/ground.h"

#include "common_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace
{
constexpr double kEps = 1e-12;

inline bool normalizePlane(Eigen::Vector4d &plane)
{
    if (!plane.allFinite())
        return false;

    const double norm = plane.head<3>().norm();
    if (!std::isfinite(norm) || norm <= 1e-6)
        return false;

    plane /= norm;
    return plane.allFinite();
}

inline bool fitMergedGround(const std::vector<const PlaneObs *> &members,
                            const std::deque<int> &keys,
                            const std::deque<PointTypePose> &poses,
                            const Eigen::Vector3d &seed_n,
                            PlaneObs &out)
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
        if (seed_n.dot(n) < 0.0)
        {
            n = -n;
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

    out = PlaneObs();
    out.center = sum_center / weight_sum;
    if (!out.center.allFinite())
        return false;

    out.plane << n.x(), n.y(), n.z(), -n.dot(out.center);
    const Eigen::Vector3d up = getGravityUp();
    if (out.plane.head<3>().dot(up) < 0.0)
        out.plane = -out.plane;
    out.plane.head<3>().normalize();
    out.n = static_cast<int>(std::llround(weight_sum));

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

        out.obs[key] = body;
    }

    return !out.obs.empty();
}
} // namespace

bool buildGroundCandidate(const std::vector<PlaneObs> &planes,
                          const std::deque<int> &keys,
                          const std::deque<PointTypePose> &poses,
                          PlaneObs &candidate)
{
    candidate = PlaneObs();
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

    std::vector<const PlaneObs *> members;
    members.push_back(seed);

    for (const auto *cand : seeds)
    {
        if (cand == seed)
            continue;

        Eigen::Vector4d cand_plane = cand->plane;
        if (!normalizePlane(cand_plane))
            continue;

        if (seed_n.norm() <= kEps || cand_plane.head<3>().norm() <= kEps)
            continue;

        const double angle = std::acos(
            clampDot(seed_n.normalized().dot(cand_plane.head<3>().normalized())));
        if (!std::isfinite(angle) || angle > rad(groundAssociationAngleDeg))
            continue;

        if (std::abs(seed_n.dot(cand->center) + seed_plane[3]) >
            groundAssociationDistance)
        {
            continue;
        }

        members.push_back(cand);
    }

    return fitMergedGround(members, keys, poses, seed_n, candidate);
}
