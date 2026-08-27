#include "s-graph/ground.h"

#include "common_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
constexpr double kEps = 1e-12;
constexpr double kGroundNzThreshold = 0.9396926207859084;

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
                            const PointTypePose &pose,
                            const Eigen::Vector3d &seed_n,
                            PlaneObs &out)
{
    if (members.empty())
        return false;

    Eigen::Vector3d sum_n = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_center = Eigen::Vector3d::Zero();
    double map_weight_sum = 0.0;

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

        const double map_w = static_cast<double>(std::max(1, plane->n));
        sum_n += map_w * n;
        sum_center += map_w * plane->center;
        map_weight_sum += map_w;
    }

    if (map_weight_sum <= 0.0)
        return false;

    Eigen::Vector3d n = sum_n / map_weight_sum;
    if (!n.allFinite() || n.norm() <= kEps)
        return false;
    n.normalize();

    out = PlaneObs();
    out.center = sum_center / map_weight_sum;
    if (!out.center.allFinite())
        return false;

    out.plane << n.x(), n.y(), n.z(), -n.dot(out.center);
    const Eigen::Vector3d up = getGravityUp();
    if (out.plane.head<3>().dot(up) < 0.0)
        out.plane = -out.plane;
    out.plane.head<3>().normalize();
    out.n = static_cast<int>(std::llround(map_weight_sum));

    const Eigen::Matrix3d R = poseRotation(pose);
    const Eigen::Vector3d t = poseTranslation(pose);
    const Eigen::Vector3d body_n = R.transpose() * out.plane.head<3>();
    const double body_d = out.plane.head<3>().dot(t) + out.plane[3];
    out.body_plane << body_n.x(), body_n.y(), body_n.z(), body_d;

    return true;
}
} // namespace

bool buildGroundCandidate(const std::vector<PlaneObs> &planes,
                          const PointTypePose &pose,
                          PlaneObs &candidate)
{
    candidate = PlaneObs();
    if (planes.empty())
        return false;

    const Eigen::Vector3d cur_pos = poseTranslation(pose);
    const Eigen::Vector3d up = getGravityUp();

    std::vector<const PlaneObs *> seeds;
    seeds.reserve(planes.size());

    for (const auto &plane : planes)
    {
        Eigen::Vector4d map_plane = plane.plane;
        if (!normalizePlane(map_plane))
            continue;

        if (map_plane.head<3>().dot(up) < kGroundNzThreshold)
            continue;

        if (map_plane.head<3>().dot(cur_pos) + map_plane[3] <= 0.0)
            continue;

        seeds.push_back(&plane);
    }

    if (seeds.empty())
        return false;

    const PlaneObs *seed = nullptr;
    int best_support = -1;
    double best_distance = std::numeric_limits<double>::infinity();
    for (const auto *cand : seeds)
    {
        const int support = cand->n;
        const double distance = (cand->center - cur_pos).norm();

        if (seed == nullptr ||
            support > best_support ||
            (support == best_support && distance < best_distance))
        {
            seed = cand;
            best_support = support;
            best_distance = distance;
        }
    }

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

    return fitMergedGround(members, pose, seed_n, candidate);
}
