#include "s-graph/plane.h"

#include "common_utils.h"
#include "ros_utils.h"
#include "utility.h"

#include "balm/bavoxel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

#include <Eigen/Eigenvalues>

namespace
{
constexpr int kMinPts = 100;
constexpr double kGroundNzThreshold = 0.9396926207859084;
constexpr double kEps = 1e-12;

inline bool fitPlane(const PointCluster &s,
                     Eigen::Vector3d &n,
                     Eigen::Vector3d &c,
                     double &d)
{
    if (s.N <= 0.0)
        return false;

    c = s.v / s.N;
    Eigen::Matrix3d cov = s.P / s.N - c * c.transpose();
    cov = 0.5 * (cov + cov.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if (solver.info() != Eigen::Success)
        return false;

    n = solver.eigenvectors().col(0);
    if (n.norm() <= std::numeric_limits<double>::epsilon())
        return false;

    n.normalize();
    d = -n.dot(c);
    return true;
}

inline bool buildPlaneObs(const OCTO_TREE_NODE *node,
                          const std::deque<int> &keys,
                          const std::deque<PointTypePose> &poses,
                          PlaneObs &obs)
{
    if (node == nullptr || node->push_state != 1)
        return false;

    const int active_count = static_cast<int>(keys.size());
    PointCluster map_sum;
    obs = PlaneObs();

    for (int slot = 0; slot < active_count; ++slot)
    {
        const PointCluster &sig_tran = node->sig_tran[slot];
        const PointCluster &sig_orig = node->sig_orig[slot];
        if (sig_tran.N <= 0.0 || sig_orig.N <= 0.0)
            continue;

        map_sum += sig_tran;
        obs.n += static_cast<int>(sig_orig.N);
    }

    if (obs.n < kMinPts)
        return false;

    Eigen::Vector3d map_n = Eigen::Vector3d::Zero();
    Eigen::Vector3d map_c = Eigen::Vector3d::Zero();
    double map_d = 0.0;
    if (!fitPlane(map_sum, map_n, map_c, map_d))
        return false;

    const Eigen::Vector3d up = getGravityUp();
    const double nz = map_n.dot(up);
    if (std::abs(nz) < kGroundNzThreshold)
        return false;
    if (nz < 0.0)
    {
        map_n = -map_n;
        map_d = -map_d;
    }

    obs.plane << map_n.x(), map_n.y(), map_n.z(), map_d;
    obs.center = map_c;

    for (int slot = 0; slot < active_count; ++slot)
    {
        const int key = keys[static_cast<size_t>(slot)];
        const PointCluster &sig_orig = node->sig_orig[slot];
        if (sig_orig.N <= 0.0)
            continue;

        Eigen::Vector3d n;
        Eigen::Vector3d c;
        double d = 0.0;
        if (!fitPlane(sig_orig, n, c, d))
            continue;

        const Eigen::Vector3d expected_body_n =
            poseRotation(poses[static_cast<size_t>(slot)]).transpose() * map_n;
        if (n.dot(expected_body_n) < 0.0)
        {
            n = -n;
            d = -d;
        }

        Eigen::Vector4d plane;
        plane << n.x(), n.y(), n.z(), d;
        obs.obs[key] = plane;
    }

    return !obs.obs.empty();
}

inline void collectPlanes(const OCTO_TREE_NODE *node,
                          const std::deque<int> &keys,
                          const std::deque<PointTypePose> &poses,
                          std::vector<PlaneObs> &planes)
{
    if (node == nullptr)
        return;

    if (node->octo_state == 1)
    {
        for (const auto *child : node->leaves)
            collectPlanes(child, keys, poses, planes);
        return;
    }

    PlaneObs obs;
    if (buildPlaneObs(node, keys, poses, obs))
        planes.push_back(std::move(obs));
}

} // namespace

Plane::~Plane()
{
    reset();
}

void Plane::reset()
{
    for (auto &entry : surf_map_)
        delete entry.second;
    surf_map_.clear();
    keys_.clear();
    poses_.clear();
}

void Plane::update(int key,
                   const pcl::PointCloud<PointTypeIndex>::Ptr &cloud,
                   const PointTypePose &pose)
{
    if (static_cast<int>(keys_.size()) == kWindowSize)
    {
        for (auto &entry : surf_map_)
            entry.second->slide_window(1, kWindowSize);

        keys_.pop_front();
        poses_.pop_front();
    }

    keys_.push_back(key);
    poses_.push_back(pose);

    const int slot = static_cast<int>(keys_.size()) - 1;
    if (cloud && !cloud->empty())
    {
        IMUST x_key;
        x_key.R = poseRotation(pose);
        x_key.p = poseTranslation(pose);
        cut_voxel(surf_map_, *cloud, x_key, slot);
    }

    const int active_count = static_cast<int>(keys_.size());
    for (auto &entry : surf_map_)
        entry.second->recut(active_count);

    for (auto it = surf_map_.begin(); it != surf_map_.end(); )
    {
        OCTO_TREE_ROOT *root = it->second;
        if (root == nullptr || !root->has_active_points(active_count))
        {
            delete root;
            it = surf_map_.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return;
}

void Plane::extract(std::vector<PlaneObs> &planes) const
{
    planes.clear();
    if (keys_.empty() || poses_.empty())
        return;

    for (const auto &entry : surf_map_)
        collectPlanes(entry.second, keys_, poses_, planes);
}

const std::deque<int> &Plane::keys() const
{
    return keys_;
}

const std::deque<PointTypePose> &Plane::poses() const
{
    return poses_;
}
