#include "s-graph/plane.h"

#include "common_utils.h"
#include "ros_utils.h"
#include "utility.h"

#include "balm/bavoxel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Eigenvalues>

namespace
{
constexpr int kMinPts = 100;

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
    if (active_count <= 0 || static_cast<int>(poses.size()) < active_count)
        return false;

    PointCluster map_sum;
    obs = PlaneObs();

    // Use all active frames to estimate one stable plane in odom/map frame.
    for (int slot = 0; slot < active_count; ++slot)
    {
        const PointCluster &sig_tran = node->sig_tran[slot];
        if (sig_tran.N <= 0.0)
            continue;

        map_sum += sig_tran;
        obs.n += static_cast<int>(sig_tran.N);
    }

    if (obs.n < kMinPts)
        return false;

    Eigen::Vector3d map_n = Eigen::Vector3d::Zero();
    Eigen::Vector3d map_c = Eigen::Vector3d::Zero();
    double map_d = 0.0;
    if (!fitPlane(map_sum, map_n, map_c, map_d))
        return false;

    const Eigen::Vector3d up = getGravityUp();
    if (map_n.dot(up) < 0.0)
    {
        map_n = -map_n;
        map_d = -map_d;
    }

    obs.plane << map_n.x(), map_n.y(), map_n.z(), map_d;
    obs.center = map_c;

    // Transform the same plane into the latest keyframe body frame.
    const PointTypePose &latest_pose = poses.back();
    const Eigen::Matrix3d R = poseRotation(latest_pose);
    const Eigen::Vector3d t = poseTranslation(latest_pose);

    Eigen::Vector3d body_n = R.transpose() * map_n;
    const double body_d = map_n.dot(t) + map_d;
    body_n.normalize();

    obs.body_plane << body_n.x(), body_n.y(), body_n.z(), body_d;
    return true;
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
