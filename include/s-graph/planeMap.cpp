#include "s-graph/planeMap.h"

#include "common_utils.h"
#include "ros_utils.h"
#include "utility.h"

#include "balm/bavoxel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>

#include <Eigen/Eigenvalues>

namespace
{
constexpr int kMinPts = 100;
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

inline void deleteSurfMap(std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> &surf_map)
{
    for (auto &entry : surf_map)
        delete entry.second;
    surf_map.clear();
}

inline bool buildPlaneObs(const OCTO_TREE_NODE *node,
                          const std::deque<int> &keys,
                          const std::deque<PointTypePose> &poses,
                          PlaneObs &obs)
{
    if (node == nullptr || node->push_state != 1)
        return false;

    const int active_count = static_cast<int>(std::min(keys.size(), poses.size()));
    if (active_count <= 0)
        return false;

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

    if (map_n.z() < 0.0)
    {
        map_n = -map_n;
        map_d = -map_d;
    }

    obs.plane << map_n.x(), map_n.y(), map_n.z(), map_d;
    obs.center = map_c;

    for (int slot = 0; slot < active_count; ++slot)
    {
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
        if (expected_body_n.norm() > kEps && n.dot(expected_body_n) < 0.0)
        {
            n = -n;
            d = -d;
        }

        Eigen::Vector4d plane;
        plane << n.x(), n.y(), n.z(), d;
        obs.obs[keys[static_cast<size_t>(slot)]] = plane;
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

PlaneMap::~PlaneMap()
{
    reset();
}

void PlaneMap::reset()
{
    if (!impl_)
        return;

    deleteSurfMap(impl_->surf_map);
    impl_->keys.clear();
    impl_->poses.clear();
}

bool PlaneMap::update(int key,
                      const pcl::PointCloud<PointTypeIndex>::Ptr &cloud,
                      const PointTypePose &pose,
                      std::vector<PlaneObs> &planes,
                      pcl::PointCloud<PointTypeIndex>::Ptr display_cloud)
{
    if (!impl_)
        impl_ = std::make_unique<Impl>();

    planes.clear();
    if (display_cloud)
        display_cloud->clear();

    if (static_cast<int>(impl_->keys.size()) == impl_->window_size)
    {
        for (auto &entry : impl_->surf_map)
            entry.second->slide_window(1, impl_->window_size);

        impl_->keys.pop_front();
        impl_->poses.pop_front();
    }

    impl_->keys.push_back(key);
    impl_->poses.push_back(pose);

    const int slot = static_cast<int>(impl_->keys.size()) - 1;
    if (cloud && !cloud->empty())
    {
        IMUST x_key;
        x_key.R = poseRotation(pose);
        x_key.p = poseTranslation(pose);
        cut_voxel(impl_->surf_map, *cloud, x_key, slot);
    }

    const int active_count = static_cast<int>(std::min(impl_->keys.size(), impl_->poses.size()));
    for (auto &entry : impl_->surf_map)
        entry.second->recut(active_count);

    for (auto it = impl_->surf_map.begin(); it != impl_->surf_map.end(); )
    {
        OCTO_TREE_ROOT *root = it->second;
        if (root == nullptr || !root->has_active_points(active_count))
        {
            delete root;
            it = impl_->surf_map.erase(it);
        }
        else
        {
            ++it;
        }
    }

    if (display_cloud)
    {
        pcl::PointCloud<PointType> balm_plane_cloud;
        for (const auto &entry : impl_->surf_map)
            entry.second->tras_display(balm_plane_cloud, active_count);

        display_cloud->reserve(balm_plane_cloud.size());
        for (const auto &src : balm_plane_cloud.points)
        {
            PointTypeIndex dst;
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.intensity;
            display_cloud->push_back(dst);
        }
    }

    for (const auto &entry : impl_->surf_map)
        collectPlanes(entry.second, impl_->keys, impl_->poses, planes);

    return true;
}

const std::deque<int> &PlaneMap::keys() const
{
    static const std::deque<int> kEmpty;
    if (!impl_)
        return kEmpty;
    return impl_->keys;
}

const std::deque<PointTypePose> &PlaneMap::poses() const
{
    static const std::deque<PointTypePose> kEmpty;
    if (!impl_)
        return kEmpty;
    return impl_->poses;
}
