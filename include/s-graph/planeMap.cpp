#include "s-graph/planeMap.h"

#include "common_utils.h"
#include "ros_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

#include "balm/bavoxel.hpp"

#include <Eigen/Eigenvalues>
#include <pcl/common/transforms.h>

namespace
{
struct PlaneLocalStats
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int n = 0;
    Eigen::Vector3d s = Eigen::Vector3d::Zero();
    Eigen::Matrix3d q = Eigen::Matrix3d::Zero();
};

struct PlanePatch
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d n_map = Eigen::Vector3d::Zero();
    Eigen::Vector3d c_map = Eigen::Vector3d::Zero();
    double d = 0.0;
    int n = 0;
    std::unordered_map<int, PlaneLocalStats> obs;
};

constexpr int kMinPts = 100;

inline bool fit(const PlaneLocalStats &s,
                Eigen::Vector3d &n,
                Eigen::Vector3d &c,
                double &d)
{
    if (s.n <= 0)
        return false;

    c = s.s / static_cast<double>(s.n);
    Eigen::Matrix3d cov = s.q / static_cast<double>(s.n) - c * c.transpose();
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

inline void acc(PlaneLocalStats &dst, const PlaneLocalStats &src)
{
    dst.n += src.n;
    dst.s += src.s;
    dst.q += src.q;
}

inline bool tf(const PlaneLocalStats &in,
               const PointTypePose &pose,
               PlaneLocalStats &out)
{
    if (in.n <= 0)
        return false;

    const Eigen::Matrix3d R = poseRotation(pose);
    const Eigen::Vector3d t = poseTranslation(pose);

    out.n = in.n;
    out.s = R * in.s + static_cast<double>(in.n) * t;
    out.q = R * in.q * R.transpose()
          + R * in.s * t.transpose()
          + t * in.s.transpose() * R.transpose()
          + static_cast<double>(in.n) * t * t.transpose();
    return true;
}

inline bool makePatch(const OCTO_TREE_NODE *node,
                      int window_start,
                      int window_count,
                      PlanePatch &p)
{
    if (node == nullptr || node->push_state != 1)
        return false;

    p = PlanePatch();
    PlaneLocalStats map_sum;

    for (int local_id = 0; local_id < window_count; ++local_id)
    {
        const PointCluster &sig_tran = node->sig_tran[local_id];
        const PointCluster &sig_orig = node->sig_orig[local_id];
        if (sig_tran.N <= 0 || sig_orig.N <= 0)
            continue;

        PlaneLocalStats o;
        o.n = static_cast<int>(sig_orig.N);
        o.s = sig_orig.v;
        o.q = sig_orig.P;
        p.obs[window_start + local_id] = o;
        p.n += o.n;

        PlaneLocalStats m;
        m.n = static_cast<int>(sig_tran.N);
        m.s = sig_tran.v;
        m.q = sig_tran.P;
        acc(map_sum, m);
    }

    if (p.n < kMinPts)
        return false;

    if (!fit(map_sum, p.n_map, p.c_map, p.d))
        return false;

    if (p.n_map.z() < 0.0)
    {
        p.n_map = -p.n_map;
        p.d = -p.d;
    }

    return true;
}

inline void collect(const OCTO_TREE_NODE *node,
                    int window_start,
                    int window_count,
                    std::vector<PlanePatch> &patches)
{
    if (node == nullptr)
        return;

    if (node->octo_state == 1)
    {
        for (const auto *child : node->leaves)
            collect(child, window_start, window_count, patches);
        return;
    }

    PlanePatch p;
    if (makePatch(node, window_start, window_count, p))
        patches.push_back(std::move(p));
}

inline bool fillObs(const PlanePatch &patch,
                    const std::vector<PointTypePose> &poses,
                    int window_start,
                    PlaneObs &obs)
{
    obs = PlaneObs();
    obs.plane << patch.n_map.x(), patch.n_map.y(), patch.n_map.z(), patch.d;
    obs.center = patch.c_map;
    obs.n = patch.n;

    for (const auto &kv : patch.obs)
    {
        const int global_key = kv.first;
        const int local_id = global_key - window_start;
        if (local_id < 0 || local_id >= static_cast<int>(poses.size()))
            continue;

        Eigen::Vector3d n;
        Eigen::Vector3d c;
        double d = 0.0;
        if (!fit(kv.second, n, c, d))
            continue;

        const Eigen::Vector3d exp = poseRotation(poses[local_id]).transpose() * patch.n_map;
        if (exp.norm() > std::numeric_limits<double>::epsilon() && n.dot(exp) < 0.0)
        {
            n = -n;
            d = -d;
        }

        Eigen::Vector4d plane;
        plane << n.x(), n.y(), n.z(), d;
        obs.obs[global_key] = plane;
    }

    return !obs.obs.empty();
}
} // namespace

bool PlaneMap::extract(const std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> &clouds,
                       const std::vector<PointTypePose> &poses,
                       int window_start,
                       std::vector<PlaneObs> &planes,
                       pcl::PointCloud<PointTypeIndex>::Ptr cloud) const
{
    planes.clear();
    if (cloud)
        cloud->clear();

    const size_t total_size = std::min(clouds.size(), poses.size());
    if (total_size == 0)
        return false;

    const int count = static_cast<int>(total_size);
    std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

    for (int local_id = 0; local_id < count; ++local_id)
    {
        IMUST x_key;
        const auto &pose = poses[local_id];
        const Eigen::Matrix3d R = poseRotation(pose);
        x_key.R = R;
        x_key.p = poseTranslation(pose);

        pcl::PointCloud<PointType> local_balm_cloud;
        if (clouds[local_id])
        {
            local_balm_cloud.reserve(clouds[local_id]->size());
            for (const auto &src : clouds[local_id]->points)
            {
                PointType dst;
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                local_balm_cloud.push_back(dst);
            }
        }

        cut_voxel(surf_map, local_balm_cloud, x_key, local_id);
    }

    for (auto &entry : surf_map)
        entry.second->recut(count);

    std::vector<PlanePatch> patches;
    patches.reserve(surf_map.size());
    for (const auto &entry : surf_map)
        collect(entry.second, window_start, count, patches);

    pcl::PointCloud<PointType> balm_plane_cloud;
    for (const auto &entry : surf_map)
        entry.second->tras_display(balm_plane_cloud, count);

    if (cloud)
    {
        cloud->reserve(balm_plane_cloud.size());
        for (const auto &src : balm_plane_cloud.points)
        {
            PointTypeIndex dst;
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.intensity;
            cloud->push_back(dst);
        }
    }

    planes.reserve(patches.size());
    for (const auto &patch : patches)
    {
        PlaneObs obs;
        if (!fillObs(patch, poses, window_start, obs))
            continue;
        planes.push_back(std::move(obs));
    }

    for (auto &entry : surf_map)
        delete entry.second;
    surf_map.clear();

    return !planes.empty();
}
