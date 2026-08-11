#ifndef S_GRAPH_PLANE_MAP_H
#define S_GRAPH_PLANE_MAP_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "balm/bavoxel.hpp"
#include "map_optimization.h"

#include <Eigen/Core>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

struct PlaneObs
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector4d plane = Eigen::Vector4d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    int n = 0;
    std::unordered_map<int, Eigen::Vector4d> obs;
};

class PlaneMap
{
public:
    PlaneMap() = default;
    ~PlaneMap();

    void reset();

    bool update(int key,
                const pcl::PointCloud<PointTypeIndex>::Ptr &cloud,
                const PointTypePose &pose,
                std::vector<PlaneObs> &planes,
                pcl::PointCloud<PointTypeIndex>::Ptr display_cloud = nullptr);

    const std::deque<int> &keys() const;
    const std::deque<PointTypePose> &poses() const;

private:
    struct Impl
    {
        int window_size = 20;
        std::deque<int> keys;
        std::deque<PointTypePose> poses;
        std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;
    };
    std::unique_ptr<Impl> impl_;
};

#endif
