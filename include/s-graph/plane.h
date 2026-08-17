#ifndef PLANE_H
#define PLANE_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "balm/bavoxel.hpp"
#include "s-graph/scene.h"
#include "map_optimization.h"

#include <deque>
#include <unordered_map>

class Plane
{
public:
    static constexpr int kWindowSize = 20;

    Plane() = default;
    ~Plane();

    void reset();

    void update(int key,
                const pcl::PointCloud<PointTypeIndex>::Ptr &cloud,
                const PointTypePose &pose);

    void extract(std::vector<PlaneObs> &planes) const;

    const std::deque<int> &keys() const;
    const std::deque<PointTypePose> &poses() const;

private:
    std::deque<int> keys_;
    std::deque<PointTypePose> poses_;
    std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map_;
};

#endif
