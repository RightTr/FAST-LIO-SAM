#ifndef S_GRAPH_PLANE_MAP_H
#define S_GRAPH_PLANE_MAP_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "map_optimization.h"

#include <Eigen/Core>
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

    bool extract(const std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> &clouds,
                 const std::vector<PointTypePose> &poses,
                 int start,
                 std::vector<PlaneObs> &planes,
                 pcl::PointCloud<PointTypeIndex>::Ptr cloud) const;
};

#endif
