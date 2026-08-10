#ifndef GROUND_MANAGER_H
#define GROUND_MANAGER_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "map_optimization.h"

#include <Eigen/Core>
#include <unordered_map>
#include <vector>

struct GroundObs
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool valid = false;
    Eigen::Vector4d map = Eigen::Vector4d::Zero();
    std::unordered_map<int, Eigen::Vector4d> body;
};

class GroundManager
{
public:
    GroundManager() = default;

    bool extract(const std::vector<pcl::PointCloud<PointTypeIndex>::Ptr> &keyframes,
                 const std::vector<PointTypePose> &poses,
                 int window_start,
                 GroundObs &gnd,
                 pcl::PointCloud<PointTypeIndex>::Ptr planes) const;
};

#endif
