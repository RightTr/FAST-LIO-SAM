#ifndef S_GRAPH_PLANE_H
#define S_GRAPH_PLANE_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "balm/bavoxel.hpp"
#include "map_optimization.h"

#include <Eigen/Core>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum class PlaneType
{
    UNKNOWN,
    GROUND,
    WALL
};

struct PlaneObs
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector4d plane = Eigen::Vector4d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    int n = 0;
    PlaneType type = PlaneType::UNKNOWN;
    std::unordered_map<int, Eigen::Vector4d> obs;
};

struct PlaneLandmark
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int id = -1;
    Eigen::Vector4d plane = Eigen::Vector4d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
};

struct PlaneFactor
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int key = -1;
    int id = -1;
    Eigen::Vector4d obs = Eigen::Vector4d::Zero();
};

struct PlaneBatch
{
    bool valid = false;
    std::vector<PlaneLandmark> init;
    std::vector<PlaneFactor> factors;
};

class Plane
{
public:
    Plane() = default;
    ~Plane();

    void reset();

    void update(int key,
                const pcl::PointCloud<PointTypeIndex>::Ptr &cloud,
                const PointTypePose &pose,
                pcl::PointCloud<PointTypeIndex>::Ptr display_cloud = nullptr);

    void extract(const std::unordered_set<int> &allowed_keys,
                 std::vector<PlaneObs> &planes) const;

    const std::deque<int> &keys() const;
    const std::deque<PointTypePose> &poses() const;

private:
    static constexpr int kWindowSize = 20;

    std::deque<int> keys_;
    std::deque<PointTypePose> poses_;
    std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map_;
};

#endif
