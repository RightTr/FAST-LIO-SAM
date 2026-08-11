#ifndef S_GRAPH_GROUND_MAP_H
#define S_GRAPH_GROUND_MAP_H

#include "s-graph/planeMap.h"
#include "utility.h"

#include <Eigen/Core>
#include <deque>
#include <unordered_map>
#include <vector>

struct Ground
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int id = -1;
    Eigen::Vector4d plane = Eigen::Vector4d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    std::vector<int> keys;
};

struct PlaneFactor
{
    int key = -1;
    int id = -1;
    Eigen::Vector4d obs = Eigen::Vector4d::Zero();
};

struct PlaneBatch
{
    bool valid = false;
    std::vector<Ground> init;
    std::vector<PlaneFactor> factors;
};

class GroundMap
{
public:
    GroundMap() = default;

    bool update(const std::vector<PlaneObs> &planes,
                const std::deque<int> &keys,
                const std::deque<PointTypePose> &poses,
                PlaneBatch &batch);

    const std::vector<Ground> &grounds() const { return grounds_; }

private:
    std::vector<Ground> grounds_;
    std::unordered_map<int, int> key_plane_;
    int current_id_ = -1;
    int next_id_ = 0;
};

#endif
