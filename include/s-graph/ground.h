#ifndef GROUND_H
#define GROUND_H

#include "s-graph/scene.h"
#include "map_optimization.h"
#include "utility.h"

#include <Eigen/Core>
#include <deque>
#include <unordered_set>
#include <vector>

struct Ground
{
    int id = -1;
    Eigen::Vector4d plane = Eigen::Vector4d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
};

bool buildGroundCandidate(const std::vector<PlaneObs> &planes,
                          const std::deque<int> &keys,
                          const std::deque<PointTypePose> &poses,
                          PlaneObs &candidate);

#endif
