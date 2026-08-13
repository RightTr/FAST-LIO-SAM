#ifndef S_GRAPH_GROUND_H
#define S_GRAPH_GROUND_H

#include "s-graph/plane.h"
#include "utility.h"

#include <Eigen/Core>
#include <deque>
#include <unordered_set>
#include <vector>

class Ground
{
public:
    Ground() = default;

    bool update(const std::vector<PlaneObs> &planes,
                const std::deque<int> &keys,
                const std::deque<PointTypePose> &poses,
                PlaneBatch &batch);

private:
    PlaneLandmark ground_;
    std::unordered_set<int> keys_;
};

#endif
