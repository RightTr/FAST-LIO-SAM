#ifndef S_GRAPH_FLOOR_MAP_H
#define S_GRAPH_FLOOR_MAP_H

#include "s-graph/ground.h"

#include <deque>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Floor
{
    double height = 0.0;
    int up = -1;
    int down = -1;
    Ground ground;
    int ground_id = -1;
};

class FloorMap
{
public:
    FloorMap() = default;

    void setEnabled(bool enabled);

    void update(const std::deque<int> &keys,
                const std::deque<PointTypePose> &poses);

    void groundKeys(const std::deque<int> &keys,
                    std::unordered_set<int> &allowed_keys) const;

    bool updateGround(const std::vector<PlaneObs> &planes,
                      const std::deque<int> &keys,
                      const std::deque<PointTypePose> &poses,
                      PlaneBatch &batch);

    bool allowLoop(int key1, int key2) const;

private:
    double trajectorySlope(const std::deque<PointTypePose> &poses) const;

private:
    mutable std::mutex mutex_;
    bool enabled_ = false;
    std::vector<Floor> floors_;
    int current_floor_id_ = -1;
    int next_plane_id_ = 0;
    int transition_ = 0;
    int last_state_key_ = -1;
    std::unordered_map<int, int> key_floor_;
};

#endif
