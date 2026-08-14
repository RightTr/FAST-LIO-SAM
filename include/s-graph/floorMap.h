#ifndef FLOOR_MAP_H
#define FLOOR_MAP_H

#include "s-graph/ground.h"
#include "map_optimization.h"

#include <deque>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Floor
{
    int up = -1;
    int down = -1;
    std::vector<Ground> grounds;
};

class FloorMap
{
public:
    FloorMap() = default;

    void update(const std::deque<int> &keys,
                const std::deque<PointTypePose> &poses);

    void groundKeys(const std::deque<int> &keys,
                    std::unordered_set<int> &allowed_keys) const;

    int floor(int key) const;

    bool updateGround(const std::vector<PlaneObs> &planes,
                      const std::deque<int> &keys,
                      const std::deque<PointTypePose> &poses,
                      bool use_floor,
                      SceneBatch &batch);

    bool allowLoop(int key1, int key2) const;

private:
    double trajectorySlope(const std::deque<PointTypePose> &poses) const;

private:
    mutable std::mutex mutex_;
    std::vector<Floor> floors_;
    int current_floor_id_ = -1;
    int next_plane_id_ = 0;
    int transition_ = 0;
    int last_ground_key_ = -1;
    std::vector<int> key_floor_;
};

#endif
