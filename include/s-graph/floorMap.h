#ifndef FLOOR_MAP_H
#define FLOOR_MAP_H

#include "s-graph/ground.h"
#include "map_optimization.h"

#include <deque>
#include <mutex>
#include <vector>

struct FloorRange
{
    int begin = -1;
    int end = -1;
};

struct Floor
{
    int up = -1;
    int down = -1;
    std::vector<Ground> grounds;
    std::vector<FloorRange> ranges;
};

class FloorMap
{
public:
    FloorMap() = default;

    void update(int key,
                const PointTypePose &pose);

    bool updateGround(const std::vector<PlaneObs> &planes,
                      const std::deque<int> &keys,
                      const std::deque<PointTypePose> &poses,
                      SceneBatch &batch);

    bool getRanges(int key,
                    std::vector<FloorRange> &ranges) const;

private:
    double trajectoryAngle() const;

private:
    mutable std::mutex mutex_;
    std::vector<Floor> floors_;
    int current_floor_id_ = 0;
    int next_plane_id_ = 0;
    int transition_dir_ = 0;
    double transition_start_z_ = 0.0;
    double flat_distance_ = 0.0;
    std::deque<PointTypePose> recent_poses_;
};

#endif
