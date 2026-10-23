#ifndef POINT_OBS_H
#define POINT_OBS_H

#include "common_lib.h"
#include "map/ikdtree/ikdtree_public.h"

namespace point_obs
{
struct MatchObservation
{
    PointType body;
    PointType normal;
    PointType target;
    double residual = 0.0;
    double row_scale = 1.0;
};

// Merges one tree's nearest points into the existing candidate set.
void nearestSearch(const PointType &point,
                   KD_TREE_PUBLIC<PointType> &tree,
                   PointVector &near,
                   std::vector<float> &sq_distances,
                   int match_point_count);

bool buildPointObservation(KD_TREE_PUBLIC<PointType> *tree,
                           const PointVector *points_near,
                           const PointType &point_body,
                           const PointType &point_world,
                           double row_scale,
                           bool use_tree_search,
                           PointVector *points_near_out,
                           MatchObservation &obs_out,
                           bool use_point2point,
                           int match_point_count,
                           float max_neighbor_sq_distance);
}

#endif
