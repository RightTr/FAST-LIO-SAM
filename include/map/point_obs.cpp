#include "map/point_obs.h"

#include <algorithm>
#include <cmath>

namespace point_obs
{
void nearestSearch(const PointType &point,
                   KD_TREE_PUBLIC<PointType> &tree,
                   PointVector &near,
                   std::vector<float> &sq_distances,
                   int match_point_count)
{
    if (match_point_count < 1)
        return;

    PointVector tree_points;
    std::vector<float> tree_distances;
    if (tree.Root_Node)
        tree.Nearest_Search(point, match_point_count, tree_points,
                            tree_distances);

    std::vector<std::pair<float, PointType>> candidates;
    for (size_t i = 0; i < std::min(near.size(), sq_distances.size()); ++i)
        candidates.emplace_back(sq_distances[i], near[i]);
    for (size_t i = 0; i < std::min(tree_points.size(), tree_distances.size()); ++i)
        candidates.emplace_back(tree_distances[i], tree_points[i]);
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    near.clear();
    sq_distances.clear();
    for (const auto &candidate : candidates)
    {
        bool duplicate = false;
        for (const auto &kept : near)
        {
            if (calc_dist(candidate.second, kept) < 1e-4f)
            {
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
        {
            near.push_back(candidate.second);
            sq_distances.push_back(candidate.first);
            if (near.size() == static_cast<size_t>(match_point_count))
                break;
        }
    }
}

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
                           float max_neighbor_sq_distance)
{
    if (match_point_count < 1)
        return false;

    VF(4) plane;
    const PointVector *near = points_near;
    PointVector near_local;

    if (use_tree_search)
    {
        if (tree == nullptr || tree->Root_Node == nullptr)
            return false;

        std::vector<float> sq_distances(match_point_count);
        tree->Nearest_Search(point_world, match_point_count, near_local,
                             sq_distances);
        if (points_near_out != nullptr)
            *points_near_out = near_local;

        if (near_local.size() < static_cast<size_t>(match_point_count) ||
            sq_distances.size() < static_cast<size_t>(match_point_count) ||
            sq_distances[match_point_count - 1] > max_neighbor_sq_distance)
            return false;
        near = &near_local;
    }
    else if (near == nullptr ||
             near->size() < static_cast<size_t>(match_point_count))
    {
        return false;
    }

    obs_out.body = point_body;
    obs_out.row_scale = row_scale;

    if (use_point2point)
    {
        obs_out.target = (*near)[0];
        obs_out.residual = std::sqrt(calc_dist(point_world, obs_out.target));
        return true;
    }

    if (match_point_count != NUM_MATCH_POINTS || !esti_plane(plane, *near, 0.1f))
        return false;

    const double signed_distance = plane(0) * point_world.x +
                                  plane(1) * point_world.y +
                                  plane(2) * point_world.z + plane(3);
    const double body_range = std::sqrt(point_body.x * point_body.x +
                                        point_body.y * point_body.y +
                                        point_body.z * point_body.z);
    const double surface_quality = 1.0 - 0.9 * std::fabs(signed_distance) /
                                   std::sqrt(std::max(1e-6, body_range));
    if (surface_quality <= 0.9)
        return false;

    obs_out.normal.x = plane(0);
    obs_out.normal.y = plane(1);
    obs_out.normal.z = plane(2);
    obs_out.normal.intensity = signed_distance;
    obs_out.residual = std::fabs(signed_distance);
    return true;
}
}
