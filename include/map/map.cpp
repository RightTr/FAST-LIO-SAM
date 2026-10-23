#include "map/map.h"

#include <cmath>
#include <omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

namespace mapping
{
void Map::setDownsampleParam(double size)
{
    onlineTree_.set_downsample_param(size);
    windowTree_.set_downsample_param(size);
}

void Map::prepareScan(size_t point_count)
{
    onlineNearestPoints.resize(point_count);
    priorNearestPoints_.resize(point_count);
    onlineSelected_.resize(point_count, 0);
    priorSelected_.resize(point_count, 0);
}

void Map::initializeOnline(PointVector &points, bool use_window)
{
    auto &tree = use_window ? windowTree_ : onlineTree_;
    if (!tree.Root_Node) tree.Build(points);
    else tree.Add_Points(points, true);
}

void Map::addOnlinePoints(PointVector &points, bool use_window,
                          bool downsample)
{
    if (points.empty()) return;
    auto &tree = use_window ? windowTree_ : onlineTree_;
    if (!tree.Root_Node) tree.Build(points);
    else tree.Add_Points(points, downsample);
}

void Map::rebuildWindow(PointVector points)
{
    windowTree_.Rebuild(std::move(points));
}

int Map::deleteBoxes(std::vector<BoxPointType> &boxes,
                     bool include_window)
{
    if (boxes.empty()) return 0;
    int deleted = onlineTree_.Delete_Point_Boxes(boxes);
    if (include_window)
        deleted += windowTree_.Delete_Point_Boxes(boxes);
    return deleted;
}

int Map::updateLocalMap(const Eigen::Vector3d &lidar_position,
                        double cube_length,
                        double detection_range,
                        bool include_window)
{
    constexpr double move_threshold = 1.5;
    if (!localMapInitialized_)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            localMapBounds_.vertex_min[axis] = lidar_position(axis) - cube_length / 2.0;
            localMapBounds_.vertex_max[axis] = lidar_position(axis) + cube_length / 2.0;
        }
        localMapInitialized_ = true;
        return 0;
    }

    double distance_to_edge[3][2];
    bool need_move = false;
    for (int axis = 0; axis < 3; ++axis)
    {
        distance_to_edge[axis][0] =
            std::fabs(lidar_position(axis) - localMapBounds_.vertex_min[axis]);
        distance_to_edge[axis][1] =
            std::fabs(lidar_position(axis) - localMapBounds_.vertex_max[axis]);
        if (distance_to_edge[axis][0] <= move_threshold * detection_range ||
            distance_to_edge[axis][1] <= move_threshold * detection_range)
            need_move = true;
    }
    if (!need_move)
        return 0;

    BoxPointType new_bounds = localMapBounds_;
    std::vector<BoxPointType> boxes_to_delete;
    const double move_distance = std::max(
        (cube_length - 2.0 * move_threshold * detection_range) * 0.5 * 0.9,
        detection_range * (move_threshold - 1.0));

    for (int axis = 0; axis < 3; ++axis)
    {
        BoxPointType box = localMapBounds_;
        if (distance_to_edge[axis][0] <= move_threshold * detection_range)
        {
            new_bounds.vertex_max[axis] -= move_distance;
            new_bounds.vertex_min[axis] -= move_distance;
            box.vertex_min[axis] = localMapBounds_.vertex_max[axis] - move_distance;
            boxes_to_delete.push_back(box);
        }
        else if (distance_to_edge[axis][1] <= move_threshold * detection_range)
        {
            new_bounds.vertex_max[axis] += move_distance;
            new_bounds.vertex_min[axis] += move_distance;
            box.vertex_max[axis] = localMapBounds_.vertex_min[axis] + move_distance;
            boxes_to_delete.push_back(box);
        }
    }

    localMapBounds_ = new_bounds;
    PointVector removed_points;
    collectRemovedPoints(removed_points);
    return deleteBoxes(boxes_to_delete, include_window);
}

bool Map::containsInLocalMap(const Eigen::Vector3d &point) const
{
    if (!localMapInitialized_)
        return true;
    for (int axis = 0; axis < 3; ++axis)
        if (point(axis) < localMapBounds_.vertex_min[axis] ||
            point(axis) > localMapBounds_.vertex_max[axis])
            return false;
    return true;
}

void Map::collectRemovedPoints(PointVector &points)
{
    onlineTree_.acquire_removed_points(points);
}

void Map::flattenTree(PointVector &points, bool use_window)
{
    points.clear();
    auto &tree = use_window ? windowTree_ : onlineTree_;
    if (tree.Root_Node)
        tree.flatten(tree.Root_Node, points, NOT_RECORD);
}

bool Map::hasTree(bool use_window) const
{
    return (use_window ? windowTree_ : onlineTree_).Root_Node != nullptr;
}

size_t Map::onlineSize(bool include_window)
{
    return onlineTree_.size() + (include_window ? windowTree_.size() : 0);
}

bool Map::loadPriorTree(const std::string &path, PointCloudXYZI &cloud)
{
    if (path.empty())
    {
        ROS_PRINT_WARN("prior mode enabled, but no prior_tree_path provided");
        return false;
    }

    if (!priorTree_.LoadIkdtree(path))
    {
        ROS_PRINT_ERROR("failed to load prior ikdtree snapshot: %s", path.c_str());
        return false;
    }

    const bool ready = priorTree_.Root_Node != nullptr;
    PointVector().swap(priorTree_.PCL_Storage);
    priorTree_.flatten(priorTree_.Root_Node, priorTree_.PCL_Storage, NOT_RECORD);
    cloud.clear();
    cloud.points = priorTree_.PCL_Storage;
    ROS_PRINT_INFO("loaded prior ikdtree snapshot: %s, nodes=%d",
                   path.c_str(), priorTree_.validnum());
    return ready;
}

void Map::setPriorTransform(const Eigen::Matrix3d &rotation,
                            const Eigen::Vector3d &translation)
{
    priorRotation_ = rotation;
    priorTranslation_ = translation;
}

bool Map::coarseIcpAlign(const PointCloudXYZI::Ptr &source,
                         const PointCloudXYZI::Ptr &target,
                         const Eigen::Matrix3d &R_map_lidar_guess,
                         const Eigen::Vector3d &t_map_lidar_guess,
                         Eigen::Matrix3d &R_map_lidar,
                         Eigen::Vector3d &t_map_lidar) const
{
    if (!source || source->size() < 20 || !target || target->empty())
        return false;

    PointCloudXYZI::Ptr target_ds(new PointCloudXYZI());
    if (target->size() > 2000)
    {
        pcl::VoxelGrid<PointType> downsample;
        downsample.setLeafSize(0.8f, 0.8f, 0.8f);
        downsample.setInputCloud(target);
        downsample.filter(*target_ds);
    }
    else
    {
        *target_ds = *target;
    }
    if (target_ds->size() < 50)
        return false;

    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(8.0);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);
    icp.setInputSource(source);
    icp.setInputTarget(target_ds);

    Eigen::Affine3f guess = Eigen::Affine3f::Identity();
    guess.linear() = R_map_lidar_guess.cast<float>();
    guess.translation() = t_map_lidar_guess.cast<float>();
    PointCloudXYZI aligned;
    icp.align(aligned, guess.matrix());
    if (!icp.hasConverged())
        return false;

    const double fitness_score = icp.getFitnessScore();
    if (!std::isfinite(fitness_score) || fitness_score > 5.0)
        return false;

    const Eigen::Matrix4f final_tf = icp.getFinalTransformation();
    R_map_lidar = final_tf.block<3, 3>(0, 0).cast<double>();
    t_map_lidar = final_tf.block<3, 1>(0, 3).cast<double>();
    return true;
}

bool Map::refinePointToPlane(const PointCloudXYZI::Ptr &scan_body,
                             const Eigen::Matrix3d &R_odom_base,
                             const Eigen::Vector3d &t_odom_base,
                             const Eigen::Matrix3d &R_base_lidar,
                             const Eigen::Vector3d &t_base_lidar,
                             Eigen::Matrix3d &R_map_odom,
                             Eigen::Vector3d &t_map_odom)
{
    if (!scan_body || scan_body->size() < 20 || priorTree_.Root_Node == nullptr)
        return false;

    for (int iter = 0; iter < 5; ++iter)
    {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        int valid_num = 0;

        for (const PointType &body : scan_body->points)
        {
            const Eigen::Vector3d p_lidar(body.x, body.y, body.z);
            const Eigen::Vector3d p_base = R_base_lidar * p_lidar + t_base_lidar;
            const Eigen::Vector3d p_odom = R_odom_base * p_base + t_odom_base;
            const Eigen::Vector3d Rp = R_map_odom * p_odom;
            const Eigen::Vector3d p_map = Rp + t_map_odom;

            PointType point_map;
            point_map.x = p_map.x();
            point_map.y = p_map.y();
            point_map.z = p_map.z();
            point_map.intensity = body.intensity;

            point_obs::MatchObservation observation;
            if (!point_obs::buildPointObservation(
                    &priorTree_, nullptr, body, point_map, 1.0, true,
                    nullptr, observation, false, NUM_MATCH_POINTS, 5.0f))
                continue;

            const Eigen::Vector3d normal(observation.normal.x,
                                         observation.normal.y,
                                         observation.normal.z);
            const double residual = observation.normal.intensity;
            Eigen::Matrix<double, 1, 6> J;
            J.block<1, 3>(0, 0) = normal.transpose();
            J.block<1, 3>(0, 3) = Rp.cross(normal).transpose();
            H += J.transpose() * J;
            b += J.transpose() * residual;
            ++valid_num;
        }

        if (valid_num < 20)
            return false;

        const Eigen::Matrix<double, 6, 1> dx = -H.ldlt().solve(b);
        if (!dx.allFinite())
            return false;

        t_map_odom += dx.head<3>();
        R_map_odom = Exp(dx(3), dx(4), dx(5)) * R_map_odom;
        if (dx.head<3>().norm() < 1e-3 && dx.tail<3>().norm() < 1e-4)
            break;
    }
    return true;
}

bool Map::coveredByPrior(const PointType &point_odom,
                         bool enabled,
                         bool initialized,
                         double map_leaf_size)
{
    if (!enabled || !initialized || priorTree_.Root_Node == nullptr)
        return false;

    const Eigen::Vector3d point_prior =
        priorRotation_ * Eigen::Vector3d(point_odom.x, point_odom.y, point_odom.z) +
        priorTranslation_;
    PointType query;
    query.x = point_prior.x();
    query.y = point_prior.y();
    query.z = point_prior.z();

    PointVector near;
    std::vector<float> sq_distances;
    priorTree_.Nearest_Search(query, 1, near, sq_distances);
    return !sq_distances.empty() &&
           sq_distances[0] < map_leaf_size * map_leaf_size;
}

std::vector<point_obs::MatchObservation> Map::buildObservations(
    const PointCloudXYZI &scan_body,
    PointCloudXYZI &scan_world,
    const Eigen::Matrix3d &R_odom_base,
    const Eigen::Vector3d &t_odom_base,
    const Eigen::Matrix3d &R_base_lidar,
    const Eigen::Vector3d &t_base_lidar,
    bool rematch,
    const ObservationOptions &options)
{
    const size_t count = scan_body.size();
    prepareScan(count);
    scan_world.resize(count);

    std::vector<point_obs::MatchObservation> online_observations(count);
    std::vector<point_obs::MatchObservation> prior_observations(count);
    std::vector<uint8_t> online_valid(count, 0);
    std::vector<uint8_t> prior_valid(count, 0);

#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
#endif
    for (int index = 0; index < static_cast<int>(count); ++index)
    {
        const size_t i = static_cast<size_t>(index);
        const PointType &body = scan_body.points[i];
        PointType &world = scan_world.points[i];
        const Eigen::Vector3d p_lidar(body.x, body.y, body.z);
        const Eigen::Vector3d p_base = R_base_lidar * p_lidar + t_base_lidar;
        const Eigen::Vector3d p_world = R_odom_base * p_base + t_odom_base;
        world.x = p_world.x();
        world.y = p_world.y();
        world.z = p_world.z();
        world.intensity = body.intensity;

        if (options.use_online)
        {
            if (rematch)
            {
                if (options.use_window)
                {
                    std::vector<float> distances;
                    onlineNearestPoints[i].clear();
                    point_obs::nearestSearch(world, onlineTree_,
                                             onlineNearestPoints[i], distances,
                                             options.match_point_count);
                    point_obs::nearestSearch(world, windowTree_,
                                             onlineNearestPoints[i], distances,
                                             options.match_point_count);
                    online_valid[i] = point_obs::buildPointObservation(
                        nullptr, &onlineNearestPoints[i], body, world, 1.0,
                        false, nullptr, online_observations[i],
                        options.point_to_point, options.match_point_count,
                        options.max_neighbor_sq_distance) &&
                        !distances.empty() &&
                        distances.back() <= options.max_neighbor_sq_distance;
                }
                else
                {
                    online_valid[i] = point_obs::buildPointObservation(
                        &onlineTree_, nullptr, body, world, 1.0, true,
                        &onlineNearestPoints[i], online_observations[i],
                        options.point_to_point, options.match_point_count,
                        options.max_neighbor_sq_distance);
                }
                onlineSelected_[i] = online_valid[i];
            }
            else if (onlineSelected_[i])
            {
                online_valid[i] = point_obs::buildPointObservation(
                    nullptr, &onlineNearestPoints[i], body, world, 1.0,
                    false, nullptr, online_observations[i],
                    options.point_to_point, options.match_point_count,
                    options.max_neighbor_sq_distance);
                if (!online_valid[i])
                    onlineSelected_[i] = 0;
            }
        }

        if (!options.use_prior)
            continue;

        const Eigen::Vector3d p_prior = priorRotation_ * p_world + priorTranslation_;
        PointType point_prior;
        point_prior.x = p_prior.x();
        point_prior.y = p_prior.y();
        point_prior.z = p_prior.z();
        point_prior.intensity = world.intensity;

        if (rematch)
        {
            prior_valid[i] = point_obs::buildPointObservation(
                &priorTree_, nullptr, body, point_prior, options.prior_scale,
                true, &priorNearestPoints_[i], prior_observations[i],
                options.point_to_point, options.match_point_count,
                options.max_neighbor_sq_distance);
            priorSelected_[i] = prior_valid[i];
        }
        else if (priorSelected_[i])
        {
            prior_valid[i] = point_obs::buildPointObservation(
                nullptr, &priorNearestPoints_[i], body, point_prior,
                options.prior_scale, false, nullptr, prior_observations[i],
                options.point_to_point, options.match_point_count,
                options.max_neighbor_sq_distance);
            if (!prior_valid[i])
                priorSelected_[i] = 0;
        }

        if (!prior_valid[i])
            continue;

        if (options.point_to_point)
        {
            const Eigen::Vector3d target_prior(
                prior_observations[i].target.x,
                prior_observations[i].target.y,
                prior_observations[i].target.z);
            const Eigen::Vector3d target_odom =
                priorRotation_.transpose() * (target_prior - priorTranslation_);
            prior_observations[i].target.x = target_odom.x();
            prior_observations[i].target.y = target_odom.y();
            prior_observations[i].target.z = target_odom.z();
        }
        else
        {
            const Eigen::Vector3d normal_prior(
                prior_observations[i].normal.x,
                prior_observations[i].normal.y,
                prior_observations[i].normal.z);
            const Eigen::Vector3d normal_odom =
                priorRotation_.transpose() * normal_prior;
            prior_observations[i].normal.x = normal_odom.x();
            prior_observations[i].normal.y = normal_odom.y();
            prior_observations[i].normal.z = normal_odom.z();
        }
    }

    std::vector<point_obs::MatchObservation> observations;
    observations.reserve(count * 2);
    for (size_t i = 0; i < count; ++i)
    {
        if (online_valid[i]) observations.push_back(online_observations[i]);
        if (prior_valid[i]) observations.push_back(prior_observations[i]);
    }

    return observations;
}
}
