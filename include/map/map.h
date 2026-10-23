#ifndef MAP_MAP_H
#define MAP_MAP_H

#include <cstdint>

#include "map/point_obs.h"

namespace mapping
{
struct ObservationOptions
{
    bool use_online{false};
    bool use_window{false};
    bool use_prior{false};
    bool point_to_point{false};
    double prior_scale{1.0};
    int match_point_count{NUM_MATCH_POINTS};
    float max_neighbor_sq_distance{5.0f};
};

class Map
{
public:
    std::vector<PointVector> onlineNearestPoints;

    KD_TREE_PUBLIC<PointType> &onlineTree() { return onlineTree_; }
    KD_TREE_PUBLIC<PointType> &windowTree() { return windowTree_; }
    KD_TREE_PUBLIC<PointType> &priorTree() { return priorTree_; }

    void setDownsampleParam(double size);
    void prepareScan(size_t point_count);
    void initializeOnline(PointVector &points, bool use_window);
    void addOnlinePoints(PointVector &points, bool use_window,
                         bool downsample);
    void rebuildWindow(PointVector points);
    int deleteBoxes(std::vector<BoxPointType> &boxes,
                    bool include_window);
    int updateLocalMap(const Eigen::Vector3d &lidar_position,
                       double cube_length,
                       double detection_range,
                       bool include_window);
    bool hasLocalMap() const { return localMapInitialized_; }
    bool containsInLocalMap(const Eigen::Vector3d &point) const;
    void flattenTree(PointVector &points, bool use_window);
    bool hasTree(bool use_window) const;
    bool hasFixedOnlineMap() const { return onlineTree_.Root_Node != nullptr; }
    size_t onlineSize(bool include_window);
    bool hasPriorMap() const { return priorTree_.Root_Node != nullptr; }
    int priorSize() { return priorTree_.validnum(); }
    bool loadPriorTree(const std::string &path, PointCloudXYZI &cloud);

    void setPriorTransform(const Eigen::Matrix3d &rotation,
                           const Eigen::Vector3d &translation);

    bool coveredByPrior(const PointType &point_odom,
                        bool enabled,
                        bool initialized,
                        double map_leaf_size);

    std::vector<point_obs::MatchObservation> buildObservations(
        const PointCloudXYZI &scan_body,
        PointCloudXYZI &scan_world,
        const Eigen::Matrix3d &R_odom_base,
        const Eigen::Vector3d &t_odom_base,
        const Eigen::Matrix3d &R_base_lidar,
        const Eigen::Vector3d &t_base_lidar,
        bool rematch,
        const ObservationOptions &options);

    bool coarseIcpAlign(const PointCloudXYZI::Ptr &source,
                        const PointCloudXYZI::Ptr &target,
                        const Eigen::Matrix3d &R_map_lidar_guess,
                        const Eigen::Vector3d &t_map_lidar_guess,
                        Eigen::Matrix3d &R_map_lidar,
                        Eigen::Vector3d &t_map_lidar) const;

    bool refinePointToPlane(const PointCloudXYZI::Ptr &scan_body,
                            const Eigen::Matrix3d &R_odom_base,
                            const Eigen::Vector3d &t_odom_base,
                            const Eigen::Matrix3d &R_base_lidar,
                            const Eigen::Vector3d &t_base_lidar,
                            Eigen::Matrix3d &R_map_odom,
                            Eigen::Vector3d &t_map_odom);

private:
    void collectRemovedPoints(PointVector &points);

    KD_TREE_PUBLIC<PointType> onlineTree_;
    KD_TREE_PUBLIC<PointType> windowTree_;
    KD_TREE_PUBLIC<PointType> priorTree_;
    std::vector<PointVector> priorNearestPoints_;
    std::vector<uint8_t> onlineSelected_;
    std::vector<uint8_t> priorSelected_;
    Eigen::Matrix3d priorRotation_{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d priorTranslation_{Eigen::Vector3d::Zero()};
    BoxPointType localMapBounds_{};
    bool localMapInitialized_{false};
};
}

#endif
