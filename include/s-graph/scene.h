#ifndef SCENE_H
#define SCENE_H

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <unordered_map>
#include <tuple>
#include <utility>
#include <vector>

enum class PlaneType
{
    UNKNOWN,
    GROUND,
    WALL
};

struct PlaneObs
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector4d plane = Eigen::Vector4d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    int n = 0;
    PlaneType type = PlaneType::UNKNOWN;
    std::unordered_map<int, Eigen::Vector4d> obs;
};

struct PlaneLandmark
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int id = -1;
    Eigen::Vector4d plane = Eigen::Vector4d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();

    PlaneLandmark() = default;
    PlaneLandmark(int id_in, const Eigen::Vector4d &plane_in, const Eigen::Vector3d &center_in)
        : id(id_in), plane(plane_in), center(center_in)
    {
    }
};

struct PlaneFactor
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int key = -1;
    int id = -1;
    Eigen::Vector4d obs = Eigen::Vector4d::Zero();

    PlaneFactor() = default;
    PlaneFactor(int key_in, int id_in, const Eigen::Vector4d &obs_in)
        : key(key_in), id(id_in), obs(obs_in)
    {
    }
};

struct SceneBatch
{
    std::vector<PlaneLandmark> plane_init;
    std::vector<PlaneFactor> plane_factors;
    std::vector<int> floor_init;
    std::vector<std::pair<int, int>> floor_factors;
};

#endif
