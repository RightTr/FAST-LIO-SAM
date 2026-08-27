#ifndef SCENE_H
#define SCENE_H

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <utility>
#include <vector>

struct PlaneObs
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector4d plane = Eigen::Vector4d::Zero();
    Eigen::Vector4d body_plane = Eigen::Vector4d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    int n = 0;
};

struct PlaneInit
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int floor = -1;
    int id = -1;
    Eigen::Vector4d plane = Eigen::Vector4d::Zero();

    PlaneInit() = default;
    PlaneInit(int floor_in, int id_in, const Eigen::Vector4d &plane_in)
        : floor(floor_in), id(id_in), plane(plane_in)
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
    std::vector<PlaneInit> planes;
    std::vector<PlaneFactor> factors;
};

#endif
