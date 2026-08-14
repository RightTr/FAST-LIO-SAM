#ifndef FLOOR_FACTOR_H
#define FLOOR_FACTOR_H

#include <Eigen/Core>

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

class FloorFactor : public gtsam::NoiseModelFactor2<gtsam::Vector1, gtsam::Pose3>
{
public:
    FloorFactor(gtsam::Key floor_key,
                gtsam::Key pose_key,
                const Eigen::Vector4d &body_plane,
                const gtsam::SharedNoiseModel &model)
        : gtsam::NoiseModelFactor2<gtsam::Vector1, gtsam::Pose3>(model, floor_key, pose_key),
          body_plane_(body_plane)
    {
    }

    gtsam::Vector evaluateError(const gtsam::Vector1 &floor_height,
                                const gtsam::Pose3 &pose,
                                boost::optional<gtsam::Matrix &> H1 = boost::none,
                                boost::optional<gtsam::Matrix &> H2 = boost::none) const override
    {
        const gtsam::Vector1 error = computeError(floor_height, pose);

        if (H1)
        {
            H1->resize(1, 1);
            (*H1)(0, 0) = -1.0;
        }

        if (H2)
        {
            *H2 = gtsam::numericalDerivative11<gtsam::Vector1, gtsam::Pose3>(
                [this, &floor_height](const gtsam::Pose3 &pose_value) -> gtsam::Vector1
                {
                    return computeError(floor_height, pose_value);
                },
                pose);
        }

        return error;
    }

private:
    gtsam::Vector1 computeError(const gtsam::Vector1 &floor_height,
                                const gtsam::Pose3 &pose) const
    {
        const Eigen::Vector3d body_z =
            pose.rotation().matrix().transpose() * Eigen::Vector3d::UnitZ();

        const double body_ground_height =
            body_plane_(3) / body_plane_.head<3>().dot(body_z);

        return (gtsam::Vector1()
                << pose.translation().z() - body_ground_height - floor_height(0))
            .finished();
    }

private:
    Eigen::Vector4d body_plane_ = Eigen::Vector4d::Zero();
};

#endif
