#ifndef FLOOR_FACTOR_H
#define FLOOR_FACTOR_H

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/OrientedPlane3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

class FloorFactor : public gtsam::NoiseModelFactor2<gtsam::Unit3, gtsam::OrientedPlane3>
{
public:
    FloorFactor(gtsam::Key floor_key,
                gtsam::Key plane_key,
                const gtsam::SharedNoiseModel &model)
        : gtsam::NoiseModelFactor2<gtsam::Unit3, gtsam::OrientedPlane3>(
              model, floor_key, plane_key)
    {
    }

    gtsam::Vector evaluateError(
        const gtsam::Unit3 &floor_normal,
        const gtsam::OrientedPlane3 &plane,
        boost::optional<gtsam::Matrix &> H1 = boost::none,
        boost::optional<gtsam::Matrix &> H2 = boost::none) const override
    {
        const gtsam::Vector2 error = computeError(floor_normal, plane);

        if (H1)
        {
            constexpr double kDelta = 1e-5;
            H1->resize(2, 2);
            for (int i = 0; i < 2; ++i)
            {
                gtsam::Vector2 delta = gtsam::Vector2::Zero();
                delta(i) = kDelta;
                const gtsam::Vector2 plus =
                    computeError(floor_normal.retract(delta), plane);
                delta(i) = -kDelta;
                const gtsam::Vector2 minus =
                    computeError(floor_normal.retract(delta), plane);
                H1->col(i) = (plus - minus) / (2.0 * kDelta);
            }
        }

        if (H2)
        {
            constexpr double kDelta = 1e-5;
            H2->resize(2, 3);
            for (int i = 0; i < 3; ++i)
            {
                gtsam::Vector3 delta = gtsam::Vector3::Zero();
                delta(i) = kDelta;
                const gtsam::Vector2 plus =
                    computeError(floor_normal, plane.retract(delta));
                delta(i) = -kDelta;
                const gtsam::Vector2 minus =
                    computeError(floor_normal, plane.retract(delta));
                H2->col(i) = (plus - minus) / (2.0 * kDelta);
            }
        }

        return error;
    }

private:
    static gtsam::Vector2 computeError(const gtsam::Unit3 &floor_normal,
                                       const gtsam::OrientedPlane3 &plane)
    {
        return floor_normal.errorVector(plane.normal());
    }
};

#endif
