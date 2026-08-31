#ifndef GNSS_YAW_FACTOR_H
#define GNSS_YAW_FACTOR_H

#include "common_utils.h"

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

class GnssYawFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
    GnssYawFactor(gtsam::Key key,
                  double measured_yaw,
                  const gtsam::SharedNoiseModel &model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key),
          measured_yaw_(measured_yaw)
    {
    }

    gtsam::Vector evaluateError(const gtsam::Pose3 &pose,
                                boost::optional<gtsam::Matrix &> H = boost::none) const override
    {
        const auto error_func = [this](const gtsam::Pose3 &p) -> gtsam::Vector1
        {
            const double yaw = p.rotation().ypr()(0);
            return (gtsam::Vector1() << normalizeYaw(yaw - measured_yaw_)).finished();
        };

        const gtsam::Vector1 error = error_func(pose);

        if (H)
        {
            *H = gtsam::numericalDerivative11<gtsam::Vector1, gtsam::Pose3>(error_func, pose);
        }

        return error;
    }

private:
    double measured_yaw_;
};

#endif
