#ifndef GNSS_XY_FACTOR_H
#define GNSS_XY_FACTOR_H

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

class GnssXYFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
    GnssXYFactor(gtsam::Key key, const gtsam::Vector2 &position,
                 const gtsam::SharedNoiseModel &model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key), position_(position)
    {
    }

    gtsam::Vector evaluateError(const gtsam::Pose3 &pose,
                               boost::optional<gtsam::Matrix &> H = boost::none) const override
    {
        if (H)
        {
            gtsam::Matrix36 translationJacobian;
            pose.translation(translationJacobian);
            *H = translationJacobian.topRows<2>();
        }
        return pose.translation().head<2>() - position_;
    }

private:
    gtsam::Vector2 position_;
};

#endif
