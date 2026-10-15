#include "simple_EKF/simple_ekf.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>

namespace
{
constexpr int ROT = 0;
constexpr int POS = 3;
constexpr int VEL = 6;
constexpr int BG = 9;
constexpr int BA = 12;
constexpr double SO3_TOLERANCE = 1e-11;
}

SimpleEKF::Matrix3 SimpleEKF::skew(const Vector3 &v)
{
    Matrix3 result;
    result << 0.0, -v.z(), v.y(),
              v.z(), 0.0, -v.x(),
              -v.y(), v.x(), 0.0;
    return result;
}

SimpleEKF::Matrix3 SimpleEKF::Exp(const Vector3 &v)
{
    const double norm = v.norm();
    Eigen::Quaterniond q;
    if (norm < SO3_TOLERANCE)
    {
        q.w() = 1.0;
        q.vec() = 0.5 * v;
    }
    else
    {
        q.w() = std::cos(0.5 * norm);
        q.vec() = std::sin(0.5 * norm) / norm * v;
    }
    return q.toRotationMatrix();
}

SimpleEKF::Vector3 SimpleEKF::Log(const Matrix3 &R)
{
    Eigen::Quaterniond q(R);
    q.normalize();
    double norm = q.vec().norm();
    if (norm < SO3_TOLERANCE)
        norm = SO3_TOLERANCE;
    return 2.0 * std::atan(norm / q.w()) / norm * q.vec();
}

void SimpleEKF::initialize(const State &state, const Covariance &covariance,
                           const Vector3 &gravity)
{
    state_ = state;
    P_ = covariance;
    gravity_ = gravity;
}

void SimpleEKF::predict(double dt, const Vector3 &acc_measurement,
                        const Vector3 &gyro_measurement,
                        const Noise &noise)
{
    const Vector3 omega = gyro_measurement - state_.bg;
    const Vector3 acc_body = acc_measurement - state_.ba;
    const Matrix3 R = state_.rot;
    const Matrix3 R_new = R * Exp(omega * dt);
    const Vector3 acc_world = R_new * acc_body + gravity_;

    state_.pos += state_.vel * dt + 0.5 * acc_world * dt * dt;
    state_.vel += acc_world * dt;
    state_.rot = R_new;

    Covariance F = Covariance::Identity();
    F.block<3, 3>(ROT, ROT) = Exp(-omega * dt);
    F.block<3, 3>(ROT, BG) = -Matrix3::Identity() * dt;
    F.block<3, 3>(POS, VEL) = Matrix3::Identity() * dt;
    F.block<3, 3>(VEL, ROT) = -R * skew(acc_body) * dt;
    F.block<3, 3>(VEL, BA) = -R * dt;

    Covariance cov_w = Covariance::Zero();
    cov_w.block<3, 3>(ROT, ROT).diagonal() = noise.gyr * dt * dt;
    cov_w.block<3, 3>(VEL, VEL) =
        R * noise.acc.asDiagonal() * R.transpose() * dt * dt;
    cov_w.block<3, 3>(BG, BG).diagonal() = noise.bg * dt * dt;
    cov_w.block<3, 3>(BA, BA).diagonal() = noise.ba * dt * dt;

    P_ = F * P_ * F.transpose() + cov_w;
}

void SimpleEKF::updateLidar(const MeasurementBuilder &builder,
                            double lidar_cov)
{
    const State propagated = state_;
    const Covariance P_propagated = P_;
    int rematch_num = 0;

    for (int iter = 0; iter < max_iterations_; ++iter)
    {
        Eigen::MatrixXd H;
        Eigen::VectorXd residual;
        const bool rematch = rematch_num == 0;
        if (!builder(state_, rematch, H, residual))
            continue;

        const Eigen::Matrix<double, 6, 6> HtH =
            H.transpose() * H / lidar_cov;
        const Eigen::Matrix<double, 6, 1> Htr =
            H.transpose() * residual / lidar_cov;

        Covariance information = P_propagated.inverse();
        information.block<6, 6>(0, 0) += HtH;
        const Covariance K1 = information.inverse();
        const Eigen::Matrix<double, DIM, 6> G =
            K1.block<DIM, 6>(0, 0) * HtH;

        const Error vec = stateDifference(propagated, state_);
        const Error dx = K1.block<DIM, 6>(0, 0) * Htr +
                         vec - G * vec.head<6>();
        applyError(dx);

        const bool converged =
            dx.segment<3>(ROT).norm() * 57.3 < 0.01 &&
            dx.segment<3>(POS).norm() * 100.0 < 0.015;
        if (converged ||
            (rematch_num == 0 && iter == max_iterations_ - 2))
            ++rematch_num;

        if (rematch_num >= 2 || iter == max_iterations_ - 1)
        {
            Covariance G_full = Covariance::Zero();
            G_full.block<DIM, 6>(0, 0) = G;
            P_ = (Covariance::Identity() - G_full) * P_propagated;
            return;
        }
    }
}

SimpleEKF::Error SimpleEKF::stateDifference(
    const State &propagated, const State &current) const
{
    Error difference = Error::Zero();
    difference.segment<3>(ROT) = Log(current.rot.transpose() * propagated.rot);
    difference.segment<3>(POS) = propagated.pos - current.pos;
    difference.segment<3>(VEL) = propagated.vel - current.vel;
    difference.segment<3>(BG) = propagated.bg - current.bg;
    difference.segment<3>(BA) = propagated.ba - current.ba;
    return difference;
}

void SimpleEKF::applyError(const Error &dx)
{
    state_.rot = state_.rot * Exp(dx.segment<3>(ROT));
    state_.pos += dx.segment<3>(POS);
    state_.vel += dx.segment<3>(VEL);
    state_.bg += dx.segment<3>(BG);
    state_.ba += dx.segment<3>(BA);
}

void SimpleEKF::setPose(const Matrix3 &rotation, const Vector3 &position)
{
    state_.rot = rotation;
    state_.pos = position;
}
