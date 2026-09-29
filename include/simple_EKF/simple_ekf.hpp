#ifndef SIMPLE_EKF_HPP
#define SIMPLE_EKF_HPP

#include <Eigen/Core>
#include <functional>

class SimpleEKF
{
public:
    static constexpr int DIM = 15;
    using Vector3 = Eigen::Vector3d;
    using Matrix3 = Eigen::Matrix3d;
    using Error = Eigen::Matrix<double, DIM, 1>;
    using Covariance = Eigen::Matrix<double, DIM, DIM>;

    struct State
    {
        Matrix3 rot = Matrix3::Identity();
        Vector3 pos = Vector3::Zero();
        Vector3 vel = Vector3::Zero();
        Vector3 bg = Vector3::Zero();
        Vector3 ba = Vector3::Zero();
    };

    struct Noise
    {
        Vector3 gyr = Vector3::Constant(0.1);
        Vector3 acc = Vector3::Constant(0.1);
        Vector3 bg = Vector3::Constant(1e-4);
        Vector3 ba = Vector3::Constant(1e-4);
    };

    using MeasurementBuilder =
        std::function<bool(const State &, bool,
                           Eigen::MatrixXd &, Eigen::VectorXd &)>;

    SimpleEKF() = default;
    SimpleEKF(const SimpleEKF &) = delete;
    SimpleEKF &operator=(const SimpleEKF &) = delete;

    void initialize(const State &state, const Covariance &covariance,
                    const Vector3 &gravity);
    void predict(double dt, const Vector3 &acc, const Vector3 &gyro,
                 const Noise &noise);
    void updateLidar(const MeasurementBuilder &builder, double lidar_cov);
    void updateZeroVelocity(double covariance);
    void setMaxIterations(int n) { max_iterations_ = n; }

    State &state() { return state_; }
    const State &state() const { return state_; }
    const Vector3 &gravity() const { return gravity_; }
    Covariance &covariance() { return P_; }
    const Covariance &covariance() const { return P_; }

private:
    static Matrix3 Exp(const Vector3 &v);
    static Vector3 Log(const Matrix3 &R);
    static Matrix3 skew(const Vector3 &v);

    Error stateDifference(const State &propagated,
                          const State &current) const;
    void applyError(const Error &dx);

    State state_;
    Vector3 gravity_ = Vector3(0.0, 0.0, -9.81);
    Covariance P_ = Covariance::Identity();
    int max_iterations_ = 4;
};

#endif
