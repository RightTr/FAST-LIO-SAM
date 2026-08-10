#ifndef BALM_TOOLS_HPP
#define BALM_TOOLS_HPP

#include <chrono>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using PointType = pcl::PointXYZI;

#ifndef DVEL
#define DVEL 6
#endif

#define PLV(n) std::vector<Eigen::Matrix<double, n, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, n, 1>>>
#define PLM(n) std::vector<Eigen::Matrix<double, n, n>, Eigen::aligned_allocator<Eigen::Matrix<double, n, n>>>

#define SKEW_SYM_MATRX(v) 0, -(v)(2), (v)(1), (v)(2), 0, -(v)(0), -(v)(1), (v)(0), 0

inline Eigen::Matrix3d hat(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return m;
}

inline Eigen::Matrix3d Exp(const Eigen::Vector3d &w)
{
    const double theta = w.norm();
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (theta < 1e-12)
        return I + hat(w);

    const Eigen::Matrix3d W = hat(w / theta);
    return I + std::sin(theta) * W + (1.0 - std::cos(theta)) * (W * W);
}

struct IMUST
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix3d R;
    Eigen::Vector3d p;

    IMUST()
    {
        R.setIdentity();
        p.setZero();
    }
};

struct PointCluster
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix3d P;
    Eigen::Vector3d v;
    double N;

    PointCluster()
    {
        P.setZero();
        v.setZero();
        N = 0.0;
    }

    explicit PointCluster(const Eigen::Vector3d &pt)
    {
        P = pt * pt.transpose();
        v = pt;
        N = 1.0;
    }

    void clear()
    {
        P.setZero();
        v.setZero();
        N = 0.0;
    }

    void push(const Eigen::Vector3d &pt)
    {
        P += pt * pt.transpose();
        v += pt;
        N += 1.0;
    }

    PointCluster &operator+=(const PointCluster &other)
    {
        P += other.P;
        v += other.v;
        N += other.N;
        return *this;
    }

    Eigen::Matrix3d cov() const
    {
        if (N <= 0.0)
            return Eigen::Matrix3d::Zero();

        const Eigen::Vector3d mean = v / N;
        return P / N - mean * mean.transpose();
    }

    void transform(const PointCluster &src, const IMUST &x)
    {
        N = src.N;
        if (N <= 0.0)
        {
            P.setZero();
            v.setZero();
            return;
        }

        const Eigen::Matrix3d Rt = x.R.transpose();
        P = x.R * src.P * Rt + x.R * src.v * x.p.transpose() + x.p * src.v.transpose() * Rt + src.N * x.p * x.p.transpose();
        v = x.R * src.v + src.N * x.p;
    }
};

struct VOXEL_LOC
{
    int64_t x;
    int64_t y;
    int64_t z;

    VOXEL_LOC() : x(0), y(0), z(0) {}
    VOXEL_LOC(int64_t x_in, int64_t y_in, int64_t z_in) : x(x_in), y(y_in), z(z_in) {}

    bool operator==(const VOXEL_LOC &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

namespace std
{
template<>
struct hash<VOXEL_LOC>
{
    size_t operator()(const VOXEL_LOC &loc) const noexcept
    {
        const size_t hx = std::hash<int64_t>{}(loc.x);
        const size_t hy = std::hash<int64_t>{}(loc.y);
        const size_t hz = std::hash<int64_t>{}(loc.z);
        return hx ^ (hy << 1U) ^ (hz << 2U);
    }
};
} // namespace std

inline void plvec_trans(PLV(3) &pvec_orig, PLV(3) &pvec_tran, const IMUST &x)
{
    pvec_tran.clear();
    pvec_tran.reserve(pvec_orig.size());
    for (const auto &p : pvec_orig)
        pvec_tran.push_back(x.R * p + x.p);
}

inline double balm_now_sec()
{
    using clock_type = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock_type::now().time_since_epoch()).count();
}

#endif // BALM_TOOLS_HPP
