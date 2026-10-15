#ifndef BA_TOOLS_HPP
#define BA_TOOLS_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cstdint>
#include <cmath>
#include <memory>
#include <functional>
#include <unordered_map>
#include <vector>

namespace lidar_ba {

#define LBA_HASH_P 116101ULL
#define LBA_MAX_N 10000000000ULL
#define LBA_SKEW(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0
#define LBA_PLM(a) std::vector<Eigen::Matrix<double, a, a>, Eigen::aligned_allocator<Eigen::Matrix<double, a, a>>>
#define LBA_PLV(a) std::vector<Eigen::Matrix<double, a, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, a, 1>>>

using PointType = pcl::PointXYZINormal;
inline const Eigen::Matrix3d I33 = Eigen::Matrix3d::Identity();

inline Eigen::Matrix3d hat(const Eigen::Vector3d &v) {
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return m;
}

class VOXEL_LOC {
public:
  int64_t x, y, z;
  VOXEL_LOC(int64_t vx=0, int64_t vy=0, int64_t vz=0): x(vx), y(vy), z(vz) {}
  bool operator==(const VOXEL_LOC &o) const { return x==o.x && y==o.y && z==o.z; }
};

inline Eigen::Matrix3d Exp(const Eigen::Vector3d &a) {
  const double n=a.norm();
  if(n < 1e-11) return I33;
  const Eigen::Vector3d u=a/n; Eigen::Matrix3d K; K << LBA_SKEW(u);
  return I33 + std::sin(n)*K + (1-std::cos(n))*K*K;
}
inline Eigen::Vector3d Log(const Eigen::Matrix3d &R) {
  const double t=(R.trace()>3.0-1e-6)?0.0:std::acos(std::clamp(0.5*(R.trace()-1),-1.0,1.0));
  Eigen::Vector3d K(R(2,1)-R(1,2),R(0,2)-R(2,0),R(1,0)-R(0,1));
  return std::abs(t)<0.001 ? 0.5*K : 0.5*t/std::sin(t)*K;
}

struct BAState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
};

} // namespace lidar_ba

namespace std {
template<> struct hash<lidar_ba::VOXEL_LOC> {
  size_t operator()(const lidar_ba::VOXEL_LOC &s) const noexcept {
    using std::hash; return (((hash<int64_t>()(s.z)*116101ULL)%10000000000ULL + hash<int64_t>()(s.y))*116101ULL)%10000000000ULL + hash<int64_t>()(s.x);
  }
};
}

#endif
