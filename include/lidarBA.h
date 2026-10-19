#ifndef LIDAR_BA_H
#define LIDAR_BA_H

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <memory>
#include <vector>

struct ImuSample {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double t = 0.0;
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyr = Eigen::Vector3d::Zero();
};

struct BAFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id = -1;
    double timestamp = 0.0;
    bool keyframe = false;
    Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d gravity_odom = Eigen::Vector3d(0.0, 0.0, -9.81);
    std::vector<ImuSample, Eigen::aligned_allocator<ImuSample>> imu;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
};

struct LidarBAConfig {
    bool enable = true;
    bool feedback_to_ekf = false;
    int window_size = 10;
    int max_iterations = 5;
    double voxel_size = 1.0;
    double point_leaf_size = 0.2;
    int max_layer = 2;
    int min_points = 20;
    int thread_num = 2;
    double min_eigen_value = 0.0025;
    int max_fixed_points = 100;
    std::vector<double> plane_eigenvalue_threshold{0.25, 0.25, 0.25};
    std::vector<double> min_layer_points{5.0, 5.0, 5.0};
    bool imu_enable = true;
    bool gravity_optimize = false;
    double gyr_cov = 0.1;
    double acc_cov = 0.1;
    double bg_cov = 1e-4;
    double ba_cov = 1e-4;
    double gravity_prior_sigma_deg = 10.0;
};

extern LidarBAConfig lidarBAConfig;

class LidarBA {
public:
    explicit LidarBA(const LidarBAConfig &config = {});
    ~LidarBA();
    LidarBA(const LidarBA &) = delete;
    LidarBA &operator=(const LidarBA &) = delete;
    void push(BAFrame frame);
    bool pushAndWait(BAFrame frame, std::vector<int> &window_ids,
                     std::vector<Eigen::Matrix4d> &window_poses,
                     bool &window_optimized);
    bool tryPop(int &id, Eigen::Isometry3d &pose,
                Eigen::Vector3d &gravity_up_odom, double &gravity_sigma);
    void finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
