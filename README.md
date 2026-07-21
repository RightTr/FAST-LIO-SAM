# FAST-LIO-SAM

A LiDAR-inertial SLAM system that integrates **FAST-LIO2** as the high-frequency frontend with a **LIO-SAM-style** factor graph backend for global optimization, supporting **RoboSense LiDARs**, **Unilidar LiDARs**, and compatible with both **ROS1** and **ROS2**.

## 🧩 Contributions

* A SLAM system that integrates FAST-LIO2 with a LIO-SAM-style factor graph backend.

* ROS1 and ROS2 adaptation

* High-frequency odometry via IMU propagation between LiDAR scans

* Manual initial pose setting for relocalization

* Stationary detection and adaptive weight handling between LiDAR update scans and ZUPT

* Support for RoboSense LiDARs, Unilidar LiDARs
  
* The standard navigation TF tree: **map** → **odom** → **base_link**

* Support localization mode: Online, Prior and Online+Prior localization

## 🛠️ Prerequisites

### Dependency

* [gtsam](https://gtsam.org/get_started/) (Georgia Tech Smoothing and Mapping library)

```bash
# Ubuntu 20.04
sudo add-apt-repository ppa:borglab/gtsam-release-4.0
# Ubuntu 22.04
sudo add-apt-repository ppa:borglab/gtsam-release-4.1

sudo apt install libgtsam-dev libgtsam-unstable-dev
```

### ROS1 Build

```bash
mkdir fastlio_sam_ws
cd fastlio_sam_ws

mkdir src && cd src
git clone https://github.com/RightTr/FAST-LIO-SAM.git

cd src/FAST-LIO-SAM
git submodule update --init --recursive

# ROS1 build
./build.sh ROS1

# ROS2 build
./build.sh humble
```

## 🚀 Usage

### LIO-SAM-style Backend

```bash
cd fastlio_ws 
source devel/setup.bash
# e.g.
roslaunch fast_lio_sam mapping_airy.launch
```

### Relocalization

The modified system supports relocalization using manually set odometry poses. Once odometry poses are published to the */reloc_topic* (according to the following .yaml file), the system will apply a pose correction and update the current state consistently.

### Localization mode

`common/mode`:
- `1`: online
  - Use only the online map for constraints.
- `2`: prior
  - Use only the prior map for constraints.
- `3`: online+prior
  - Use both the online map and the prior map for constraints.

`prior_map/prior_init` enables one-shot startup alignment in `prior` mode.
It uses the prior map to refine the initial pose before tracking starts.

You can generate a prior `ikdtree` snapshot from a PCD file with `pcd_to_ikdtree_bin`.

### Keyframe export

Enable `lio_sam/keyframe_export_en` to export each accepted keyframe as a full-resolution PCD built from `feats_undistort`, together with its pose, under `ROOT_DIR/RESULTS/KEY_FRAMES/scans/` and `ROOT_DIR/RESULTS/KEY_FRAMES/pose.txt`.

Enable `lio_sam/keyframe_global_pcd_en` to additionally stitch all exported keyframes into one global PCD at `ROOT_DIR/RESULTS/KEY_FRAMES/global.pcd`.

### Result save

* `result_save/feat_accum_save_en` to accumulate each undistorted scan in the world frame. The merged feature cloud is maintained in memory during runtime and written to `ROOT_DIR/RESULTS/PCD/<lidar_time>.pcd` when the system shuts down. 
* `result_save/imu_state_save_en` saves each IMU state to `ROOT_DIR/RESULTS/IMU_STATES/imu_state.txt`. 
* `result_save/scan_frame_save_en` saves each undistorted scan in LiDAR body frame under `ROOT_DIR/RESULTS/SCAN_FRAMES/scans/<lidar_time>.pcd`, saves the corresponding timestamped cloud under `ROOT_DIR/RESULTS/SCAN_FRAMES/scans_tstamp/<lidar_time>.pcd`, and writes the pose to `ROOT_DIR/RESULTS/SCAN_FRAMES/scan_pose.txt`.

## Stationary detection and adaptive weight handling

The system will adjust the confidence of the ZUPT and LiDAR updates based on the detected motion state, using **accelerometer and gyroscope variances** as well as **the EMA of velocity**. When the system detects a stationary state, it will increase the confidence of the ZUPT update and decrease the confidence of the LiDAR update, and vice versa when in motion.

Check the related parameters in the .yaml files.

```yaml
zupt:
    use_zupt: true                    # enable adaptive zero velocity update
    zupt_acc_var_threshold:  0.0001   # (m/s²)² per-axis acc variance → full-confidence threshold
    zupt_gyro_var_threshold: 0.00001  # (rad/s)² per-axis gyro variance → full-confidence threshold

    # Static confidence: exp(-3 * max_normalized_variance), in [0,1]
    zupt_confidence_min: 0.05         # below this no ZUPT is applied

    # Dynamic ZUPT weight: R = clamp(zupt_r_min / eff_confidence, zupt_r_max)
    zupt_r_min: 1.0e-5                # ZUPT measurement noise at full confidence
    zupt_r_max: 1.0                   # ZUPT measurement noise cap (≈ no constraint)

    # Dynamic LiDAR weight: lidar_cov = LASER_POINT_COV * (1 + scale*conf) * max(1, residual/ref)
    lidar_cov_static_scale: 5.0       # LiDAR cov multiplier at full confidence
    lidar_residual_ref: 0.05          # (m) reference residual; above this LiDAR trust reduces

    # Covariance inflation (lock-in prevention)
    cov_inflate_start: 200            # IMU steps of continuous static before inflation kicks in
    cov_inflate_pos:   1.0e-7         # per-step position covariance inflation
    cov_inflate_rot:   1.0e-8         # per-step rotation covariance inflation
```

### High frequency odometry via IMU propagation between LiDAR scans

Subscribe the topic */OdometryHighFreq* to receive high frequency odometry output via IMU propagation between LiDAR scans.

### Extended LiDAR support

Now, FAST-LIO supports tracking and mapping using the RoboSense LiDARs (e.g., RoboSense Airy) and Unilidar LiDARs (e.g., Unilidar L2). Check the related files in ./config, ./launch_ROS1/odom, ./launch_ROS1/mapping, ./launch_ROS2/odom, and ./launch_ROS2/mapping.

```bash
# e.g.
roslaunch fast_lio_sam odom_airy.launch
```

## 📝 TODO List

* [x] Full ROS2 adaptation
* [x] ROS2 adaptation Test
* [] ZUPT parameter tuning and testing
* [] Prior localization test

## 📚 Related Works

[FAST-LIO2: Fast Direct LiDAR-inertial Odometry](doc/Fast_LIO_2.pdf)

[FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter](https://arxiv.org/abs/2010.08196)

[FAST-LIO official repository](https://github.com/hku-mars/FAST_LIO.git)

[FAST_LIO_SAM](https://github.com/kahowang/FAST_LIO_SAM.git)

[LIO-SAM](https://github.com/TixiaoShan/LIO-SAM.git)

[robosense_fast_lio](https://github.com/RuanJY/robosense_fast_lio.git)

[point_lio_unilidar](https://github.com/unitreerobotics/point_lio_unilidar.git)
