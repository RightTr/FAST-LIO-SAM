# FAST-LIO-SAM

A LiDAR-inertial SLAM system that integrates **FAST-LIO2** as the high-frequency front-end with a **LIO-SAM-style** factor graph back-end for global optimization, supporting **RoboSense series LiDARs**, and compatible with both **ROS1** and **ROS2**.

## Related Works

[FAST-LIO2: Fast Direct LiDAR-inertial Odometry](doc/Fast_LIO_2.pdf)

[FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter](https://arxiv.org/abs/2010.08196)

[FAST_LIO_SAM](https://github.com/kahowang/FAST_LIO_SAM.git)

[LIO-SAM](https://github.com/TixiaoShan/LIO-SAM.git)

## Contributions

* A SLAM system that integrates FAST-LIO2 with a LIO-SAM-style factor graph backend.

* ROS2 adaptation

* High-frequency odometry via IMU propagation between LiDAR scans

* Manual initial pose setting for relocalization

* Support for RoboSense series LiDARs

## Prerequisites

### ROS1 Build

```bash
mkdir fastlio_sam_ws
cd fastlio_sam_ws

mkdir src && cd src
git clone https://github.com/RightTr/FAST-LIO-SAM.git
git clone https://github.com/RightTr/fast_lio_interfaces.git

cd src/FAST-LIO-SAM
git submodule update --init --recursive

./build.sh ROS1
```

**Directly Run:**

```bash
cd fastlio_sam_ws 

source devel/setup.bash

# e.g.
roslaunch fast_lio mapping_mid360.launch
```

### ROS2 Build

```bash
mkdir fastlio_ws
cd fastlio_ws

mkdir src && cd src
git clone https://github.com/RightTr/FAST_LIO.git
git clone https://github.com/RightTr/fast_lio_interfaces.git

cd src/FAST-LIO
git submodule update --init --recursive

./build.sh humble
```

**Directly Run:**

```bash
cd fastlio_ws 

source install/setup.bash

# e.g.
ros2 launch fast_lio mapping_mid360.launch.py
```

### Relocalization Mode

The proposed system supports relocalization using manually set odometry poses. Once odometry poses are published to the */reloc_topic* (according to the following .yaml file), the system will reset the system and the initial pose according to your input.

Run Relocalization Mode:

```bash
# e.g.
# ROS1
cd fastlio_ws 
source devel/setup.bash
roslaunch fast_lio reloc_mid360.launch
# Publish geometry_msgs::PoseStamped to the /reloc_topic

# ROS2
cd fastlio_ws 
source install/setup.bash
ros2 launch fast_lio reloc_mid360.launch.py
# Publish geometry_msgs::msg::PoseStamped to the /reloc_topic
```

### High Frequency Odometry based on IMU Propagation

Subscribe the topic */OdometryHighFreq* to receive high frequency odometry output via IMU propagation between LiDAR scans.

### RoboSense Series LiDAR Adpatation

Now, FAST-LIO supports tracking and mapping using the RoboSense LiDARs (e.g., RoboSense Airy). Check the related files in ./config and ./launch folder.

```bash
# e.g.
roslaunch fast_lio mapping_airy.launch
```

## TODO List

* Robosense Airy LiDARs Support ✅ (2026.1.24)
* Robosense Airy Adaption Test ✅ (2026.3.6)
* FastLIO frontend with LIO-SAM-style factor graph backend
