#ifndef ROS_PARAMETERS_H
#define ROS_PARAMETERS_H

#include <string>
#include <vector>
#include <atomic>
#include <Eigen/Geometry>
#include "lidarBA.h"

class Preprocess;

// Runtime configuration shared by ROS parameter loading and mapping modules.
extern bool sam_enable, path_en, scan_pub_en, dense_pub_en, scan_body_pub_en;
extern bool online_tree_pub_en, ba_tree_pub_en;
extern bool effect_pub_en, time_sync_en, grav_align, timeRepair;
extern bool prior_init_en, point2point, geo_map_enable;
extern int NUM_MAX_ITERATIONS, lidar_type, res_save_interval, feats_down_size;
extern double time_diff_lidar_to_imu, imu_dt, prior_lidar_cov;
extern double filter_size_corner_min, filter_size_surf_min, filter_size_map_min;
extern double cube_len, fov_deg, gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
extern double geo_support, geo_plane_dist;
extern float DET_RANGE;
extern std::string map_file_path, lid_topic, imu_topic, reloc_topic, prior_tree_path;
extern std::vector<double> extrinT, extrinR;
extern bool feat_accum_save_en, scan_frame_save_en, imu_state_save_en;
extern bool ikdtree_output_save_en;

extern int numberOfCores;
extern float surroundingkeyframeAddingDistThreshold, surroundingkeyframeAddingAngleThreshold;
extern float surroundingKeyframeDensity, surroundingKeyframeSearchRadius;
extern bool loopClosureEnableFlag, scanContextEnableFlag;
extern float historyKeyframeSearchRadius;
extern float historyKeyframeSearchTimeDiff, historyKeyframeFitnessScore, icpMaxCorrDistance;
extern int surroundingKeyframeSize, historyKeyframeSearchNum, loopCurNum, loopPreNum;
extern double priorRotVar, priorPosVar, odomRotVar, odomPosVar, loopRotVar, loopPosVar;
extern float globalMapVisualizationSearchRadius;
extern float globalMapVisualizationLeafSize, mappingICPSize;
extern int ikdtreeSearchNeighborNum;
extern bool keyframe_export_en, keyframe_global_pcd_en, sceneEnableFlag;
extern std::string map_frame, odom_frame, base_frame, high_freq_base_frame;
extern std::string gnss_topic, gnss_heading_topic;
extern bool gnssEnableFlag, gnssPathVis;
extern double gpsFactorMinDis, gnss_dt, heading_offset;
extern std::vector<double> gnss_extrinsic_T_raw, gnss_extrinsic_R_raw;
extern Eigen::Vector3d gnss_extrinsic_T;
extern Eigen::Matrix3d gnss_extrinsic_R;
extern bool useGnssYawFactor, useGnssElevation, flip_en;
extern double gnssPosXYVar, gnssPosZVar, gnssYawVar;
extern std::atomic<bool> gnss_aligned;
extern const Eigen::Matrix3d IMU_FLIP_R;
extern int mapping_mode;
extern bool use_online_map, use_prior_map;

void set_mapping_mode();

void read_mapping_params(Preprocess &preprocess, int &lidarType);
void read_liosam_params();
void read_lidar_ba_params();
void read_sgraph_params();
void read_gnss_params();

#endif
