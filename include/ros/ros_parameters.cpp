#include "ros/ros_parameters.h"
#include "lidarBA.h"
#include "ros/ros_utils.h"
#include "preprocess.h"

#include <cmath>

// Shared configuration state used by the frontend and backend.
bool online_tree_pub_en = false;
bool ba_tree_pub_en = false;
int numberOfCores;

float surroundingkeyframeAddingDistThreshold;
float surroundingkeyframeAddingAngleThreshold;
float surroundingKeyframeDensity;
float surroundingKeyframeSearchRadius;

bool loopClosureEnableFlag;
bool scanContextEnableFlag;
int surroundingKeyframeSize;
float historyKeyframeSearchRadius;
float historyKeyframeSearchTimeDiff;
int historyKeyframeSearchNum;
float historyKeyframeFitnessScore;
float icpMaxCorrDistance;
double priorRotVar = 1e-4;
double priorPosVar = 1e-4;
double odomRotVar = 1e-3;
double odomPosVar = 1e-3;
double loopRotVar = 0.1;
double loopPosVar = 0.1;
int loopCurNum;
int loopPreNum;

float globalMapVisualizationSearchRadius;
float globalMapVisualizationLeafSize;
float mappingICPSize;
int ikdtreeSearchNeighborNum;

bool keyframe_export_en = false;
bool keyframe_global_pcd_en = false;
LidarBAConfig lidarBAConfig;
bool sceneEnableFlag = false;

std::string map_frame = "map";
std::string odom_frame = "odom";
std::string base_frame = "base_link";
std::string high_freq_base_frame = "base_link_hf";
std::string gnss_topic = "handsfree/rtk/gnss";
std::string gnss_heading_topic = "handsfree/rtk/heading";
bool gnssEnableFlag = false;
bool gnssPathVis = false;
double gpsFactorMinDis = 5.0;
double gnss_dt = 0.2;
std::vector<double> gnss_extrinsic_T_raw(3, 0.0);
std::vector<double> gnss_extrinsic_R_raw{1.0, 0.0, 0.0,
                                         0.0, 1.0, 0.0,
                                         0.0, 0.0, 1.0};
Eigen::Vector3d gnss_extrinsic_T = Eigen::Vector3d::Zero();
Eigen::Matrix3d gnss_extrinsic_R = Eigen::Matrix3d::Identity();
double heading_offset = 0.0;
bool useGnssYawFactor = false;
double gnssPosXYVar = 0.25;
double gnssPosZVar = 100.0;
double gnssYawVar = 0.01;
bool useGnssElevation = false;
std::atomic<bool> gnss_aligned(false);
bool flip_en = false;
const Eigen::Matrix3d IMU_FLIP_R = (Eigen::Matrix3d() <<
    1.0,  0.0,  0.0,
    0.0, -1.0,  0.0,
    0.0,  0.0, -1.0).finished();

int mapping_mode = 1;
bool use_online_map = true;
bool use_prior_map = false;

void set_mapping_mode()
{
    switch (mapping_mode)
    {
        case 1:
            use_online_map = true;
            use_prior_map = false;
            return;
        case 2:
            use_online_map = false;
            use_prior_map = true;
            return;
        case 3:
            use_online_map = true;
            use_prior_map = true;
            return;
        default:
            ROS_PRINT_WARN("unknown common/mode: %d, fallback to 1(online)", mapping_mode);
            mapping_mode = 1;
            use_online_map = true;
            use_prior_map = false;
            return;
    }
}

void read_mapping_params(Preprocess &preprocess, int &lidarType)
{
    rosparam_get("sam_enable", sam_enable, false);
    rosparam_get("publish/path_en", path_en, true);
    rosparam_get("publish/scan_publish_en", scan_pub_en, true);
    rosparam_get("publish/dense_publish_en", dense_pub_en, true);
    rosparam_get("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    rosparam_get("publish/online_tree_pub_en", online_tree_pub_en, false);
    rosparam_get("publish/ba_tree_pub_en", ba_tree_pub_en, false);
    rosparam_get("publish/effect_pub_en", effect_pub_en, false);
    rosparam_get("max_iteration", NUM_MAX_ITERATIONS, 4);
    rosparam_get("map_file_path", map_file_path, std::string(""));
    rosparam_get("common/lid_topic", lid_topic, std::string("/livox/lidar"));
    rosparam_get("common/imu_topic", imu_topic, std::string("/livox/imu"));
    rosparam_get("common/map_frame", map_frame, std::string("map"));
    rosparam_get("common/odom_frame", odom_frame, std::string("odom"));
    rosparam_get("common/base_frame", base_frame, std::string("base_link"));
    rosparam_get("common/high_freq_base_frame", high_freq_base_frame,
                 std::string("base_link_hf"));
    rosparam_get("reloc/reloc_topic", reloc_topic, std::string("/reloc/manual"));
    rosparam_get("common/time_sync_en", time_sync_en, false);
    rosparam_get("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    rosparam_get("common/flip_en", flip_en, false);
    rosparam_get("common/grav_align", grav_align, false);
    rosparam_get("common/mode", mapping_mode, 1);
    rosparam_get("mapping/imu_dt", imu_dt, 0.005);
    rosparam_get("mapping/timeRepair", timeRepair, true);
    rosparam_get("prior_map/prior_tree_path", prior_tree_path, std::string(""));
    rosparam_get("prior_map/prior_init", prior_init_en, false);
    rosparam_get("prior_map/prior_lidar_cov", prior_lidar_cov, 0.001);
    rosparam_get("filter_size_corner", filter_size_corner_min, 0.5);
    rosparam_get("filter_size_surf", filter_size_surf_min, 0.5);
    rosparam_get("filter_size_map", filter_size_map_min, 0.5);
    rosparam_get("cube_side_length", cube_len, 200.0);
    rosparam_get("mapping/det_range", DET_RANGE, 300.f);
    rosparam_get("mapping/fov_degree", fov_deg, 180.0);
    rosparam_get("mapping/gyr_cov", gyr_cov, 0.1);
    rosparam_get("mapping/acc_cov", acc_cov, 0.1);
    rosparam_get("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    rosparam_get("mapping/b_acc_cov", b_acc_cov, 0.0001);
    rosparam_get("mapping/point2point", point2point, false);
    rosparam_get("geo_map/enable", geo_map_enable, false);
    rosparam_get("geo_map/support", geo_support, 0.15);
    rosparam_get("geo_map/plane_dist", geo_plane_dist, 0.05);
    rosparam_get("preprocess/blind", preprocess.blind, 0.01);
    rosparam_get("preprocess/lidar_type", lidarType, (int)AVIA);
    rosparam_get("preprocess/scan_line", preprocess.N_SCANS, 16);
    rosparam_get("preprocess/timestamp_unit", preprocess.time_unit, (int)US);
    rosparam_get("preprocess/scan_rate", preprocess.SCAN_RATE, 10);
    rosparam_get("point_filter_num", preprocess.point_filter_num, 2);
    rosparam_get("feature_extract_enable", preprocess.feature_enabled, false);
    rosparam_get("result_save/imu_state_save_en", imu_state_save_en, false);
    rosparam_get("result_save/scan_frame_save_en", scan_frame_save_en, false);
    rosparam_get("result_save/ikdtree_output_save_en", ikdtree_output_save_en, false);
    rosparam_get("result_save/feat_accum_save_en", feat_accum_save_en, false);
    rosparam_get("result_save/interval", res_save_interval, -1);
    rosparam_get("mapping/extrinsic_T", extrinT, std::vector<double>());
    rosparam_get("mapping/extrinsic_R", extrinR, std::vector<double>());
}

void read_liosam_params() {

    // CPU parameters
    rosparam_get("lio_sam/numberOfCores", numberOfCores, 2);

    // Keyframe Strategy
    rosparam_get("lio_sam/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0f);
    rosparam_get("lio_sam/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2f);
    rosparam_get("lio_sam/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0f);
    rosparam_get("lio_sam/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0f);

    // Loop closure parameters
    rosparam_get("lio_sam/loopClosureEnableFlag", loopClosureEnableFlag, false);
    rosparam_get("lio_sam/scanContextEnableFlag", scanContextEnableFlag, false);
    rosparam_get("lio_sam/surroundingKeyframeSize", surroundingKeyframeSize, 50);
    rosparam_get("lio_sam/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0f);
    rosparam_get("lio_sam/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0f);
    rosparam_get("lio_sam/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
    rosparam_get("lio_sam/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3f);
    rosparam_get("lio_sam/icpMaxCorrDistance", icpMaxCorrDistance, 2.0f);
    rosparam_get("lio_sam/priorRotVar", priorRotVar, 1e-4);
    rosparam_get("lio_sam/priorPosVar", priorPosVar, 1e-4);
    rosparam_get("lio_sam/odomRotVar", odomRotVar, 1e-3);
    rosparam_get("lio_sam/odomPosVar", odomPosVar, 1e-3);
    rosparam_get("lio_sam/loopRotVar", loopRotVar, 0.1);
    rosparam_get("lio_sam/loopPosVar", loopPosVar, 0.1);
    rosparam_get("lio_sam/loopCurNum", loopCurNum, 5);
    rosparam_get("lio_sam/loopPreNum", loopPreNum, 3);

    // Global pointcloud visualization
    rosparam_get("lio_sam/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3f);
    rosparam_get("lio_sam/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0f);

    rosparam_get("lio_sam/mappingICPSize", mappingICPSize, 0.2f);
    rosparam_get("lio_sam/ikdtreeSearchNeighborNum", ikdtreeSearchNeighborNum, 8);
    rosparam_get("lio_sam/keyframe_export_en", keyframe_export_en, false);
    rosparam_get("lio_sam/keyframe_global_pcd_en", keyframe_global_pcd_en, false);

}

void read_lidar_ba_params()
{
    rosparam_get("lidar_ba/enable", lidarBAConfig.enable, true);
    rosparam_get("lidar_ba/feedback_to_ekf", lidarBAConfig.feedback_to_ekf, false);
    rosparam_get("lidar_ba/window_size", lidarBAConfig.window_size, 10);
    rosparam_get("lidar_ba/max_iterations", lidarBAConfig.max_iterations, 5);
    rosparam_get("lidar_ba/voxel_size", lidarBAConfig.voxel_size, 1.0);
    rosparam_get("lidar_ba/point_leaf_size", lidarBAConfig.point_leaf_size, 0.2);
    rosparam_get("lidar_ba/max_layer", lidarBAConfig.max_layer, 2);
    rosparam_get("lidar_ba/min_points", lidarBAConfig.min_points, 20);
    rosparam_get("lidar_ba/thread_num", lidarBAConfig.thread_num, 2);
    rosparam_get("lidar_ba/min_eigen_value", lidarBAConfig.min_eigen_value, 0.0025);
    rosparam_get("lidar_ba/max_fixed_points", lidarBAConfig.max_fixed_points, 100);
    rosparam_get("lidar_ba/plane_eigenvalue_threshold", lidarBAConfig.plane_eigenvalue_threshold, std::vector<double>{0.25, 0.25, 0.25});
    rosparam_get("lidar_ba/min_layer_points", lidarBAConfig.min_layer_points, std::vector<double>{5.0, 5.0, 5.0});
    rosparam_get("lidar_ba/imu_enable", lidarBAConfig.imu_enable, true);
}

void read_sgraph_params()
{
    rosparam_get("s-graph/sceneEnableFlag", sceneEnableFlag, true);
}

void read_gnss_params() {
    rosparam_get("gnss/topic", gnss_topic, std::string("handsfree/rtk/gnss"));
    rosparam_get("gnss/heading_topic", gnss_heading_topic, std::string("handsfree/rtk/heading"));
    rosparam_get("gnss/gnssPathVis", gnssPathVis, false);
    rosparam_get("gnss/gnssEnableFlag", gnssEnableFlag, false);
    rosparam_get("gnss/gpsFactorMinDis", gpsFactorMinDis, 5.0);
    rosparam_get("gnss/gnss_dt", gnss_dt, 0.2);
    rosparam_get("gnss/extrinsic_T", gnss_extrinsic_T_raw,
                 std::vector<double>{0.0, 0.0, 0.0});
    rosparam_get("gnss/extrinsic_R", gnss_extrinsic_R_raw,
                 std::vector<double>{1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0});
    gnss_extrinsic_T = Eigen::Vector3d(
        gnss_extrinsic_T_raw[0],
        gnss_extrinsic_T_raw[1],
        gnss_extrinsic_T_raw[2]);
    gnss_extrinsic_R <<
        gnss_extrinsic_R_raw[0], gnss_extrinsic_R_raw[1], gnss_extrinsic_R_raw[2],
        gnss_extrinsic_R_raw[3], gnss_extrinsic_R_raw[4], gnss_extrinsic_R_raw[5],
        gnss_extrinsic_R_raw[6], gnss_extrinsic_R_raw[7], gnss_extrinsic_R_raw[8];
    double heading_offset_deg = 0.0;
    rosparam_get("gnss/heading_offset_deg", heading_offset_deg, 0.0);
    heading_offset = heading_offset_deg * M_PI / 180.0;
    rosparam_get("gnss/useYawFactor", useGnssYawFactor, false);
    rosparam_get("gnss/posXYVar", gnssPosXYVar, 0.25);
    rosparam_get("gnss/posZVar", gnssPosZVar, 100.0);
    rosparam_get("gnss/yawVar", gnssYawVar, 0.01);
    rosparam_get("gnss/useGnssElevation", useGnssElevation, false);
}
