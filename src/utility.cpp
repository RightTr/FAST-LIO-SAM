#include "utility.h"
#include "ros_utils.h"
using namespace std;

// CPU Params
int numberOfCores;

// Surrounding map
float surroundingkeyframeAddingDistThreshold; 
float surroundingkeyframeAddingAngleThreshold; 
float surroundingKeyframeDensity;
float surroundingKeyframeSearchRadius;

// Loop closure
bool  loopClosureEnableFlag;
float loopClosureFrequency;
int   surroundingKeyframeSize;
float historyKeyframeSearchRadius;
float historyKeyframeSearchTimeDiff;
float historyKeyframeSearchAngleThreshold;
int   historyKeyframeSearchNum;
float historyKeyframeFitnessScore;

// global map visualization radius
float globalMapVisualizationSearchRadius;
float globalMapVisualizationPoseDensity;
float globalMapVisualizationLeafSize;

float mappingICPSize;

int ikdtreeSearchNeighborNum;

bool keyframe_export_en = false;
bool keyframe_global_pcd_en = false;

std::string map_frame = "map";
std::string odom_frame = "odom";
std::string base_frame = "base_link";
std::string high_freq_base_frame = "base_hf_link";
std::string gnss_topic = "handsfree/rtk/gnss";
std::string gnss_heading_topic = "handsfree/rtk/heading";
bool gpsEnableFlag = false;
std::vector<double> gnss_extrinsic_T_raw(3, 0.0);
std::vector<double> gnss_extrinsic_R_raw{1.0, 0.0, 0.0,
                                         0.0, 1.0, 0.0,
                                         0.0, 0.0, 1.0};
Eigen::Vector3d gnss_extrinsic_T = Eigen::Vector3d::Zero();
Eigen::Matrix3d gnss_extrinsic_R = Eigen::Matrix3d::Identity();
double gnss_heading_offset_deg = 0.0;
double gnss_time_offset = 0.0;
bool useGpsElevation = false;
double gpsCovThreshold = 2.0;
double poseCovThreshold = 25.0;
double gnss_heading_deg = 0.0;
bool gnss_heading_valid = false;
std::atomic<bool> gnss_aligned(false);
std::deque<OdometryMsg> gps_buffer;
std::mutex mtx_gps;
std::mutex mtx_gnss_heading;
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
    rosparam_get("lio_sam/loopClosureFrequency", loopClosureFrequency, 1.0f);
    rosparam_get("lio_sam/surroundingKeyframeSize", surroundingKeyframeSize, 50);
    rosparam_get("lio_sam/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0f);
    rosparam_get("lio_sam/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0f);
    rosparam_get("lio_sam/historyKeyframeSearchAngleThreshold", historyKeyframeSearchAngleThreshold, 0.5f);
    rosparam_get("lio_sam/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
    rosparam_get("lio_sam/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3f);

    // Global pointcloud visualization
    rosparam_get("lio_sam/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3f);
    rosparam_get("lio_sam/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0f);
    rosparam_get("lio_sam/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0f);

    rosparam_get("lio_sam/mappingICPSize", mappingICPSize, 0.2f);
    rosparam_get("lio_sam/ikdtreeSearchNeighborNum", ikdtreeSearchNeighborNum, 8);
    rosparam_get("lio_sam/keyframe_export_en", keyframe_export_en, false);
    rosparam_get("lio_sam/keyframe_global_pcd_en", keyframe_global_pcd_en, false);
}

void read_gnss_params() {
    rosparam_get("gnss/topic", gnss_topic, std::string("handsfree/rtk/gnss"));
    rosparam_get("gnss/heading_topic", gnss_heading_topic, std::string("handsfree/rtk/heading"));
    rosparam_get("gnss/gpsEnableFlag", gpsEnableFlag, false);
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
    rosparam_get("gnss/heading_offset_deg", gnss_heading_offset_deg, 0.0);
    rosparam_get("gnss/time_offset", gnss_time_offset, 0.0);
    rosparam_get("gnss/useGpsElevation", useGpsElevation, false);
    rosparam_get("gnss/gpsCovThreshold", gpsCovThreshold, 2.0);
    rosparam_get("gnss/poseCovThreshold", poseCovThreshold, 25.0);
}
