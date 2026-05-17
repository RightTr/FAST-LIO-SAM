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
bool flip_en = false;
const Eigen::Matrix3d IMU_FLIP_R = (Eigen::Matrix3d() <<
    1.0,  0.0,  0.0,
    0.0, -1.0,  0.0,
    0.0,  0.0, -1.0).finished();

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
