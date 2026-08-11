#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <iomanip>
#include <csignal>
#include <unistd.h>
#include <limits>
#include <Python.h>
#include <so3_math.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include "GNSS_Processing.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include "preprocess.h"
#include "ikd-Tree/ikdtree_public.h"
#include <atomic>
#include "posebuffer.h"
#include <thread>
#include "ros_utils.h"
#include "map_optimization.h"
#include "utility.h"
#include "common_utils.h"
#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define PUBFRAME_PERIOD     (20)

double kdtree_incremental_time = 0.0, kdtree_delete_time = 0.0;
double match_time = 0, solve_time = 0;
int    kdtree_size_st = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   feat_accum_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
bool   imu_state_save_en = false, scan_frame_save_en = false, ikdtree_output_save_en = false;

bool feature_pub_en = false, effect_pub_en = false;

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;
string reloc_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0;
double last_raw_timestamp_lidar = -1.0, last_raw_timestamp_imu = -1.0;
double lidar_timestamp_offset_sec = 0.0;
double imu_timestamp_offset_sec = 0.0;
double imu_dt = 0.005;
double lidar_dt = 0.01;
bool   timeRepair = true;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, res_save_interval = -1;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_EKF_inited;
std::atomic<bool> flg_exit(false);
Eigen::Matrix3d R_map_odom = Eigen::Matrix3d::Identity();
Eigen::Vector3d t_map_odom = Eigen::Vector3d::Zero();
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool sam_enable = false;
bool grav_align = false;
int lidar_type;
bool use_zupt = false;
std::string prior_tree_path;
double zupt_acc_var_threshold;
double zupt_gyro_var_threshold;
double prior_lidar_cov = 0.001;
bool prior_init_en = false;
bool prior_init_done = false;
// Adaptive ZUPT params
double zupt_r_min              = 1e-5;
double zupt_r_max              = 1.0;
double zupt_confidence_min     = 0.05;
double zupt_inflate_pos        = 1e-7;
double zupt_inflate_rot        = 1e-8;
int    zupt_inflate_start      = 200;
// Adaptive LiDAR weight params
double lidar_cov_static_scale  = 5.0;
double lidar_residual_ref      = 0.05;

std::shared_ptr<GnssProcess> p_gnss = std::make_shared<GnssProcess>();
PathPublisher pubGnssPath;

static void updateMapOdom(const Eigen::Matrix3d &R_map_body,
                          const Eigen::Vector3d &t_map_body,
                          const Eigen::Matrix3d &R_odom_body,
                          const Eigen::Vector3d &t_odom_body)
{
    R_map_odom = R_map_body * R_odom_body.transpose();
    t_map_odom = t_map_body - R_map_odom * t_odom_body;
}

void setMapOdom(const Eigen::Matrix3d &R_map_odom_,
                const Eigen::Vector3d &t_map_odom_)
{
    R_map_odom = R_map_odom_;
    t_map_odom = t_map_odom_;
}

void publishGnssPath(const PosData &pos);
bool initGnssMap(double lidar_stamp_sec);

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points;
vector<PointVector>  Prior_Nearest_Points;
vector<uint8_t>      prior_point_selected;
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<ImuMsgConstPtr> imu_buffer;

mutex mtx_reloc;
condition_variable sig_reloc;
Pose reloc_state;
std::atomic<bool> relocalize_flag(false);

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr featsFromPriorMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE_PUBLIC<PointType> ikdtree;
KD_TREE_PUBLIC<PointType> prior_ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

PathMsg path;
PathMsg gnssPath;
OdomMsg odomAftMapped;
QuaternionMsg geoQuat;
PoseStampedMsg msg_body_pose;

int    scan_frame_idx = 0;
std::ofstream scan_frame_pose_file;
std::ofstream imu_pose_file;
std::ofstream keyframe_pose_file;
pcl::PointCloud<PointTypeIndex>::Ptr keyframe_global_cloud(new pcl::PointCloud<PointTypeIndex>());

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

struct MatchObservation
{
    PointType body;
    PointType normal;
    double residual = 0.0;
    double row_scale = 1.0;
};

double adapt_lidar_weight()
{
    if (!use_zupt)
        return LASER_POINT_COV;

    // confidence cached by UndistortPcl inside p_imu->Process()
    const double static_confidence = p_imu->get_static_confidence();

    // Adaptive LiDAR cov: rises with static confidence and previous-frame residual
    const double lidar_static_scale   = 1.0 + lidar_cov_static_scale * static_confidence;
    const double lidar_residual_scale = std::max(1.0, res_mean_last / lidar_residual_ref);
    return std::min(LASER_POINT_COV * lidar_static_scale * lidar_residual_scale, 0.1);
}

bool build_point_observation(KD_TREE_PUBLIC<PointType> *tree,
                             const PointVector *points_near,
                             const PointType &point_body,
                             const PointType &point_world,
                             double row_scale,
                             bool use_tree_search,
                             PointVector *points_near_out,
                             MatchObservation &obs_out)
{
    VF(4) pabcd;
    double pd2 = 0.0;

    if (use_tree_search)
    {
        if (tree == nullptr)
            return false;

        vector<float> point_search_sq_dis(NUM_MATCH_POINTS);
        PointVector points_near;
        tree->Nearest_Search(point_world, NUM_MATCH_POINTS, points_near,
                             point_search_sq_dis);

        if (points_near_out != nullptr)
            *points_near_out = points_near;

        if (points_near.size() < NUM_MATCH_POINTS ||
            point_search_sq_dis[NUM_MATCH_POINTS - 1] > 5)
            return false;
        if (!esti_plane(pabcd, points_near, 0.1f))
            return false;

        pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y +
              pabcd(2) * point_world.z + pabcd(3);
    }
    else
    {
        if (points_near == nullptr || points_near->size() < NUM_MATCH_POINTS)
            return false;
        if (!esti_plane(pabcd, *points_near, 0.1f))
            return false;
        pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y +
              pabcd(2) * point_world.z + pabcd(3);
    }

    const double body_range = std::sqrt(point_body.x * point_body.x +
                                        point_body.y * point_body.y +
                                        point_body.z * point_body.z);
    const double surf_quality = 1.0 - 0.9 * std::fabs(pd2) /
                                std::sqrt(std::max(1e-6, body_range));
    if (surf_quality <= 0.9)
        return false;

    obs_out.body = point_body;
    obs_out.normal.x = pabcd(0);
    obs_out.normal.y = pabcd(1);
    obs_out.normal.z = pabcd(2);
    obs_out.normal.intensity = pd2;
    obs_out.residual = std::fabs(pd2);
    obs_out.row_scale = row_scale;
    return true;
}

bool loadPriorTreeMap()
{

    if (prior_tree_path.empty())
    {
        ROS_PRINT_WARN("prior mode enabled, but no prior_tree_path provided");
        return false;
    }

    if (prior_ikdtree.LoadIkdtree(prior_tree_path))
    {
        const bool prior_map_ready = prior_ikdtree.Root_Node != nullptr;
        PointVector().swap(prior_ikdtree.PCL_Storage);
        prior_ikdtree.flatten(prior_ikdtree.Root_Node, prior_ikdtree.PCL_Storage, NOT_RECORD);
        featsFromPriorMap->clear();
        featsFromPriorMap->points = prior_ikdtree.PCL_Storage;
        ROS_PRINT_INFO("loaded prior ikdtree snapshot: %s, nodes=%d",
                       prior_tree_path.c_str(), prior_ikdtree.validnum());
        return prior_map_ready;
    }

    ROS_PRINT_ERROR("failed to load prior ikdtree snapshot: %s", prior_tree_path.c_str());
    return false;
}

bool coarsePriorIcpAlign(Eigen::Matrix3d &R_map_lidar,
                         Eigen::Vector3d &t_map_lidar,
                         double &fitness_score)
{
    fitness_score = std::numeric_limits<double>::infinity();

    if (!use_prior_map || featsFromPriorMap == nullptr || featsFromPriorMap->empty() ||
        feats_down_body == nullptr || feats_down_body->size() < 20)
    {
        return false;
    }

    PointCloudXYZI::Ptr prior_map_ds(new PointCloudXYZI());
    if (featsFromPriorMap->size() > 2000)
    {
        pcl::VoxelGrid<PointType> downsample;
        downsample.setLeafSize(0.8f, 0.8f, 0.8f);
        downsample.setInputCloud(featsFromPriorMap);
        downsample.filter(*prior_map_ds);
    }
    else
    {
        *prior_map_ds = *featsFromPriorMap;
    }

    if (prior_map_ds->size() < 50)
    {
        return false;
    }

    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(8.0);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);
    icp.setInputSource(feats_down_body);
    icp.setInputTarget(prior_map_ds);

    Eigen::Matrix3d R_map_base_guess;
    Eigen::Vector3d t_map_base_guess;
    composeMapPose(state_point.rot.toRotationMatrix(), state_point.pos,
                   R_map_base_guess, t_map_base_guess);

    const Eigen::Matrix3d R_base_lidar = state_point.offset_R_L_I.toRotationMatrix();
    const Eigen::Vector3d t_base_lidar = state_point.offset_T_L_I;
    const Eigen::Matrix3d R_map_lidar_guess = R_map_base_guess * R_base_lidar;
    const Eigen::Vector3d t_map_lidar_guess = t_map_base_guess +
                                              R_map_base_guess * t_base_lidar;

    Eigen::Affine3f guess = Eigen::Affine3f::Identity();
    guess.linear() = R_map_lidar_guess.cast<float>();
    guess.translation() = t_map_lidar_guess.cast<float>();
    const Eigen::Matrix4f guess_matrix = guess.matrix();

    PointCloudXYZI aligned;
    icp.align(aligned, guess_matrix);
    if (!icp.hasConverged())
    {
        return false;
    }

    fitness_score = icp.getFitnessScore();
    if (!std::isfinite(fitness_score) || fitness_score > 5.0)
    {
        return false;
    }

    const Eigen::Matrix4f final_tf = icp.getFinalTransformation();
    R_map_lidar = final_tf.block<3, 3>(0, 0).cast<double>();
    t_map_lidar = final_tf.block<3, 1>(0, 3).cast<double>();

    return true;
}

void save_scan_frame(const string& scan_frames_dir)
{
    // Save undistorted scan in LiDAR body frame.
    const std::string stamp_str = format_unix_time(lidar_end_time);

    pcl::PointCloud<pcl::PointXYZ> scan_xyz;
    pcl::PointCloud<pcl::PointXYZI> scan_xyz_tstamp;
    scan_xyz.reserve(feats_undistort->size());
    scan_xyz_tstamp.reserve(feats_undistort->size());
    for (const auto &pt : feats_undistort->points)
    {
        pcl::PointXYZ p_xyz;
        p_xyz.x = pt.x;
        p_xyz.y = pt.y;
        p_xyz.z = pt.z;
        scan_xyz.push_back(p_xyz);

        pcl::PointXYZI p_xyz_tstamp;
        p_xyz_tstamp.x = pt.x;
        p_xyz_tstamp.y = pt.y;
        p_xyz_tstamp.z = pt.z;
        p_xyz_tstamp.intensity = pt.curvature;
        scan_xyz_tstamp.push_back(p_xyz_tstamp);
    }

    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(scan_frames_dir + "scans/" + stamp_str + ".pcd", scan_xyz);
    pcd_writer.writeBinary(scan_frames_dir + "scans_tstamp/" + stamp_str + ".pcd", scan_xyz_tstamp);

    // Save LiDAR pose in world frame, TUM format: timestamp tx ty tz qx qy qz qw
    V3D p_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    auto q_lid = state_point.rot * state_point.offset_R_L_I;
    scan_frame_pose_file << lidar_end_time << " "
              << p_lid.x() << " " << p_lid.y() << " " << p_lid.z() << " "
              << q_lid.coeffs()[0] << " " << q_lid.coeffs()[1] << " "
              << q_lid.coeffs()[2] << " " << q_lid.coeffs()[3] << "\n";

    scan_frame_idx++;
}

void save_ikdtree_cloud(const string& ikdtree_cloud_path)
{
    if (ikdtree.Root_Node == nullptr)
        return;

    PointVector tree_points;
    ikdtree.flatten(ikdtree.Root_Node, tree_points, NOT_RECORD);
    if (tree_points.empty())
        return;

    PointCloudXYZI tree_cloud;
    tree_cloud.points = tree_points;
    tree_cloud.width = tree_cloud.points.size();
    tree_cloud.height = 1;
    tree_cloud.is_dense = true;

    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(ikdtree_cloud_path, tree_cloud);
}

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_PRINT_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

static inline V3D transformBodyPointToGlobal(const V3D &p_body,
                                             const Eigen::Matrix3d &R_world_body,
                                             const Eigen::Vector3d &t_world_body,
                                             const state_ikfom &state);

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    const V3D p_body(pi->x, pi->y, pi->z);
    const V3D p_global = transformBodyPointToGlobal(
        p_body, s.rot.toRotationMatrix(), s.pos, s);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

static inline V3D transformBodyPointToGlobal(const V3D &p_body,
                                             const Eigen::Matrix3d &R_world_body,
                                             const Eigen::Vector3d &t_world_body,
                                             const state_ikfom &state)
{
    return R_world_body * (state.offset_R_L_I * p_body + state.offset_T_L_I) + t_world_body;
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    const V3D p_body(pi->x, pi->y, pi->z);
    const V3D p_global = transformBodyPointToGlobal(
        p_body, state_point.rot.toRotationMatrix(), state_point.pos, state_point);
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToMap(PointType const * const pi, PointType * const po)
{
    const V3D p_body(pi->x, pi->y, pi->z);
    Eigen::Matrix3d R_map_body;
    Eigen::Vector3d t_map_body;
    composeMapPose(state_point.rot.toRotationMatrix(), state_point.pos, R_map_body, t_map_body);
    const V3D p_global = transformBodyPointToGlobal(p_body, R_map_body, t_map_body, state_point);
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    const V3D p_body(pi[0], pi[1], pi[2]);
    const V3D p_global = transformBodyPointToGlobal(
        p_body, state_point.rot.toRotationMatrix(), state_point.pos, state_point);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

template<typename T>
void pointBodyToMap(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    const V3D p_body(pi[0], pi[1], pi[2]);
    Eigen::Matrix3d R_map_body;
    Eigen::Vector3d t_map_body;
    composeMapPose(state_point.rot.toRotationMatrix(), state_point.pos, R_map_body, t_map_body);
    const V3D p_global = transformBodyPointToGlobal(p_body, R_map_body, t_map_body, state_point);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    const V3D p_body(pi->x, pi->y, pi->z);
    const V3D p_global = transformBodyPointToGlobal(
        p_body, state_point.rot.toRotationMatrix(), state_point.pos, state_point);
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyToMap(PointType const * const pi, PointType * const po)
{
    const V3D p_body(pi->x, pi->y, pi->z);
    Eigen::Matrix3d R_map_body;
    Eigen::Vector3d t_map_body;
    composeMapPose(state_point.rot.toRotationMatrix(), state_point.pos, R_map_body, t_map_body);
    const V3D p_global = transformBodyPointToGlobal(p_body, R_map_body, t_map_body, state_point);
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const Pcl2MsgConstPtr &msg)
{
    mtx_buffer.lock();
    const double raw_timestamp = get_ros_time_sec(msg->header.stamp);
    double corrected_timestamp = raw_timestamp;

    if (timeRepair)
    {
        if (!repairTimestamp(raw_timestamp, lidar_dt,
            lidar_timestamp_offset_sec, last_raw_timestamp_lidar,
            last_timestamp_lidar, corrected_timestamp))
        {
            mtx_buffer.unlock();
            return;
        }
    }
    else
    {
        last_raw_timestamp_lidar = raw_timestamp;
        last_timestamp_lidar = raw_timestamp;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(corrected_timestamp);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const LivoxCustomMsgConstPtr &msg) 
{
    mtx_buffer.lock();
    const double raw_timestamp = get_ros_time_sec(msg->header.stamp);
    double corrected_timestamp = raw_timestamp;

    if (timeRepair)
    {
        if (!repairTimestamp(raw_timestamp, lidar_dt,
            lidar_timestamp_offset_sec, last_raw_timestamp_lidar,
            last_timestamp_lidar, corrected_timestamp))
        {
            mtx_buffer.unlock();
            return;
        }
    }
    else
    {
        last_raw_timestamp_lidar = raw_timestamp;
        last_timestamp_lidar = raw_timestamp;
    }
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && last_timestamp_imu > 0.0 && last_timestamp_lidar > 0.0 &&
        abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(corrected_timestamp);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const ImuMsgConstPtr &msg_in) 
{   
    publish_count ++;
    // cout<<"IMU got at: "<<get_ros_time_sec(msg_in->header.stamp)<<endl;
    ImuMsgPtr msg(new ImuMsg(*msg_in));

    if (flip_en)
    {
        // Use a proper rotation (det=+1) instead of a reflection.
        const V3D gyr_raw(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        const V3D acc_raw(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
        const V3D gyr_flip = IMU_FLIP_R * gyr_raw;
        const V3D acc_flip = IMU_FLIP_R * acc_raw;

        msg->angular_velocity.x = gyr_flip.x();
        msg->angular_velocity.y = gyr_flip.y();
        msg->angular_velocity.z = gyr_flip.z();
        msg->linear_acceleration.x = acc_flip.x();
        msg->linear_acceleration.y = acc_flip.y();
        msg->linear_acceleration.z = acc_flip.z();
    }

    const double msg_in_stamp_sec = get_ros_time_sec(msg_in->header.stamp);
    msg->header.stamp = get_ros_time(msg_in_stamp_sec - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = get_ros_time(timediff_lidar_wrt_imu + msg_in_stamp_sec);
    }
    const double raw_timestamp = get_ros_time_sec(msg->header.stamp);

    mtx_buffer.lock();
    double corrected_timestamp = raw_timestamp;
    if (timeRepair)
    {
        if (!repairTimestamp(raw_timestamp, imu_dt,
            imu_timestamp_offset_sec, last_raw_timestamp_imu,
            last_timestamp_imu, corrected_timestamp))
        {
            mtx_buffer.unlock();
            return;
        }
    }
    else
    {
        last_raw_timestamp_imu = raw_timestamp;
        last_timestamp_imu = raw_timestamp;
    }

    msg->header.stamp = get_ros_time(corrected_timestamp);

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

/*** relocation callback ***/
void reloc_cbk(const PoseStampedMsgConstPtr &msg_in) 
{
    double timestamp = get_ros_time_sec(msg_in->header.stamp);
    double x = msg_in->pose.position.x;
    double y = msg_in->pose.position.y;
    double z = msg_in->pose.position.z;

    double qx = msg_in->pose.orientation.x;
    double qy = msg_in->pose.orientation.y;
    double qz = msg_in->pose.orientation.z;
    double qw = msg_in->pose.orientation.w;
    Eigen::Quaterniond q(qw, qx, qy, qz);
    
    std::lock_guard<std::mutex> lock(mtx_reloc);
    reloc_state = Pose(x, y, z,
                    q.x(), q.y(), q.z(), q.w(), timestamp);
    relocalize_flag.store(true); 
    ROS_PRINT_INFO("Reloc received: (%.3f, %.3f, %.3f), quat=(%.3f, %.3f, %.3f, %.3f)",
        x, y, z, q.x(), q.y(), q.z(), q.w());
}

void gnss_cbk(const GnssFixMsgConstPtr &msg_in)
{
    static double last_gnss_timestamp = -1.0;

    if (!p_gnss) {
        return;
    }

    const double t = get_ros_time_sec(msg_in->header.stamp);
    if (!std::isfinite(t))
    {
        return;
    }

    if (last_gnss_timestamp >= 0.0 && t <= last_gnss_timestamp)
    {
        ROS_PRINT_WARN("GNSS fix time rollback: cur=%.9f last=%.9f", t, last_gnss_timestamp);
        return;
    }

    last_gnss_timestamp = t;
    p_gnss->pushFix(msg_in);

    if (!(gnssEnableFlag || gnssPathVis) || !gnss_aligned.load())
    {
        return;
    }

    PosData pos;
    if (p_gnss->latestPos(pos) && pos.t >= 0.0)
    {
        publishGnssPath(pos);
    }
}

void gnss_heading_cbk(const OdometryMsgConstPtr &msg_in)
{
    static double last_heading_timestamp = -1.0;

    if (!p_gnss) {
        return;
    }

    const double t = get_ros_time_sec(msg_in->header.stamp);
    if (!std::isfinite(t))
    {
        return;
    }

    if (last_heading_timestamp >= 0.0 && t <= last_heading_timestamp)
    {
        ROS_PRINT_WARN("GNSS yaw time rollback: cur=%.9f last=%.9f", t, last_heading_timestamp);
        return;
    }

    last_heading_timestamp = t;
    p_gnss->pushYaw(msg_in);
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();


        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_PRINT_WARN("Too few input point cloud!");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }
        if(lidar_type == MARSIM)
            lidar_end_time = meas.lidar_beg_time;

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    const double imu_front_stamp = imu_buffer.empty()
        ? -1.0
        : get_ros_time_sec(imu_buffer.front()->header.stamp);
    double imu_time = imu_front_stamp;
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_ros_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    if (meas.imu.empty())
    {
        ROS_PRINT_WARN(
            "No IMU for lidar scan: lidar=[%.9f, %.9f], imu_front=%.9f, imu_last=%.9f",
            meas.lidar_beg_time,
            lidar_end_time,
            imu_front_stamp,
            last_timestamp_imu);

        lidar_buffer.pop_front();
        time_buffer.pop_front();
        lidar_pushed = false;
        return false;
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    setLaserCurTime(lidar_end_time);
    return true;
}

int process_increments = 0;
void map_incremental()
{
    if (!use_online_map)
        return;

    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

void getCurrPose(const state_ikfom& curr_state) {
    Eigen::Vector3d eulerAngle = curr_state.rot.matrix().eulerAngles(2,1,0); 
    
    transformTobeMapped[0] = eulerAngle(2);             
    transformTobeMapped[1] = eulerAngle(1);          
    transformTobeMapped[2] = eulerAngle(0);        
    transformTobeMapped[3] = curr_state.pos(0);
    transformTobeMapped[4] = curr_state.pos(1);
    transformTobeMapped[5] = curr_state.pos(2);
}

void getCurrOffset(const state_ikfom& curr_state) {
    translationLidarToIMU = curr_state.offset_T_L_I;
    rotationLidarToIMU = curr_state.offset_R_L_I.toRotationMatrix();
}

template<typename T>
void fillPoseMsg(T& pose_msg, const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
{
    Eigen::Quaterniond q(R);
    q.normalize();
    pose_msg.position.x = t.x();
    pose_msg.position.y = t.y();
    pose_msg.position.z = t.z();
    pose_msg.orientation.x = q.x();
    pose_msg.orientation.y = q.y();
    pose_msg.orientation.z = q.z();
    pose_msg.orientation.w = q.w();
}

void fillTransformMsg(TransformStampedMsg& tf_msg, const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
{
    Eigen::Quaterniond q(R);
    q.normalize();
    tf_msg.transform.translation.x = t.x();
    tf_msg.transform.translation.y = t.y();
    tf_msg.transform.translation.z = t.z();
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();
}

void fillOdometryMsg(OdomMsg& odom_msg,
                     const std::string& parent_frame,
                     const std::string& child_frame,
                     const TimeType& stamp,
                     const Eigen::Matrix3d& R,
                     const Eigen::Vector3d& t)
{
    odom_msg.header.stamp = stamp;
    odom_msg.header.frame_id = parent_frame;
    odom_msg.child_frame_id = child_frame;
    fillPoseMsg(odom_msg.pose.pose, R, t);
}

template<typename CovT>
void fillOdometryCovariance(OdomMsg& odom_msg, const CovT& P)
{
    for (int i = 0; i < 6; i ++)
    {
        const int k = i < 3 ? i + 3 : i - 3;
        odom_msg.pose.covariance[i*6 + 0] = P(k, 3);
        odom_msg.pose.covariance[i*6 + 1] = P(k, 4);
        odom_msg.pose.covariance[i*6 + 2] = P(k, 5);
        odom_msg.pose.covariance[i*6 + 3] = P(k, 0);
        odom_msg.pose.covariance[i*6 + 4] = P(k, 1);
        odom_msg.pose.covariance[i*6 + 5] = P(k, 2);
    }
}

void publishGnssPath(const PosData &pos)
{
    if (!gnssPathVis) return;
    PoseStampedMsg gnss_pose;
    gnss_pose.header.stamp = get_ros_time(pos.t);
    gnss_pose.header.frame_id = map_frame;
    gnss_pose.pose.position.x = pos.p.x();
    gnss_pose.pose.position.y = pos.p.y();
    gnss_pose.pose.position.z = pos.p.z();
    gnss_pose.pose.orientation.w = 1.0;

    gnssPath.header.stamp = gnss_pose.header.stamp;
    gnssPath.header.frame_id = map_frame;
    gnssPath.poses.push_back(gnss_pose);
    ros_publish(pubGnssPath, gnssPath);
}

bool initGnssMap(double lidar_stamp_sec)
{
    if (!p_gnss)
    {
        return false;
    }

    PosData gnss_pos;
    YawData heading;
    if (!p_gnss->pickInitPair(gnss_pos, heading))
    {
        return false;
    }

    const Eigen::Matrix3d R_odom = state_point.rot.toRotationMatrix();
    const double current_yaw = std::atan2(R_odom(1, 0), R_odom(0, 0));
    const double dyaw = normalizeYaw(heading.yaw - current_yaw);
    const Eigen::Matrix3d R_init_now = Eigen::AngleAxisd(dyaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    const Eigen::Vector3d p_odom_ant = state_point.pos + R_odom * p_gnss->lever();
    Eigen::Vector3d t_map_odom_now = gnss_pos.p - R_init_now * p_odom_ant;
    if (!useGnssElevation)
    {
        t_map_odom_now.z() = t_map_odom.z();
    }

    setMapOdom(R_init_now, t_map_odom_now);

    publishMapToOdomTf(get_ros_time(lidar_stamp_sec));
    gnss_aligned.store(true);
    return true;
}

void publishMapToOdomTf(const TimeType& stamp)
{
    TransformStampedMsg map_to_odom_msg;
    map_to_odom_msg.header.stamp = stamp;
    map_to_odom_msg.header.frame_id = map_frame;
    map_to_odom_msg.child_frame_id = odom_frame;
    map_to_odom_msg.transform.translation.x = t_map_odom.x();
    map_to_odom_msg.transform.translation.y = t_map_odom.y();
    map_to_odom_msg.transform.translation.z = t_map_odom.z();

    Eigen::Quaterniond q_map_odom(R_map_odom);
    q_map_odom.normalize();
    map_to_odom_msg.transform.rotation.x = q_map_odom.x();
    map_to_odom_msg.transform.rotation.y = q_map_odom.y();
    map_to_odom_msg.transform.rotation.z = q_map_odom.z();
    map_to_odom_msg.transform.rotation.w = q_map_odom.w();

#ifdef USE_ROS1
    static tf::TransformBroadcaster br;
#elif defined(USE_ROS2)
    static tf2_ros::TransformBroadcaster br(get_ros_node());
#endif
    br.sendTransform(map_to_odom_msg);
}

bool refinePriorPointToPlane(Eigen::Matrix3d &R_map_odom_,
                             Eigen::Vector3d &t_map_odom_)
{
    if (feats_down_body == nullptr || feats_down_body->size() < 20 ||
        prior_ikdtree.Root_Node == nullptr)
        return false;

    for (int iter = 0; iter < 5; ++iter)
    {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        int valid_num = 0;

        for (const PointType &point_body : feats_down_body->points)
        {
            const Eigen::Vector3d p_lidar(point_body.x, point_body.y, point_body.z);
            const Eigen::Vector3d p_base = state_point.offset_R_L_I * p_lidar +
                                           state_point.offset_T_L_I;
            const Eigen::Vector3d p_odom = state_point.rot * p_base + state_point.pos;
            const Eigen::Vector3d Rp = R_map_odom_ * p_odom;
            const Eigen::Vector3d p_map = Rp + t_map_odom_;

            PointType point_map;
            point_map.x = p_map.x();
            point_map.y = p_map.y();
            point_map.z = p_map.z();
            point_map.intensity = point_body.intensity;

            MatchObservation obs;
            if (!build_point_observation(&prior_ikdtree, nullptr, point_body, point_map,
                                         1.0, true, nullptr, obs))
                continue;

            const Eigen::Vector3d normal(obs.normal.x, obs.normal.y, obs.normal.z);
            const double residual = obs.normal.intensity;
            Eigen::Matrix<double, 1, 6> J;
            J.block<1, 3>(0, 0) = normal.transpose();
            J.block<1, 3>(0, 3) = Rp.cross(normal).transpose();

            H += J.transpose() * J;
            b += J.transpose() * residual;
            ++valid_num;
        }

        if (valid_num < 20)
            return false;

        const Eigen::Matrix<double, 6, 1> dx = -H.ldlt().solve(b);
        if (!dx.allFinite())
            return false;

        t_map_odom_ += dx.head<3>();
        R_map_odom_ = Exp(dx(3), dx(4), dx(5)) * R_map_odom_;

        if (dx.head<3>().norm() < 1e-3 && dx.tail<3>().norm() < 1e-4)
            break;
    }

    return true;
}

bool priorInitAlign()
{
    if (!use_prior_map || !prior_init_en || prior_init_done ||
        prior_ikdtree.Root_Node == nullptr)
        return false;

    Eigen::Matrix3d R_map_lidar;
    Eigen::Vector3d t_map_lidar;
    double fitness_score = 0.0;
    if (!coarsePriorIcpAlign(R_map_lidar, t_map_lidar, fitness_score))
        return false;

    const Eigen::Matrix3d R_base_lidar = state_point.offset_R_L_I.toRotationMatrix();
    const Eigen::Vector3d t_base_lidar = state_point.offset_T_L_I;
    const Eigen::Matrix3d R_map_base = R_map_lidar * R_base_lidar.transpose();
    const Eigen::Vector3d t_map_base = t_map_lidar - R_map_base * t_base_lidar;

    const Eigen::Matrix3d R_odom_base = state_point.rot.toRotationMatrix();
    const Eigen::Vector3d t_odom_base = state_point.pos;
    Eigen::Matrix3d R_map_odom_refined = R_map_base * R_odom_base.transpose();
    Eigen::Vector3d t_map_odom_refined = t_map_base - R_map_odom_refined * t_odom_base;

    if (!refinePriorPointToPlane(R_map_odom_refined, t_map_odom_refined))
        return false;

    setMapOdom(R_map_odom_refined, t_map_odom_refined);
    publishMapToOdomTf(get_ros_time(lidar_end_time));
    prior_init_done = true;

    ROS_PRINT_INFO("prior init done: ICP fitness=%.6f", fitness_score);
    return true;
}

void publish_frame_world(const Pcl2Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToMap(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }
        Pcl2Msg laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = map_frame;
        ros_publish(pubLaserCloudFull, laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save accumulated feature scans ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that PCD saving will influence real-time performance **/
    if (feat_accum_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToMap(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && res_save_interval > 0  && scan_wait_num >= res_save_interval)
        {
            const std::string stamp_str = format_unix_time(lidar_end_time);
            const std::string all_points_dir = string(string(ROOT_DIR) + "PCD/") + stamp_str + string(".pcd");
            pcl::PCDWriter pcd_writer;
            cout << "current accumulated feature cloud saved to /PCD/" << stamp_str << ".pcd" << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const Pcl2Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    Pcl2Msg laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = base_frame;
    ros_publish(pubLaserCloudFull_body, laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const Pcl2Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToMap(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    Pcl2Msg laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
    laserCloudFullRes3.header.frame_id = map_frame;
    ros_publish(pubLaserCloudEffect, laserCloudFullRes3);
}

void publish_map(const Pcl2Publisher & pubLaserCloudMap)
{
    Pcl2Msg laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
    // featsFromMap is maintained in the local odom frame by the frontend map.
    // Let TF apply map->odom for visualization instead of relabeling it as map.
    laserCloudMap.header.frame_id = odom_frame;
    ros_publish(pubLaserCloudMap, laserCloudMap);
}

void publish_prior_map(const Pcl2Publisher & pubLaserCloudPriorMap)
{
    if (featsFromPriorMap->empty())
        return;

    Pcl2Msg laserCloudPriorMap;
    pcl::toROSMsg(*featsFromPriorMap, laserCloudPriorMap);
    laserCloudPriorMap.header.stamp = get_ros_time(lidar_end_time);
    laserCloudPriorMap.header.frame_id = map_frame;
    ros_publish(pubLaserCloudPriorMap, laserCloudPriorMap);
}

void publish_odometryhighfreq(PoseBuffer& pbuffer,
                              const OdomPublisher& pubOdomHighFreqLocal)
{
    while (ros_ok() && !flg_exit){
        Pose pose;
        if (!pbuffer.TryPop(pose))
        {
            usleep(1000);
            continue;
        }
        OdomMsg msg_local;
        const auto stamp = get_ros_time(pose._timestamp);

        const Eigen::Matrix3d R_odom_base = Eigen::Quaterniond(pose._qw, pose._qx, pose._qy, pose._qz).toRotationMatrix();
        const Eigen::Vector3d t_odom_base(pose._x, pose._y, pose._z);
        fillOdometryMsg(msg_local, odom_frame, high_freq_base_frame, stamp, R_odom_base, t_odom_base);

        TransformStampedMsg tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = odom_frame;
        tf_msg.child_frame_id = high_freq_base_frame;
        fillTransformMsg(tf_msg, R_odom_base, t_odom_base);

        ros_publish(pubOdomHighFreqLocal, msg_local);

#ifdef USE_ROS1
        static tf::TransformBroadcaster br_hf;
#elif defined(USE_ROS2)
        static tf2_ros::TransformBroadcaster br_hf(get_ros_node());
#endif
        br_hf.sendTransform(tf_msg);

        if (imu_state_save_en && imu_pose_file.is_open())
        {
            imu_pose_file << pose._timestamp << " "
                << pose._x << " " << pose._y << " " << pose._z << " "
                << pose._qx << " " << pose._qy << " " << pose._qz << " " << pose._qw << " "
                << pose._vx << " " << pose._vy << " " << pose._vz << " "
                << pose._bgx << " " << pose._bgy << " " << pose._bgz << " "
                << pose._bax << " " << pose._bay << " " << pose._baz << " "
                << pose._gravx << " " << pose._gravy << " " << pose._gravz << " "
                << pose._exqx << " " << pose._exqy << " " << pose._exqz << " " << pose._exqw << " "
                << pose._extx << " " << pose._exty << " " << pose._extz << "\n";
            imu_pose_file.flush();
        }
    }
}

template<typename T>
void set_posestamp(T & out)
{
    Eigen::Matrix3d R_map_base;
    Eigen::Vector3d t_map_base;
    composeMapPose(state_point.rot.toRotationMatrix(), state_point.pos, R_map_base, t_map_base);
    const Eigen::Quaterniond q_map_base(R_map_base);

    out.pose.position.x = t_map_base(0);
    out.pose.position.y = t_map_base(1);
    out.pose.position.z = t_map_base(2);
    out.pose.orientation.x = q_map_base.x();
    out.pose.orientation.y = q_map_base.y();
    out.pose.orientation.z = q_map_base.z();
    out.pose.orientation.w = q_map_base.w();
    
}

void publish_odometry(const OdomPublisher & pubOdomAftMappedLocal)
{
    const auto stamp = get_ros_time(lidar_end_time);

    const Eigen::Matrix3d R_odom_base = state_point.rot.toRotationMatrix();
    const Eigen::Vector3d t_odom_base = state_point.pos;
    fillOdometryMsg(odomAftMapped, odom_frame, base_frame, stamp, R_odom_base, t_odom_base);

    const auto& P = kf.get_P();
    fillOdometryCovariance(odomAftMapped, P);

    TransformStampedMsg tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = odom_frame;
    tf_msg.child_frame_id  = base_frame;
    fillTransformMsg(tf_msg, R_odom_base, t_odom_base);

    ros_publish(pubOdomAftMappedLocal, odomAftMapped);
    publishMapToOdomTf(stamp);
#ifdef USE_ROS1
    static tf::TransformBroadcaster br;
#elif defined(USE_ROS2)
    static tf2_ros::TransformBroadcaster br(get_ros_node());
#endif
    br.sendTransform(tf_msg);
}

void publish_path(const PathPublisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time);

    msg_body_pose.header.frame_id = map_frame;

    /*** if path is too large, rviz will crash ***/
    static int jjj = 0;
    jjj++;

    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        path.header.stamp = msg_body_pose.header.stamp;
        ros_publish(pubPath, path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    if (feats_down_size < 1)
    {
        ekfom_data.valid = false;
        return;
    }

    const bool use_prior = use_prior_map;
    const double prior_scale =
        std::sqrt(LASER_POINT_COV / std::max(prior_lidar_cov, 1e-9));
    if (use_prior)
    {
        Prior_Nearest_Points.resize(feats_down_size);
        prior_point_selected.resize(feats_down_size, 0);
    }

    std::vector<MatchObservation> online_candidates(feats_down_size);
    std::vector<MatchObservation> prior_candidates(feats_down_size);
    std::vector<uint8_t> online_valid(feats_down_size, 0);
    std::vector<uint8_t> prior_valid(feats_down_size, 0);

#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        if (ekfom_data.converge)
        {
            if (use_online_map &&
                build_point_observation(&ikdtree, nullptr, point_body, point_world, 1.0, true, &Nearest_Points[i], online_candidates[i]))
            {
                online_valid[i] = 1;
                point_selected_surf[i] = true;
                normvec->points[i] = online_candidates[i].normal;
                res_last[i] = online_candidates[i].residual;
            }
            else if (use_online_map)
            {
                point_selected_surf[i] = false;
            }
        }
        else if (use_online_map && point_selected_surf[i])
        {
            if (build_point_observation(nullptr, &Nearest_Points[i],
                                        point_body, point_world, 1.0, false, nullptr, online_candidates[i]))
            {
                online_valid[i] = 1;
                normvec->points[i] = online_candidates[i].normal;
                res_last[i] = online_candidates[i].residual;
            }
            else
            {
                point_selected_surf[i] = false;
            }
        }

        if (use_prior)
        {
            if (ekfom_data.converge)
            {
                if (build_point_observation(&prior_ikdtree, nullptr, point_body, point_world,
                                            prior_scale, true, &Prior_Nearest_Points[i], prior_candidates[i]))
                {
                    prior_valid[i] = 1;
                    prior_point_selected[i] = 1;
                }
                else
                {
                    prior_point_selected[i] = 0;
                }
            }
            else if (i < static_cast<int>(prior_point_selected.size()) && prior_point_selected[i])
            {
                if (build_point_observation(nullptr, &Prior_Nearest_Points[i],
                                            point_body, point_world,
                                            prior_scale, false, nullptr, prior_candidates[i]))
                {
                    prior_valid[i] = 1;
                }
                else
                {
                    prior_point_selected[i] = 0;
                }
            }
        }
    }

    std::vector<MatchObservation> matches;
    matches.reserve(feats_down_size * 2);
    double online_residual_sum = 0.0;
    int online_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (online_valid[i])
        {
            matches.push_back(online_candidates[i]);
            online_residual_sum += online_candidates[i].residual;
            online_feat_num++;
        }
        if (prior_valid[i])
        {
            matches.push_back(prior_candidates[i]);
        }
    }

    effct_feat_num = matches.size();
    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        return;
    }

    laserCloudOri->points.resize(effct_feat_num);
    corr_normvect->points.resize(effct_feat_num);
    laserCloudOri->width = effct_feat_num;
    laserCloudOri->height = 1;
    corr_normvect->width = effct_feat_num;
    corr_normvect->height = 1;

    for (int i = 0; i < effct_feat_num; i++)
    {
        laserCloudOri->points[i] = matches[i].body;
        corr_normvect->points[i] = matches[i].normal;
        total_residual += matches[i].residual;
    }

    if (online_feat_num > 0)
        res_mean_last = online_residual_sum / online_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        const double row_scale = matches[i].row_scale;
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << row_scale * norm_p.x, row_scale * norm_p.y, row_scale * norm_p.z,
                                                row_scale * VEC_FROM_ARRAY(A), row_scale * VEC_FROM_ARRAY(B), row_scale * VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << row_scale * norm_p.x, row_scale * norm_p.y, row_scale * norm_p.z,
                                                row_scale * VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -row_scale * norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char** argv)
{
    #ifdef USE_ROS1
    ros::init(argc, argv, "fast_lio_sam");
    init_ros_node();
    
    #elif defined(USE_ROS2)
    rclcpp::init(argc, argv);
    init_ros_node(rclcpp::Node::make_shared("fast_lio_sam"));
    #endif

    // Load parameters (unified ROS1/ROS2)
    rosparam_get("sam_enable", sam_enable, false);
    rosparam_get("publish/path_en", path_en, true);
    rosparam_get("publish/scan_publish_en", scan_pub_en, true);
    rosparam_get("publish/dense_publish_en", dense_pub_en, true);
    rosparam_get("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    rosparam_get("publish/feature_pub_en", feature_pub_en, false);
    rosparam_get("publish/effect_pub_en", effect_pub_en, false);
    rosparam_get("max_iteration", NUM_MAX_ITERATIONS, 4);
    rosparam_get("map_file_path", map_file_path, std::string(""));
    rosparam_get("common/lid_topic", lid_topic, std::string("/livox/lidar"));
    rosparam_get("common/imu_topic", imu_topic, std::string("/livox/imu"));
    rosparam_get("common/map_frame", map_frame, std::string("map"));
    rosparam_get("common/odom_frame", odom_frame, std::string("odom"));
    rosparam_get("common/base_frame", base_frame, std::string("base_link"));
    rosparam_get("common/high_freq_base_frame", high_freq_base_frame, std::string("base_link_hf"));
    rosparam_get("reloc/reloc_topic", reloc_topic, std::string("/reloc/manual"));
    rosparam_get("common/time_sync_en", time_sync_en, false);
    rosparam_get("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    rosparam_get("common/flip_en", flip_en, false);
    rosparam_get("common/grav_align", grav_align, false);
    rosparam_get("common/mode", mapping_mode, 1);
    rosparam_get("mapping/imu_dt", imu_dt, 0.005);
    rosparam_get("mapping/lidar_dt", lidar_dt, 0.1);
    rosparam_get("mapping/timeRepair", timeRepair, true);
    set_mapping_mode();
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
    rosparam_get("preprocess/blind", p_pre->blind, 0.01);
    rosparam_get("preprocess/lidar_type", lidar_type, (int)AVIA);
    rosparam_get("preprocess/scan_line", p_pre->N_SCANS, 16);
    rosparam_get("preprocess/timestamp_unit", p_pre->time_unit, (int)US);
    rosparam_get("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    rosparam_get("point_filter_num", p_pre->point_filter_num, 2);
    rosparam_get("feature_extract_enable", p_pre->feature_enabled, false);
    rosparam_get("result_save/imu_state_save_en", imu_state_save_en, false);
    rosparam_get("result_save/scan_frame_save_en", scan_frame_save_en, false);
    rosparam_get("result_save/ikdtree_output_save_en", ikdtree_output_save_en, false);
    rosparam_get("mapping/extrinsic_est_en", extrinsic_est_en, true);
    rosparam_get("result_save/feat_accum_save_en", feat_accum_save_en, false);
    rosparam_get("result_save/interval", res_save_interval, -1);
    rosparam_get("mapping/extrinsic_T", extrinT, std::vector<double>());
    rosparam_get("mapping/extrinsic_R", extrinR, std::vector<double>());
    rosparam_get("zupt/use_zupt",                use_zupt,                false);
    rosparam_get("zupt/zupt_acc_var_threshold",  zupt_acc_var_threshold,  0.001);
    rosparam_get("zupt/zupt_gyro_var_threshold", zupt_gyro_var_threshold, 0.0001);
    rosparam_get("zupt/zupt_r_min",              zupt_r_min,              1e-5);
    rosparam_get("zupt/zupt_r_max",              zupt_r_max,              1.0);
    rosparam_get("zupt/zupt_confidence_min",     zupt_confidence_min,     0.05);
    rosparam_get("zupt/cov_inflate_pos",         zupt_inflate_pos,        1e-7);
    rosparam_get("zupt/cov_inflate_rot",         zupt_inflate_rot,        1e-8);
    rosparam_get("zupt/cov_inflate_start",       zupt_inflate_start,      200);
    rosparam_get("zupt/lidar_cov_static_scale",  lidar_cov_static_scale,  5.0);
    rosparam_get("zupt/lidar_residual_ref",      lidar_residual_ref,      0.05);

    path.header.stamp = get_ros_now();
    path.header.frame_id = map_frame;

    if (sam_enable) {
        read_liosam_params();
    }
    read_gnss_params();
    if (p_gnss)
    {
        p_gnss->setOffset(heading_offset);
    }
    p_pre->lidar_type = lidar_type;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;

    /*** variables definition ***/
    int effect_feat_num = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    // extrinT/extrinR must stay in the sensor's native Airy/DIFOP IMU convention.
    // When flip_en is enabled, IMU measurements, local point clouds, and these
    // extrinsics are all standardized to the Mid360 convention before estimation.
    if (gnssEnableFlag || gnssPathVis)
    {
        grav_align = true;
    }
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    Lidar_T_wrt_IMU = standardize(Lidar_T_wrt_IMU);
    Lidar_R_wrt_IMU = standardize(Lidar_R_wrt_IMU);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_grav_align(grav_align);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->set_use_zupt(use_zupt);
    p_imu->set_zupt_thresholds(zupt_acc_var_threshold, zupt_gyro_var_threshold);
    p_imu->set_zupt_adaptive_params(zupt_r_min, zupt_r_max, zupt_confidence_min,
                                     zupt_inflate_pos, zupt_inflate_rot, zupt_inflate_start);
    p_imu->lidar_type = lidar_type;

    if (p_gnss)
    {
        const Eigen::Vector3d lever_imu =
            Lidar_T_wrt_IMU + Lidar_R_wrt_IMU * gnss_extrinsic_T;
        p_gnss->setLever(lever_imu);
    }

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    if (use_prior_map)
    {
        if (!loadPriorTreeMap())
        {
            use_prior_map = false;
            prior_init_en = false;
        }
    }

    /*** debug record ***/
    prepareResultDirs();

    /*** ROS subscribe initialization ***/
    if (p_pre->lidar_type == AVIA) {
        static auto sub_pcl = create_subscriber<LivoxMsg>(lid_topic, 200000, livox_pcl_cbk);
    } else {
        static auto sub_pcl = create_subscriber<PointCloud2Msg>(lid_topic, 200000, standard_pcl_cbk);
    }
    
    auto sub_reloc = create_subscriber<PoseStampedMsg>(reloc_topic, 10, reloc_cbk);
    auto sub_imu = create_subscriber<ImuMsg>(imu_topic, 200000, imu_cbk);
    static auto sub_gnss = create_subscriber<GnssFixMsg>(gnss_topic, 10, gnss_cbk);
    static auto sub_gnss_heading = create_subscriber<OdometryMsg>(gnss_heading_topic, 10, gnss_heading_cbk);
    auto pubLaserCloudFull = create_publisher<PointCloud2Msg>("/cloud_registered", 100000);
    auto pubLaserCloudFull_body = create_publisher<PointCloud2Msg>("/cloud_registered_body", 100000);
    auto pubLaserCloudEffect = create_publisher<PointCloud2Msg>("/cloud_effected", 100000);
    auto pubLaserCloudMap = create_publisher<PointCloud2Msg>("/Laser_map", 100000);
    auto pubLaserCloudPriorMap = create_publisher<PointCloud2Msg>("/Laser_map_prior", 100000);
    #ifdef USE_ROS1
    int best_qos = 0;  // ROS1 ignores this parameter
    int reliable_qos = 0;
    #elif defined(USE_ROS2)
    auto best_qos = rclcpp::QoS(10).best_effort(); // avoid latency caused by QoS reliability in ROS2
    auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    #endif
    auto pubOdomAftMapped = create_publisher_qos<OdometryMsg>("/Odometry", reliable_qos);
    auto pubPath = create_publisher_qos<PathMsg>("/path", reliable_qos);
    pubGnssPath = create_publisher_qos<PathMsg>("/gnss_path", reliable_qos);
    auto pubOdomHighFreq = create_publisher_qos<OdometryMsg>("/OdometryHighFreq", best_qos);
    p_pre->pub_corn = create_publisher<PointCloud2Msg>("/corn_feature", 100000);
    p_pre->pub_surf = create_publisher<PointCloud2Msg>("/surf_feature", 100000);

    if (sam_enable) {
        MapOptimizationInit();
        printf("...... Pose graph optimization backend start......\n");
    }

    std::thread odomhighthread([&](){
        publish_odometryhighfreq(p_imu->pbuffer, pubOdomHighFreq);
    });

    std::thread loopthread;
    std::thread globalthread;
    std::thread gnssthread;
    std::thread structurethread;
    if (sam_enable)
    {
        loopthread = std::thread(&loopClosureThread);
        globalthread = std::thread(&visualizeGlobalMapThread);
        structurethread = std::thread(&structureMatchingThread);
        if (gnssEnableFlag)
        {
            gnssthread = std::thread(&gnssMatchingThread);
        }
    }

//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    RateType rate(5000);
    while (ros_ok() && !flg_exit)
    {
        spin_once();

        // relocalization trigger
        if(relocalize_flag.load())
        {
            Pose reloc_pose;
            {
                std::lock_guard<std::mutex> lock(mtx_reloc);
                reloc_pose = reloc_state;
            }
            Eigen::Quaterniond reloc_rot(reloc_pose._qw, reloc_pose._qx, reloc_pose._qy, reloc_pose._qz);
            reloc_rot.normalize();
            const Eigen::Vector3d reloc_pos(reloc_pose._x, reloc_pose._y, reloc_pose._z);

            const Eigen::Matrix3d R_map_base = reloc_rot.toRotationMatrix();
            const Eigen::Matrix3d R_odom_base = state_point.rot.toRotationMatrix();
            const Eigen::Vector3d t_odom_base = state_point.pos;
            updateMapOdom(R_map_base, reloc_pos, R_odom_base, t_odom_base);
            publishMapToOdomTf(get_ros_time(reloc_pose._timestamp));

            ROS_PRINT_INFO("Reloc: pos=(%.2f %.2f %.2f), quat=(%.2f %.2f %.2f %.2f)",
                reloc_pos.x(), reloc_pos.y(), reloc_pos.z(),
                reloc_rot.x(), reloc_rot.y(), reloc_rot.z(), reloc_rot.w());
            relocalize_flag.store(false);
        }

        if(sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            match_time = 0;
            solve_time = 0;

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // LiDAR position in the world coordinate frame

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_PRINT_WARN("No point, skip this scan!");
                continue;
            }

            // Try to establish map as the local GNSS ENU frame when GNSS is
            // enabled for fusion or path visualization. If GNSS is not ready
            // yet, keep running normal LIO.
            if ((gnssEnableFlag || gnssPathVis) && !gnss_aligned.load())
            {
                initGnssMap(Measures.lidar_end_time);
            }

            if (scan_frame_save_en) save_scan_frame(scan_frames_dir);

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            if (use_online_map)
                lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            feats_down_size = feats_down_body->points.size();
            /*** initialize the online map kdtree ***/
            if (use_online_map && ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                if (!use_prior_map)
                    continue;
            }
            kdtree_size_st = use_online_map ? ikdtree.size() : 0;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_PRINT_WARN("No point, skip this scan!");
                continue;
            }

            const double adaptive_lidar_cov = adapt_lidar_weight();
            
            feats_down_world->resize(feats_down_size);

            if(feature_pub_en && use_online_map)
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            Nearest_Points.resize(feats_down_size);

            /*** iterated state estimation ***/
            double solve_H_time = 0;
            if (use_prior_map && prior_init_en && !prior_init_done)
            {
                if (!priorInitAlign())
                {
                    ROS_PRINT_WARN("prior init alignment failed, retry on next scan");
                    continue;
                }
            }
            else
            {
                kf.update_iterated_dyn_share_modified(adaptive_lidar_cov, solve_H_time);
                state_point = kf.get_x();
                euler_cur = SO3ToEuler(state_point.rot);
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                geoQuat.x = state_point.rot.coeffs()[0];
                geoQuat.y = state_point.rot.coeffs()[1];
                geoQuat.z = state_point.rot.coeffs()[2];
                geoQuat.w = state_point.rot.coeffs()[3];
            }

            bool keyframe = false;
            if (sam_enable)
            {
                getCurrPose(state_point);
                getCurrOffset(state_point);
                keyframe = isKeyFrame();
            }

            // GNSS callbacks are serviced on the same executor as the main loop.
            // Spin once more here so GNSS messages that arrived during the heavy
            // LiDAR/IMU processing are visible before factor selection.
            spin_once();

            if (sam_enable) {
                if (keyframe)
                {
                    saveKeyFramesAndFactor(feats_undistort);
                }
                correctPoses();
                publishSamMsg();
            }

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            map_incremental();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || feat_accum_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            if (effect_pub_en) publish_effect_world(pubLaserCloudEffect);
            if (feature_pub_en && use_online_map) publish_map(pubLaserCloudMap);
            if (feature_pub_en && use_prior_map) publish_prior_map(pubLaserCloudPriorMap);
            /*** Debug variables ***/
        }

        rate.sleep();
    }            

    if (!flg_exit && pcl_wait_save->size() > 0 && feat_accum_save_en)
    {
        const std::string stamp_str = format_unix_time(lidar_end_time);
        const std::string file_name = stamp_str + string(".pcd");
        const std::string all_points_dir = string(string(ROOT_DIR) + "PCD/") + file_name;
        pcl::PCDWriter pcd_writer;
        cout << "current accumulated feature cloud saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (keyframe_export_en && keyframe_pose_file.is_open())
    {
        for (int i = 0; i < cloudKeyPoses6D->points.size(); ++i)
        {
            PointTypePose thisPose6D = cloudKeyPoses6D->points[i];
            keyframe_pose_file << thisPose6D.time << " "
                       << thisPose6D.x << " " << thisPose6D.y << " " << thisPose6D.z << " "
                       << thisPose6D.roll << " " << thisPose6D.pitch << " " << thisPose6D.yaw << "\n";
        }
        keyframe_pose_file.close();
    }

    if (keyframe_global_pcd_en)
    {
        string global_keyframe_path = keyframe_frames_dir + "global.pcd";
        pcl::PCDWriter pcd_writer;
        cout << "current global keyframe map saved to " << global_keyframe_path << endl;

        for (int i = 0; i < featCloudKeyFrames.size(); ++i)
        {
            *keyframe_global_cloud += *transformPointCloud(featCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
        }

        pcd_writer.writeBinary(global_keyframe_path, *keyframe_global_cloud);
    }

    flg_exit = true;
    if (odomhighthread.joinable()) {
        odomhighthread.join();
    }
    if (loopthread.joinable()) {
        loopthread.join();
    }
    if (globalthread.joinable()) {
        globalthread.join();
    }
    if (gnssthread.joinable()) {
        gnssthread.join();
    }
    if (structurethread.joinable()) {
        structurethread.join();
    }

    if (ikdtree_output_save_en && use_online_map && ikdtree.Root_Node != nullptr)
    {
        save_ikdtree_cloud(ikdtree_output_dir + "prior_cloud.pcd");
        const string ikdtree_snapshot_path = ikdtree_output_dir + "prior_tree.bin";
        if (ikdtree.SaveIkdtree(ikdtree_snapshot_path))
            ROS_PRINT_INFO("saved final ikdtree snapshot to %s", ikdtree_snapshot_path.c_str());
        else
            ROS_PRINT_WARN("failed to save final ikdtree snapshot to %s", ikdtree_snapshot_path.c_str());
    }

    return 0;
}
