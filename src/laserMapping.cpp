#include <omp.h>
#include <mutex>
#include <condition_variable>
#include <math.h>
#include <thread>
#include <fstream>
#include <iomanip>
#include <csignal>
#include <unistd.h>
#include <algorithm>
#include <deque>
#include <unordered_map>
#include <Python.h>
#include <so3_math.h>
#include "simple_EKF/simple_ekf.hpp"
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include "synchronizer.hpp"
#include "GNSS_Processing.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include "preprocess.h"
#include "map/map.h"
#include <atomic>
#include "ros/ros_utils.h"
#include "ros/ros_manager.h"
#include "ros/rosbag_input.h"
#include "ros/ros_parameters.h"
#include "debug/result_writer.h"
#include "global_graph.h"
#include "lidarBA.h"
#include "balm/tools.hpp"
#include "common_utils.h"
#ifdef USE_ROS2
#include <rclcpp/executors/multi_threaded_executor.hpp>
#endif
#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)

double kdtree_incremental_time = 0.0, kdtree_delete_time = 0.0;
double match_time = 0, solve_time = 0;
int    kdtree_size_st = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   feat_accum_save_en = false, time_sync_en = false, path_en = true;
bool   imu_state_save_en = false, scan_frame_save_en = false, ikdtree_output_save_en = false;

bool effect_pub_en = false;
std::mutex mapDisplayMutex;
std::condition_variable mapDisplayCondition;
std::thread mapDisplayThread;
PointVector mapDisplayPoints;
PointVector baMapDisplayPoints;
double displayStamp = 0.0;
double latestMapTime = 0.0;
bool mapTimeUpdated = false;
bool mapDisplayPending = false;
bool mapDisplayCloudRequested = false;
bool mapDisplayStopping = false;

float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;
string reloc_topic;

double last_timestamp_lidar = -1.0;
TimeSyncronizer imu_timestamp_sync;
double imu_dt = 0.005;
bool   timeRepair = true;
std::deque<double> imuRollbackTimes;
bool imuRollbackScan = false;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
bool geo_map_enable = false;
double geo_support = 0.15;
double geo_plane_dist = 0.05;
bool point2point = false;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0;
int    feats_down_size = 0, NUM_MAX_ITERATIONS = 0, res_save_interval = -1;
bool   lidar_pushed, flg_first_scan = true, flg_EKF_inited;
std::atomic<bool> flg_exit(false);
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool sam_enable = false;
float transformTobeMapped[6]{};
std::unique_ptr<LidarBA> lidarBA;
bool useBAMap()
{
    return lidarBAConfig.enable && lidarBAConfig.feedback_to_ekf && use_online_map;
}
std::unique_ptr<GlobalGraph> globalGraph;
PointTypePose lastOdomPose{};
int nextLidarBAFrameId = 0;
int keyFrameCount = 0;
bool grav_align = false;
int lidar_type;
std::string prior_tree_path;
double prior_lidar_cov = 0.001;
bool prior_init_en = false;
bool prior_init_done = false;
std::atomic<int64_t> last_input_time{-1};

std::shared_ptr<GnssProcess> p_gnss = std::make_shared<GnssProcess>();

void publishGnssPath(const GnssPositionData &pos);
void publishGnssLinkTf();
bool initGnssMap(double lidar_stamp_sec);

vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<ImuMsgConstPtr> imu_buffer;

mutex mtx_reloc;
condition_variable sig_reloc;
State reloc_state;
std::atomic<bool> relocalize_flag(false);

PointCloudXYZI::Ptr featsFromPriorMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr geo_map_dropped(new PointCloudXYZI());
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

mapping::Map mapManager;

V3D euler_cur;
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
SimpleEKF kf;
SimpleEKF::State state_point;
V3D pos_lid;

PathMsg path;
PathMsg odomPath;
PathMsg lidarBAPath;
MarkerArrayMsg baFrameMarkers;
PathMsg gnssPath;
OdomMsg odomAftMapped;
QuaternionMsg geoQuat;
PoseStampedMsg msg_body_pose;


shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int)
{
    flg_exit = true;
}

static inline V3D transformBodyPointToGlobal(const V3D &p_body,
                                             const Eigen::Matrix3d &R_world_body,
                                             const Eigen::Vector3d &t_world_body,
                                             const SimpleEKF::State &state);

void pointBodyToWorldState(PointType const * const pi, PointType * const po, SimpleEKF::State &s)
{
    const V3D p_body(pi->x, pi->y, pi->z);
    const V3D p_global = transformBodyPointToGlobal(
        p_body, s.rot, s.pos, s);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

static inline V3D transformBodyPointToGlobal(const V3D &p_body,
                                             const Eigen::Matrix3d &R_world_body,
                                             const Eigen::Vector3d &t_world_body,
                                             const SimpleEKF::State &state)
{
    return R_world_body * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + t_world_body;
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    const V3D p_body(pi->x, pi->y, pi->z);
    const V3D p_global = transformBodyPointToGlobal(
        p_body, state_point.rot, state_point.pos, state_point);
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
    composeMapPose(state_point.rot, state_point.pos, R_map_body, t_map_body);
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
        p_body, state_point.rot, state_point.pos, state_point);

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
    composeMapPose(state_point.rot, state_point.pos, R_map_body, t_map_body);
    const V3D p_global = transformBodyPointToGlobal(p_body, R_map_body, t_map_body, state_point);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    const V3D p_body(pi->x, pi->y, pi->z);
    const V3D p_global = transformBodyPointToGlobal(
        p_body, state_point.rot, state_point.pos, state_point);
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
    composeMapPose(state_point.rot, state_point.pos, R_map_body, t_map_body);
    const V3D p_global = transformBodyPointToGlobal(p_body, R_map_body, t_map_body, state_point);
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void standard_pcl_cbk(const Pcl2MsgConstPtr &msg)
{
    last_input_time.store(nowTimeUs());

    const double raw_timestamp = get_ros_time_sec(msg->header.stamp);
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);

    std::lock_guard<std::mutex> lock(mtx_buffer);
    const double corrected_timestamp = raw_timestamp + imu_timestamp_sync.offset();

    if (last_timestamp_lidar >= 0.0 && corrected_timestamp <= last_timestamp_lidar)
        return;

    last_timestamp_lidar = corrected_timestamp;

    lidar_buffer.push_back(ptr);
    time_buffer.push_back(corrected_timestamp);
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const LivoxCustomMsgConstPtr &msg) 
{
    last_input_time.store(nowTimeUs());

    const double raw_timestamp = get_ros_time_sec(msg->header.stamp);
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);

    std::lock_guard<std::mutex> lock(mtx_buffer);
    const double corrected_timestamp = raw_timestamp + imu_timestamp_sync.offset(); // vital!!!
    // const double corrected_timestamp = raw_timestamp;

    if (last_timestamp_lidar >= 0.0 && corrected_timestamp <= last_timestamp_lidar)
    {
        return;
    }

    last_timestamp_lidar = corrected_timestamp;
    
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

    lidar_buffer.push_back(ptr);
    time_buffer.push_back(corrected_timestamp);
}

void imu_cbk(const ImuMsgConstPtr &msg_in) 
{   
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
    std::lock_guard<std::mutex> lock(mtx_buffer);
    msg->header.stamp = get_ros_time(msg_in_stamp_sec - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = get_ros_time(timediff_lidar_wrt_imu + msg_in_stamp_sec);
    }
    const double raw_timestamp = get_ros_time_sec(msg->header.stamp);
    bool imuRolledBack = false;

    double corrected_timestamp = raw_timestamp;
    if (timeRepair)
    {
        if (!imu_timestamp_sync.repair(raw_timestamp, imu_dt, corrected_timestamp,
                                       imuRolledBack))
        {
            return;
        }
    }
    else
    {
        imu_timestamp_sync.accept(raw_timestamp);
    }

    last_timestamp_imu = imu_timestamp_sync.lastCorrectedStamp();

    msg->header.stamp = get_ros_time(corrected_timestamp);

    imu_buffer.push_back(msg);
    if (imuRolledBack)
        imuRollbackTimes.push_back(corrected_timestamp);
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
    reloc_state = State(x, y, z,
                    q.x(), q.y(), q.z(), q.w(), timestamp);
    relocalize_flag.store(true); 
}

void gnss_cbk(const GnssFixMsgConstPtr &msg_in)
{
    GnssPositionData pos;
    if (!p_gnss->processGnssFix(msg_in, pos)) return;

    if (gnssPathVis && gnss_aligned.load())
        publishGnssPath(pos);

    publishGnssLinkTf();
}

void gnss_heading_cbk(const OdometryMsgConstPtr &msg_in)
{
    if (!p_gnss->processGnssYaw(msg_in)) return;
    publishGnssLinkTf();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    std::lock_guard<std::mutex> lock(mtx_buffer);
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
    imuRollbackScan = false;
    while (!imuRollbackTimes.empty() && imuRollbackTimes.front() <= lidar_end_time)
    {
        // The first scan after an IMU timestamp repair uses a synthetic
        // continuous timestamp.  Let the EKF consume it to keep state
        // continuity, but never let this transition scan contaminate the
        // map or the BA window.
        imuRollbackScan = true;
        imuRollbackTimes.pop_front();
    }
    return true;
}

void map_incremental(PointCloudXYZI::Ptr acceptedCloud = nullptr)
{
    geo_map_dropped->clear();
    if (acceptedCloud) acceptedCloud->clear();
    if (!use_online_map)
        return;

    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    for (int i = 0; i < feats_down_size; ++i)
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

    if (geo_map_enable)
    {
        for (int i = 0; i < feats_down_size; ++i)
        {
            const PointType &point = feats_down_world->points[i];
            if (mapManager.coveredByPrior(point, use_prior_map, prior_init_done,
                                   filter_size_map_min))
                continue;

            const PointVector &points_near = mapManager.onlineNearestPoints[i];
            bool keep = !flg_EKF_inited || points_near.size() < NUM_MATCH_POINTS ||
                        std::sqrt(calc_dist(point, points_near[NUM_MATCH_POINTS - 1])) > geo_support;
            if (!keep)
            {
                VF(4) plane;
                keep = !esti_plane(plane, points_near, 0.1f) ||
                       std::fabs(plane(0) * point.x + plane(1) * point.y +
                                 plane(2) * point.z + plane(3)) > geo_plane_dist;
            }

            if (keep)
            {
                PointToAdd.push_back(point);
                if (acceptedCloud) acceptedCloud->push_back(feats_down_body->points[i]);
            }
            else
                geo_map_dropped->push_back(point);
        }
    }
    else
    {
        for (int i = 0; i < feats_down_size; ++i)
        {
            const PointType &point = feats_down_world->points[i];
            if (mapManager.coveredByPrior(point, use_prior_map, prior_init_done,
                                   filter_size_map_min))
                continue;

            if (mapManager.onlineNearestPoints[i].empty() || !flg_EKF_inited)
            {
                PointToAdd.push_back(point);
                if (acceptedCloud) acceptedCloud->push_back(feats_down_body->points[i]);
                continue;
            }

            const PointVector &points_near = mapManager.onlineNearestPoints[i];
            PointType mid_point;
            mid_point.x = floor(point.x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(point.y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(point.z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            if (fabs(points_near[0].x - mid_point.x) >
                    0.5 * filter_size_map_min &&
                fabs(points_near[0].y - mid_point.y) >
                    0.5 * filter_size_map_min &&
                fabs(points_near[0].z - mid_point.z) >
                    0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(point);
                if (acceptedCloud) acceptedCloud->push_back(feats_down_body->points[i]);
                continue;
            }

            const float dist = calc_dist(point, mid_point);
            bool need_add = true;
            for (int j = 0; j < NUM_MATCH_POINTS &&
                            j < static_cast<int>(points_near.size()); ++j)
            {
                if (calc_dist(points_near[j], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
            {
                PointToAdd.push_back(point);
                if (acceptedCloud) acceptedCloud->push_back(feats_down_body->points[i]);
            }
        }
    }

    const double st_time = omp_get_wtime();
    if (!mapManager.hasTree(useBAMap()))
    {
        PointToAdd.insert(PointToAdd.end(),
                          PointNoNeedDownsample.begin(),
                          PointNoNeedDownsample.end());
        add_point_size = PointToAdd.size();
        mapManager.initializeOnline(PointToAdd, useBAMap());
        kdtree_incremental_time = omp_get_wtime() - st_time;
        return;
    }

    mapManager.addOnlinePoints(PointToAdd, useBAMap(), true);

    if (!geo_map_enable && !PointNoNeedDownsample.empty())
        mapManager.addOnlinePoints(PointNoNeedDownsample, useBAMap(), false);

    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

void getCurrPose(const SimpleEKF::State& curr_state) {
    Eigen::Vector3d eulerAngle = curr_state.rot.eulerAngles(2,1,0);
    
    transformTobeMapped[0] = eulerAngle(2);             
    transformTobeMapped[1] = eulerAngle(1);          
    transformTobeMapped[2] = eulerAngle(0);        
    transformTobeMapped[3] = curr_state.pos(0);
    transformTobeMapped[4] = curr_state.pos(1);
    transformTobeMapped[5] = curr_state.pos(2);
}

bool applyBA(bool optimized, const Eigen::Isometry3d &pose)
{
    if (!lidarBAConfig.enable || !lidarBAConfig.feedback_to_ekf || !optimized)
        return false;

    const SimpleEKF::State before = kf.state();
    const Eigen::Matrix3d relativeRotation =
        before.rot.transpose() * pose.linear();
    kf.setPose(pose.linear(), pose.translation());
    p_imu->syncPose(relativeRotation, kf.gravity());
    state_point = kf.state();
    euler_cur = state_point.rot.eulerAngles(0, 1, 2);
    pos_lid = state_point.pos + state_point.rot * Lidar_T_wrt_IMU;
    const Eigen::Quaterniond q_state(state_point.rot);
    geoQuat.x = q_state.x();
    geoQuat.y = q_state.y();
    geoQuat.z = q_state.z();
    geoQuat.w = q_state.w();
    getCurrPose(state_point);
    return true;
}

PointTypePose currentOdomPose(double timestamp)
{
    PointTypePose pose;
    pose.x = transformTobeMapped[3];
    pose.y = transformTobeMapped[4];
    pose.z = transformTobeMapped[5];
    pose.roll = transformTobeMapped[0];
    pose.pitch = transformTobeMapped[1];
    pose.yaw = transformTobeMapped[2];
    pose.time = timestamp;
    return pose;
}

bool isKeyframe()
{
    if (keyFrameCount == 0)
        return true;

    const Eigen::Affine3f previous = pcl::getTransformation(
        lastOdomPose.x, lastOdomPose.y, lastOdomPose.z,
        lastOdomPose.roll, lastOdomPose.pitch, lastOdomPose.yaw);
    const Eigen::Affine3f current = pcl::getTransformation(
        transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    const Eigen::Affine3f relative = previous.inverse() * current;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(relative, x, y, z, roll, pitch, yaw);

    return !(std::abs(roll) < surroundingkeyframeAddingAngleThreshold &&
             std::abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
             std::abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
             std::sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold);
}

void publishLidarBA(const BAFramePtr &frame);

PointVector frameMapPoints(const BAFramePtr &frame, const Eigen::Isometry3d &pose)
{
    PointVector points;
    if (!frame || !frame->map_cloud) return points;
    points.reserve(frame->map_cloud->size());
    for (const auto &src : frame->map_cloud->points)
    {
        const Eigen::Vector3d p = pose * Eigen::Vector3d(src.x, src.y, src.z);
        PointType point;
        point.x = p.x(); point.y = p.y(); point.z = p.z();
        point.intensity = src.intensity;
        if (mapManager.coveredByPrior(point, use_prior_map, prior_init_done,
                               filter_size_map_min)) continue;
        if (mapManager.hasLocalMap() && !mapManager.containsInLocalMap(p))
            continue;
        points.push_back(point);
    }
    return points;
}

void rebuildBAMap()
{
    if (!useBAMap() || !lidarBA || !mapManager.hasLocalMap()) return;
    PointVector points;
    std::unordered_map<lidar_ba::VOXEL_LOC, size_t> voxelIndices;
    const double leaf = std::max(filter_size_map_min, 0.01);
    for (const BAFramePtr &frame : lidarBA->ActiveFrames())
    {
        PointVector framePoints = frameMapPoints(frame, frame->optimized_pose);
        for (const PointType &point : framePoints)
        {
            const Eigen::Vector3d world(point.x, point.y, point.z);
            if (!world.allFinite()) continue;
            const lidar_ba::VOXEL_LOC voxel(
                static_cast<int64_t>(std::floor(point.x / leaf)),
                static_cast<int64_t>(std::floor(point.y / leaf)),
                static_cast<int64_t>(std::floor(point.z / leaf)));
            const auto [it, inserted] = voxelIndices.emplace(voxel, points.size());
            if (inserted) points.push_back(point);
            else
            {
                const Eigen::Vector3d center((voxel.x + 0.5) * leaf,
                                             (voxel.y + 0.5) * leaf,
                                             (voxel.z + 0.5) * leaf);
                const PointType &old = points[it->second];
                if ((world - center).squaredNorm() <
                    (Eigen::Vector3d(old.x, old.y, old.z) - center).squaredNorm())
                    points[it->second] = point;
            }
        }
    }
    mapManager.rebuildWindow(std::move(points));
    kdtree_size_st = mapManager.onlineSize(true);
}

bool processBAFrame(
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud,
    PointCloudXYZI::Ptr mapCloud, double timestamp, bool keyFrame,
    Eigen::Isometry3d &pose,
    std::vector<BAFramePtr> &baFrames)
{
    if (!lidarBA)
    {
        lidarBA.reset(new LidarBA(lidarBAConfig, kf.gravity()));
    }

    const PointTypePose framePose = currentOdomPose(timestamp);

    BAFramePtr frame(new BAFrame());
    frame->id = nextLidarBAFrameId++;
    frame->timestamp = timestamp;
    frame->keyframe = keyFrame;
    frame->odom_pose = Eigen::Isometry3d::Identity();
    frame->odom_pose.linear() = poseRotation(framePose);
    frame->odom_pose.translation() = poseTranslation(framePose);
    frame->velocity = kf.state().vel;
    frame->bg = kf.state().bg;
    frame->ba = kf.state().ba;

    frame->map_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    frame->map_cloud->reserve(mapCloud ? mapCloud->size() : 0);
    if (mapCloud) for (const auto &point : mapCloud->points)
    {
        const Eigen::Vector3d body = Lidar_R_wrt_IMU.cast<double>() *
            Eigen::Vector3d(point.x, point.y, point.z) +
            Lidar_T_wrt_IMU.cast<double>();
        pcl::PointXYZI converted;
        converted.x = body.x(); converted.y = body.y(); converted.z = body.z();
        converted.intensity = point.intensity;
        frame->map_cloud->push_back(converted);
    }

    if (lidarBAConfig.imu_enable)
    {
        ImuMsgConstPtr right_imu;
        {
            std::lock_guard<std::mutex> lock(mtx_buffer);
            if (!imu_buffer.empty()) right_imu = imu_buffer.front();
        }
        frame->imu = p_imu->takeBAImuInterval(timestamp, right_imu);
    }

    frame->cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    frame->cloud->reserve(cloud ? cloud->size() : 0);
    if (cloud)
    {
        for (const auto &point : cloud->points)
        {
            const Eigen::Vector3d body = Lidar_R_wrt_IMU.cast<double>() *
                Eigen::Vector3d(point.x, point.y, point.z) +
                Lidar_T_wrt_IMU.cast<double>();
            pcl::PointXYZI converted;
            converted.x = body.x(); converted.y = body.y(); converted.z = body.z();
            converted.intensity = point.intensity;
            frame->cloud->push_back(converted);
        }
    }

    return lidarBA->Process(std::move(frame), pose, baFrames);
}

void publishLidarBA(const BAFramePtr &frame)
{
    Eigen::Affine3f baAffine = Eigen::Affine3f::Identity();
    baAffine.linear() = frame->optimized_pose.linear().cast<float>();
    baAffine.translation() = frame->optimized_pose.translation().cast<float>();
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(baAffine, x, y, z, roll, pitch, yaw);

    PointTypePose pose;
    pose.x = x; pose.y = y; pose.z = z;
    pose.roll = roll; pose.pitch = pitch; pose.yaw = yaw;
    pose.time = frame->timestamp;

    if (baFrameMarkers.markers.empty())
    {
        MarkerMsg marker;
        marker.ns = "ba_frames";
        marker.id = 0;
        marker.type = MarkerMsg::SPHERE_LIST;
        marker.action = MarkerMsg::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.14;
        marker.scale.y = 0.14;
        marker.scale.z = 0.14;
        marker.color.r = 0.0;
        marker.color.g = 0.9;
        marker.color.b = 1.0;
        marker.color.a = 1.0;
        baFrameMarkers.markers.push_back(marker);
    }

    PoseStampedMsg poseMessage;
    poseMessage.header.stamp = get_ros_time(pose.time);
    poseMessage.header.frame_id = odom_frame;
    poseMessage.pose.position.x = pose.x;
    poseMessage.pose.position.y = pose.y;
    poseMessage.pose.position.z = pose.z;
    poseMessage.pose.orientation = quaternion_from_rpy(
        pose.roll, pose.pitch, pose.yaw);
    lidarBAPath.poses.push_back(poseMessage);
    lidarBAPath.header.stamp = poseMessage.header.stamp;
    lidarBAPath.header.frame_id = odom_frame;

    MarkerMsg &marker = baFrameMarkers.markers.front();
    marker.header.stamp = poseMessage.header.stamp;
    marker.header.frame_id = odom_frame;
    PointMsg point;
    point.x = pose.x;
    point.y = pose.y;
    point.z = pose.z;
    marker.points.push_back(point);
    const size_t frameCount = lidarBAPath.poses.size();
    if (frameCount == 1 || frameCount % 10 == 0)
    {
        ros_manager.publish(
            ros_manager.pubLidarBAPath, lidarBAPath);
        ros_manager.publish(
            ros_manager.pubBAFrameMarkers, baFrameMarkers);
    }
}

void processBAFrames(const std::vector<BAFramePtr> &baFrames)
{
    for (const BAFramePtr &frame : baFrames)
    {
        if (!frame)
            continue;
        if (useBAMap())
        {
            PointVector points = frameMapPoints(frame, frame->optimized_pose);
            if (!points.empty())
            {
                mapManager.addOnlinePoints(points, false, true);
            }
        }
        publishLidarBA(frame);
        if (sam_enable && frame->keyframe && globalGraph &&
            !globalGraph->InsertKeyFrame(frame))
        {
            ROS_PRINT_ERROR("GlobalGraph rejected BA keyframe %d", frame->id);
        }
        if (!frame->keyframe)
            frame->cloud.reset();
        frame->map_cloud.reset();
    }
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

template<typename CovT>
void fillOdometryCovariance(OdomMsg& odom_msg, const CovT& P)
{
    static constexpr int idx[6] = {
        3, 4, 5,
        0, 1, 2
    };

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            odom_msg.pose.covariance[i * 6 + j] =
                P(idx[i], idx[j]);
        }
    }
}

void publishGnssPath(const GnssPositionData &pos)
{
    if (!gnssPathVis || !gnss_aligned.load()) return;

    Eigen::Matrix3d R_enu_map;
    Eigen::Vector3d t_enu_map;
    ros_manager.getEnuMap(R_enu_map, t_enu_map);
    const Eigen::Vector3d position_map =
        R_enu_map.transpose() * (pos.position - t_enu_map);

    PoseStampedMsg gnss_pose;
    gnss_pose.header.stamp = get_ros_time(pos.time);
    gnss_pose.header.frame_id = map_frame;
    gnss_pose.pose.position.x = position_map.x();
    gnss_pose.pose.position.y = position_map.y();
    gnss_pose.pose.position.z = position_map.z();
    gnss_pose.pose.orientation.w = 1.0;

    gnssPath.header.stamp = gnss_pose.header.stamp;
    gnssPath.header.frame_id = map_frame;
    gnssPath.poses.push_back(gnss_pose);
    ros_manager.publish(ros_manager.pubGnssPath, gnssPath);
}

void publishGnssLinkTf()
{
    GnssPositionData pos;
    GnssYawData yaw;
    {
        std::lock_guard<std::mutex> lock(p_gnss->gnssMutex);
        if (p_gnss->latestPos.time < 0.0 || p_gnss->latestYaw.time < 0.0)
            return;

        pos = p_gnss->latestPos;
        yaw = p_gnss->latestYaw;
    }

    TransformStampedMsg tf_msg;
    tf_msg.header.stamp = get_ros_time(std::max(pos.time, yaw.time));
    tf_msg.header.frame_id = "enu";
    tf_msg.child_frame_id = "gnss_link";

    const Eigen::Matrix3d R_enu_gnss =
        Eigen::AngleAxisd(yaw.yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    fillTransformMsg(tf_msg, R_enu_gnss, pos.position);

    ros_manager.sendTf(tf_msg);
}

bool initGnssMap(double lidar_stamp_sec)
{
    GnssPositionData gnss_pos;
    GnssYawData heading;
    if (!p_gnss->syncGnssInit(gnss_pos, heading)) return false;

    Eigen::Matrix3d R_map_body;
    Eigen::Vector3d t_map_body;
    composeMapPose(state_point.rot, state_point.pos, R_map_body, t_map_body);

    ros_manager.initializeEnuMap(
        gnss_pos.position, heading.yaw, R_map_body, t_map_body,
        p_gnss->leverArm, get_ros_time(lidar_stamp_sec));
    return true;
}

bool priorInitAlign()
{
    if (!use_prior_map || !prior_init_en || prior_init_done ||
        !mapManager.hasPriorMap() || featsFromPriorMap == nullptr ||
        featsFromPriorMap->empty() ||
        feats_down_body == nullptr || feats_down_body->size() < 20)
        return false;

    Eigen::Matrix3d R_map_base_guess;
    Eigen::Vector3d t_map_base_guess;
    composeMapPose(state_point.rot, state_point.pos,
                   R_map_base_guess, t_map_base_guess);

    const Eigen::Matrix3d R_map_lidar_guess =
        R_map_base_guess * Lidar_R_wrt_IMU;
    const Eigen::Vector3d t_map_lidar_guess =
        t_map_base_guess + R_map_base_guess * Lidar_T_wrt_IMU;

    Eigen::Matrix3d R_map_lidar;
    Eigen::Vector3d t_map_lidar;
    if (!mapManager.coarseIcpAlign(
            feats_down_body, featsFromPriorMap, R_map_lidar_guess,
            t_map_lidar_guess, R_map_lidar, t_map_lidar))
        return false;

    const Eigen::Matrix3d R_base_lidar = Lidar_R_wrt_IMU;
    const Eigen::Vector3d t_base_lidar = Lidar_T_wrt_IMU;
    const Eigen::Matrix3d R_map_base = R_map_lidar * R_base_lidar.transpose();
    const Eigen::Vector3d t_map_base = t_map_lidar - R_map_base * t_base_lidar;

    const Eigen::Matrix3d R_odom_base = state_point.rot;
    const Eigen::Vector3d t_odom_base = state_point.pos;
    Eigen::Matrix3d R_map_odom_refined = R_map_base * R_odom_base.transpose();
    Eigen::Vector3d t_map_odom_refined = t_map_base - R_map_odom_refined * t_odom_base;

    if (!mapManager.refinePointToPlane(
            feats_down_body, R_odom_base, t_odom_base,
            R_base_lidar, t_base_lidar,
            R_map_odom_refined, t_map_odom_refined))
        return false;

    mapManager.setPriorTransform(R_map_odom_refined, t_map_odom_refined);
    ros_manager.setMapOdom(R_map_odom_refined, t_map_odom_refined);
    prior_init_done = true;

    return true;
}

void publish_frame_world(ResultWriter &resultWriter)
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
        ros_manager.publish(ros_manager.pubLaserCloudFull, laserCloudmsg);
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
        resultWriter.accumulateFeatureCloud(
            *laserCloudWorld, lidar_end_time, res_save_interval);
    }
}

void publish_frame_body()
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
    ros_manager.publish(ros_manager.pubLaserCloudFullBody, laserCloudmsg);
}

void publish_effect_world()
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
    ros_manager.publish(ros_manager.pubLaserCloudEffect, laserCloudFullRes3);
}

void display_map_loop()
{
    double lastMapPublishTime = -1.0;
    while (true)
    {
        PointVector points;
        PointVector baPoints;
        double stamp;
        {
            std::unique_lock<std::mutex> lock(mapDisplayMutex);
            mapDisplayCondition.wait(lock, [&] {
                return mapTimeUpdated || mapDisplayStopping;
            });
            if (mapDisplayStopping) return;

            mapTimeUpdated = false;
            const double scanTime = latestMapTime;
            if (lastMapPublishTime >= 0.0 &&
                scanTime - lastMapPublishTime < 1.0 &&
                scanTime >= lastMapPublishTime)
                continue;

            lastMapPublishTime = scanTime;
            mapDisplayCloudRequested = true;
            mapDisplayCondition.notify_all();
            mapDisplayCondition.wait(lock, [] {
                return mapDisplayPending || mapDisplayStopping;
            });
            if (mapDisplayStopping) return;
            points.swap(mapDisplayPoints);
            baPoints.swap(baMapDisplayPoints);
            stamp = displayStamp;
            mapDisplayPending = false;
        }

        PointCloudXYZI cloud;
        cloud.height = 1;
        if (online_tree_pub_en)
        {
            cloud.points = std::move(points);
            cloud.width = cloud.points.size();
            Pcl2Msg message;
            pcl::toROSMsg(cloud, message);
            message.header.stamp = get_ros_time(stamp);
            message.header.frame_id = odom_frame;
            ros_manager.publish(ros_manager.pubLaserCloudMap, message);
        }
        if (ba_tree_pub_en)
        {
            cloud.points = std::move(baPoints);
            cloud.width = cloud.points.size();
            Pcl2Msg message;
            pcl::toROSMsg(cloud, message);
            message.header.stamp = get_ros_time(stamp);
            message.header.frame_id = odom_frame;
            ros_manager.publish(ros_manager.pubLaserCloudBAMap, message);
        }
    }
}

void update_map_display_time(double stamp)
{
    {
        std::lock_guard<std::mutex> lock(mapDisplayMutex);
        latestMapTime = stamp;
        mapTimeUpdated = true;
    }
    mapDisplayCondition.notify_one();
}

void prepare_map_display_cloud()
{
    double stamp;
    {
        std::lock_guard<std::mutex> lock(mapDisplayMutex);
        if (!mapDisplayCloudRequested || mapDisplayStopping)
            return;
        mapDisplayCloudRequested = false;
        stamp = lidar_end_time;
    }

    PointVector mapPoints, baMapPoints;
    if (online_tree_pub_en)
        mapManager.flattenTree(mapPoints, false);
    if (ba_tree_pub_en)
        mapManager.flattenTree(baMapPoints, true);
    {
        std::lock_guard<std::mutex> lock(mapDisplayMutex);
        if (mapDisplayStopping)
            return;
        mapDisplayPoints = std::move(mapPoints);
        baMapDisplayPoints = std::move(baMapPoints);
        displayStamp = stamp;
        mapDisplayPending = true;
    }
    mapDisplayCondition.notify_all();
}

void publish_geo_map()
{
    if (!geo_map_enable || geo_map_dropped->empty())
        return;

    Pcl2Msg geoMapDroppedMsg;
    pcl::toROSMsg(*geo_map_dropped, geoMapDroppedMsg);
    geoMapDroppedMsg.header.stamp = get_ros_time(lidar_end_time);
    geoMapDroppedMsg.header.frame_id = odom_frame;
    ros_manager.publish(ros_manager.pubGeoMapDropped, geoMapDroppedMsg);
}

void publish_prior_map()
{
    if (featsFromPriorMap->empty())
        return;

    Pcl2Msg laserCloudPriorMap;
    pcl::toROSMsg(*featsFromPriorMap, laserCloudPriorMap);
    laserCloudPriorMap.header.stamp = get_ros_time(lidar_end_time);
    laserCloudPriorMap.header.frame_id = map_frame;
    ros_manager.publish(ros_manager.pubLaserCloudPriorMap, laserCloudPriorMap);
}

void publish_odometryhighfreq(ImuProcess &imuProcess,
                              ResultWriter &resultWriter)
{
    auto &ros = ros_manager;
    Eigen::Matrix3d R_map_odom_hf = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_map_odom_hf = Eigen::Vector3d::Zero();

    while (ros_ok() && !flg_exit){
        State pose;
        if (!imuProcess.TryPopPose(pose))
        {
            usleep(1000);
            continue;
        }
        OdomMsg msg_local;
        const auto stamp = get_ros_time(pose._timestamp);

        ros.tryTakeMapOdomUpdate(R_map_odom_hf, t_map_odom_hf);

        TransformStampedMsg map_to_odom_msg;
        map_to_odom_msg.header.stamp = stamp;
        map_to_odom_msg.header.frame_id = map_frame;
        map_to_odom_msg.child_frame_id = odom_frame;
        fillTransformMsg(map_to_odom_msg, R_map_odom_hf, t_map_odom_hf);

        ros.sendTf(map_to_odom_msg);

        const Eigen::Matrix3d R_odom_base = Eigen::Quaterniond(pose._qw, pose._qx, pose._qy, pose._qz).toRotationMatrix();
        const Eigen::Vector3d t_odom_base(pose._x, pose._y, pose._z);
        msg_local.header.stamp = stamp;
        msg_local.header.frame_id = odom_frame;
        msg_local.child_frame_id = high_freq_base_frame;
        {
            Eigen::Quaterniond q(R_odom_base);
            q.normalize();
            msg_local.pose.pose.position.x = t_odom_base.x();
            msg_local.pose.pose.position.y = t_odom_base.y();
            msg_local.pose.pose.position.z = t_odom_base.z();
            msg_local.pose.pose.orientation.x = q.x();
            msg_local.pose.pose.orientation.y = q.y();
            msg_local.pose.pose.orientation.z = q.z();
            msg_local.pose.pose.orientation.w = q.w();
        }

        TransformStampedMsg tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = odom_frame;
        tf_msg.child_frame_id = high_freq_base_frame;
        fillTransformMsg(tf_msg, R_odom_base, t_odom_base);

        ros.publish(ros.pubOdomHighFreq, msg_local);

        ros.sendTf(tf_msg);

        resultWriter.saveImuPose(pose);
    }
}

template<typename T>
void set_posestamp(T & out)
{
    Eigen::Matrix3d R_map_base;
    Eigen::Vector3d t_map_base;
    composeMapPose(state_point.rot, state_point.pos, R_map_base, t_map_base);
    const Eigen::Quaterniond q_map_base(R_map_base);

    out.pose.position.x = t_map_base(0);
    out.pose.position.y = t_map_base(1);
    out.pose.position.z = t_map_base(2);
    out.pose.orientation.x = q_map_base.x();
    out.pose.orientation.y = q_map_base.y();
    out.pose.orientation.z = q_map_base.z();
    out.pose.orientation.w = q_map_base.w();
    
}

void publish_odometry()
{
    auto &ros = ros_manager;
    const auto stamp = get_ros_time(lidar_end_time);

    const Eigen::Matrix3d R_odom_base = state_point.rot;
    const Eigen::Vector3d t_odom_base = state_point.pos;
    odomAftMapped.header.stamp = stamp;
    odomAftMapped.header.frame_id = odom_frame;
    odomAftMapped.child_frame_id = base_frame;
    {
        Eigen::Quaterniond q(R_odom_base);
        q.normalize();
        odomAftMapped.pose.pose.position.x = t_odom_base.x();
        odomAftMapped.pose.pose.position.y = t_odom_base.y();
        odomAftMapped.pose.pose.position.z = t_odom_base.z();
        odomAftMapped.pose.pose.orientation.x = q.x();
        odomAftMapped.pose.pose.orientation.y = q.y();
        odomAftMapped.pose.pose.orientation.z = q.z();
        odomAftMapped.pose.pose.orientation.w = q.w();
    }

    const auto& P = kf.covariance();
    fillOdometryCovariance(odomAftMapped, P);

    TransformStampedMsg tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = odom_frame;
    tf_msg.child_frame_id  = base_frame;
    fillTransformMsg(tf_msg, R_odom_base, t_odom_base);

    ros.publish(ros.pubOdom, odomAftMapped);
    ros.publishEnuToMapTf(stamp);
    ros.sendTf(tf_msg);
}

void publish_path()
{
    const PointTypePose pose = currentOdomPose(lidar_end_time);
    msg_body_pose.header.stamp = get_ros_time(pose.time);
    msg_body_pose.header.frame_id = odom_frame;
    msg_body_pose.pose.position.x = pose.x;
    msg_body_pose.pose.position.y = pose.y;
    msg_body_pose.pose.position.z = pose.z;
    msg_body_pose.pose.orientation = quaternion_from_rpy(pose.roll, pose.pitch, pose.yaw);

    /*** if path is too large, rviz will crash ***/
    static int jjj = 0;
    jjj++;

    if (jjj % 10 == 0)
    {
        odomPath.poses.push_back(msg_body_pose);
        odomPath.header.stamp = msg_body_pose.header.stamp;
        odomPath.header.frame_id = odom_frame;

        Eigen::Matrix3d R_map_body;
        Eigen::Vector3d t_map_body;
        composeMapPose(poseRotation(pose), poseTranslation(pose), R_map_body, t_map_body);
        PoseStampedMsg mapPose = msg_body_pose;
        const Eigen::Quaterniond q_map_body(R_map_body);
        mapPose.header.frame_id = map_frame;
        mapPose.pose.position.x = t_map_body.x();
        mapPose.pose.position.y = t_map_body.y();
        mapPose.pose.position.z = t_map_body.z();
        mapPose.pose.orientation.x = q_map_body.x();
        mapPose.pose.orientation.y = q_map_body.y();
        mapPose.pose.orientation.z = q_map_body.z();
        mapPose.pose.orientation.w = q_map_body.w();
        path.poses.push_back(mapPose);
        path.header.stamp = msg_body_pose.header.stamp;
        path.header.frame_id = map_frame;
        ros_manager.publish(ros_manager.pubPath, path);
        ros_manager.publish(ros_manager.pubOdomPath, odomPath);
    }
}

bool buildLidarMeasurement(const SimpleEKF::State &s,
                           bool rematch,
                           Eigen::MatrixXd &H,
                           Eigen::VectorXd &residual)
{
    laserCloudOri->clear();
    corr_normvect->clear();
    if (feats_down_size < 1)
        return false;

    mapping::ObservationOptions options;
    options.use_online = use_online_map;
    options.use_window = useBAMap();
    options.use_prior = use_prior_map && prior_init_done;
    options.point_to_point = point2point;
    options.prior_scale = std::sqrt(LASER_POINT_COV);

    const double match_start = omp_get_wtime();
    std::vector<point_obs::MatchObservation> matches = mapManager.buildObservations(
        *feats_down_body, *feats_down_world, s.rot, s.pos,
        Lidar_R_wrt_IMU, Lidar_T_wrt_IMU, rematch, options);
    effct_feat_num = matches.size();
    if (effct_feat_num < 1)
        return false;
    match_time += omp_get_wtime() - match_start;

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
    }
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of measurement Jacobian and residual vector. ***/
    const int measurement_rows = point2point ? 3 * effct_feat_num : effct_feat_num;
    H = MatrixXd::Zero(measurement_rows, 6);
    residual.resize(measurement_rows);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        V3D point_this = Lidar_R_wrt_IMU * point_this_be + Lidar_T_wrt_IMU;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        const double row_scale = matches[i].row_scale;

        if (point2point)
        {
            const PointType &target = matches[i].target;
            const V3D target_world(target.x, target.y, target.z);
            const V3D point_world = s.rot * point_this + s.pos;
            const int row = 3 * i;

            H.block<3, 3>(row, 0) = -row_scale * s.rot * point_crossmat;
            H.block<3, 3>(row, 3) = row_scale * M3D::Identity();
            residual.segment<3>(row) =
                row_scale * (target_world - point_world);
        }
        else
        {
            /*** get the normal vector of closest surface/corner ***/
            const PointType &norm_p = corr_normvect->points[i];
            V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

            /*** calculate the Measuremnt Jacobian matrix H ***/
            V3D C(s.rot.transpose() * norm_vec);
            V3D A(point_crossmat * C);
            H.block<1, 3>(i, 0) = row_scale * A.transpose();
            H.block<1, 3>(i, 3) = row_scale * norm_vec.transpose();

            /*** Measuremnt: distance to the closest surface/corner ***/
            residual(i) = -row_scale * norm_p.intensity;
        }
    }
    solve_time += omp_get_wtime() - solve_start_;
    return true;
}

int main(int argc, char** argv)
{
    ResultWriter resultWriter;
    #ifdef USE_ROS1
    ros::init(argc, argv, "fast_lio_sam");
    init_ros_node();
    
    #elif defined(USE_ROS2)
    rclcpp::init(argc, argv);
    init_ros_node(rclcpp::Node::make_shared("fast_lio_sam"));
    #endif

    read_mapping_params(*p_pre, lidar_type);
    std::string offlineBagPath;
    rosparam_get("offline/bag_path", offlineBagPath, std::string(""));
    std::string inputMode;
    rosparam_get("offline/input_mode", inputMode, std::string("online"));
    set_mapping_mode();
    mapManager.setDownsampleParam(filter_size_map_min);

    path.header.stamp = get_ros_now();
    path.header.frame_id = map_frame;

    read_lidar_ba_params();
    lidarBAConfig.imu_enable = false;
    lidarBAConfig.gyr_cov = gyr_cov;
    lidarBAConfig.acc_cov = acc_cov;
    lidarBAConfig.bg_cov = b_gyr_cov;
    lidarBAConfig.ba_cov = b_acc_cov;
    if (sam_enable) {
        read_liosam_params();
        read_sgraph_params();
    }
    read_gnss_params();
    p_gnss->headingOffset = heading_offset;
    p_pre->lidar_type = lidar_type;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;

    /*** variables definition ***/
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
    p_imu->lidar_type = lidar_type;

    kf.setMaxIterations(NUM_MAX_ITERATIONS);

    p_gnss->leverArm =
        Lidar_T_wrt_IMU + Lidar_R_wrt_IMU * gnss_extrinsic_T;

    if (use_prior_map)
    {
        if (!mapManager.loadPriorTree(prior_tree_path, *featsFromPriorMap))
        {
            use_prior_map = false;
            prior_init_en = false;
        }
    }

    /*** debug record ***/
    resultWriter.prepareDirectories();

    {
    RosManager::SubscriberOptions rosOptions;
    rosOptions.avia = p_pre->lidar_type == AVIA;
    rosOptions.lidarTopic = lid_topic;
    rosOptions.imuTopic = imu_topic;
    rosOptions.relocTopic = reloc_topic;
    rosOptions.gnssTopic = gnss_topic;
    rosOptions.gnssHeadingTopic = gnss_heading_topic;
    rosOptions.livoxCallback = livox_pcl_cbk;
    rosOptions.cloudCallback = standard_pcl_cbk;
    rosOptions.relocCallback = reloc_cbk;
    rosOptions.imuCallback = imu_cbk;
    rosOptions.gnssCallback = gnss_cbk;
    rosOptions.headingCallback = gnss_heading_cbk;
    ros_manager.initPublishers(
        lidarBAConfig.enable, sam_enable, gnssPathVis);
    std::unique_ptr<RosbagInput> bagInput;
    if (inputMode == "offline")
        bagInput = std::make_unique<RosbagInput>(offlineBagPath, lid_topic, imu_topic,
            rosOptions.avia, rosOptions.cloudCallback,
            rosOptions.livoxCallback, rosOptions.imuCallback);
    else
        ros_manager.initSubscribers(rosOptions);
    if ((online_tree_pub_en || ba_tree_pub_en) && use_online_map)
        mapDisplayThread = std::thread(display_map_loop);

    if (sam_enable)
        globalGraph.reset(new GlobalGraph());
    nextLidarBAFrameId = 0;
    keyFrameCount = 0;
    lastOdomPose = PointTypePose{};
        std::thread odomhighthread([&](){
            publish_odometryhighfreq(*p_imu, resultWriter);
        });

//------------------------------------------------------------------------------------------------------
        signal(SIGINT, SigHandle);
#ifdef USE_ROS2
        rclcpp::executors::MultiThreadedExecutor callbackExecutor(
            rclcpp::ExecutorOptions(), 2);
        callbackExecutor.add_node(get_ros_node());
        std::thread callbackThread([&callbackExecutor]() { callbackExecutor.spin(); });
#endif
        RateType rate(5000);
        bool input_finished = false;
        bool bagEof = false;
        while (ros_ok() && !flg_exit)
        {
#ifdef USE_ROS1
            spin_once();
#endif
            if (bagInput && !bagEof)
                bagEof = !bagInput->feedNext();
        // relocalization trigger
        if(relocalize_flag.load())
        {
            State reloc_pose;
            {
                std::lock_guard<std::mutex> lock(mtx_reloc);
                reloc_pose = reloc_state;
            }
            Eigen::Quaterniond reloc_rot(reloc_pose._qw, reloc_pose._qx, reloc_pose._qy, reloc_pose._qz);
            reloc_rot.normalize();
            const Eigen::Vector3d reloc_pos(reloc_pose._x, reloc_pose._y, reloc_pose._z);

            const Eigen::Matrix3d R_map_base = reloc_rot.toRotationMatrix();
            const Eigen::Matrix3d R_odom_base = state_point.rot;
            const Eigen::Vector3d t_odom_base = state_point.pos;
            ros_manager.updateMapOdom(
                R_map_base, reloc_pos, R_odom_base, t_odom_base);

            relocalize_flag.store(false);
        }

        const bool frameReady = !input_finished && sync_packages(Measures);
        if(frameReady)
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
            if (lidarBAConfig.imu_enable)
                p_imu->accumulateBAImu(Measures.imu);
            state_point = kf.state();
            pos_lid = state_point.pos + state_point.rot * Lidar_T_wrt_IMU; // LiDAR position in the world coordinate frame

            if (!feats_undistort || feats_undistort->empty())
            {
                ROS_PRINT_WARN("No point, skip this scan!");
                continue;
            }

            if (!sam_enable &&
                (gnssEnableFlag || gnssPathVis) && !gnss_aligned.load())
            {
                initGnssMap(Measures.lidar_end_time);
            }
            if (scan_frame_save_en)
                resultWriter.saveScanFrame(
                    *feats_undistort, lidar_end_time, state_point.rot,
                    state_point.pos, Lidar_R_wrt_IMU, Lidar_T_wrt_IMU);

            const bool globalReady = mapManager.hasTree(false) ||
                (useBAMap() && mapManager.hasTree(true));

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            if (use_online_map && globalReady)
            {
                const double delete_start = omp_get_wtime();
                kdtree_delete_counter = mapManager.updateLocalMap(
                    pos_lid, cube_len, DET_RANGE, useBAMap());
                kdtree_delete_time = omp_get_wtime() - delete_start;
            }

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            feats_down_size = feats_down_body->points.size();
            if (imuRollbackScan && use_online_map && !use_prior_map &&
                !globalReady)
                continue;
            /*** initialize the online map kdtree ***/
            if (!imuRollbackScan && use_online_map && !use_prior_map &&
                !globalReady)
            {
                if(feats_down_size > 5)
                {
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                        pointBodyToWorld(&(feats_down_body->points[i]),
                                         &(feats_down_world->points[i]));
                    mapManager.initializeOnline(feats_down_world->points, useBAMap());
                }
                bool baOptimized = false;
                Eigen::Isometry3d baPose = Eigen::Isometry3d::Identity();
                std::vector<BAFramePtr> baFrames;
                bool baFrameProcessed = false;
                bool keyFrameAdded = false;
                if (lidarBAConfig.enable && feats_down_size >= 5)
                {
                    getCurrPose(state_point);
                    const bool keyFrame = sam_enable && isKeyframe();
                    keyFrameAdded = keyFrame;
                    baOptimized = processBAFrame(
                        feats_undistort, feats_down_body, lidar_end_time, keyFrame,
                        baPose, baFrames);
                    baFrameProcessed = true;
                }
                else if (sam_enable && mapManager.hasFixedOnlineMap())
                {
                    getCurrPose(state_point);
                    const bool keyFrame = sam_enable && isKeyframe();
                    if (keyFrame)
                    {
                        keyFrameAdded = true;
                        baOptimized = processBAFrame(
                            feats_undistort, feats_down_body, lidar_end_time, true,
                            baPose, baFrames);
                        baFrameProcessed = true;
                    }
                }
                if (baFrameProcessed) {
                    applyBA(baOptimized, baPose);
                    processBAFrames(baFrames);
                    rebuildBAMap();
                    if (keyFrameAdded) {
                        lastOdomPose = currentOdomPose(lidar_end_time);
                        ++keyFrameCount;
                    }
                }
                continue;
            }
            kdtree_size_st = use_online_map ? mapManager.onlineSize(useBAMap()) : 0;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_PRINT_WARN("No point, skip this scan!");
                continue;
            }

            feats_down_world->resize(feats_down_size);


            /*** iterated state estimation ***/
            double solve_H_time = 0;
            if (use_prior_map && prior_init_en && !prior_init_done)
            {
                if (!priorInitAlign())
                {
                    ROS_PRINT_WARN("prior init alignment failed, retry on next scan");
                    continue;
                }
                state_point = kf.state();
                euler_cur = state_point.rot.eulerAngles(0, 1, 2);
                pos_lid = state_point.pos + state_point.rot * Lidar_T_wrt_IMU;
                const Eigen::Quaterniond q_state(state_point.rot);
                geoQuat.x = q_state.x();
                geoQuat.y = q_state.y();
                geoQuat.z = q_state.z();
                geoQuat.w = q_state.w();
                getCurrPose(state_point);
            }
            else
            {
                const SimpleEKF::MeasurementBuilder measurementBuilder =
                    [](const SimpleEKF::State &state, bool rematch,
                       Eigen::MatrixXd &H, Eigen::VectorXd &residual) {
                        return buildLidarMeasurement(state, rematch, H, residual);
                    };
                kf.updateLidar(measurementBuilder, LASER_POINT_COV);
                state_point = kf.state();
                euler_cur = state_point.rot.eulerAngles(0, 1, 2);
                pos_lid = state_point.pos + state_point.rot * Lidar_T_wrt_IMU;
                const Eigen::Quaterniond q_state(state_point.rot);
                geoQuat.x = q_state.x();
                geoQuat.y = q_state.y();
                geoQuat.z = q_state.z();
                geoQuat.w = q_state.w();
                getCurrPose(state_point);
            }

            bool keyFrame = false;
            if (sam_enable)
            {
                getCurrPose(state_point);
                if (sam_enable)
                    keyFrame = isKeyframe();
            }

            if (imuRollbackScan)
                geo_map_dropped->clear();

            PointCloudXYZI::Ptr acceptedMapCloud;
            if (useBAMap() && !imuRollbackScan)
            {
                acceptedMapCloud.reset(new PointCloudXYZI());
                map_incremental(acceptedMapCloud);
            }

            bool baOptimized = false;
            Eigen::Isometry3d baPose = Eigen::Isometry3d::Identity();
            std::vector<BAFramePtr> baFrames;
            if (!imuRollbackScan && (lidarBAConfig.enable || keyFrame))
            {
                getCurrPose(state_point);
                baOptimized = processBAFrame(
                    feats_undistort,
                    acceptedMapCloud ? acceptedMapCloud : feats_down_body,
                    lidar_end_time, keyFrame,
                    baPose, baFrames);
                applyBA(baOptimized, baPose);
                processBAFrames(baFrames);
                if (keyFrame)
                {
                    lastOdomPose = currentOdomPose(lidar_end_time);
                    ++keyFrameCount;
                }
            }

            /*** add the feature points to map kdtree ***/
            if (use_online_map && !imuRollbackScan && !useBAMap())
                map_incremental();
            rebuildBAMap();
            publish_geo_map();

            /******* Publish odometry after BA feedback and map update *******/
            publish_odometry();
            
            /******* Publish points *******/
            if (path_en)                         publish_path();
            if (scan_pub_en || feat_accum_save_en)
                publish_frame_world(resultWriter);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body();
            if (effect_pub_en) publish_effect_world();
            if ((online_tree_pub_en || ba_tree_pub_en) && use_online_map)
            {
                update_map_display_time(lidar_end_time);
                prepare_map_display_cloud();
            }
            if ((online_tree_pub_en || ba_tree_pub_en) && use_prior_map)
                publish_prior_map();
        }
        
            if (bagInput && bagEof && !frameReady) {
                break;
            }
            if (!bagInput)
            {
                const int64_t input_time = last_input_time.load();
                if (!input_finished && input_time > 0 && nowTimeUs() - input_time >= 2000000)
                    input_finished = true;
                if (input_finished)
                    break;
                rate.sleep();
            }
        }

        {
            std::lock_guard<std::mutex> lock(mapDisplayMutex);
            mapDisplayStopping = true;
        }
        mapDisplayCondition.notify_one();
        if (mapDisplayThread.joinable()) mapDisplayThread.join();
#ifdef USE_ROS1
        // Drain callbacks already queued before stopping Tracking submissions.
        spin_once();
#elif defined(USE_ROS2)
        callbackExecutor.cancel();
        callbackThread.join();
#endif

        if (lidarBA)
        {
            const std::vector<BAFramePtr> baFrames = lidarBA->Finish();
            processBAFrames(baFrames);
            if (!lidarBAPath.poses.empty())
            {
                ros_manager.publish(
                    ros_manager.pubLidarBAPath, lidarBAPath);
                ros_manager.publish(
                    ros_manager.pubBAFrameMarkers, baFrameMarkers);
            }
        }
        if (globalGraph)
            globalGraph->Shutdown();
        flg_exit = true;
        if (odomhighthread.joinable()) {
            odomhighthread.join();
        }

    resultWriter.flushAccumulatedCloud(lidar_end_time);

    if (sam_enable && keyframe_export_en)
    {
        resultWriter.saveKeyframePoses(
            globalGraph ? globalGraph->GetKeyframePoses()
                        : std::vector<PointTypePose>{});
    }

    if (sam_enable && keyframe_global_pcd_en)
    {
        resultWriter.saveGlobalCloud(globalGraph.get());
    }

        if (ikdtree_output_save_en && use_online_map && mapManager.hasFixedOnlineMap())
        {
            resultWriter.saveTree(mapManager.onlineTree());
        }

        lidarBA.reset();
        globalGraph.reset();
        ros_manager.reset();
    }

#ifdef USE_ROS2
    get_ros_node().reset();
#endif
    ros_shutdown();

    return 0;
}
