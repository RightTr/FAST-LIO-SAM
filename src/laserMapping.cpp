#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <iomanip>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
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

double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   feat_accum_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
bool   imu_state_save_en = false, scan_frame_save_en = false;

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
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
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
double zupt_acc_var_threshold;
double zupt_gyro_var_threshold;
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

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
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

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_PRINT_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

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
    const double stamp_sec = get_ros_time_sec(msg->header.stamp);
    if (stamp_sec < last_timestamp_lidar)
    {
        ROS_PRINT_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(stamp_sec);
    last_timestamp_lidar = stamp_sec;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const LivoxCustomMsgConstPtr &msg) 
{
    mtx_buffer.lock();
    const double stamp_sec = get_ros_time_sec(msg->header.stamp);
    if (stamp_sec < last_timestamp_lidar)
    {
        ROS_PRINT_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = stamp_sec;
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
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
    double timestamp = get_ros_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_PRINT_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

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
    
    std::lock_guard<std::mutex> lock(mtx_reloc);
    reloc_state = Pose(x, y, z,
                    qx, qy, qz, qw, timestamp);
    relocalize_flag.store(true); 
    ROS_PRINT_INFO("Reloc received: (%.3f, %.3f, %.3f), quat=(%.3f, %.3f, %.3f, %.3f)",
        x, y, z, qx, qy, qz, qw);
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
    double imu_time = get_ros_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_ros_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
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

void update_state_ikfom()
{
    state_ikfom state_updated = kf.get_x();
    Eigen::Vector3d pos(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]);
    Eigen::Quaterniond q = Eigen::Quaterniond(Eigen::AngleAxisd(transformTobeMapped[2], Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(transformTobeMapped[1], Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(transformTobeMapped[0], Eigen::Vector3d::UnitX()));

    // Only update pose
    state_updated.pos = pos;
    state_updated.rot =  q;
    state_point = state_updated;

    kf.change_x(state_updated);
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
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
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
            RGBpointBodyToWorld(&feats_undistort->points[i], \
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
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
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
    laserCloudMap.header.frame_id = map_frame;
    ros_publish(pubLaserCloudMap, laserCloudMap);
}

void publish_odometryhighfreq(PoseBuffer& pbuffer,
                              const OdomPublisher& pubOdomHighFreqLocal,
                              const OdomPublisher& pubOdomHighFreqGlobal)
{
    while (ros_ok() && !flg_exit){
        Pose pose;
        if (!pbuffer.TryPop(pose))
        {
            usleep(1000);
            continue;
        }
        OdomMsg msg_local;
        OdomMsg msg_global;
        const auto stamp = get_ros_time(pose._timestamp);

        const Eigen::Matrix3d R_odom_map = R_map_odom.transpose();
        const Eigen::Matrix3d R_map_base = Eigen::Quaterniond(pose._qw, pose._qx, pose._qy, pose._qz).toRotationMatrix();
        const Eigen::Vector3d t_map_base(pose._x, pose._y, pose._z);

        const Eigen::Matrix3d R_odom_base = R_odom_map * R_map_base;
        const Eigen::Vector3d t_odom_base = R_odom_map * (t_map_base - t_map_odom);
        fillOdometryMsg(msg_local, odom_frame, high_freq_base_frame, stamp, R_odom_base, t_odom_base);
        fillOdometryMsg(msg_global, map_frame, high_freq_base_frame, stamp, R_map_base, t_map_base);

        TransformStampedMsg tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = odom_frame;
        tf_msg.child_frame_id = high_freq_base_frame;
        fillTransformMsg(tf_msg, R_odom_base, t_odom_base);

        ros_publish(pubOdomHighFreqLocal, msg_local);
        ros_publish(pubOdomHighFreqGlobal, msg_global);

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
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const OdomPublisher & pubOdomAftMappedLocal,
                      const OdomPublisher & pubOdomAftMappedGlobal)
{
    OdomMsg odomAftMappedGlobal;
    const auto stamp = get_ros_time(lidar_end_time);

    const Eigen::Matrix3d R_odom_map = R_map_odom.transpose();

    const Eigen::Matrix3d R_map_base = state_point.rot.toRotationMatrix();
    const Eigen::Vector3d t_map_base = state_point.pos;
    const Eigen::Matrix3d R_odom_base = R_odom_map * state_point.rot.toRotationMatrix();
    const Eigen::Vector3d t_odom_base = R_odom_map * (state_point.pos - t_map_odom);
    fillOdometryMsg(odomAftMapped, odom_frame, base_frame, stamp, R_odom_base, t_odom_base);
    fillOdometryMsg(odomAftMappedGlobal, map_frame, base_frame, stamp, R_map_base, t_map_base);

    const auto& P = kf.get_P();
    fillOdometryCovariance(odomAftMapped, P);
    fillOdometryCovariance(odomAftMappedGlobal, P);

    TransformStampedMsg tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = odom_frame;
    tf_msg.child_frame_id  = base_frame;
    fillTransformMsg(tf_msg, R_odom_base, t_odom_base);

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

    ros_publish(pubOdomAftMappedLocal, odomAftMapped);
    ros_publish(pubOdomAftMappedGlobal, odomAftMappedGlobal);

#ifdef USE_ROS1
    static tf::TransformBroadcaster br;
#elif defined(USE_ROS2)
    static tf2_ros::TransformBroadcaster br(get_ros_node());
#endif
    br.sendTransform(map_to_odom_msg);
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

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        return;
    }

    // compute the residual mean
    res_mean_last = total_residual / effct_feat_num;
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
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
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
    rosparam_get("common/high_freq_base_frame", high_freq_base_frame, base_frame);
    rosparam_get("reloc/reloc_topic", reloc_topic, std::string("/reloc/manual"));
    rosparam_get("common/time_sync_en", time_sync_en, false);
    rosparam_get("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    rosparam_get("common/flip_en", flip_en, false);
    rosparam_get("common/grav_align", grav_align, false);
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

    p_pre->lidar_type = lidar_type;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;

    /*** variables definition ***/
    int effect_feat_num = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    // extrinT/extrinR must stay in the sensor's native Airy/DIFOP IMU convention.
    // When flip_en is enabled, IMU measurements, local point clouds, and these
    // extrinsics are all standardized to the Mid360 convention before estimation.
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
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    const string scan_frames_dir = root_dir + "/SCAN_FRAMES/";
    const string imu_states_dir = root_dir + "/IMU_STATES/";
    const string keyframe_frames_dir = root_dir + "/KEY_FRAMES/";
    
    if (scan_frame_save_en)
    {
        if (!create_directory(scan_frames_dir + "scans/"))
        {
            ROS_PRINT_ERROR("Failed to create scan frame save directories, disable scan_frame_save_en");
            scan_frame_save_en = false;
        }
        else if (!create_directory(scan_frames_dir + "scans_tstamp/"))
        {
            ROS_PRINT_ERROR("Failed to create tstamp scan frame save directories, disable scan_frame_save_en");
            scan_frame_save_en = false;
        }
        else
        {
            string scan_frame_pose_path = scan_frames_dir + "scan_pose.txt";
            scan_frame_pose_file.open(scan_frame_pose_path.c_str(), ios::out | ios::app);
            scan_frame_pose_file << std::fixed << std::setprecision(9);
        }
    }
    if (imu_state_save_en)
    {
        if (!create_directory(imu_states_dir))
        {
            ROS_PRINT_ERROR("Failed to create IMU state save directory, disable imu_state_save_en");
            imu_state_save_en = false;
        }
        else
        {
            string imu_state_path = imu_states_dir + "imu_state.txt";
            imu_pose_file.open(imu_state_path.c_str(), ios::out);
            imu_pose_file << std::fixed << std::setprecision(9);
            imu_pose_file << "# imu_states.txt columns:\n";
            imu_pose_file << "# timestamp[s] pos_x[m] pos_y[m] pos_z[m] qx qy qz qw "
                             "vel_x[m/s] vel_y[m/s] vel_z[m/s] "
                             "bg_x[rad/s] bg_y[rad/s] bg_z[rad/s] "
                             "ba_x[m/s^2] ba_y[m/s^2] ba_z[m/s^2] "
                             "grav_x[m/s^2] grav_y[m/s^2] grav_z[m/s^2] "
                             "ext_qx ext_qy ext_qz ext_qw ext_tx[m] ext_ty[m] ext_tz[m]\n";
        }
    }

    if (keyframe_export_en || keyframe_global_pcd_en)
    {
        if (!create_directory(keyframe_frames_dir + "scans/"))
        {
            ROS_PRINT_ERROR("Failed to create keyframe save directories, disable keyframe export");
            keyframe_export_en = false;
            keyframe_global_pcd_en = false;
        }
        else if (keyframe_export_en)
        {
            string keyframe_pose_path = keyframe_frames_dir + "keyframe_pose.txt";
            keyframe_pose_file.open(keyframe_pose_path.c_str(), ios::out | ios::app);
            keyframe_pose_file << std::fixed << std::setprecision(9);
        }
    }

    /*** ROS subscribe initialization ***/
    if (p_pre->lidar_type == AVIA) {
        static auto sub_pcl = create_subscriber<LivoxMsg>(lid_topic, 200000, livox_pcl_cbk);
    } else {
        static auto sub_pcl = create_subscriber<PointCloud2Msg>(lid_topic, 200000, standard_pcl_cbk);
    }
    
    auto sub_reloc = create_subscriber<PoseStampedMsg>(reloc_topic, 10, reloc_cbk);
    auto sub_imu = create_subscriber<ImuMsg>(imu_topic, 200000, imu_cbk);
    auto pubLaserCloudFull = create_publisher<PointCloud2Msg>("/cloud_registered", 100000);
    auto pubLaserCloudFull_body = create_publisher<PointCloud2Msg>("/cloud_registered_body", 100000);
    auto pubLaserCloudEffect = create_publisher<PointCloud2Msg>("/cloud_effected", 100000);
    auto pubLaserCloudMap = create_publisher<PointCloud2Msg>("/Laser_map", 100000);
    #ifdef USE_ROS1
    int odom_qos = 0;  // ROS1 ignores this parameter
    #elif defined(USE_ROS2)
    auto odom_qos = rclcpp::QoS(10).best_effort(); // avoid latency caused by QoS reliability in ROS2
    #endif
    auto pubOdomAftMapped = create_publisher_qos<OdometryMsg>("/Odometry", odom_qos);
    auto pubOdomAftMappedGlobal = create_publisher_qos<OdometryMsg>("/OdometryGlobal", odom_qos);
    auto pubPath = create_publisher_qos<PathMsg>("/path", odom_qos);
    auto pubOdomHighFreq = create_publisher_qos<OdometryMsg>("/OdometryHighFreq", odom_qos);
    auto pubOdomHighFreqGlobal = create_publisher_qos<OdometryMsg>("/OdometryHighFreqGlobal", odom_qos);
    p_pre->pub_corn = create_publisher<PointCloud2Msg>("/corn_feature", 100000);
    p_pre->pub_surf = create_publisher<PointCloud2Msg>("/surf_feature", 100000);

    if (sam_enable) {
        MapOptimizationInit();
        printf("...... LIO-SAM Backend Start......\n");

    }

    std::thread odomhighthread([&](){
        publish_odometryhighfreq(p_imu->pbuffer, pubOdomHighFreq, pubOdomHighFreqGlobal);
    });

    std::thread loopthread;
    std::thread globalthread;
    if (sam_enable)
    {
        loopthread = std::thread(&loopClosureThread);
        globalthread = std::thread(&visualizeGlobalMapThread);
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
            feats_down_world->clear();
            p_imu->Reset();
            state_ikfom state_point_reloc;
            {
                std::lock_guard<std::mutex> lock(mtx_reloc);
                state_point_reloc.pos = Eigen::Vector3d(reloc_state._x, reloc_state._y, reloc_state._z);
                state_point_reloc.rot = Eigen::Quaterniond(reloc_state._qw, reloc_state._qx,
                                            reloc_state._qy, reloc_state._qz);
            }
            state_point_reloc.rot.normalize();
            kf.reset(state_point_reloc);
            ikdtree.delete_tree_nodes(&ikdtree.Root_Node);

            ROS_PRINT_INFO("Reloc: pos=(%.2f %.2f %.2f), quat=(%.2f %.2f %.2f %.2f)",
                state_point_reloc.pos.x(), state_point_reloc.pos.y(), state_point_reloc.pos.z(),
                state_point_reloc.rot.x(), state_point_reloc.rot.y(), state_point_reloc.rot.z(), state_point_reloc.rot.w());
            relocalize_flag.store(false);
            flg_first_scan = true;
            continue;
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

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // LiDAR position in the world coordinate frame

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_PRINT_WARN("No point, skip this scan!");
                continue;
            }

            if (scan_frame_save_en) save_scan_frame(scan_frames_dir);

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
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
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_PRINT_WARN("No point, skip this scan!");
                continue;
            }

            // confidence cached by UndistortPcl inside p_imu->Process()
            const double static_confidence = use_zupt ? p_imu->get_static_confidence() : 0.0;

            // Adaptive LiDAR cov: rises with static confidence and previous-frame residual
            const double lidar_static_scale   = 1.0 + lidar_cov_static_scale * static_confidence;
            const double lidar_residual_scale = std::max(1.0, res_mean_last / lidar_residual_ref);
            const double adaptive_lidar_cov   = std::min(
                LASER_POINT_COV * lidar_static_scale * lidar_residual_scale, 0.1);
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            if(feature_pub_en) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; 

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(adaptive_lidar_cov, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();
            
            if (sam_enable) {
                const Eigen::Matrix3d R_odom_map_prev = R_map_odom.transpose();
                const Eigen::Matrix3d R_odom_base_before_sam = R_odom_map_prev * state_point.rot.toRotationMatrix();
                const Eigen::Vector3d t_odom_base_before_sam = R_odom_map_prev * (state_point.pos - t_map_odom);

                getCurrPose(state_point);
                getCurrOffset(state_point);
                saveKeyFramesAndFactor(feats_undistort);
                const Eigen::Matrix3d R_map_base =
                    (Eigen::AngleAxisd(transformTobeMapped[2], Eigen::Vector3d::UnitZ()) *
                     Eigen::AngleAxisd(transformTobeMapped[1], Eigen::Vector3d::UnitY()) *
                     Eigen::AngleAxisd(transformTobeMapped[0], Eigen::Vector3d::UnitX())).toRotationMatrix();
                const Eigen::Vector3d t_map_base(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]);
                R_map_odom = R_map_base * R_odom_base_before_sam.transpose();
                t_map_odom = t_map_base - R_map_odom * t_odom_base_before_sam;
                update_state_ikfom(); // Update current state_point
                correctPoses();

                publishSamMsg();
            }

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped, pubOdomAftMappedGlobal);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || feat_accum_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            if (effect_pub_en) publish_effect_world(pubLaserCloudEffect);
            if (feature_pub_en) publish_map(pubLaserCloudMap);
            /*** Debug variables ***/
        }

        rate.sleep();
    }            

    /**************** save accumulated feature scans ****************/
    /* 1. make sure you have enough memories
    /* 2. PCD saving will largely influence real-time performance **/
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
        string global_keyframe_path = root_dir + "/KEY_FRAMES/global.pcd";
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

    return 0;
}
