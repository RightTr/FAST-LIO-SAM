#include "lidarBA.h"
#include "balm/voxel_map.hpp"
#include "ros_utils.h"

#include <Eigen/Cholesky>
#include <gtsam/navigation/PreintegrationParams.h>
#include <pcl/filters/voxel_grid.h>
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <thread>
#include <tuple>
#include <unordered_map>

using namespace lidar_ba;

namespace {
using namespace lidar_ba;

void recutMap(std::unordered_map<VOXEL_LOC, OctoTree*> &map, int count,
              std::vector<BAState> &poses, LidarFactor &factor,
              std::vector<std::vector<SlideWindow*>> &sws) {
    std::vector<OctoTree*> nodes; nodes.reserve(map.size());
    for (auto &entry : map) nodes.push_back(entry.second);
    const size_t nthreads=std::min(nodes.size(),sws.size());
    if (nthreads < 2) {
        for (OctoTree *node : nodes) node->recut(count,poses,sws[0]);
    } else {
        std::vector<std::thread> workers;
        for(size_t t=0;t<nthreads;++t) workers.emplace_back([&,t]{
            const size_t begin=nodes.size()*t/nthreads, end=nodes.size()*(t+1)/nthreads;
            for(size_t i=begin;i<end;++i) nodes[i]->recut(count,poses,sws[t]);
        });
        for(auto &worker:workers) worker.join();
    }
    for (size_t i=1; i<sws.size(); ++i) {
        sws[0].insert(sws[0].end(), sws[i].begin(), sws[i].end());
        sws[i].clear();
    }
    for (auto &entry : map) entry.second->tras_opt(factor);
}

void marginalize(std::unordered_map<VOXEL_LOC, OctoTree*> &map, int count,
                 std::vector<BAState> &poses, const LidarFactor &factor,
                 std::vector<SlideWindow*> &sw) {
    for (auto it=map.begin(); it!=map.end();) {
        OctoTree *node=it->second;
        node->margi(count, 1, poses, factor);
        if (node->isexist) ++it;
        else { node->clear_slwd(sw); it=map.erase(it); }
    }
}

void releaseMap(std::unordered_map<VOXEL_LOC, OctoTree*> &map,
                std::vector<SlideWindow*> &sw) {
    for (auto &entry : map) {
        std::vector<OctoTree*> children;
        entry.second->tras_ptr(children);
        entry.second->clear_slwd(sw);
        delete entry.second;
        for (OctoTree *child : children) delete child;
    }
    map.clear();
    for (SlideWindow *window : sw) delete window;
    sw.clear();
}

Eigen::Isometry3d toIsometry(const BAState &state) {
    Eigen::Isometry3d out=Eigen::Isometry3d::Identity();
    out.linear()=state.R; out.translation()=state.p; return out;
}

} // namespace

struct LidarBA::Impl {
    explicit Impl(const LidarBAConfig &cfg) : config(cfg) {
        config.window_size=std::max(2, config.window_size);
        config.thread_num=std::max(1, config.thread_num);
        config.max_iterations=std::max(1,config.max_iterations);
        config.max_layer=std::clamp(config.max_layer,0,3);
        config.min_points=std::max(3,config.min_points);
        config.max_fixed_points=std::max(config.min_points,config.max_fixed_points);
        if(!(config.voxel_size>0.0)) config.voxel_size=1.0;
        lidar_ba::thread_num=config.thread_num;
        lidar_ba::voxel_size=config.voxel_size;
        lidar_ba::max_layer=config.max_layer;
        lidar_ba::max_points=config.max_fixed_points;
        lidar_ba::min_ba_point=config.min_points;
        lidar_ba::min_eigen_value=config.min_eigen_value;
        lidar_ba::plane_eigen_value_thre=config.plane_eigenvalue_threshold;
        if (lidar_ba::plane_eigen_value_thre.empty())
            lidar_ba::plane_eigen_value_thre.push_back(0.25);
        while (lidar_ba::plane_eigen_value_thre.size() <= size_t(config.max_layer))
            lidar_ba::plane_eigen_value_thre.push_back(lidar_ba::plane_eigen_value_thre.back());
        lidar_ba::min_point.setConstant(config.min_points);
        for (size_t i=0; i<config.min_layer_points.size() && i<4; ++i)
            lidar_ba::min_point[i]=static_cast<int>(config.min_layer_points[i]);
        window_size=config.window_size;
        slots.resize(window_size);
        for (int i=0;i<window_size;++i) slots[i]=i;
        lidar_ba::mp=slots.data();
        windows.resize(config.thread_num);
        factor=LidarFactor(window_size);
        imuParams=gtsam::PreintegrationParams::MakeSharedU(9.81);
        imuParams->gyroscopeCovariance=Eigen::Matrix3d::Identity()*config.gyr_cov;
        imuParams->accelerometerCovariance=Eigen::Matrix3d::Identity()*config.acc_cov;
        imuParams->integrationCovariance=Eigen::Matrix3d::Identity()*1e-8;
        worker=std::thread(&Impl::run,this);
    }
    ~Impl() { finish(); releaseMap(map, windows[0]); lidar_ba::mp=nullptr; }

    void push(BAFrame f) {
        if (!f.cloud) f.cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        std::lock_guard<std::mutex> lock(mutex);
        if (stopping) return;
        input.push_back(std::move(f)); cv.notify_one();
    }
    bool pushAndWait(BAFrame f, std::vector<int> &window_ids,
                     std::vector<Eigen::Matrix4d> &window_poses,
                     bool &window_optimized) {
        if (!f.cloud) f.cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        const int id=f.id;
        std::unique_lock<std::mutex> lock(mutex);
        if (stopping) return false;
        input.push_back(std::move(f));
        cv.notify_one();
        cv.wait(lock,[&]{return stopping||completedInputId>=id;});
        if (completedInputId<id) return false;
        window_ids=completedWindowIds;
        window_poses=completedWindowPoses;
        window_optimized=completedWindowOptimized;
        return true;
    }
    bool tryPop(int &id, Eigen::Isometry3d &pose,
                Eigen::Vector3d &gravity_up_odom, double &gravity_sigma) {
        std::lock_guard<std::mutex> lock(mutex);
        if (output.empty()) return false;
        std::tie(id, pose, gravity_up_odom, gravity_sigma) = output.front();
        output.pop_front(); return true;
    }
    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (joined) return;
            stopping=true;
        }
        cv.notify_all();
        if (worker.joinable()) worker.join();
        joined=true;
    }
    void publish(const BAFrame &f, const BAState &s) {
        const int id=f.id;
        const Eigen::Isometry3d pose=toIsometry(s);
        const Eigen::Vector3d gravity_up_odom=gravityInitialized
            ? gravityUp.point3() : (-f.gravity_odom).normalized();
        std::lock_guard<std::mutex> lock(mutex);
        output.emplace_back(id, pose, gravity_up_odom, gravitySigma);
    }
    void process(BAFrame f) {
        BAState state; state.R=f.odom_pose.linear(); state.p=f.odom_pose.translation();
        InertialState inertial;
        inertial.v=f.velocity; inertial.bg=f.bg; inertial.ba=f.ba;
        if(!gravityInitialized)
        {
            gravityUp=gtsam::Unit3(-f.gravity_odom);
            gravityInitialized=true;
        }
        if(!poses.empty() && config.imu_enable)
        {
            const gtsam::imuBias::ConstantBias biasHat(
                imuStates.back().ba,imuStates.back().bg);
            ImuPreintegration edge;
            edge.pim=std::make_shared<gtsam::PreintegratedImuMeasurements>(
                imuParams,biasHat);
            for(size_t i=1;i<f.imu.size();++i)
            {
                const double dt=f.imu[i].t-f.imu[i-1].t;
                edge.pim->integrateMeasurement(
                    0.5*(f.imu[i-1].acc+f.imu[i].acc),
                    0.5*(f.imu[i-1].gyr+f.imu[i].gyr),dt);
            }
            preintegrations.push_back(std::move(edge));
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr source(new pcl::PointCloud<pcl::PointXYZI>());
        source->reserve(f.cloud->size());
        for (const auto &p : f.cloud->points)
            if (std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)) source->push_back(p);
        pcl::PointCloud<pcl::PointXYZI> filtered;
        if (config.point_leaf_size > 0.0 && !source->empty()) {
            pcl::VoxelGrid<pcl::PointXYZI> downsample;
            const float leaf=static_cast<float>(config.point_leaf_size);
            downsample.setLeafSize(leaf,leaf,leaf); downsample.setInputCloud(source); downsample.filter(filtered);
        } else filtered=*source;
        PVecPtr points(new PVec()); points->reserve(filtered.size());
        for (const auto &p : filtered.points) {
            pointVar pv; pv.pnt=Eigen::Vector3d(p.x,p.y,p.z);
            pv.var=Eigen::Matrix3d::Identity()*1e-4; points->push_back(pv);
        }
        poses.push_back(state); imuStates.push_back(inertial);
        clouds.push_back(points); frames.push_back(std::move(f));
        const int idx=static_cast<int>(poses.size())-1;
        LBA_PLV(3) world(points->size());
        for (size_t i=0;i<points->size();++i) world[i]=state.R*(*points)[i].pnt+state.p;
        if (!world.empty()) cut_voxel_multi(map,clouds.back(),idx,slideMap,window_size,world,windows);
        recutMap(slideMap,static_cast<int>(poses.size()),poses,factor,windows);
        if (static_cast<int>(poses.size()) < window_size) {
            saveWindowSnapshot(false);
            return;
        }

        bool optimized=false;
        if (factor.plvec_voxels.size() >= 2) {
            Lidar_BA_Optimizer optimizer; optimizer.thd_num=config.thread_num;
            Eigen::MatrixXd hessian; std::vector<double> residuals;
            const std::vector<BAState> previous_poses=poses;
            const std::vector<InertialState> previous_imu_states=imuStates;
            const gtsam::Unit3 previous_gravity=gravityUp;
            if(config.imu_enable)
            {
                optimized=optimizer.damping_iter_joint(
                    poses,imuStates,gravityUp,gravityPrior,factor,preintegrations,
                    gravityMagnitude,config.gravity_prior_sigma_deg*M_PI/180.0,
                    config.bg_cov,config.ba_cov,config.gravity_optimize,
                    config.max_iterations);
                gravitySigma=optimizer.last_gravity_sigma;
            }
            else
            {
                optimized=optimizer.damping_iter(
                    poses,factor,&hessian,residuals,config.max_iterations,false);
            }
            if (!optimized)
            {
                poses=previous_poses; imuStates=previous_imu_states;
                gravityUp=previous_gravity;
            }
        }
        saveWindowSnapshot(optimized);
        publish(frames.front(),poses.front());
        marginalize(slideMap,window_size,poses,factor,windows[0]);
        for (int &slot: slots) { ++slot; if(slot>=window_size) slot=0; }
        poses.erase(poses.begin()); imuStates.erase(imuStates.begin());
        clouds.erase(clouds.begin()); frames.erase(frames.begin());
        if(!preintegrations.empty()) preintegrations.erase(preintegrations.begin());
        factor=LidarFactor(window_size);
    }
    void saveWindowSnapshot(bool optimized) {
        completedWindowIds.clear();
        completedWindowPoses.clear();
        completedWindowIds.reserve(frames.size());
        completedWindowPoses.reserve(poses.size());
        for (size_t i=0; i<frames.size() && i<poses.size(); ++i) {
            completedWindowIds.push_back(frames[i].id);
            completedWindowPoses.push_back(toIsometry(poses[i]).matrix());
        }
        completedWindowOptimized=optimized;
    }
    void run() {
        for (;;) {
            BAFrame f;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock,[&]{return stopping||!input.empty();});
                if(input.empty()&&stopping) break;
                f=std::move(input.front());input.pop_front();
            }
            if(config.enable) process(f);
            else {
                BAState s; s.R=f.odom_pose.linear();s.p=f.odom_pose.translation();publish(f,s);
                completedWindowIds.assign(1,f.id);
                completedWindowPoses.assign(1,f.odom_pose.matrix());
                completedWindowOptimized=false;
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                completedInputId=f.id;
            }
            cv.notify_all();
        }
        // A partial final window cannot use the fixed-size upstream Hessian safely.
        // Emit every remaining frame in order with its EKF pose.
        for(size_t i=0;i<frames.size();++i) publish(frames[i],poses[i]);
        frames.clear();poses.clear();imuStates.clear();preintegrations.clear();clouds.clear();
    }

    LidarBAConfig config; int window_size=10; double gravityMagnitude=9.81; std::vector<int> slots;
    std::vector<BAState> poses;
    std::vector<InertialState> imuStates;
    std::vector<ImuPreintegration> preintegrations;
    std::vector<PVecPtr> clouds; std::vector<BAFrame> frames;
    boost::shared_ptr<gtsam::PreintegrationParams> imuParams;
    gtsam::Unit3 gravityUp{Eigen::Vector3d::UnitZ()};
    gtsam::Unit3 gravityPrior{Eigen::Vector3d::UnitZ()};
    bool gravityInitialized=false;
    std::unordered_map<lidar_ba::VOXEL_LOC,lidar_ba::OctoTree*> map, slideMap;
    std::vector<std::vector<lidar_ba::SlideWindow*>> windows;
    lidar_ba::LidarFactor factor{10};
    std::mutex mutex; std::condition_variable cv; std::deque<BAFrame> input;
    std::deque<std::tuple<int, Eigen::Isometry3d, Eigen::Vector3d, double>> output;
    double gravitySigma=5.0*M_PI/180.0;
    std::vector<int> completedWindowIds;
    std::vector<Eigen::Matrix4d> completedWindowPoses;
    int completedInputId=-1;
    bool completedWindowOptimized=false;
    bool stopping=false,joined=false; std::thread worker;
};

LidarBA::LidarBA(const LidarBAConfig &config) : impl_(new Impl(config)) {}
LidarBA::~LidarBA() = default;
void LidarBA::push(BAFrame frame) { impl_->push(std::move(frame)); }
bool LidarBA::pushAndWait(BAFrame frame, std::vector<int> &window_ids,
                          std::vector<Eigen::Matrix4d> &window_poses,
                          bool &window_optimized) {
    return impl_->pushAndWait(std::move(frame),window_ids,window_poses,window_optimized);
}
bool LidarBA::tryPop(int &id, Eigen::Isometry3d &pose,
                     Eigen::Vector3d &gravity_up_odom, double &gravity_sigma) {
    return impl_->tryPop(id, pose, gravity_up_odom, gravity_sigma);
}
void LidarBA::finish() { impl_->finish(); }
