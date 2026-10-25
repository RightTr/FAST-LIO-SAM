#include "lidarBA.h"
#include "balm/voxel_map.hpp"
#include "common_lib.h"
#include "ros/ros_utils.h"

#include <Eigen/Cholesky>
#include <gtsam/navigation/PreintegrationParams.h>
#include <pcl/filters/voxel_grid.h>
#include <algorithm>
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
    explicit Impl(const LidarBAConfig &cfg, const Eigen::Vector3d &gravity)
        : config(cfg), gravityUp(-gravity),
          factor(std::max(2, cfg.window_size)) {
        config.window_size=std::max(2, config.window_size);
        config.thread_num=std::max(1, config.thread_num);
        config.max_iterations=std::max(1,config.max_iterations);
        config.max_layer=std::clamp(config.max_layer,0,3);
        config.min_points=std::max(1,config.min_points);
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
        slots.resize(config.window_size);
        for (int i=0;i<config.window_size;++i) slots[i]=i;
        lidar_ba::mp=slots.data();
        windows.resize(config.thread_num);
        imuParams=gtsam::PreintegrationParams::MakeSharedU(G_m_s2);
        imuParams->gyroscopeCovariance=Eigen::Matrix3d::Identity()*config.gyr_cov;
        imuParams->accelerometerCovariance=Eigen::Matrix3d::Identity()*config.acc_cov;
        imuParams->integrationCovariance=Eigen::Matrix3d::Identity()*1e-8;
    }
    ~Impl() { releaseMap(map, windows[0]); lidar_ba::mp=nullptr; }

    void finalize(const BAFramePtr &frame, const BAState &s,
                  std::vector<BAFramePtr> &finalizedFrames) {
        frame->optimized_pose = toIsometry(s);
        frame->finalized.store(true, std::memory_order_release);
        finalizedFrames.push_back(frame);
    }

    bool process(const BAFramePtr &frame, Eigen::Isometry3d &pose,
                 std::vector<BAFramePtr> &finalizedFrames) {
        pose = Eigen::Isometry3d::Identity();
        finalizedFrames.clear();
        if (!frame || finished)
            return false;

        if (!config.enable) {
            BAState state;
            state.R = frame->odom_pose.linear();
            state.p = frame->odom_pose.translation();
            finalize(frame, state, finalizedFrames);
            return false;
        }

        const BAFrame &f = *frame;
        BAState state; state.R=f.odom_pose.linear(); state.p=f.odom_pose.translation();
        InertialState inertial;
        inertial.v=f.velocity; inertial.bg=f.bg; inertial.ba=f.ba;
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
        clouds.push_back(points); frames.push_back(frame);
        const int idx=static_cast<int>(poses.size())-1;
        LBA_PLV(3) world(points->size());
        for (size_t i=0;i<points->size();++i) world[i]=state.R*(*points)[i].pnt+state.p;
        if (!world.empty()) cut_voxel_multi(
            map, clouds.back(), idx, slideMap, config.window_size, world, windows);
        recutMap(slideMap,static_cast<int>(poses.size()),poses,factor,windows);
        for (size_t i = 0; i < frames.size() && i < poses.size(); ++i)
            frames[i]->optimized_pose = toIsometry(poses[i]);
        if (static_cast<int>(poses.size()) < config.window_size) {
            return false;
        }

        bool optimized=false;
        if (factor.plvec_voxels.size() >= 2) {
            Lidar_BA_Optimizer optimizer; optimizer.thd_num=config.thread_num;
            Eigen::MatrixXd hessian; std::vector<double> residuals;
            const std::vector<BAState> previous_poses=poses;
            const std::vector<InertialState> previous_imu_states=imuStates;
            if(config.imu_enable)
            {
                optimized=optimizer.damping_iter_joint(
                    poses,imuStates,gravityUp,factor,preintegrations,
                    G_m_s2,
                    config.bg_cov,config.ba_cov,
                    config.max_iterations);
            }
            else
            {
                optimized=optimizer.damping_iter(
                    poses,factor,&hessian,residuals,config.max_iterations,false);
            }
            if (!optimized)
            {
                poses=previous_poses; imuStates=previous_imu_states;
            }
        }
        if (optimized)
            pose = toIsometry(poses.back());
        for (size_t i = 0; i < frames.size() && i < poses.size(); ++i)
            frames[i]->optimized_pose = toIsometry(poses[i]);
        finalize(frames.front(), poses.front(), finalizedFrames);
        marginalize(slideMap,config.window_size,poses,factor,windows[0]);
        for (int &slot: slots) {
            ++slot;
            if (slot >= config.window_size) slot = 0;
        }
        poses.erase(poses.begin()); imuStates.erase(imuStates.begin());
        clouds.erase(clouds.begin()); frames.erase(frames.begin());
        if(!preintegrations.empty()) preintegrations.erase(preintegrations.begin());
        factor=LidarFactor(config.window_size);
        return optimized;
    }
    std::vector<BAFramePtr> Finish() {
        std::vector<BAFramePtr> finalizedFrames;
        if (finished)
            return finalizedFrames;
        finished = true;
        finalizedFrames.reserve(frames.size());
        for (size_t i = 0; i < frames.size() && i < poses.size(); ++i)
            finalize(frames[i], poses[i], finalizedFrames);
        frames.clear(); poses.clear(); imuStates.clear(); preintegrations.clear(); clouds.clear();
        return finalizedFrames;
    }

    LidarBAConfig config;
    std::vector<int> slots;
    std::vector<BAState> poses;
    std::vector<InertialState> imuStates;
    std::vector<ImuPreintegration> preintegrations;
    std::vector<PVecPtr> clouds; std::vector<BAFramePtr> frames;
    boost::shared_ptr<gtsam::PreintegrationParams> imuParams;
    gtsam::Unit3 gravityUp{Eigen::Vector3d::UnitZ()};
    std::unordered_map<lidar_ba::VOXEL_LOC,lidar_ba::OctoTree*> map, slideMap;
    std::vector<std::vector<lidar_ba::SlideWindow*>> windows;
    lidar_ba::LidarFactor factor;
    bool finished = false;
};

LidarBA::LidarBA(const LidarBAConfig &config, const Eigen::Vector3d &gravity)
    : impl_(new Impl(config, gravity)) {}
LidarBA::~LidarBA() = default;
bool LidarBA::Process(BAFramePtr frame, Eigen::Isometry3d &pose,
                      std::vector<BAFramePtr> &finalized) {
    return impl_->process(frame, pose, finalized);
}
std::vector<BAFramePtr> LidarBA::ActiveFrames() const { return impl_->frames; }
std::vector<BAFramePtr> LidarBA::Finish() { return impl_->Finish(); }
