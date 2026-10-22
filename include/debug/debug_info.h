#ifndef DEBUG_INFO_H
#define DEBUG_INFO_H

#include "ros/ros_utils.h"

#include <algorithm>
#include <cstddef>

class DebugInfo
{
public:
    struct Scan
    {
        double ekfMs = 0.0;
        double baMs = 0.0;
        double scanMs = 0.0;
        bool optimized = false;
        bool feedbackApplied = false;
        double correctionRotationDeg = 0.0;
        double correctionPositionMeters = 0.0;
        size_t lidarBacklog = 0;
        bool hasMapResidual = false;
        double mapResidual = 0.0;
    };

    explicit DebugInfo(size_t interval = 100) : interval_(interval) {}

    void record(const Scan &scan)
    {
        ++scans_;
        backlogPeak_ = std::max(backlogPeak_, scan.lidarBacklog);
        ekfTimeSumMs_ += scan.ekfMs;
        baTimeSumMs_ += scan.baMs;
        scanTimeSumMs_ += scan.scanMs;
        baWorstMs_ = std::max(baWorstMs_, scan.baMs);
        optimized_ += scan.optimized ? 1 : 0;

        if (scan.feedbackApplied)
        {
            ++feedbackCount_;
            feedbackMaxRotationDeg_ = std::max(
                feedbackMaxRotationDeg_, scan.correctionRotationDeg);
            feedbackMaxPositionMeters_ = std::max(
                feedbackMaxPositionMeters_, scan.correctionPositionMeters);
        }
        if (scan.hasMapResidual && scan.mapResidual > 0.0)
        {
            mapResidualSum_ += scan.mapResidual;
            ++mapResidualCount_;
        }

        if (interval_ == 0 || scans_ < interval_)
            return;

        ROS_PRINT_INFO("[Tracking] EKF=%.2f ms Scan=%.2f ms backlog=%zu peak=%zu",
                       ekfTimeSumMs_ / interval_, scanTimeSumMs_ / interval_,
                       scan.lidarBacklog, backlogPeak_);
        ROS_PRINT_INFO("[LocalBA] Process=%.2f ms Optimized=%zu/%zu Worst=%.2f ms",
                       baTimeSumMs_ / interval_, optimized_, interval_, baWorstMs_);
        ROS_PRINT_INFO("[Feedback] corrections=%zu dR=%.3f deg dP=%.3f m map_residual=%.4f",
                       feedbackCount_, feedbackMaxRotationDeg_,
                       feedbackMaxPositionMeters_,
                       mapResidualCount_ ? mapResidualSum_ / mapResidualCount_ : 0.0);
        reset();
    }

private:
    void reset()
    {
        scans_ = 0;
        optimized_ = 0;
        backlogPeak_ = 0;
        feedbackCount_ = 0;
        ekfTimeSumMs_ = 0.0;
        baTimeSumMs_ = 0.0;
        scanTimeSumMs_ = 0.0;
        baWorstMs_ = 0.0;
        feedbackMaxRotationDeg_ = 0.0;
        feedbackMaxPositionMeters_ = 0.0;
        mapResidualSum_ = 0.0;
        mapResidualCount_ = 0;
    }

    size_t interval_;
    size_t scans_ = 0;
    size_t optimized_ = 0;
    size_t backlogPeak_ = 0;
    size_t feedbackCount_ = 0;
    size_t mapResidualCount_ = 0;
    double ekfTimeSumMs_ = 0.0;
    double baTimeSumMs_ = 0.0;
    double scanTimeSumMs_ = 0.0;
    double baWorstMs_ = 0.0;
    double feedbackMaxRotationDeg_ = 0.0;
    double feedbackMaxPositionMeters_ = 0.0;
    double mapResidualSum_ = 0.0;
};

#endif
