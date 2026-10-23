#ifndef DEBUG_INFO_H
#define DEBUG_INFO_H

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

    explicit DebugInfo(size_t = 100) {}
    void record(const Scan &) {}
};

#endif
