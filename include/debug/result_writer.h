#ifndef RESULT_WRITER_H
#define RESULT_WRITER_H

#include "common_lib.h"
#include "global_graph.h"

#include <fstream>
#include <string>
#include <vector>

template<typename PointT>
class KD_TREE_PUBLIC;

class ResultWriter
{
public:
    static ResultWriter &instance();

    void saveAccumulatedCloud(const PointCloudXYZI &cloud, double timestamp) const;
    void saveKeyframePoses(const std::vector<PointTypePose> &poses,
                           std::ofstream &file) const;
    bool saveGlobalCloud(GlobalGraph *graph) const;
    bool saveTree(KD_TREE_PUBLIC<PointType> &tree) const;
    void saveKeyframeCloud(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, double timestamp);

private:
    ResultWriter();
    std::string accumulatedCloudDir_;
    std::string keyframeScanDir_;
    std::string globalKeyframePath_;
    std::string ikdtreeCloudPath_;
    std::string ikdtreePath_;
};

#endif
