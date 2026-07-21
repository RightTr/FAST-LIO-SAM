#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/conversions.h>

#include "ikdtree_public.h"

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "usage: " << argv[0] << " input.pcd [output.bin]\n";
        return 1;
    }

    const std::filesystem::path input_path(argv[1]);
    std::filesystem::path output_path;
    if (argc == 3)
    {
        output_path = argv[2];
    }
    else
    {
        output_path = input_path;
        output_path.replace_extension(".bin");
    }

    using PointT = pcl::PointXYZINormal;

    pcl::PCLPointCloud2 cloud_blob;
    if (pcl::io::loadPCDFile(input_path.string(), cloud_blob) != 0)
    {
        std::cerr << "failed to load pcd: " << input_path << "\n";
        return 1;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    cloud->reserve(cloud_blob.width * cloud_blob.height);

    auto field_offset = [&](const std::string &name) -> int {
        for (const auto &field : cloud_blob.fields)
        {
            if (field.name == name)
                return static_cast<int>(field.offset);
        }
        return -1;
    };

    const int x_off = field_offset("x");
    const int y_off = field_offset("y");
    const int z_off = field_offset("z");
    const int i_off = field_offset("intensity");
    const int nx_off = field_offset("normal_x");
    const int ny_off = field_offset("normal_y");
    const int nz_off = field_offset("normal_z");
    const int c_off = field_offset("curvature");

    if (x_off < 0 || y_off < 0 || z_off < 0)
    {
        std::cerr << "pcd is missing x/y/z fields\n";
        return 1;
    }

    const std::size_t point_count = static_cast<std::size_t>(cloud_blob.width) * cloud_blob.height;
    for (std::size_t idx = 0; idx < point_count; ++idx)
    {
        const std::uint8_t *base = cloud_blob.data.data() + idx * cloud_blob.point_step;
        PointT pt;

        std::memcpy(&pt.x, base + x_off, sizeof(float));
        std::memcpy(&pt.y, base + y_off, sizeof(float));
        std::memcpy(&pt.z, base + z_off, sizeof(float));
        if (i_off >= 0) std::memcpy(&pt.intensity, base + i_off, sizeof(float));
        if (nx_off >= 0) std::memcpy(&pt.normal_x, base + nx_off, sizeof(float));
        if (ny_off >= 0) std::memcpy(&pt.normal_y, base + ny_off, sizeof(float));
        if (nz_off >= 0) std::memcpy(&pt.normal_z, base + nz_off, sizeof(float));
        if (c_off >= 0) std::memcpy(&pt.curvature, base + c_off, sizeof(float));
        cloud->push_back(pt);
    }

    std::vector<int> valid_index;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, valid_index);
    if (cloud->empty())
    {
        std::cerr << "pcd has no valid points\n";
        return 1;
    }

    auto *tree = new KD_TREE_PUBLIC<PointT>();
    typename KD_TREE_PUBLIC<PointT>::PointVector points;
    points.reserve(cloud->size());
    for (const auto &pt : cloud->points)
    {
        points.push_back(pt);
    }

    tree->Build(points);
    if (tree->Root_Node == nullptr)
    {
        std::cerr << "failed to build ikdtree from pcd\n";
        return 1;
    }

    if (!output_path.parent_path().empty())
    {
        std::filesystem::create_directories(output_path.parent_path());
    }
    if (!tree->SaveStaticSnapshot(output_path.string()))
    {
        std::cerr << "failed to save ikdtree snapshot: " << output_path << "\n";
        return 1;
    }

    std::cout << "saved ikdtree snapshot: " << output_path
              << " (nodes=" << tree->validnum() << ")\n";
    return 0;
}
