#define private public
#include "ikd_Tree.h"
#undef private
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

template<typename PointType>
class KD_TREE_PUBLIC : public KD_TREE<PointType>
{
    public:
        using KD_TREE<PointType>::delete_tree_nodes;
        using Ptr = std::shared_ptr<KD_TREE_PUBLIC<PointType>>;
        using Node = typename KD_TREE<PointType>::KD_TREE_NODE;

        int Add_Point(const PointType& point)
        {
            typename KD_TREE<PointType>::PointVector points;
            points.push_back(point);
            return this->Add_Points(points, false);
        }

        bool SaveIkdtree(const std::string &path)
        {
            if (path.empty() || this->Root_Node == nullptr)
                return false;

            std::ofstream out(path, std::ios::binary);
            if (!out.is_open())
                return false;

            IkdtreeHeader header = {{'I', 'K', 'D', 'P', 'R', 'I', 'O', 'R'}, 1,
                                    static_cast<uint32_t>(sizeof(PointType)),
                                    CountNodes(this->Root_Node)};
            out.write(reinterpret_cast<const char *>(&header), sizeof(header));
            WriteNode(out, this->Root_Node);
            return out.good();
        }

        bool LoadIkdtree(const std::string &path)
        {
            if (path.empty())
                return false;

            std::ifstream in(path, std::ios::binary);
            if (!in.is_open())
                return false;

            IkdtreeHeader header;
            in.read(reinterpret_cast<char *>(&header), sizeof(header));
            const char expected_magic[8] = {'I', 'K', 'D', 'P', 'R', 'I', 'O', 'R'};
            if (!in.good() || std::memcmp(header.magic, expected_magic, sizeof(expected_magic)) != 0 ||
                header.version != 1 || header.point_size != sizeof(PointType))
            {
                return false;
            }

            Node *root = ReadNode(in);
            if (!in.good() || root == nullptr || CountNodes(root) != header.node_count)
            {
                DeleteStaticTree(root);
                return false;
            }

            this->delete_tree_nodes(&this->Root_Node);
            RefreshStaticNode(root, nullptr);
            this->Root_Node = root;
            return true;
        }

    private:
        struct IkdtreeHeader
        {
            char magic[8];
            uint32_t version;
            uint32_t point_size;
            uint64_t node_count;
        };

        static uint64_t CountNodes(const Node *node)
        {
            if (node == nullptr)
                return 0;
            return 1 + CountNodes(node->left_son_ptr) + CountNodes(node->right_son_ptr);
        }

        static void DeleteStaticTree(Node *node)
        {
            if (node == nullptr)
                return;
            DeleteStaticTree(node->left_son_ptr);
            DeleteStaticTree(node->right_son_ptr);
            delete node;
        }

        static void WriteNode(std::ofstream &out, const Node *node)
        {
            const uint8_t has_node = node != nullptr ? 1 : 0;
            out.write(reinterpret_cast<const char *>(&has_node), sizeof(has_node));
            if (node == nullptr)
                return;

            out.write(reinterpret_cast<const char *>(&node->point), sizeof(PointType));
            out.write(reinterpret_cast<const char *>(&node->division_axis), sizeof(node->division_axis));
            WriteNode(out, node->left_son_ptr);
            WriteNode(out, node->right_son_ptr);
        }

        static Node *ReadNode(std::ifstream &in)
        {
            uint8_t has_node = 0;
            in.read(reinterpret_cast<char *>(&has_node), sizeof(has_node));
            if (!in.good() || has_node == 0)
                return nullptr;

            Node *node = new Node;
            node->left_son_ptr = nullptr;
            node->right_son_ptr = nullptr;
            node->father_ptr = nullptr;
            in.read(reinterpret_cast<char *>(&node->point), sizeof(PointType));
            in.read(reinterpret_cast<char *>(&node->division_axis), sizeof(node->division_axis));
            node->left_son_ptr = ReadNode(in);
            node->right_son_ptr = ReadNode(in);
            return node;
        }

        static void RefreshStaticNode(Node *node, Node *father)
        {
            if (node == nullptr)
                return;

            node->father_ptr = father;
            RefreshStaticNode(node->left_son_ptr, node);
            RefreshStaticNode(node->right_son_ptr, node);

            node->TreeSize = 1;
            node->invalid_point_num = 0;
            node->down_del_num = 0;
            node->point_deleted = false;
            node->tree_deleted = false;
            node->point_downsample_deleted = false;
            node->tree_downsample_deleted = false;
            node->need_push_down_to_left = false;
            node->need_push_down_to_right = false;
            node->working_flag = false;
            pthread_mutex_init(&(node->push_down_mutex_lock), NULL);

            float min_x = node->point.x, max_x = node->point.x;
            float min_y = node->point.y, max_y = node->point.y;
            float min_z = node->point.z, max_z = node->point.z;

            if (node->left_son_ptr != nullptr)
            {
                node->TreeSize += node->left_son_ptr->TreeSize;
                min_x = std::min(min_x, node->left_son_ptr->node_range_x[0]);
                max_x = std::max(max_x, node->left_son_ptr->node_range_x[1]);
                min_y = std::min(min_y, node->left_son_ptr->node_range_y[0]);
                max_y = std::max(max_y, node->left_son_ptr->node_range_y[1]);
                min_z = std::min(min_z, node->left_son_ptr->node_range_z[0]);
                max_z = std::max(max_z, node->left_son_ptr->node_range_z[1]);
            }
            if (node->right_son_ptr != nullptr)
            {
                node->TreeSize += node->right_son_ptr->TreeSize;
                min_x = std::min(min_x, node->right_son_ptr->node_range_x[0]);
                max_x = std::max(max_x, node->right_son_ptr->node_range_x[1]);
                min_y = std::min(min_y, node->right_son_ptr->node_range_y[0]);
                max_y = std::max(max_y, node->right_son_ptr->node_range_y[1]);
                min_z = std::min(min_z, node->right_son_ptr->node_range_z[0]);
                max_z = std::max(max_z, node->right_son_ptr->node_range_z[1]);
            }

            node->node_range_x[0] = min_x;
            node->node_range_x[1] = max_x;
            node->node_range_y[0] = min_y;
            node->node_range_y[1] = max_y;
            node->node_range_z[0] = min_z;
            node->node_range_z[1] = max_z;

            const float x_l = (max_x - min_x) * 0.5f;
            const float y_l = (max_y - min_y) * 0.5f;
            const float z_l = (max_z - min_z) * 0.5f;
            node->radius_sq = x_l * x_l + y_l * y_l + z_l * z_l;
            node->alpha_del = 0.0f;
            node->alpha_bal = 0.0f;
            if (node->TreeSize > 1)
            {
                const int left_size = node->left_son_ptr != nullptr ? node->left_son_ptr->TreeSize : 0;
                const int right_size = node->right_son_ptr != nullptr ? node->right_son_ptr->TreeSize : 0;
                node->alpha_bal = static_cast<float>(std::max(left_size, right_size)) /
                                  static_cast<float>(node->TreeSize - 1);
            }
        }
};
