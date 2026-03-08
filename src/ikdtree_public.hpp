#define private public
#include <ikd-Tree/ikd_Tree.h>
#undef private

template<typename PointType>
class KD_TREE_PUBLIC : public KD_TREE<PointType>
{
    public:
        using KD_TREE<PointType>::delete_tree_nodes;
        using Ptr = std::shared_ptr<KD_TREE_PUBLIC<PointType>>;

        int Add_Point(const PointType& point)
        {
            typename KD_TREE<PointType>::PointVector points;
            points.push_back(point);
            return this->Add_Points(points, false);
        }
};