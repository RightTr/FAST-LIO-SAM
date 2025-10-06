#define private public
#include <ikd-Tree/ikd_Tree.h>
#undef private

template<typename PointType>
class KD_TREE_PUBLIC : public KD_TREE<PointType>
{
    public:
        using KD_TREE<PointType>::delete_tree_nodes; 
};