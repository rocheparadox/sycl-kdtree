//
// Created by brazenparadox on 02/02/24.
//

#ifndef KHALID_KD_TREE_KHALID_NEAREST_NEIGHBOR_HPP
#define KHALID_KD_TREE_KHALID_NEAREST_NEIGHBOR_HPP

#include "khalid_kdtree_sycl.hpp"


/*------------------------- Nearest Neighbor ------------------------*/

/*
template<typename PointDatatype, typename Point>
void get_nearest_neighbor_kernel(khalid::Node<Point>* device_tree, Point  query_point, int total_tree_level,
                                 int vertex_count, sycl::queue device_queue){
    int *nearest_neighbor_index  = sycl::malloc_shared<int>(1, device_queue);
    device_queue.submit([&] (sycl::handler &hndlr){
        hndlr.parallel_for(vertex_count, [=] (sycl::id<1> idx){
            *nearest_neighbor_index = get_nearest_neighbor(device_tree, query_point, total_tree_level);
        });
    });

    device_queue.wait();
    std::cout << "\nThe nearest Neighbor of " << query_point << " is " << device_tree[*nearest_neighbor_index].point << std::flush;
}
*/

template<typename PointDatatype>
void get_nearest_neighbor_kernel(Node<PointDatatype>* modelview_device_tree, PointDatatype* dataview,
                                 int total_tree_level, int vertex_count, sycl::queue device_queue){

    int* nearest_neighbor_indices = sycl::malloc_shared<int>(vertex_count, device_queue);

    device_queue.submit([&] (sycl::handler &hndlr){
        hndlr.parallel_for(vertex_count, [=] (sycl::id<1> idx){
            nearest_neighbor_indices[idx] = get_nearest_neighbor(modelview_device_tree, dataview[idx], total_tree_level);
        });
    });

    device_queue.wait();
/*        for(int idx=0; idx<vertex_count; idx++)
            std::cout << "\nThe nearest Neighbor of " << dataview[idx] << " is " <<
                modelview_device_tree[nearest_neighbor_indices[idx]].point << std::flush;*/
}

template<typename PointDatatype>
float euclidean_distance(khalid::Point3D<PointDatatype> pointa, khalid::Point3D<PointDatatype> pointb){
    return sqrtf(powf(pointa.x - pointb.x, 2) + powf(pointa.y - pointb.y, 2) + powf(pointa.z - pointb.z, 2));
}

template<typename Point>
int get_nearest_neighbor(Node<Point>* device_tree, Point query_point, int total_tree_levels,
                         int dimensions, int size_of_tree) {
    bool traverse_down = true;
    int level = 0;
    int nearest_neighbor_index;
    float nearest_neighbor_distance = -1;
    int search_index = 0; // root node
    while (true) {
        // TODO: check if the node is leaf node

        int sorting_dimension = level % dimensions;
        float euc_distance = euclidean_distance(device_tree[search_index].point, query_point);
        if (euc_distance < nearest_neighbor_distance || nearest_neighbor_distance == -1) {
            nearest_neighbor_distance = euc_distance;
            nearest_neighbor_index = search_index;
        }
        // traverse through the tree
        if (query_point[sorting_dimension] < device_tree[search_index].point[sorting_dimension]) {
            search_index = khalid::l_child(search_index);
        } else {
            search_index = khalid::r_child(search_index);
        }

        //std::cout << "\n\nlevel: " << level << " , search index: " << search_index << " nearest neighbor index : " << nearest_neighbor_index \
                            << " nearest n dist: " << nearest_neighbor_distance << " euc distance : " << euc_distance << std::flush;
        if (++level == total_tree_levels)
            break;
    }
    return nearest_neighbor_index;
}

#endif //KHALID_KD_TREE_KHALID_NEAREST_NEIGHBOR_HPP
