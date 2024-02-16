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

namespace khalid {
    template<typename ModelviewDatatype, typename DataviewDatatype>
    struct Correspondence {
        DataviewDatatype *dataview_point;
        ModelviewDatatype *modelview_point;
        float euclidean_distance;
    };
}


template<typename PointDatatype>
float euclidean_distance(khalid::Point3D<PointDatatype> pointa, khalid::Point3D<PointDatatype> pointb){
    return sqrtf(powf(pointa.x - pointb.x, 2) + powf(pointa.y - pointb.y, 2) + powf(pointa.z - pointb.z, 2));
}


template<typename Point>
int get_nearest_neighbor(khalid::Node<Point>* device_tree, Point query_point, int vertex_count, int dimensions) {

    bool traverse_down = true;
    int nearest_neighbor_index;
    int level = 0;
    float nearest_neighbor_distance = -1;
    int search_index = 0; // root node
    int sorting_dimension;
    int previous_search_index;

    //int iteration = 0;

    while (true) {

        sorting_dimension = level % dimensions;
        //std::cout << "\nLE: " <<level << " TD: " << traverse_down << " SI: " << search_index << " SD: " << sorting_dimension;
        if(traverse_down){
            // check if this is the leaf node.
            if(khalid::l_child(search_index) > vertex_count && khalid::r_child(search_index) > vertex_count ){
                //std::cout << " :: Going to traverse up from next iteration." << std::flush;
                traverse_down = false;
                continue;
            }
            else{
                // traverse through the tree
                if (query_point[sorting_dimension] < device_tree[search_index][sorting_dimension]) {
                    //std::cout << " and we are going to left child" << std::flush;
                    search_index = khalid::l_child(search_index);
                } else {
                    search_index = khalid::r_child(search_index);
                    //std::cout << " and we are going to right child" << std::flush;
                }
                level++;
            }
        }
        else{

            //std::cout << " Previous index is " << previous_search_index << std::flush;
            // traverse up
            float euc_distance = euclidean_distance(device_tree[search_index], query_point);
            if (euc_distance < nearest_neighbor_distance || nearest_neighbor_distance == -1) {
                nearest_neighbor_distance = euc_distance;
                nearest_neighbor_index = search_index;
            }

            if (khalid::l_child(search_index) < vertex_count && khalid::r_child(search_index) < vertex_count &&
                static_cast<float>(abs(query_point[sorting_dimension] - device_tree[search_index][sorting_dimension])) < nearest_neighbor_distance) {
                /*        std::cout << "\n\n-------XXX Node with point " << current_node_point[splitting_dim] <<
                        " has lower axis distance XXX--------- axis distance : " << std::abs(query_point[splitting_dim] - current_node_point[splitting_dim]) <<
                        " min distance: " << *min_distance <<std::flush;*/

                // set search index to farther child
                int farther_child;
                if (query_point[sorting_dimension] < device_tree[search_index][sorting_dimension]) {
                    farther_child = khalid::r_child(search_index);
                } else {
                    farther_child = khalid::l_child(search_index);
                }

                //std::cout << " FC: " << farther_child << std::flush;

                if(farther_child == previous_search_index){
                    // keep traversing up
                    previous_search_index = search_index;
                    search_index = khalid::parent(search_index);
                    level--;
                }
                else {
                    //std::cout << " We are taking a pivot at index " << search_index << " due to hyperplane with farther child at index " << farther_child;
                    search_index = farther_child;
                    traverse_down = true;
                    level++;
                }

                // check if we are trying to traverse through the child we have already done.

            }

            else {
                previous_search_index = search_index;
                search_index = khalid::parent(search_index);
                level--;
            }

            if(search_index == 0){
                break;
            }
        }
    }
    //std::cout << "\n\nThe nearest neighbour index is " << nearest_neighbor_index << std::flush;
    return nearest_neighbor_index;
}

template<typename Point, typename Node>
void get_nearest_neighbor_kernel(Node* modelview_device_tree, Point* dataview,
                                 int vertex_count, int* nearest_neighbour_index, sycl::queue device_queue
){
    device_queue.submit([&] (sycl::handler &hndlr){
        hndlr.parallel_for(vertex_count, [=] (sycl::id<1> idx){
            nearest_neighbour_index[idx] = get_nearest_neighbor(modelview_device_tree, dataview[idx], vertex_count, 3);
        });
    });

    device_queue.wait();
/*        for(int idx=0; idx<vertex_count; idx++)
            std::cout << "\nThe nearest Neighbor of " << dataview[idx] << " is " <<
                modelview_device_tree[nearest_neighbor_indices[idx]].point << std::flush;*/
}

template<typename Node, typename Point>
void get_nearest_neighbor_kernel(Node* modelview_device_tree, Point* dataview,
                                 khalid::Correspondence<Node, Point>* correspondences,
                                 int vertex_count, sycl::queue device_queue){

    device_queue.submit([&] (sycl::handler &hndlr){
        hndlr.parallel_for(vertex_count, [=] (sycl::id<1> idx){
            correspondences[idx].dataview_point = &dataview[idx];
            int nearest_neighbour_index = get_nearest_neighbor(modelview_device_tree, dataview[idx], vertex_count, 3);
            correspondences[idx].modelview_point = &modelview_device_tree[nearest_neighbour_index];
            correspondences[idx].euclidean_distance = euclidean_distance(modelview_device_tree[nearest_neighbour_index], dataview[idx]);
        });
    });

    device_queue.wait();
/*        for(int idx=0; idx<vertex_count; idx++)
            std::cout << "\nThe nearest Neighbor of " << dataview[idx] << " is " <<
                modelview_device_tree[nearest_neighbor_indices[idx]].point << std::flush;*/
}

#endif //KHALID_KD_TREE_KHALID_NEAREST_NEIGHBOR_HPP
