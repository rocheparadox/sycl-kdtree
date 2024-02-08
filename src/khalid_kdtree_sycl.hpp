//
// Created by brazenparadox on 01/02/24.
//

#ifndef KDTREE_SYCL_KHALID_KDTREE_SYCL_H
#define KDTREE_SYCL_KHALID_KDTREE_SYCL_H

#include<tbb/task.h>
#include<CL/sycl.hpp>
#include<cmath>
#include<algorithm>
#include<oneapi/dpl/algorithm>
#include<chrono>
//#include "khalid_nearest_neighbor.h"


namespace khalid{

    template<typename PointDatatype>
    struct Point2D{
        PointDatatype x;
        PointDatatype y;

        // Overload the subscript operator
        PointDatatype& operator[](int index) {
            if (index == 0) {
                return x;
            } else if (index == 1) {
                return y;
            }
            else{
                return x;
            }
        }

        friend std::ostream& operator<<(std::ostream& os, const Point2D& point) {
            os << point.x << " " << point.y << " ";
            return os;
        }
    };

    template<typename PointDatatype>
    struct Point3D{
        PointDatatype x;
        PointDatatype y;
        PointDatatype z;

        // Overload the subscript operator
        PointDatatype& operator[](int index) {
            if (index == 0) {
                return x;
            } else if (index == 1) {
                return y;
            }
            else{
                return z;
            }
        }

        friend std::ostream& operator<<(std::ostream& os, const Point3D& point) {
            os << point.x << " " << point.y << " "  << point.z << " ";
            return os;
        }
    };

    template<typename Point>
    struct Node{
        Point point;
        int lla_index;
    };


    template<typename Point>
    bool less_than_compare(Node<Point> node_a, Node<Point> node_b, int dim){
        return node_a.lla_index < node_b.lla_index || node_a.lla_index == node_b.lla_index && \
        node_a.point[dim] < node_b.point[dim];
    }

    inline int r_child(int index){
        return (index << 1 ) + 2;
    }

    inline int l_child(int index){
        return (index << 1 ) + 1;
    }

    inline int parent(int index){
        return (index - 1)/2;
    }

    inline int number_of_nodes_in_level(int level){
        return (1 << level) - 1;
    }


    int tree_level_of_node(int index){
        return 31 - sycl::clz(index + 1);
    }

    int get_total_level(int vertex_count){
        return 31 - __builtin_clz(vertex_count) + 1;
    }

    int subtree_size_of_node(int index, int total_level, int current_level, int vertex_count){

        int fllc_s = ~((~index) << (total_level - current_level - 1));
        return (1 << (total_level - current_level - 1)) - 1 + \
        std::min(std::max(0, vertex_count - fllc_s), 1 << (total_level - current_level - 1));
    }

    int segment_s_begin(int index, int total_level, int current_level, int vertex_count){
        int number_of_left_siblings = index - ((1 << current_level) - 1);
        return ((1 << current_level) - 1 ) \
            + number_of_left_siblings * ((1 << (total_level - current_level - 1)) -1) + \
                    std::min(number_of_left_siblings*(1 << (total_level - current_level - 1)),
                    vertex_count - (( 1 << (total_level - 1)) - 1 ));
    }

    template<typename PointDatatype>
    void update_lla_index_tags(Node<PointDatatype>* tree, int total_level,  int current_level, int vertex_count,
                               sycl::queue device_queue) {

        device_queue.submit([&] (sycl::handler &hndlr){
            hndlr.parallel_for(vertex_count, [=](sycl::id<1> idx){

            int _idx = static_cast<int>(idx);
            if (!(_idx >= vertex_count || _idx < number_of_nodes_in_level(current_level))) {
                int pivot_pos = segment_s_begin(tree[_idx].lla_index, total_level, current_level, vertex_count) +
                                subtree_size_of_node(l_child(tree[_idx].lla_index), total_level, current_level + 1, vertex_count);
                //std::cout << "\nThe pivot pos for array index " << idx << " i.e points " << tree[idx].point << " is " << pivot_pos << std::flush;
                if (_idx < pivot_pos) {
                    tree[idx].lla_index = l_child(tree[_idx].lla_index);
                } else if (_idx > pivot_pos) {
                    tree[idx].lla_index = r_child(tree[_idx].lla_index);
                }
            }
            });
        });

        //std::cout << "\n\nThe lla is updated!!!\n\n" << std::flush;
    }

    template<typename PointDatatype>
    void build_kdtree(Node<PointDatatype>* device_tree, int total_tree_levels, int vertex_count,
                      int dimensions, sycl::queue device_queue){

        auto start = std::chrono::steady_clock::now();
        for(int idx=0; idx<static_cast<int>(std::log2(vertex_count)); idx++){

            // sort according to the dimension
            int sort_dimension = idx%dimensions;
            std::sort(device_tree, device_tree + vertex_count,
                 [=](const Node<PointDatatype> node_a, const Node<PointDatatype> node_b){
                     return less_than_compare(node_a, node_b, sort_dimension);});

            /*if (++idx >= std::log2(vertex_count))
                break;*/

            /*std::cout << "\nThe idx is " << idx << " and the dimension is " << sort_dimension << " " << std::flush;
            std::cout << "\n\nSorted based on dimension " << sort_dimension <<"\n" <<std::flush;*/

/*            for(int idx=0; idx<tree.size(); idx++)
                std::cout << device_tree[idx].point << " " << device_tree[idx].lla_index << "\n" << std::flush;*/

            update_lla_index_tags(device_tree, total_tree_levels, idx, vertex_count, device_queue);
            device_queue.wait();

/*            std::cout << "\n\nlla updated\n\n" << std::flush;*/
        }

        std::sort(device_tree, device_tree + vertex_count,
                  [=](const Node<PointDatatype> node_a, const Node<PointDatatype> node_b){
                      return less_than_compare(node_a, node_b, 0);});

        auto end = std::chrono::steady_clock::now();
        auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

        //std::cout << "\nTime taken to build the kd tree is " << time_taken << " milliseconds" << std::flush;

/*        for(int idx=0; idx<tree.size(); idx++)
            std::cout << device_tree[idx].point << " " << device_tree[idx].lla_index << "\n" << std::flush;*/
    }
}


#endif //KDTREE_SYCL_KHALID_KDTREE_SYCL_H
