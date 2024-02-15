//
// Created by brazenparadox on 10/02/24.
//

#include<iostream>
#include<vector>
#include<CL/sycl.hpp>
#include "../include/khalid_kdtree_sycl.hpp"
#include "chrono"
#include<happly.h>
#include "../include/khalid_nearest_neighbor.hpp"


#ifdef VERTEX_COUNT
const int vertex_count = VERTEX_COUNT;
#else
const int vertex_count = 453;
#endif


auto selector = sycl::default_selector_v;
sycl::queue device_queue(selector);

using Point = khalid::Point3D<float>;
using Node3D = khalid::Node<Point>;

std::vector<Point> get_pcl_from_plyfile(std::string plyfile_location){

    std::vector<Point> pointcloud;

    happly::PLYData plyIn(plyfile_location);
    std::vector<float> x = plyIn.getElement("vertex").getProperty<float>("x");
    std::vector<float> y = plyIn.getElement("vertex").getProperty<float>("y");
    std::vector<float> z = plyIn.getElement("vertex").getProperty<float>("z");

    for(int col=0; col<x.size(); col++){
        Point point;
        point.x = x[col];
        point.y = y[col];
        point.z = z[col];

        pointcloud.push_back(point);
    }

    return pointcloud;
}

int main(){

    std::string dataview_location = "/home/brazenparadox/bunnies/rotated_bun_zipper_res4_10degs.ply";
    std::string modelview_location = "/home/brazenparadox/bunnies/bun_zipper_res4.ply";
    std::vector<Point> modelview_pcl = get_pcl_from_plyfile(modelview_location);
    std::vector<Point> dataview_pcl = get_pcl_from_plyfile(dataview_location);
    std::vector<Node3D> modelview_tree;

    std::cout << "\nThe device is " << device_queue.get_device().get_info<sycl::info::device::name>() << " and the vertex count is " << vertex_count << "\n\n";

    for(int idx=0; idx<modelview_pcl.size(); idx++){
        Node3D node;
        node.x = modelview_pcl[idx].x;
        node.y = modelview_pcl[idx].y;
        node.z = modelview_pcl[idx].z;
        node.lla_index = 0;
        modelview_tree.push_back(node);
    }

    int total_tree_levels = khalid::get_total_level(vertex_count);
    std::cout <<"\nThe total tree level is " << total_tree_levels << "\n\n\n" << std::flush;

    Node3D* device_tree = sycl::malloc_shared<Node3D>(vertex_count, device_queue);
    Point* dataview = sycl::malloc_shared<Point>(vertex_count, device_queue);
    Node3D* tree_host = modelview_tree.data();
    Point* dataview_host = dataview_pcl.data();

    //std::cout<< "\nsize mv: " << modelview.size();

    device_queue.submit([&](sycl::handler &hndlr){
        hndlr.memcpy(device_tree, tree_host, sizeof(Node3D) * vertex_count);
    });

    device_queue.submit([&](sycl::handler &hndlr) {
        hndlr.memcpy(dataview, dataview_host, sizeof(Point) * vertex_count);
    });
    device_queue.wait();

    //std::cout << "\n\nModelview tree and dataview points are copied to the device memory\n";
    khalid::build_kdtree(device_tree, vertex_count, 3, device_queue);
    //std::cout << "\n\nKD-tree for modelview points built successfully\n";

/*    for(int idx=0; idx<modelview.size(); idx++)
        std::cout << device_tree[idx].point << " " << device_tree[idx].lla_index << "\n" << std::flush;*/

    int* nearest_neighbour_index = sycl::malloc_shared<int>(vertex_count, device_queue);
    get_nearest_neighbor_kernel(device_tree, dataview, vertex_count, nearest_neighbour_index, device_queue);
    for(int idx=0; idx<dataview_pcl.size(); idx++){
        std::cout << "\nThe correspondent point of " << dataview[idx] << " is " << device_tree[nearest_neighbour_index[idx]] << std::flush; //" with euc distance " << kdtree_correspondent_points[idx].euc_distance <<std::flush;
    }

    std::cout << "\n\n" << std::flush;
    return 0;
}

