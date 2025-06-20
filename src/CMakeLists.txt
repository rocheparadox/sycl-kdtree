cmake_minimum_required(VERSION 3.26)
project(sycl-kdtree)

message(STATUS "Setting icpx as the main compiler")
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_STANDARD 14)

find_package(oneDPL REQUIRED)

set(TARGET_NAME khalid_kd_tree)
set(SOURCE_FILE "main.cpp")
set(COMPILE_AUX_FILES "../include/khalid_kdtree_sycl.hpp" "../include/khalid_nearest_neighbor.hpp")
set(SYCL_LINKER "-fsycl")

if(DEFINED VERTEX_COUNT)
    message(STATUS "Number of vertex count is given as ${VERTEX_COUNT}")
    set(VERTEX_COUNT_FLAG "-DVERTEX_COUNT=${VERTEX_COUNT}")
endif ()

include_directories(/opt/intel/oneapi/2024.0/include)
include_directories(/home/brazenparadox/Documents/paradox_custom_libs)

add_executable(${TARGET_NAME}.${VERTEX_COUNT} ${SOURCE_FILE} ${COMPILE_AUX_FILES})
set_target_properties(${TARGET_NAME}.${VERTEX_COUNT} PROPERTIES COMPILE_FLAGS ${SYCL_LINKER})
set_target_properties(${TARGET_NAME}.${VERTEX_COUNT} PROPERTIES LINK_FLAGS ${SYCL_LINKER})

set(NVIDIA_GPU_COMPILE_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

## This section compiles for nvidia-gpu test for a single point
set(NVIDIA_SOURCE_FILE test_nvidia.cpp)
set(NVIDIA_TARGET ${TARGET_NAME}.${VERTEX_COUNT}.nvidia-gpu)
add_executable(${NVIDIA_TARGET} EXCLUDE_FROM_ALL ${NVIDIA_SOURCE_FILE} ${AUX_FILES})
set_target_properties(${NVIDIA_TARGET} PROPERTIES COMPILE_FLAGS "-fsycl  ${NVIDIA_GPU_COMPILE_FLAGS} ${VERTEX_COUNT_FLAG}" )
set_target_properties(${NVIDIA_TARGET} PROPERTIES LINK_FLAGS "-fsycl ${NVIDIA_GPU_COMPILE_FLAGS}")
add_custom_target(nvidiagpu-single-point DEPENDS ${NVIDIA_TARGET})


## This section compiles for nvidia-gpu correspondence for plyfiles
set(NVIDIA_CORRESPONDENCE_SOURCE_FILE nvidia_correspondence.cpp)
SET(NVIDIA_CORRESPONDENCE_TARGET ${TARGET_NAME}${VERTEX_COUNT}.nvidiagpu-plyfile)
add_executable(${NVIDIA_CORRESPONDENCE_TARGET} EXCLUDE_FROM_ALL ${NVIDIA_CORRESPONDENCE_SOURCE_FILE} ${AUX_FILES})
set_target_properties(${NVIDIA_CORRESPONDENCE_TARGET} PROPERTIES COMPILE_FLAGS "-fsycl  ${NVIDIA_GPU_COMPILE_FLAGS} ${VERTEX_COUNT_FLAG}" )
set_target_properties(${NVIDIA_CORRESPONDENCE_TARGET} PROPERTIES LINK_FLAGS "-fsycl ${NVIDIA_GPU_COMPILE_FLAGS}")
add_custom_target(nvidiagpu-plyfile DEPENDS ${NVIDIA_CORRESPONDENCE_TARGET})