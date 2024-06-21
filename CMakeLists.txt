cmake_minimum_required(VERSION 3.5)

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc")

project(SpTRSV LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_FLAGS "-O3 -w -m64 -Xptxas -dlcm=cg")

include_directories(${PROJECT_SOURCE_DIR}/include)

set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(SRC_FILE ${SRC_DIR}/common.cu
${SRC_DIR}/GPU_setup.cpp
${SRC_DIR}/schedule.cu
${SRC_DIR}/scheduling_new.cu
${SRC_DIR}/preprocessing.cu
${SRC_DIR}/partition_row.cu
${SRC_DIR}/partition_level.cu
${SRC_DIR}/preprocessing_new.cu
${SRC_DIR}/finalize.cu
${SRC_DIR}/SpTRSV_executor.cu
${SRC_DIR}/SpTRSV_executor_variant.cu
${SRC_DIR}/specialization.cpp
${SRC_DIR}/format_def.cpp
${SRC_DIR}/transformation.cu
${SRC_DIR}/elimination.cpp
${SRC_DIR}/search.cu
)

set(UTILS_DIR ${PROJECT_SOURCE_DIR}/utils)
set(UTILS_FILE ${UTILS_DIR}/utils.cpp)

cuda_add_library(SpTRSV ${SRC_FILE})
add_library(utils ${UTILS_FILE})
