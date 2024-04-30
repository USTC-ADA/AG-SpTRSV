#include "../include/AG-SpTRSV.h"
#include "template.h"
#include <cuda_runtime.h>

#pragma once

// In this implementation, we assume that
// diagonal elements of the matrix are explicitly stored, 
// together with off-diagonal elements in CSR format

template <typename T, int subwarp_size>
__global__ void SpTRSV_code_subwarp(int info_start_n, int info_end_n, subwarpinfo<subwarp_size> *info, 
            const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
            const T* b, T* x, int *get_value, int tb_size, int tb_num)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    //int tb_size = blockDim.x;
    int warp_pb = tb_size / WARP_SIZE;

    int wid = tid / WARP_SIZE;
    int global_wid = bid * warp_pb + wid;
    int lane_id = tid % WARP_SIZE;

    extern __shared__ int s[];
    int *ColIdx_buf = (int*)&s[0];
    T *csrValue_buf = (T*)&s[BUF_SIZE * tb_size];

    int sw_num = subwarp_size;
    int sw_size = WARP_SIZE / sw_num;
    if (lane_id >= sw_num * sw_size) return;
    int sw_id = lane_id / sw_size;
    int sw_off = lane_id % sw_size;

    if (tid >= tb_size) return;

    for (int info_id = global_wid + info_start_n; info_id < info_end_n; info_id += tb_num * tb_size / WARP_SIZE)
    {
        if (info_id >= info_end_n) continue;
        subwarpinfo<subwarp_size> current_info = info[info_id];

        #define READ_FLAG_TEMPLATE READ_FLAG
        #define READ_FENCE_TEMPLATE READ_FENCE
        #define WRITE_FLAG_TEMPLATE WRITE_FLAG
        #include "SpTRSV_code_subwarp.cu"
        #undef READ_FLAG_TEMPLATE
        #undef READ_FENCE_TEMPLATE
        #undef WRITE_FLAG_TEMPLATE
    }

}

template <typename T, int subwarp_size>
__global__ void SpTRSV_code_subwarp_nosync(int info_start_n, int info_end_n, subwarpinfo<subwarp_size> *info, 
            const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
            const T* b, T* x, int *get_value, int tb_size, int tb_num)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    //int tb_size = blockDim.x;
    int warp_pb = tb_size / WARP_SIZE;

    int wid = tid / WARP_SIZE;
    int global_wid = bid * warp_pb + wid;
    int lane_id = tid % WARP_SIZE;

    extern __shared__ int s[];
    int *ColIdx_buf = (int*)&s[0];
    T *csrValue_buf = (T*)&s[BUF_SIZE * tb_size];

    int sw_num = subwarp_size;
    int sw_size = WARP_SIZE / sw_num;
    if (lane_id >= sw_num * sw_size) return;
    int sw_id = lane_id / sw_size;
    int sw_off = lane_id % sw_size;

    if (tid >= tb_size) return;

    for (int info_id = global_wid + info_start_n; info_id < info_end_n; info_id += tb_num * tb_size / WARP_SIZE)
    {
        if (info_id >= info_end_n) continue;
        subwarpinfo<subwarp_size> current_info = info[info_id];

        #define READ_FLAG_TEMPLATE NO_READ_FLAG
        #define READ_FENCE_TEMPLATE NO_READ_FENCE
        #define WRITE_FLAG_TEMPLATE WRITE_FLAG_NOFENCE
        #include "SpTRSV_code_subwarp.cu"
        #undef READ_FLAG_TEMPLATE
        #undef READ_FENCE_TEMPLATE
        #undef WRITE_FLAG_TEMPLATE
    }
}

template <typename T, int subwarp_size>
__global__ void SpTRSV_code_subwarp_determ(subwarpinfo<subwarp_size> **info, int *info_n,
            const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
            const T* b, T* x, int *get_value, int tb_size)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    //int tb_size = tb_size;
    int total_warp_num = tb_size * BLOCK_NUM;
    int warp_pb = tb_size / WARP_SIZE;

    int wid = tid / WARP_SIZE;
    int global_wid = bid * warp_pb + wid;
    int lane_id = tid % WARP_SIZE;

    extern __shared__ int s[];
    int *ColIdx_buf = (int*)&s[0];
    T *csrValue_buf = (T*)&s[BUF_SIZE * tb_size];

    int sw_num = subwarp_size;
    int sw_size = WARP_SIZE / sw_num;
    if (lane_id >= sw_num * sw_size) return;
    int sw_id = lane_id / sw_size;
    int sw_off = lane_id % sw_size;

    if (tid >= tb_size) return;
    if (global_wid >= total_warp_num) return;

    //if (!lane_id && global_wid < 5) printf("len %d\n", info_n[global_wid]);

    for (int info_id = 0; info_id < info_n[global_wid]; info_id++)
    {
        subwarpinfo<subwarp_size> current_info = info[global_wid][info_id];

        #define READ_FLAG_TEMPLATE READ_FLAG
        #define READ_FENCE_TEMPLATE READ_FENCE
        #define WRITE_FLAG_TEMPLATE WRITE_FLAG
        #include "SpTRSV_code_subwarp.cu"
        #undef READ_FLAG_TEMPLATE
        #undef READ_FENCE_TEMPLATE
        #undef WRITE_FLAG_TEMPLATE
    }
}

// template __global__ void SpTRSV_code_subwarp<subwarpinfo_2>(int info_start_n, int info_end_n, subwarpinfo_2 *info, 
//             const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
//             const T* b, T* x, int *get_value);

// template __global__ void SpTRSV_code_subwarp<subwarpinfo_4>(int info_start_n, int info_end_n, subwarpinfo_4 *info, 
//             const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
//             const T* b, T* x, int *get_value);

// template __global__ void SpTRSV_code_subwarp<subwarpinfo_8>(int info_start_n, int info_end_n, subwarpinfo_8 *info, 
//             const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
//             const T* b, T* x, int *get_value);

// template __global__ void SpTRSV_code_subwarp<subwarpinfo_16>(int info_start_n, int info_end_n, subwarpinfo_16 *info, 
//             const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
//             const T* b, T* x, int *get_value);