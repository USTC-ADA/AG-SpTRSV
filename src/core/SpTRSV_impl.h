#include "../include/AG-SpTRSV.h"
#include "template.h"
#include <cuda_runtime.h>

// In this implementation, we assume that
// diagonal elements of the matrix are explicitly stored, 
// together with off-diagonal elements in CSR format
template<typename T>
__global__ void SpTRSV_code_sync(int info_start_n, int info_end_n, warpinfo *info, 
            const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
            const T* b, T* x, int *get_value)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    int global_wid = bid * WARP_NUM_PER_BLOCK + wid;
    int lane_id = tid % WARP_SIZE;

    extern __shared__ int s[];
    int *ColIdx_buf = (int*)&s[0];
    T *csrValue_buf = (T*)&s[BUF_SIZE * THREAD_NUM_PER_BLOCK];

    if (tid >= THREAD_NUM_PER_BLOCK) return;
    if (global_wid + info_start_n >= info_end_n) return;

    // for (int schedule_i = 0; schedule_i < level[global_wid]; schedule_i++)
    // {
        warpinfo current_info = info[info_start_n + global_wid];

        int row_st = current_info.row_st;
        int row_ed = current_info.row_ed;
        
        int using_shared_mem = current_info.info & 0x10;

        #define READ_FLAG_TEMPLATE READ_FLAG
        #define READ_FENCE_TEMPLATE READ_FENCE
        #define WRITE_FLAG_TEMPLATE WRITE_FLAG
        #include "SpTRSV_code.cu"
        #undef READ_FLAG_TEMPLATE
        #undef READ_FENCE_TEMPLATE
        #undef WRITE_FLAG_TEMPLATE

    // }
}

// In this implementation, we assume that
// diagonal elements of the matrix are explicitly stored, 
// together with off-diagonal elements in CSR format
template<typename T>
__global__ void SpTRSV_code_nosync(int info_start_n, int info_end_n, warpinfo *info, 
            const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
            const T* b, T* x, int *get_value)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    int global_wid = bid * WARP_NUM_PER_BLOCK + wid;
    int lane_id = tid % WARP_SIZE;

    extern __shared__ int s[];
    int *ColIdx_buf = (int*)&s[0];
    T *csrValue_buf = (T*)&s[BUF_SIZE * THREAD_NUM_PER_BLOCK];

    if (tid >= THREAD_NUM_PER_BLOCK) return;
    if (global_wid + info_start_n >= info_end_n) return;

    // for (int schedule_i = 0; schedule_i < level[global_wid]; schedule_i++)
    // {
        warpinfo current_info = info[info_start_n + global_wid];

        int row_st = current_info.row_st;
        int row_ed = current_info.row_ed;
        
        int using_shared_mem = current_info.info & 0x10;

        #define READ_FLAG_TEMPLATE NO_READ_FLAG
        #define READ_FENCE_TEMPLATE NO_READ_FENCE
        #define WRITE_FLAG_TEMPLATE NO_WRITE_FLAG
        #include "SpTRSV_code.cu"
        #undef READ_FLAG_TEMPLATE
        #undef READ_FENCE_TEMPLATE
        #undef WRITE_FLAG_TEMPLATE

    // }
}
