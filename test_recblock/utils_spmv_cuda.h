#ifndef _UTILS_SPMV_CUDA_
#define _UTILS_SPMV_CUDA_

#include "common.h"
#include <cuda_runtime.h>
#include "cusparse.h"
#include "utils_reordering.h"

#define LONGROW_THRESHOLD 2048
#define SHORTROW_THRESHOLD 8

typedef struct SpMV_block
{
    int method;
    int num_threads;
    int num_blocks;
    int num_threads_l;
    int num_blocks_l;
    int m;
    int *d_csrRowPtr_l;
    int *d_csrColIdx_l;
    VALUE_TYPE *d_csrVal_l;
    VALUE_TYPE *d_x;
    VALUE_TYPE *d_y;
    int m_new;
    int *d_longrow_idx;
    int longrow;
} SpMV_block;

__global__ void spmv_longrow_csr_cuda_executor(const int *d_csrRowPtr,
                                               const int *d_csrColIdx,
                                               const VALUE_TYPE *d_csrVal,
                                               const VALUE_TYPE *d_x,
                                               VALUE_TYPE *d_y,
                                               const int longrow,
                                               const int *d_longrow_idx)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    for (int i = 0; i < longrow; i++)
    {
        const int rowid = d_longrow_idx[i];
        const int start = d_csrRowPtr[rowid];
        const int stop = d_csrRowPtr[rowid + 1];
        const int len = stop - start;
        VALUE_TYPE sum = 0;

        if (global_id < len)
        {
            sum = d_x[d_csrColIdx[start + global_id]] * d_csrVal[start + global_id];
        }

        sum = sum_32_shfl(sum);
        if (lane_id == 0 && sum != 0)
            atomicAdd(&d_y[rowid], sum);
    }
}

__global__ void spmv_threadsca_csr_cuda_executor(const int *d_csrRowPtr,
                                                 const int *d_csrColIdx,
                                                 const VALUE_TYPE *d_csrVal,
                                                 const int m,
                                                 const VALUE_TYPE *d_x,
                                                 VALUE_TYPE *d_y)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < m)
    {
        const int rowid = global_id;
        const int start = d_csrRowPtr[rowid] - d_csrRowPtr[0];
        const int stop = d_csrRowPtr[rowid + 1] - d_csrRowPtr[0];
        VALUE_TYPE sum = 0;
        if (stop - start <= LONGROW_THRESHOLD)
        {
            for (int j = start; j < stop; j++)
                sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
        }
        d_y[rowid] = sum;
    }
}

__global__ void spmv_threadsca_dcsr_cuda_executor(const int *d_csrRowPtr,
                                                  const int *d_csrColIdx,
                                                  const VALUE_TYPE *d_csrVal,
                                                  const int m,
                                                  const VALUE_TYPE *d_x,
                                                  VALUE_TYPE *d_y,
                                                  const int *d_row_perm)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < m)
    {
        const int rowid = global_id;
        const int start = d_csrRowPtr[rowid] - d_csrRowPtr[0];
        const int stop = d_csrRowPtr[rowid + 1] - d_csrRowPtr[0];
        VALUE_TYPE sum = 0;
        if (stop - start <= LONGROW_THRESHOLD)
        {
            for (int j = start; j < stop; j++)
                sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
        }
        d_y[d_row_perm[rowid]] = sum;
    }

    // const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    // if (!global_id)
    // {
    //     for (int i = 0; i < m; i++)
    //     {
    //         const int rowid = i;
    //         const int start = d_csrRowPtr[rowid] - d_csrRowPtr[0];
    //         const int stop = d_csrRowPtr[rowid + 1] - d_csrRowPtr[0];
    //         VALUE_TYPE sum = 0;
    //         if (stop - start <= LONGROW_THRESHOLD)
    //         {
    //             for (int j = start; j < stop; j++)
    //                 sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
    //         }
    //         d_y[d_row_perm[rowid]] = sum;
    //         printf("id = %d  sum = %.1lf\n", d_row_perm[rowid], sum);
    //     }
    // }
}

__global__ void spmv_warpvec_csr_cuda_executor(const int *d_csrRowPtr,
                                               const int *d_csrColIdx,
                                               const VALUE_TYPE *d_csrVal,
                                               const int m,
                                               const VALUE_TYPE *d_x,
                                               VALUE_TYPE *d_y)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    const int rowid = global_id / WARP_SIZE;
    if (rowid >= m)
        return;

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int start = d_csrRowPtr[rowid] - d_csrRowPtr[0];
    const int stop = d_csrRowPtr[rowid + 1] - d_csrRowPtr[0];
    if (start == stop)
    {
        if (!lane_id)
            d_y[rowid] = 0;
        return;
    }

    VALUE_TYPE sum = 0;
    if (stop - start <= LONGROW_THRESHOLD)
    {
        for (int j = start + lane_id; j < stop; j += WARP_SIZE)
        {
            sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
        }
        sum = sum_32_shfl(sum);
    }

    //finish
    if (!lane_id)
        d_y[rowid] = sum;
}

__global__ void spmv_warpvec_dcsr_cuda_executor(const int *d_csrRowPtr,
                                                const int *d_csrColIdx,
                                                const VALUE_TYPE *d_csrVal,
                                                const int m,
                                                const VALUE_TYPE *d_x,
                                                VALUE_TYPE *d_y,
                                                const int *d_row_perm)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    const int rowid = global_id / WARP_SIZE;
    if (rowid >= m)
        return;

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int start = d_csrRowPtr[rowid] - d_csrRowPtr[0];
    const int stop = d_csrRowPtr[rowid + 1] - d_csrRowPtr[0];
    if (start == stop)
    {
        if (!lane_id)
            d_y[rowid] = 0;
        return;
    }

    VALUE_TYPE sum = 0;
    if (stop - start <= LONGROW_THRESHOLD)
    {
        for (int j = start + lane_id; j < stop; j += WARP_SIZE)
        {
            sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
        }
        sum = sum_32_shfl(sum);
    }

    //finish
    if (!lane_id)
        d_y[d_row_perm[rowid]] = sum;
}

__global__ void subKernel(VALUE_TYPE *b,
                          VALUE_TYPE *y,
                          int m)
{
    const int global_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (global_id < m)
        b[global_id] = b[global_id] - y[global_id];
}

#endif
