#ifndef _UTILS_SPTRSV_CUDA_
#define _UTILS_SPTRSV_CUDA_

#include "common.h"
#include <cuda_runtime.h>
#include "cusparse.h"
#include "utils_reordering.h"

typedef struct SpTRSV_block
{
    int method;
    int num_threads;
    int num_blocks;
    int m;
    int nnzTR;
    int substitution;
    VALUE_TYPE *d_b;
    VALUE_TYPE *d_x;
    cusparseSolvePolicy_t policy;
    cusparseOperation_t trans;
    cusparseHandle_t handle;
    double alpha_double;
    float alpha_float;
    cusparseMatDescr_t descr;
    csrsv2Info_t info;
    void *pBuffer;
    int m_lv;
    int offset;
    int *d_graphInDegree;
    VALUE_TYPE *d_left_sum;
    int *d_while_profiler;
    int *d_id_extractor;
    int *d_levelItem;
    int nlv;
    int *levelPtr;
    int *m_lv_array;
    int *offset_array;
    int *nnz_lv_array;
} SpTRSV_block;

__global__ void sptrsv_syncfree_csc_cuda_analyser(const int *d_cscRowIdx,
                                                  const int m,
                                                  const int nnz,
                                                  int *d_graphInDegree)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (global_id < nnz)
    {
        atomicAdd(&d_graphInDegree[d_cscRowIdx[global_id]], 1);
    }
}

__global__ void sptrsv_syncfree_warpvec_csc_cuda_executor(const int *d_cscColPtr,
                                                          const int *d_cscRowIdx,
                                                          const VALUE_TYPE *d_cscVal,
                                                          int *d_graphInDegree,
                                                          VALUE_TYPE *d_left_sum,
                                                          const int m,
                                                          const int substitution,
                                                          const VALUE_TYPE *d_b,
                                                          VALUE_TYPE *d_x,
                                                          int *d_while_profiler,
                                                          int *d_id_extractor,
                                                          int *d_levelItem)
{
    // Initialize
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl_sync(0xffffffff, global_x_id, 0);

    if (global_x_id >= m)
        return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? global_x_id : m - 1 - global_x_id;

    // Prefetch
    const int colstart = d_cscColPtr[global_x_id] - d_cscColPtr[0];
    const int colstop = d_cscColPtr[global_x_id + 1] - d_cscColPtr[0];
    const int pos = substitution == SUBSTITUTION_FORWARD ? colstart : colstop - 1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];

    const int perm_id = global_x_id;
    const VALUE_TYPE valb = d_b[perm_id];

    // Consumer
    do
    {
        __threadfence_block();
    } while (d_graphInDegree[perm_id] != 1);

    VALUE_TYPE xi = d_left_sum[perm_id];
    xi = (valb - xi) * coef;

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? colstart + 1 : colstart;
    const int stop_ptr = substitution == SUBSTITUTION_FORWARD ? colstop : colstop - 1;
    for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const int rowIdx = d_cscRowIdx[j];

        atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
        __threadfence();

        atomicSub(&d_graphInDegree[rowIdx], 1);
    }

    //finish
    if (!lane_id)
        d_x[perm_id] = xi;
}

__global__ void sptrsv_levelset_threadsca_csr_cuda_executor_fasttrack(const int *d_csrRowPtr,
                                                                      const int *d_csrColIdx,
                                                                      const VALUE_TYPE *d_csrVal,
                                                                      const int m,
                                                                      const int m_total,
                                                                      const int offset,
                                                                      const int substitution,
                                                                      const VALUE_TYPE *d_b,
                                                                      VALUE_TYPE *d_x)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
    {
        int rowidx = global_x_id + offset;
        rowidx = substitution == SUBSTITUTION_FORWARD ? rowidx : m_total - 1 - global_x_id - offset;
        const int pos = substitution == SUBSTITUTION_FORWARD ? (d_csrRowPtr[rowidx + 1] - d_csrRowPtr[0]) - 1 : (d_csrRowPtr[rowidx] - d_csrRowPtr[0]);
        d_x[rowidx] = d_b[rowidx] / d_csrVal[pos];
    }
}

__global__ void sptrsv_levelset_threadsca_csr_cuda_executor(const int *d_csrRowPtr,
                                                            const int *d_csrColIdx,
                                                            const VALUE_TYPE *d_csrVal,
                                                            const int m,
                                                            const int m_total,
                                                            const int offset,
                                                            const int substitution,
                                                            const VALUE_TYPE *d_b,
                                                            VALUE_TYPE *d_x)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
    {
        int rowidx = global_x_id + offset;
        rowidx = substitution == SUBSTITUTION_FORWARD ? rowidx : m_total - 1 - global_x_id - offset;

        const int start = substitution == SUBSTITUTION_FORWARD ? (d_csrRowPtr[rowidx] - d_csrRowPtr[0]) : (d_csrRowPtr[rowidx] - d_csrRowPtr[0]) + 1;
        const int stop = substitution == SUBSTITUTION_FORWARD ? (d_csrRowPtr[rowidx + 1] - d_csrRowPtr[0]) : (d_csrRowPtr[rowidx + 1] - d_csrRowPtr[0]) + 1;
        VALUE_TYPE sum = 0;
        for (int j = start; j < stop - 1; j++)
            sum += d_x[d_csrColIdx[j]] * d_csrVal[j];

        const int pos = substitution == SUBSTITUTION_FORWARD ? (d_csrRowPtr[rowidx + 1] - d_csrRowPtr[0]) - 1 : d_csrRowPtr[rowidx] - d_csrRowPtr[0];
        d_x[rowidx] = (d_b[rowidx] - sum) / d_csrVal[pos];
    }
}

__global__ void sptrsv_levelset_warpvec_csr_cuda_executor(const int *d_csrRowPtr,
                                                          const int *d_csrColIdx,
                                                          const VALUE_TYPE *d_csrVal,
                                                          const int m,
                                                          const int m_total,
                                                          const int offset,
                                                          const int substitution,
                                                          const VALUE_TYPE *d_b,
                                                          VALUE_TYPE *d_x)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize
    int rowidx = global_id / WARP_SIZE;
    if (rowidx >= m)
        return;

    rowidx += offset;

    rowidx = substitution == SUBSTITUTION_FORWARD ? rowidx : m_total - 1 - rowidx;

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int start = substitution == SUBSTITUTION_FORWARD ? (d_csrRowPtr[rowidx] - d_csrRowPtr[0]) : (d_csrRowPtr[rowidx] - d_csrRowPtr[0]) + 1;
    const int stop = substitution == SUBSTITUTION_FORWARD ? (d_csrRowPtr[rowidx + 1] - d_csrRowPtr[0]) : (d_csrRowPtr[rowidx + 1] - d_csrRowPtr[0]) + 1;

    VALUE_TYPE sum = 0;
    for (int j = start + lane_id; j < stop - 1; j += WARP_SIZE)
    {
        sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
    }
    sum = sum_32_shfl(sum);

    //finish
    if (!lane_id)
    {
        const int pos = substitution == SUBSTITUTION_FORWARD ? (d_csrRowPtr[rowidx + 1] - d_csrRowPtr[0]) - 1 : (d_csrRowPtr[rowidx] - d_csrRowPtr[0]);
        d_x[rowidx] = (d_b[rowidx] - sum) / d_csrVal[pos];
    }
}

__global__ void sptrsv_syncfree_csc_cuda_executor_fasttrack(const int *d_cscColPtr,
                                                            const int *d_cscRowIdx,
                                                            const VALUE_TYPE *d_cscVal,
                                                            const int m,
                                                            const int substitution,
                                                            const VALUE_TYPE *d_b,
                                                            VALUE_TYPE *d_x)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
    {
        const int pos = substitution == SUBSTITUTION_FORWARD ? (d_cscColPtr[global_x_id] - d_cscColPtr[0]) : (d_cscColPtr[global_x_id + 1] - d_cscColPtr[0]) - 1;
        d_x[global_x_id] = d_b[global_x_id] / d_cscVal[pos];
    }
}

#endif
