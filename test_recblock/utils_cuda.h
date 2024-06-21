#ifndef __UTILS_CUDA__
#define __UTILS_CUDA__
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"
#include <cuda_runtime.h>

__global__ void matrix_transposition_litelite_cuda(int nnz,
                                                   int *csrColIdx,
                                                   int *cscColPtr)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < nnz)
        atomicAdd(&(cscColPtr[csrColIdx[global_x_id]]), 1);
}

__global__ void get_indegree_cuda(int *d_indegree,
                                  int *d_csrRowPtrTR,
                                  int m)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
    {
        d_indegree[global_x_id] = d_csrRowPtrTR[global_x_id + 1] - d_csrRowPtrTR[global_x_id];
    }
}

__global__ void findlevel_cu(int *cscColPtr,
                             int *cscRowIdx,
                             int *csrRowPtr,
                             int m,
                             int *nlevel,
                             int *levelPtr,
                             int *levelItem,
                             int *indegree,
                             int *ptr)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id == 0)
    {
        levelPtr[0] = 0;
        ptr[0] = 0;
        for (int i = 0; i < m; i++)
        {
            if (indegree[i] == 1)
            {
                levelItem[ptr[0]] = i;
                ptr[0]++;
            }
        }

        levelPtr[1] = ptr[0];
        int lvi = 1;
        while (levelPtr[lvi] != m)
        {
            for (int i = levelPtr[lvi - 1]; i < levelPtr[lvi]; i++)
            {
                int node = levelItem[i];
                for (int j = cscColPtr[node]; j < cscColPtr[node + 1]; j++)
                {
                    int visit_node = cscRowIdx[j];
                    indegree[visit_node]--;
                    if (indegree[visit_node] == 1)
                    {
                        levelItem[ptr[0]] = visit_node;
                        ptr[0]++;
                    }
                }
            }
            lvi++;
            levelPtr[lvi] = ptr[0];
        }
        nlevel[0] = lvi;
    }
}

void findlevel_cuda(int *cscColPtr,
                    int *cscRowIdx,
                    int *csrRowPtr,
                    int m,
                    int *nlevel,
                    int *levelPtr,
                    int *levelItem)
{
    int *d_indegree;
    cudaMalloc((void **)&d_indegree, m * sizeof(int));
    int *ptr;
    cudaMalloc((void **)&ptr, 1 * sizeof(int));
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil((double)(m) / (double)num_threads);
    get_indegree_cuda<<<num_blocks, num_threads>>>(d_indegree, csrRowPtr, m);

    findlevel_cu<<<1, 1>>>(cscColPtr, cscRowIdx, csrRowPtr,
                           m, nlevel, levelPtr, levelItem, d_indegree, ptr);
    cudaFree(d_indegree);
}

__global__ void get_levelperm(int m,
                              int substitution,
                              int *d_levelperm,
                              int *d_levelItem_tmp)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (substitution == SUBSTITUTION_FORWARD)
    {
        if (global_x_id < m)
            d_levelperm[global_x_id] = global_x_id;
    }
    else
    {
        if (global_x_id < m)
        {
            int id = m - global_x_id - 1;
            d_levelperm[id] = id;
        }
    }
}

__global__ void perm_reorder_ptr(int *d_cscColPtrTR,
                                 int *d_cscColPtrTR_new,
                                 int *d_levelItem,
                                 int n,
                                 int substitution)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < n)
    {
        int idx = substitution == SUBSTITUTION_FORWARD ? d_levelItem[global_x_id] : d_levelItem[n - global_x_id - 1];
        int nnzr = d_cscColPtrTR[idx + 1] - d_cscColPtrTR[idx];
        d_cscColPtrTR_new[global_x_id] = nnzr;
    }
}

__global__ void perm_reorder_idxval(int *d_cscColPtrTR,
                                    int *d_cscRowIdxTR,
                                    VALUE_TYPE *d_cscValTR,
                                    int *d_cscColPtrTR_new,
                                    int *d_cscRowIdxTR_new,
                                    VALUE_TYPE *d_cscValTR_new,
                                    int *d_levelItem,
                                    int n,
                                    int substitution)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int colid = global_x_id / WARP_SIZE;
    if (colid < n)
    {
        const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
        int idx = substitution == SUBSTITUTION_FORWARD ? d_levelItem[colid] : d_levelItem[n - colid - 1];
        int nnzr = d_cscColPtrTR[idx + 1] - d_cscColPtrTR[idx];
        for (int j = lane_id; j < nnzr; j += WARP_SIZE)
        {
            int off = d_cscColPtrTR[idx] + j;
            int off_new = d_cscColPtrTR_new[colid] + j;
            d_cscRowIdxTR_new[off_new] = d_cscRowIdxTR[off];
            d_cscValTR_new[off_new] = d_cscValTR[off];
        }
    }
}

__global__ void get_backward_perm(int *d_levelperm,
                                  int *d_levelItem,
                                  int m)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
        d_levelperm[d_levelItem[global_x_id]] = m - global_x_id - 1;
}

__global__ void reorder_idx(int *d_cscRowIdxTR_new,
                            int *d_levelperm,
                            int nnzTR)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < nnzTR)
    {
        d_cscRowIdxTR_new[global_x_id] = d_levelperm[d_cscRowIdxTR_new[global_x_id]];
    }
}

__global__ void get_backward_item(int *d_levelItem_tmp,
                                  int *d_levelItem,
                                  int m)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
        d_levelItem_tmp[m - global_x_id - 1] = d_levelItem[global_x_id];
}

__global__ void cal_triblk_nnz(int *d_cscColPtrTR,
                               int *d_cscRowIdxTR,
                               int tri_up,
                               int tri_down,
                               int substitution,
                               int *blk_nnz)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int len = tri_down - tri_up;
    if (global_x_id < len)
    {
        int start = d_cscColPtrTR[global_x_id + tri_up];
        int end = d_cscColPtrTR[global_x_id + tri_up + 1];
        if (substitution == SUBSTITUTION_FORWARD)
            for (int i = start; i < end; i++)
            {
                if (d_cscRowIdxTR[i] < tri_down)
                    atomicAdd(&blk_nnz[0], 1);
            }
        else
            for (int i = start; i < end; i++)
            {
                if (d_cscRowIdxTR[i] >= tri_up)
                    atomicAdd(&blk_nnz[0], 1);
            }
    }
}

__global__ void cal_recblk_nnz(int *d_cscColPtrTR,
                               int *d_cscRowIdxTR,
                               int rec_up,
                               int rec_down,
                               int rec_left,
                               int rec_right,
                               int *blk_nnz)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int len = rec_right - rec_left;
    if (global_x_id < len)
    {
        int start = d_cscColPtrTR[global_x_id + rec_left];
        int end = d_cscColPtrTR[global_x_id + rec_left + 1];
        for (int i = start; i < end; i++)
        {
            if (d_cscRowIdxTR[i] >= rec_up && d_cscRowIdxTR[i] < rec_down)
                atomicAdd(&blk_nnz[0], 1);
        }
    }
}

void mat_preprocessing_cuda(int *cscColPtrTR,
                            int *cscRowIdxTR,
                            VALUE_TYPE *cscValTR,
                            int m,
                            int n,
                            int nlevel,
                            int *loc_off,
                            int *tmp_off,
                            int *blk_m,
                            int *blk_n,
                            int *d_blk_nnz,
                            int *subtri_upbound,
                            int *subtri_downbound,
                            int *subrec_upbound,
                            int *subrec_downbound,
                            int *subrec_rightbound,
                            int *subrec_leftbound,
                            int substitution)
{
    int tri_block = pow(2, nlevel);
    int sqr_block = tri_block - 1;
    int sum_block = tri_block + sqr_block;
    int step_size = m / tri_block;

    int *figure;
    figure = (int *)malloc(sizeof(int) * nlevel + 1);

    for (int i = 0; i <= nlevel; i++)
        figure[i] = (int)pow(2, i);

    int *judge;
    judge = (int *)malloc(sizeof(int) * nlevel);
    memset(judge, 0, sizeof(int) * nlevel);

    for (int i = 0; i < nlevel; i++)
    {
        for (int j = nlevel - 1; j >= nlevel - 1 - i; j--)
        {
            judge[i] += figure[j];
        }
    }

    if (substitution == SUBSTITUTION_FORWARD)
    {
        int tri_up = 0;
        int tri_down = step_size;
        int rec_up;
        int rec_down;
        int rec_left;
        int rec_right;

        int tri_count = 0;
        for (int i = 0; i < sum_block; i++)
        {
            if (i % 2 == 0)
            {
                blk_m[i] = tri_down - tri_up;
                blk_n[i] = tri_down - tri_up;

                subtri_upbound[i] = tri_up;
                subtri_downbound[i] = tri_down;
                int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                int num_blocks = ceil((double)(blk_m[i]) / (double)num_threads);
                cal_triblk_nnz<<<num_blocks, num_threads>>>(cscColPtrTR, cscRowIdxTR, tri_up, tri_down,
                                                            substitution, &d_blk_nnz[i]);

                loc_off[i] = tri_up;

                tri_up += step_size;
                tri_down += step_size;

                tri_count++;
            }
            else
            {
                rec_up = tri_down - step_size;

                int flag = 0;
                int temp = tri_count;
                while (1)
                {
                    for (int j = 0; j <= nlevel; j++)
                    {
                        if (temp == figure[j])
                        {
                            rec_down = rec_up + temp * step_size;

                            flag = 1;
                            break;
                        }
                    }

                    if (flag == 1)
                        break;

                    for (int j = 0; j <= nlevel; j++)
                    {
                        if (temp < figure[j])
                        {
                            temp -= figure[j - 1];
                            break;
                        }
                    }
                }

                rec_right = rec_up;
                rec_left = rec_right - (rec_down - rec_up);

                for (int j = 0; j < nlevel; j++)
                {
                    if (judge[j] == tri_count)
                    {
                        rec_down = m;
                        break;
                    }
                }

                blk_m[i] = rec_down - rec_up;
                blk_n[i] = rec_right - rec_left;

                int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                int num_blocks = ceil((double)(blk_n[i]) / (double)num_threads);
                cal_recblk_nnz<<<num_blocks, num_threads>>>(cscColPtrTR, cscRowIdxTR, rec_up, rec_down,
                                                            rec_left, rec_right, &d_blk_nnz[i]);

                tmp_off[i] = rec_up;
                loc_off[i] = rec_left;

                if (i == sum_block - 2)
                {
                    tri_down = m;
                }

                subrec_upbound[i] = rec_up;
                subrec_downbound[i] = rec_down;
                subrec_rightbound[i] = rec_right;
                subrec_leftbound[i] = rec_left;
            }
        }
    }
    else
    {
        int tri_up = m - step_size;
        int tri_down = m;
        int rec_up;
        int rec_down;
        int rec_left;
        int rec_right;
        int tri_count = 0;
        for (int i = 0; i < sum_block; i++)
        {
            if (i % 2 == 0)
            {
                blk_m[i] = tri_down - tri_up;
                blk_n[i] = tri_down - tri_up;

                subtri_upbound[i] = tri_up;
                subtri_downbound[i] = tri_down;

                int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                int num_blocks = ceil((double)(blk_m[i]) / (double)num_threads);
                cal_triblk_nnz<<<num_blocks, num_threads>>>(cscColPtrTR, cscRowIdxTR, tri_up, tri_down,
                                                            substitution, &d_blk_nnz[i]);

                loc_off[i] = tri_up;

                tri_up -= step_size;
                tri_down -= step_size;

                tri_count++;
            }
            else
            {
                rec_down = tri_up + step_size;

                int flag = 0;
                int temp = tri_count;
                while (1)
                {
                    for (int j = 0; j <= nlevel; j++)
                    {
                        if (temp == figure[j])
                        {
                            rec_up = rec_down - temp * step_size;

                            flag = 1;
                            break;
                        }
                    }

                    if (flag == 1)
                        break;

                    for (int j = 0; j <= nlevel; j++)
                    {
                        if (temp < figure[j])
                        {
                            temp -= figure[j - 1];
                            break;
                        }
                    }
                }

                rec_left = rec_down;
                rec_right = rec_left + (rec_down - rec_up);

                for (int j = 0; j < nlevel; j++)
                {
                    if (judge[j] == tri_count)
                    {
                        rec_up = 0;
                        break;
                    }
                }

                blk_m[i] = rec_down - rec_up;
                blk_n[i] = rec_right - rec_left;

                int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                int num_blocks = ceil((double)(blk_n[i]) / (double)num_threads);
                cal_recblk_nnz<<<num_blocks, num_threads>>>(cscColPtrTR, cscRowIdxTR, rec_up, rec_down,
                                                            rec_left, rec_right, &d_blk_nnz[i]);

                tmp_off[i] = rec_up; //record rec upbound
                loc_off[i] = rec_left;

                if (i == sum_block - 2)
                {
                    tri_up = 0;
                }

                subrec_upbound[i] = rec_up;
                subrec_downbound[i] = rec_down;
                subrec_rightbound[i] = rec_right;
                subrec_leftbound[i] = rec_left;
            }
        }
    }

    free(figure);
    free(judge);
}

__global__ void store_into_subtrimat_ptr(int upbound,
                                         int downbound,
                                         int *d_cscColPtr,
                                         int *d_cscRowIdx,
                                         int *d_cscColPtr_sub,
                                         int substitution)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int len = downbound - upbound;
    if (global_x_id < len)
    {
        int idx = global_x_id + upbound;
        int nnz = 0;
        if (substitution == SUBSTITUTION_FORWARD)
        {
            for (int i = d_cscColPtr[idx]; i < d_cscColPtr[idx + 1]; i++)
            {
                if (d_cscRowIdx[i] < downbound)
                {
                    nnz++;
                }
            }
            d_cscColPtr_sub[global_x_id] = nnz;
        }
        else
        {
            for (int i = d_cscColPtr[idx]; i < d_cscColPtr[idx + 1]; i++)
            {
                if (d_cscRowIdx[i] >= upbound)
                {
                    nnz++;
                }
            }
            d_cscColPtr_sub[global_x_id] = nnz;
        }
    }
}

__global__ void store_into_subrecmat_ptr(int upbound,
                                         int downbound,
                                         int leftbound,
                                         int rightbound,
                                         int *d_cscColPtr,
                                         int *d_cscRowIdx,
                                         int *d_cscColPtr_sub)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int len = rightbound - leftbound;
    if (global_x_id < len)
    {
        int idx = global_x_id + leftbound;
        int nnz = 0;
        for (int i = d_cscColPtr[idx]; i < d_cscColPtr[idx + 1]; i++)
        {
            if (d_cscRowIdx[i] >= upbound && d_cscRowIdx[i] < downbound)
            {
                nnz++;
            }
        }
        d_cscColPtr_sub[global_x_id] = nnz;
    }
}

__global__ void store_into_subtrimat_idxval(int upbound,
                                            int downbound,
                                            int *d_cscColPtr,
                                            int *d_cscRowIdx,
                                            VALUE_TYPE *d_cscVal,
                                            int *d_cscColPtr_sub,
                                            int *d_cscRowIdx_sub,
                                            VALUE_TYPE *d_cscVal_sub,
                                            int substitution)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int len = downbound - upbound;
    if (global_x_id < len)
    {
        int idx = global_x_id + upbound;
        int nnz = 0;
        if (substitution == SUBSTITUTION_FORWARD)
        {
            for (int i = d_cscColPtr[idx]; i < d_cscColPtr[idx + 1]; i++)
            {
                if (d_cscRowIdx[i] < downbound)
                {
                    int id = d_cscColPtr_sub[global_x_id] + nnz;
                    d_cscRowIdx_sub[id] = d_cscRowIdx[i] - upbound;
                    d_cscVal_sub[id] = d_cscVal[i];
                    nnz++;
                }
            }
        }
        else
        {
            for (int i = d_cscColPtr[idx]; i < d_cscColPtr[idx + 1]; i++)
            {
                if (d_cscRowIdx[i] >= upbound)
                {
                    int id = d_cscColPtr_sub[global_x_id] + nnz;
                    d_cscRowIdx_sub[id] = d_cscRowIdx[i] - upbound;
                    d_cscVal_sub[id] = d_cscVal[i];
                    nnz++;
                }
            }
        }
    }
}

__global__ void store_into_subrecmat_idxval(int upbound,
                                            int downbound,
                                            int leftbound,
                                            int rightbound,
                                            int *d_cscColPtr,
                                            int *d_cscRowIdx,
                                            VALUE_TYPE *d_cscVal,
                                            int *d_cscColPtr_sub,
                                            int *d_cscRowIdx_sub,
                                            VALUE_TYPE *d_cscVal_sub)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int len = rightbound - leftbound;
    if (global_x_id < len)
    {
        int idx = global_x_id + leftbound;
        int nnz = 0;
        for (int i = d_cscColPtr[idx]; i < d_cscColPtr[idx + 1]; i++)
        {
            if (d_cscRowIdx[i] >= upbound && d_cscRowIdx[i] < downbound)
            {
                int id = d_cscColPtr_sub[global_x_id] + nnz;
                d_cscRowIdx_sub[id] = d_cscRowIdx[i] - upbound;
                d_cscVal_sub[id] = d_cscVal[i];
                nnz++;
            }
        }
    }
}

__global__ void insert_nnz_to_mat(int m,
                                  int *csrRowPtr,
                                  int *csrColIdx,
                                  VALUE_TYPE *csrVal,
                                  int *cscColIncr,
                                  int *cscRowIdx,
                                  VALUE_TYPE *cscVal)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
    {
        for (int row = 0; row < m; row++)
            for (int i = csrRowPtr[row]; i < csrRowPtr[row + 1]; i++)
            {
                int col = csrColIdx[i];
                cscRowIdx[cscColIncr[col]] = row;
                cscVal[cscColIncr[col]] = csrVal[i];
                cscColIncr[col]++;
            }
    }
}

void matrix_transposition_cuda(int m,
                               int n,
                               int nnz,
                               int *csrRowPtr,
                               int *csrColIdx,
                               VALUE_TYPE *csrVal,
                               int *cscRowIdx,
                               int *cscColPtr,
                               VALUE_TYPE *cscVal)
{
    cudaMemset(cscColPtr, 0, sizeof(int) * (n + 1));
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil((double)(nnz) / (double)num_threads);
    matrix_transposition_litelite_cuda<<<num_blocks, num_threads>>>(nnz, csrColIdx,
                                                                    cscColPtr);
    thrust::exclusive_scan(thrust::device, cscColPtr,
                           cscColPtr + n + 1, cscColPtr, 0);
    int *cscColIncr;
    cudaMalloc((void **)&cscColIncr, sizeof(int) * (n + 1));
    cudaMemcpy(cscColIncr, cscColPtr, sizeof(int) * (n + 1), cudaMemcpyDeviceToDevice);

    num_threads = WARP_PER_BLOCK * WARP_SIZE;
    num_blocks = ceil((double)(m) / (double)num_threads);
    insert_nnz_to_mat<<<1, 1>>>(m, csrRowPtr, csrColIdx, csrVal,
                                cscColIncr, cscRowIdx, cscVal);
    cudaFree(cscColIncr);
}

__global__ void pre_store_to_recblockdata(int n,
                                          int *cscColPtrTR_sub,
                                          int *offset)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < n)
    {
        int nnzc = cscColPtrTR_sub[global_x_id + 1] - cscColPtrTR_sub[global_x_id];
        offset[global_x_id] = nnzc;
    }
}

__global__ void store_to_recblockdata(int n,
                                      int *cscColPtrTR_sub,
                                      int *cscRowIdxTR_sub,
                                      VALUE_TYPE *cscValTR_sub,
                                      int *recblock_Index,
                                      VALUE_TYPE *recblock_Val,
                                      int *offset,
                                      int base)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < n)
    {
        int idx = offset[global_x_id] - offset[0] + base;
        for (int j = cscColPtrTR_sub[global_x_id]; j < cscColPtrTR_sub[global_x_id + 1]; j++)
        {
            recblock_Index[idx] = cscRowIdxTR_sub[j];
            recblock_Val[idx] = cscValTR_sub[j];
            idx++;
        }
    }
}

void levelset_reordering_colrow_csc_cuda(int *cscColPtrTR,
                                         int *cscRowIdxTR,
                                         VALUE_TYPE *cscValTR,
                                         int *cscColPtrTR_new,
                                         int *cscRowIdxTR_new,
                                         VALUE_TYPE *cscValTR_new,
                                         int *levelItem,
                                         int m,
                                         int n,
                                         int nnzTR,
                                         int substitution)
{
    int *d_csrRowPtrTR_tmp;
    cudaMalloc((void **)&d_csrRowPtrTR_tmp, (m + 1) * sizeof(int));
    cudaMemset(d_csrRowPtrTR_tmp, 0, (m + 1) * sizeof(int));

    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil((double)(nnzTR) / (double)num_threads);
    matrix_transposition_litelite_cuda<<<num_blocks, num_threads>>>(nnzTR, cscRowIdxTR, d_csrRowPtrTR_tmp);
    thrust::exclusive_scan(thrust::device, d_csrRowPtrTR_tmp,
                           d_csrRowPtrTR_tmp + m + 1, d_csrRowPtrTR_tmp, 0);

    int *levelPtr;
    cudaMalloc((void **)&levelPtr, (m + 1) * sizeof(int));
    cudaMemset(levelPtr, 0, sizeof(int));
    int *nlv;
    cudaMalloc((void **)&nlv, 1 * sizeof(int));
    findlevel_cuda(cscColPtrTR, cscRowIdxTR, d_csrRowPtrTR_tmp,
                   m, nlv, levelPtr, levelItem);
    int *d_levelItem_tmp;
    int *d_levelperm;
    cudaMalloc((void **)&d_levelItem_tmp, m * sizeof(int));
    cudaMemcpy(d_levelItem_tmp, levelItem, m * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMalloc((void **)&d_levelperm, m * sizeof(int));
    get_levelperm<<<num_blocks, num_threads>>>(m, substitution, d_levelperm, d_levelItem_tmp);
    thrust::stable_sort_by_key(thrust::device, d_levelItem_tmp, d_levelItem_tmp + m,
                               d_levelperm);

    if (substitution == SUBSTITUTION_BACKWARD)
        get_backward_perm<<<num_blocks, num_threads>>>(d_levelperm, levelItem, m);

    num_threads = WARP_PER_BLOCK * WARP_SIZE;
    num_blocks = ceil((double)(n) / (double)(num_threads));
    perm_reorder_ptr<<<num_blocks, num_threads>>>(cscColPtrTR, cscColPtrTR_new,
                                                  levelItem, n, substitution);

    thrust::exclusive_scan(thrust::device, cscColPtrTR_new,
                           cscColPtrTR_new + n + 1, cscColPtrTR_new, 0);
    num_threads = WARP_PER_BLOCK * WARP_SIZE;
    num_blocks = ceil((double)(n) / (double)WARP_PER_BLOCK);
    perm_reorder_idxval<<<num_blocks, num_threads>>>(cscColPtrTR, cscRowIdxTR, cscValTR,
                                                     cscColPtrTR_new, cscRowIdxTR_new, cscValTR_new,
                                                     levelItem, n, substitution);
    num_threads = WARP_PER_BLOCK * WARP_SIZE;
    num_blocks = ceil((double)(nnzTR) / (double)num_threads);
    reorder_idx<<<num_blocks, num_threads>>>(cscRowIdxTR_new, d_levelperm, nnzTR);

    if (substitution == SUBSTITUTION_BACKWARD)
    {
        int *d_levelItem_tmp;
        cudaMalloc((void **)&d_levelItem_tmp, m * sizeof(int));
        num_threads = WARP_PER_BLOCK * WARP_SIZE;
        num_blocks = ceil((double)(m) / (double)num_threads);
        get_backward_item<<<num_blocks, num_threads>>>(d_levelItem_tmp, levelItem, m);
        cudaMemcpy(levelItem, d_levelItem_tmp, m * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaFree(d_levelItem_tmp);
    }

    cudaFree(d_csrRowPtrTR_tmp);
    cudaFree(levelPtr);
    cudaFree(d_levelperm);
    cudaFree(d_levelItem_tmp);
    cudaFree(nlv);
}

__global__ void find_level_nnz(int *csrRowPtrTR_sub,
                               int *levelPtr,
                               int *nnz_lv_array,
                               int lv)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < lv)
    {
        int nnzlv = 0;
        for (int lvi = levelPtr[global_x_id]; lvi < levelPtr[global_x_id + 1]; lvi++)
        {
            nnzlv += csrRowPtrTR_sub[lvi + 1] - csrRowPtrTR_sub[lvi];
        }
        nnz_lv_array[global_x_id] = nnzlv;
    }
}

__global__ void cal_longrow(int *i_new,
                            int *lenmax,
                            int *longrow,
                            int *longlen,
                            int *longrow_idx,
                            int *csrRowPtr_sqr,
                            int m)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (!global_x_id)
    {
        i_new[0] = 1;
        lenmax[0] = csrRowPtr_sqr[1] - csrRowPtr_sqr[0];
        longrow[0] = 0;
        longlen[0] = 0;
        for (int i = 1; i <= m; i++)
        {
            int len = csrRowPtr_sqr[i] - csrRowPtr_sqr[i - 1];
            lenmax[0] = len > lenmax[0] ? len : lenmax[0];

            if (csrRowPtr_sqr[i] != csrRowPtr_sqr[i - 1])
            {
                if (len > LONGROW_THRESHOLD)
                {
                    longrow_idx[longrow[0]] = i - 1;
                    longrow[0]++;
                    longlen[0] += len;
                }
                i_new[0]++;
            }
        }
    }
}

__global__ void dcsr_recblockdata_ptr(int m,
                                      int *csrRowPtr_sqr,
                                      int ptr_offset,
                                      int dcsrindex_offset,
                                      int *dcsr_i,
                                      int *recblock_Ptr,
                                      int *recblock_dcsr_rowidx)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (!global_x_id)
    {
        for (int i = 0; i < m; i++)
        {
            if (csrRowPtr_sqr[i + 1] != csrRowPtr_sqr[i])
            {
                int row_nnz = csrRowPtr_sqr[i + 1] - csrRowPtr_sqr[i];
                int index = ptr_offset + dcsr_i[0];
                int index_dcsr = dcsrindex_offset + dcsr_i[0];
                recblock_Ptr[index] = recblock_Ptr[index - 1] + row_nnz;
                recblock_dcsr_rowidx[index_dcsr] = i;
                dcsr_i[0]++;
            }
        }
    }
}

__global__ void levelset_reordering_vecb_cuda(VALUE_TYPE *b,
                                              VALUE_TYPE *b_perm,
                                              int *levelItem,
                                              int m)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < m)
        b_perm[global_x_id] = b[levelItem[global_x_id]];
}

__global__ void levelset_reordering_vecx_cuda(VALUE_TYPE *x_perm,
                                              VALUE_TYPE *x,
                                              int *levelItem,
                                              int n)
{
    const int global_x_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_x_id < n)
        x[levelItem[global_x_id]] = x_perm[global_x_id];
}

#endif