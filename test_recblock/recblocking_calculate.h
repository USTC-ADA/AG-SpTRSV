#ifndef _LORU_CALCULATE_
#define _LORU_CALCULATE_
#include <stdio.h>
#include "common.h"
#include "findlevel.h"
#include "tranpose.h"
#include <cuda_runtime.h>
#include "cusparse.h"
#include "utils.h"
#include "utils_sptrsv_cuda.h"
#include "utils_spmv_cuda.h"
#include "utils_reordering.h"
#include <cuda_runtime.h>

void L_calculate(SpMV_block *mv_blk,
                 SpTRSV_block *trsv_blk,
                 int sum_block,
                 int *blk_m,
                 int *blk_n,
                 int *loc_off,
                 int *tmp_off,
                 int m,
                 int rhs,
                 VALUE_TYPE *x_t,
                 VALUE_TYPE *b_t,
                 VALUE_TYPE *b_perm,
                 const int *d_recblock_Ptr,
                 const int *d_recblock_Index,
                 const int *d_recblock_dcsr_rowidx,
                 const double *d_recblock_Val,
                 int *ptr_offset,
                 int *index_offset,
                 int *dcsrindex_offset,
                 double *cal_time)
{
    struct timeval t1, t2;
    for (int re = 0; re < BENCH_REPEAT; re++)
    {
        cudaMemcpy(b_t, b_perm, rhs * m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToDevice);

        int b_offset = 0;
        int x_offset = 0;
        int tri_index = 0;
        int squ_index = 0;
        gettimeofday(&t1, NULL);
        for (int i = 0; i < sum_block; i++)
        {
            if (i % 2 == 0)
            {
                if (trsv_blk[tri_index].method == 0)
                {
                    sptrsv_syncfree_csc_cuda_executor_fasttrack<<<trsv_blk[tri_index].num_blocks, trsv_blk[tri_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                                     trsv_blk[tri_index].m, trsv_blk[tri_index].substitution, &b_t[b_offset], &x_t[x_offset]);
                }
                else if (trsv_blk[tri_index].method == 1)
                {
                    if (sizeof(VALUE_TYPE) == 8)
                        cusparseDcsrsv2_solve(trsv_blk[tri_index].handle, trsv_blk[tri_index].trans, trsv_blk[tri_index].m, trsv_blk[tri_index].nnzTR, &(trsv_blk[tri_index].alpha_double), trsv_blk[tri_index].descr,
                                              (double *)(&d_recblock_Val[index_offset[i]]), &d_recblock_Ptr[ptr_offset[i]], &d_recblock_Index[index_offset[i]], trsv_blk[tri_index].info,
                                              (double *)&(b_t[b_offset]), (double *)&(x_t[x_offset]), trsv_blk[tri_index].policy, trsv_blk[tri_index].pBuffer);
                    else if (sizeof(VALUE_TYPE) == 4)
                        cusparseScsrsv2_solve(trsv_blk[tri_index].handle, trsv_blk[tri_index].trans, trsv_blk[tri_index].m, trsv_blk[tri_index].nnzTR, &(trsv_blk[tri_index].alpha_float), trsv_blk[tri_index].descr,
                                              (float *)(&d_recblock_Val[index_offset[i]]), &d_recblock_Ptr[ptr_offset[i]], &d_recblock_Index[index_offset[i]], trsv_blk[tri_index].info,
                                              (float *)&(b_t[b_offset]), (float *)&(x_t[x_offset]), trsv_blk[tri_index].policy, trsv_blk[tri_index].pBuffer);
                }
                else if (trsv_blk[tri_index].method == 2)
                {
                    for (int li = 0; li < trsv_blk[tri_index].nlv; li++)
                    {
                        if (li == 0)
                        {
                            trsv_blk[tri_index].num_threads = WARP_PER_BLOCK * WARP_SIZE;
                            trsv_blk[tri_index].num_blocks = ceil((double)(trsv_blk[tri_index].m_lv_array[li]) / (double)(trsv_blk[tri_index].num_threads));
                            sptrsv_levelset_threadsca_csr_cuda_executor_fasttrack<<<trsv_blk[tri_index].num_blocks, trsv_blk[tri_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                                                       trsv_blk[tri_index].m_lv_array[li], trsv_blk[tri_index].m, trsv_blk[tri_index].offset_array[li], trsv_blk[tri_index].substitution, &b_t[b_offset], &x_t[x_offset]);
                        }
                        else
                        {
                            if ((trsv_blk[tri_index].nnz_lv_array[li] / trsv_blk[tri_index].m_lv_array[li]) <= 15)
                            {
                                trsv_blk[tri_index].num_threads = WARP_PER_BLOCK * WARP_SIZE;
                                trsv_blk[tri_index].num_blocks = ceil((double)(trsv_blk[tri_index].m_lv_array[li]) / (double)(trsv_blk[tri_index].num_threads));
                                sptrsv_levelset_threadsca_csr_cuda_executor<<<trsv_blk[tri_index].num_blocks, trsv_blk[tri_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                                                 trsv_blk[tri_index].m_lv_array[li], trsv_blk[tri_index].m, trsv_blk[tri_index].offset_array[li], trsv_blk[tri_index].substitution, &b_t[b_offset], &x_t[x_offset]);
                            }
                            else
                            {
                                trsv_blk[tri_index].num_threads = WARP_PER_BLOCK * WARP_SIZE;
                                trsv_blk[tri_index].num_blocks = ceil((double)(trsv_blk[tri_index].m_lv_array[li]) / (double)((trsv_blk[tri_index].num_threads) / WARP_SIZE));
                                sptrsv_levelset_warpvec_csr_cuda_executor<<<trsv_blk[tri_index].num_blocks, trsv_blk[tri_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                                               trsv_blk[tri_index].m_lv_array[li], trsv_blk[tri_index].m, trsv_blk[tri_index].offset_array[li], trsv_blk[tri_index].substitution, &b_t[b_offset], &x_t[x_offset]);
                            }
                        }
                    }
                }
                else if (trsv_blk[tri_index].method == 3)
                {
                    sptrsv_syncfree_warpvec_csc_cuda_executor<<<trsv_blk[tri_index].num_blocks, trsv_blk[tri_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                                   trsv_blk[tri_index].d_graphInDegree, trsv_blk[tri_index].d_left_sum,
                                                                                                                                   trsv_blk[tri_index].m, trsv_blk[tri_index].substitution, &b_t[b_offset], &x_t[x_offset], trsv_blk[tri_index].d_while_profiler,
                                                                                                                                   trsv_blk[tri_index].d_id_extractor, trsv_blk[tri_index].d_levelItem);
                }
                tri_index++;
                b_offset += blk_m[i];
                x_offset += blk_n[i];
                cudaDeviceSynchronize();
            }
            else
            {
                if (mv_blk[squ_index].method == 0)
                {
                    spmv_threadsca_csr_cuda_executor<<<mv_blk[squ_index].num_blocks, mv_blk[squ_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                      mv_blk[squ_index].m, &x_t[loc_off[i]], mv_blk[squ_index].d_y);
                    if (mv_blk[squ_index].longrow != 0)
                        spmv_longrow_csr_cuda_executor<<<mv_blk[squ_index].num_blocks_l, mv_blk[squ_index].num_threads_l>>>(mv_blk[squ_index].d_csrRowPtr_l, mv_blk[squ_index].d_csrColIdx_l, mv_blk[squ_index].d_csrVal_l,
                                                                                                                            &x_t[loc_off[i]], mv_blk[squ_index].d_y, mv_blk[squ_index].longrow, mv_blk[squ_index].d_longrow_idx);

                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)(blk_m[i]) / (double)num_threads);
                    subKernel<<<num_blocks, num_threads>>>(&(b_t[tmp_off[i]]), mv_blk[squ_index].d_y, blk_m[i]);
                }
                else if (mv_blk[squ_index].method == 1)
                {
                    spmv_threadsca_dcsr_cuda_executor<<<mv_blk[squ_index].num_blocks, mv_blk[squ_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                       mv_blk[squ_index].m_new, &x_t[loc_off[i]], mv_blk[squ_index].d_y, &d_recblock_dcsr_rowidx[dcsrindex_offset[i]]);

                    if (mv_blk[squ_index].longrow != 0)
                        spmv_longrow_csr_cuda_executor<<<mv_blk[squ_index].num_blocks_l, mv_blk[squ_index].num_threads_l>>>(mv_blk[squ_index].d_csrRowPtr_l, mv_blk[squ_index].d_csrColIdx_l, mv_blk[squ_index].d_csrVal_l,
                                                                                                                            &x_t[loc_off[i]], mv_blk[squ_index].d_y, mv_blk[squ_index].longrow, mv_blk[squ_index].d_longrow_idx);

                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)(blk_m[i]) / (double)num_threads);
                    subKernel<<<num_blocks, num_threads>>>(&(b_t[tmp_off[i]]), mv_blk[squ_index].d_y, blk_m[i]);
                }
                else if (mv_blk[squ_index].method == 2)
                {
                    spmv_warpvec_csr_cuda_executor<<<mv_blk[squ_index].num_blocks, mv_blk[squ_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                    mv_blk[squ_index].m, &x_t[loc_off[i]], mv_blk[squ_index].d_y);
                    if (mv_blk[squ_index].longrow != 0)
                        spmv_longrow_csr_cuda_executor<<<mv_blk[squ_index].num_blocks_l, mv_blk[squ_index].num_threads_l>>>(mv_blk[squ_index].d_csrRowPtr_l, mv_blk[squ_index].d_csrColIdx_l, mv_blk[squ_index].d_csrVal_l,
                                                                                                                            &x_t[loc_off[i]], mv_blk[squ_index].d_y, mv_blk[squ_index].longrow, mv_blk[squ_index].d_longrow_idx);

                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)(blk_m[i]) / (double)num_threads);
                    subKernel<<<num_blocks, num_threads>>>(&(b_t[tmp_off[i]]), mv_blk[squ_index].d_y, blk_m[i]);
                }
                else if (mv_blk[squ_index].method == 3)
                {
                    spmv_warpvec_dcsr_cuda_executor<<<mv_blk[squ_index].num_blocks, mv_blk[squ_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                     mv_blk[squ_index].m_new, &x_t[loc_off[i]], mv_blk[squ_index].d_y, &d_recblock_dcsr_rowidx[dcsrindex_offset[i]]);
                    if (mv_blk[squ_index].longrow != 0)
                        spmv_longrow_csr_cuda_executor<<<mv_blk[squ_index].num_blocks_l, mv_blk[squ_index].num_threads_l>>>(mv_blk[squ_index].d_csrRowPtr_l, mv_blk[squ_index].d_csrColIdx_l, mv_blk[squ_index].d_csrVal_l,
                                                                                                                            &x_t[loc_off[i]], mv_blk[squ_index].d_y, mv_blk[squ_index].longrow, mv_blk[squ_index].d_longrow_idx);

                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)(blk_m[i]) / (double)num_threads);
                    subKernel<<<num_blocks, num_threads>>>(&(b_t[tmp_off[i]]), mv_blk[squ_index].d_y, blk_m[i]);
                }
                squ_index++;
                cudaDeviceSynchronize();
            }
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        *cal_time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }
    *cal_time /= BENCH_REPEAT;
}

void U_calculate(SpMV_block *mv_blk,
                 SpTRSV_block *trsv_blk,
                 int sum_block,
                 int *blk_m,
                 int *blk_n,
                 int *loc_off,
                 int *tmp_off,
                 int m,
                 int nnzTR,
                 int rhs,
                 VALUE_TYPE *x_t,
                 VALUE_TYPE *b_t,
                 VALUE_TYPE *b_perm,
                 const int *d_recblock_Ptr,
                 const int *d_recblock_Index,
                 const int *d_recblock_dcsr_rowidx,
                 const double *d_recblock_Val,
                 int *ptr_offset,
                 int *index_offset,
                 int *dcsrindex_offset,
                 double *cal_time)
{
    struct timeval t1, t2;
    double total_cal_time = 0;
    for (int re = 0; re < BENCH_REPEAT; re++)
    {
        cudaMemcpy(b_t, b_perm, rhs * m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToDevice);
        int b_offset = m;
        int x_offset = m;
        int tri_index = 0;
        int squ_index = 0;

        gettimeofday(&t1, NULL);
        for (int i = 0; i < sum_block; i++)
        {
            if (i % 2 == 0)
            {
                b_offset -= blk_m[i];
                x_offset -= blk_n[i];
                if (trsv_blk[tri_index].method == 0)
                {
                    sptrsv_syncfree_csc_cuda_executor_fasttrack<<<trsv_blk[tri_index].num_blocks, trsv_blk[tri_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                                     trsv_blk[tri_index].m, trsv_blk[tri_index].substitution, &b_t[b_offset], &x_t[x_offset]);
                }
                else if (trsv_blk[tri_index].method == 1)
                {
                    if (sizeof(VALUE_TYPE) == 8)
                        cusparseDcsrsv2_solve(trsv_blk[tri_index].handle, trsv_blk[tri_index].trans, trsv_blk[tri_index].m, trsv_blk[tri_index].nnzTR, &(trsv_blk[tri_index].alpha_double), trsv_blk[tri_index].descr,
                                              (double *)(&d_recblock_Val[index_offset[i]]), &d_recblock_Ptr[ptr_offset[i]], &d_recblock_Index[index_offset[i]], trsv_blk[tri_index].info,
                                              (double *)&(b_t[b_offset]), (double *)&(x_t[x_offset]), trsv_blk[tri_index].policy, trsv_blk[tri_index].pBuffer);
                    else if (sizeof(VALUE_TYPE) == 4)
                        cusparseScsrsv2_solve(trsv_blk[tri_index].handle, trsv_blk[tri_index].trans, trsv_blk[tri_index].m, trsv_blk[tri_index].nnzTR, &(trsv_blk[tri_index].alpha_float), trsv_blk[tri_index].descr,
                                              (float *)(&d_recblock_Val[index_offset[i]]), &d_recblock_Ptr[ptr_offset[i]], &d_recblock_Index[index_offset[i]], trsv_blk[tri_index].info,
                                              (float *)&(b_t[b_offset]), (float *)&(x_t[x_offset]), trsv_blk[tri_index].policy, trsv_blk[tri_index].pBuffer);
                }
                else if (trsv_blk[tri_index].method == 2)
                {
                    for (int li = 0; li < trsv_blk[tri_index].nlv; li++)
                    {
                        if (li == 0)
                        {
                            trsv_blk[tri_index].num_threads = WARP_PER_BLOCK * WARP_SIZE;
                            trsv_blk[tri_index].num_blocks = ceil((double)(trsv_blk[tri_index].m_lv_array[li]) / (double)(trsv_blk[tri_index].num_threads));
                            sptrsv_levelset_threadsca_csr_cuda_executor_fasttrack<<<trsv_blk[tri_index].num_blocks, trsv_blk[tri_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                                                       trsv_blk[tri_index].m_lv_array[li], trsv_blk[tri_index].m, trsv_blk[tri_index].offset_array[li], trsv_blk[tri_index].substitution, &b_t[b_offset], &x_t[x_offset]);
                        }
                        else
                        {
                            if ((trsv_blk[tri_index].nnz_lv_array[li] / trsv_blk[tri_index].m_lv_array[li]) <= 15)
                            {
                                trsv_blk[tri_index].num_threads = WARP_PER_BLOCK * WARP_SIZE;
                                trsv_blk[tri_index].num_blocks = ceil((double)(trsv_blk[tri_index].m_lv_array[li]) / (double)(trsv_blk[tri_index].num_threads));
                                sptrsv_levelset_threadsca_csr_cuda_executor<<<trsv_blk[tri_index].num_blocks, trsv_blk[tri_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                                                 trsv_blk[tri_index].m_lv_array[li], trsv_blk[tri_index].m, trsv_blk[tri_index].offset_array[li], trsv_blk[tri_index].substitution, &b_t[b_offset], &x_t[x_offset]);
                            }
                            else
                            {
                                trsv_blk[tri_index].num_threads = WARP_PER_BLOCK * WARP_SIZE;
                                trsv_blk[tri_index].num_blocks = ceil((double)(trsv_blk[tri_index].m_lv_array[li]) / (double)((trsv_blk[tri_index].num_threads) / WARP_SIZE));
                                sptrsv_levelset_warpvec_csr_cuda_executor<<<trsv_blk[tri_index].num_blocks, trsv_blk[tri_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                                               trsv_blk[tri_index].m_lv_array[li], trsv_blk[tri_index].m, trsv_blk[tri_index].offset_array[li], trsv_blk[tri_index].substitution, &b_t[b_offset], &x_t[x_offset]);
                            }
                        }
                    }
                }
                else if (trsv_blk[tri_index].method == 3)
                {
                    sptrsv_syncfree_warpvec_csc_cuda_executor<<<trsv_blk[tri_index].num_blocks, trsv_blk[tri_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                                   trsv_blk[tri_index].d_graphInDegree, trsv_blk[tri_index].d_left_sum,
                                                                                                                                   trsv_blk[tri_index].m, trsv_blk[tri_index].substitution, &b_t[b_offset], &x_t[x_offset], trsv_blk[tri_index].d_while_profiler,
                                                                                                                                   trsv_blk[tri_index].d_id_extractor, trsv_blk[tri_index].d_levelItem);
                }
                tri_index++;

                cudaDeviceSynchronize();
            }
            else
            {
                if (mv_blk[squ_index].method == 0)
                {
                    spmv_threadsca_csr_cuda_executor<<<mv_blk[squ_index].num_blocks, mv_blk[squ_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                      mv_blk[squ_index].m, &x_t[loc_off[i]], mv_blk[squ_index].d_y);
                    if (mv_blk[squ_index].longrow != 0)
                        spmv_longrow_csr_cuda_executor<<<mv_blk[squ_index].num_blocks_l, mv_blk[squ_index].num_threads_l>>>(mv_blk[squ_index].d_csrRowPtr_l, mv_blk[squ_index].d_csrColIdx_l, mv_blk[squ_index].d_csrVal_l,
                                                                                                                            &x_t[loc_off[i]], mv_blk[squ_index].d_y, mv_blk[squ_index].longrow, mv_blk[squ_index].d_longrow_idx);

                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)(blk_m[i]) / (double)num_threads);
                    subKernel<<<num_blocks, num_threads>>>(&(b_t[tmp_off[i]]), mv_blk[squ_index].d_y, blk_m[i]);
                }
                else if (mv_blk[squ_index].method == 1)
                {
                    spmv_threadsca_dcsr_cuda_executor<<<mv_blk[squ_index].num_blocks, mv_blk[squ_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                       mv_blk[squ_index].m_new, &x_t[loc_off[i]], mv_blk[squ_index].d_y, &d_recblock_dcsr_rowidx[dcsrindex_offset[i]]);
                    if (mv_blk[squ_index].longrow != 0)
                        spmv_longrow_csr_cuda_executor<<<mv_blk[squ_index].num_blocks_l, mv_blk[squ_index].num_threads_l>>>(mv_blk[squ_index].d_csrRowPtr_l, mv_blk[squ_index].d_csrColIdx_l, mv_blk[squ_index].d_csrVal_l,
                                                                                                                            &x_t[loc_off[i]], mv_blk[squ_index].d_y, mv_blk[squ_index].longrow, mv_blk[squ_index].d_longrow_idx);

                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)(blk_m[i]) / (double)num_threads);
                    subKernel<<<num_blocks, num_threads>>>(&(b_t[tmp_off[i]]), mv_blk[squ_index].d_y, blk_m[i]);
                }
                else if (mv_blk[squ_index].method == 2)
                {
                    spmv_warpvec_csr_cuda_executor<<<mv_blk[squ_index].num_blocks, mv_blk[squ_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                    mv_blk[squ_index].m, &x_t[loc_off[i]], mv_blk[squ_index].d_y);
                    if (mv_blk[squ_index].longrow != 0)
                        spmv_longrow_csr_cuda_executor<<<mv_blk[squ_index].num_blocks_l, mv_blk[squ_index].num_threads_l>>>(mv_blk[squ_index].d_csrRowPtr_l, mv_blk[squ_index].d_csrColIdx_l, mv_blk[squ_index].d_csrVal_l,
                                                                                                                            &x_t[loc_off[i]], mv_blk[squ_index].d_y, mv_blk[squ_index].longrow, mv_blk[squ_index].d_longrow_idx);

                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)(blk_m[i]) / (double)num_threads);
                    subKernel<<<num_blocks, num_threads>>>(&(b_t[tmp_off[i]]), mv_blk[squ_index].d_y, blk_m[i]);
                }
                else if (mv_blk[squ_index].method == 3)
                {
                    spmv_warpvec_dcsr_cuda_executor<<<mv_blk[squ_index].num_blocks, mv_blk[squ_index].num_threads>>>(&d_recblock_Ptr[ptr_offset[i] - 1], &d_recblock_Index[index_offset[i]], &d_recblock_Val[index_offset[i]],
                                                                                                                     mv_blk[squ_index].m_new, &x_t[loc_off[i]], mv_blk[squ_index].d_y, &d_recblock_dcsr_rowidx[dcsrindex_offset[i]]);
                    if (mv_blk[squ_index].longrow != 0)
                        spmv_longrow_csr_cuda_executor<<<mv_blk[squ_index].num_blocks_l, mv_blk[squ_index].num_threads_l>>>(mv_blk[squ_index].d_csrRowPtr_l, mv_blk[squ_index].d_csrColIdx_l, mv_blk[squ_index].d_csrVal_l,
                                                                                                                            &x_t[loc_off[i]], mv_blk[squ_index].d_y, mv_blk[squ_index].longrow, mv_blk[squ_index].d_longrow_idx);

                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)(blk_m[i]) / (double)num_threads);
                    subKernel<<<num_blocks, num_threads>>>(&(b_t[tmp_off[i]]), mv_blk[squ_index].d_y, blk_m[i]);
                }
                squ_index++;
                cudaDeviceSynchronize();
            }
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        *cal_time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }
    *cal_time /= BENCH_REPEAT;
}

void device_memfree(SpMV_block *mv_blk,
                    SpTRSV_block *trsv_blk,
                    int tri_block,
                    int squ_block)
{
    for (int i = 0; i < tri_block; i++)
    {
        if (trsv_blk[i].method == 0)
        {
        }
        else if (trsv_blk[i].method == 1)
        {
            cudaFree(trsv_blk[i].pBuffer);
        }
        else if (trsv_blk[i].method == 2)
        {
            free(trsv_blk[i].nnz_lv_array);
            free(trsv_blk[i].m_lv_array);
            free(trsv_blk[i].offset_array);
        }
        else if (trsv_blk[i].method == 3)
        {
            cudaFree(trsv_blk[i].d_graphInDegree);
            cudaFree(trsv_blk[i].d_left_sum);
            cudaFree(trsv_blk[i].d_id_extractor);
            cudaFree(trsv_blk[i].d_levelItem);
        }
    }
    for (int i = 0; i < squ_block; i++)
    {
        if (mv_blk[i].method == 0)
        {
            cudaFree(mv_blk[i].d_y);
            if (mv_blk[i].longrow != 0)
            {
                cudaFree(mv_blk[i].d_csrRowPtr_l);
                cudaFree(mv_blk[i].d_csrColIdx_l);
                cudaFree(mv_blk[i].d_csrVal_l);
                cudaFree(mv_blk[i].d_longrow_idx);
            }
        }
        else if (mv_blk[i].method == 1)
        {
            cudaFree(mv_blk[i].d_y);
            if (mv_blk[i].longrow != 0)
            {
                cudaFree(mv_blk[i].d_csrRowPtr_l);
                cudaFree(mv_blk[i].d_csrColIdx_l);
                cudaFree(mv_blk[i].d_csrVal_l);
                cudaFree(mv_blk[i].d_longrow_idx);
            }
        }
        else if (mv_blk[i].method == 2)
        {
            cudaFree(mv_blk[i].d_y);
            if (mv_blk[i].longrow != 0)
            {
                cudaFree(mv_blk[i].d_csrRowPtr_l);
                cudaFree(mv_blk[i].d_csrColIdx_l);
                cudaFree(mv_blk[i].d_csrVal_l);
                cudaFree(mv_blk[i].d_longrow_idx);
            }
        }
        else if (mv_blk[i].method == 3)
        {
            cudaFree(mv_blk[i].d_y);
            if (mv_blk[i].longrow != 0)
            {
                cudaFree(mv_blk[i].d_csrRowPtr_l);
                cudaFree(mv_blk[i].d_csrColIdx_l);
                cudaFree(mv_blk[i].d_csrVal_l);
                cudaFree(mv_blk[i].d_longrow_idx);
            }
        }
    }
    free(mv_blk);
    free(trsv_blk);
}

#endif