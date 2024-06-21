#ifndef __RECBLOCKING_SOLVER_CUDA__
#define __RECBLOCKING_SOLVER_CUDA__
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"
#include "recblocking_preprocess.h"
#include "recblocking_calculate.h"
#include "utils_spmv_cuda.h"
#include "utils_sptrsv_cuda.h"
#include "tranpose.h"
#include "utils_reordering.h"
#include "findlevel.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "utils_cuda.h"

void recblocking_solver_cuda(int *d_cscColPtrTR,
                             int *d_cscRowIdxTR,
                             VALUE_TYPE *d_cscValTR,
                             int m,
                             int n,
                             int nnzTR,
                             VALUE_TYPE *d_x,
                             VALUE_TYPE *d_b,
                             int substitution,
                             int lv,
                             double *cal_time,
                             double *preprocess_time)
{
    int tri_block = pow(2, lv);
    int squ_block = tri_block - 1;
    int sum_block = tri_block + squ_block;

    SpTRSV_block *trsv_blk = (SpTRSV_block *)malloc(sizeof(SpTRSV_block) * tri_block);
    SpMV_block *mv_blk = (SpMV_block *)malloc(sizeof(SpMV_block) * squ_block);
    for (int i = 0; i < squ_block; i++)
        mv_blk[i].method = -1;

    int *blk_m = (int *)malloc(sizeof(int) * (squ_block + tri_block));
    int *blk_n = (int *)malloc(sizeof(int) * (squ_block + tri_block));

    int *blk_nnz = (int *)malloc(sizeof(int) * (squ_block + tri_block));
    int *d_blk_nnz;
    cudaMalloc((void **)&d_blk_nnz, sizeof(int) * (squ_block + tri_block));
    cudaMemset(d_blk_nnz, 0, sizeof(int) * (squ_block + tri_block));

    int *loc_off = (int *)malloc(sizeof(int) * (squ_block + tri_block));
    memset(loc_off, 0, sizeof(int) * (squ_block + tri_block));
    int *tmp_off = (int *)malloc(sizeof(int) * (squ_block + tri_block));
    memset(tmp_off, 0, sizeof(int) * (squ_block + tri_block));
    int *subtri_upbound = (int *)malloc(sizeof(int) * sum_block);
    int *subtri_downbound = (int *)malloc(sizeof(int) * sum_block);
    int *subrec_upbound = (int *)malloc(sizeof(int) * sum_block);
    int *subrec_downbound = (int *)malloc(sizeof(int) * sum_block);
    int *subrec_rightbound = (int *)malloc(sizeof(int) * sum_block);
    int *subrec_leftbound = (int *)malloc(sizeof(int) * sum_block);

    int *d_recblock_Ptr;
    int *d_recblock_Index;
    int *d_recblock_dcsr_rowidx;
    double *d_recblock_Val;
    int *ptr_offset;
    int *index_offset;
    int *dcsrindex_offset;
    int ptr_size = 1;
    int idx_size = 0;
    int dcsr_size = 0;

    if (substitution == SUBSTITUTION_FORWARD)
    {
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);

        int *d_cscColPtrTR_new;
        int *d_cscRowIdxTR_new;
        VALUE_TYPE *d_cscValTR_new;
        cudaMalloc((void **)&d_cscColPtrTR_new, (n + 1) * sizeof(int));
        cudaMalloc((void **)&d_cscRowIdxTR_new, nnzTR * sizeof(int));
        cudaMalloc((void **)&d_cscValTR_new, nnzTR * sizeof(VALUE_TYPE));

        // ----------------levelset_reordering_colrow_csc-----------------
        int *d_levelItem;
        cudaMalloc((void **)&d_levelItem, m * sizeof(int));
        levelset_reordering_colrow_csc_cuda(d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
                                            d_cscColPtrTR_new, d_cscRowIdxTR_new, d_cscValTR_new,
                                            d_levelItem, m, n, nnzTR, substitution);

        // ---------------------reorder end----------------------
        mat_preprocessing_cuda(d_cscColPtrTR_new, d_cscRowIdxTR_new, d_cscValTR_new, m, n,
                               lv, loc_off, tmp_off, blk_m, blk_n, d_blk_nnz, subtri_upbound,
                               subtri_downbound, subrec_upbound, subrec_downbound, subrec_rightbound,
                               subrec_leftbound, substitution);
        cudaMemcpy(blk_nnz, d_blk_nnz, sizeof(int) * (squ_block + tri_block), cudaMemcpyDeviceToHost);

        for (int i = 0; i < sum_block; i++)
        {
            if (i % 2 == 0)
            {
                ptr_size += blk_n[i];
                idx_size += blk_nnz[i];
                ptr_size += 1;
            }
            else
            {
                ptr_size += blk_m[i];
                idx_size += blk_nnz[i];
                dcsr_size += blk_m[i];
            }
        }
        // ---------------------get_recblock_size end--------------------------

        cudaMalloc((void **)&d_recblock_Ptr, ptr_size * sizeof(int));
        cudaMemset(d_recblock_Ptr, 0, sizeof(int));
        cudaMalloc((void **)&d_recblock_Index, idx_size * sizeof(int));
        cudaMalloc((void **)&d_recblock_dcsr_rowidx, dcsr_size * sizeof(int));
        cudaMalloc((void **)&d_recblock_Val, idx_size * sizeof(double));
        ptr_offset = (int *)malloc(sizeof(int) * (sum_block + 1));
        index_offset = (int *)malloc(sizeof(int) * (sum_block + 1));
        dcsrindex_offset = (int *)malloc(sizeof(int) * (sum_block + 1));
        dcsrindex_offset[0] = 0;
        ptr_offset[0] = 1;
        index_offset[0] = 0;

        int blk_count = 0;
        // store sub-matrix into device
        int trsv_count = 0;
        int mv_count = 0;
        int recblock_nnz_ptr = 0;
        for (blk_count = 0; blk_count < sum_block; blk_count++)
        {
            if (blk_count % 2 == 0)
            {
                int cu_flag = 0;
                int *d_cscColPtrTR_sub;
                int *d_cscRowIdxTR_sub;
                VALUE_TYPE *d_cscValTR_sub;
                cudaMalloc((void **)&d_cscColPtrTR_sub, sizeof(int) * (blk_n[blk_count] + 1));
                cudaMalloc((void **)&d_cscRowIdxTR_sub, sizeof(int) * blk_nnz[blk_count]);
                cudaMalloc((void **)&d_cscValTR_sub, sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

                int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                int num_blocks = ceil((double)(blk_m[blk_count]) / (double)num_threads);
                int upbound = subtri_upbound[blk_count];
                int downbound = subtri_downbound[blk_count];
                store_into_subtrimat_ptr<<<num_blocks, num_threads>>>(upbound, downbound, d_cscColPtrTR_new,
                                                                      d_cscRowIdxTR_new, d_cscColPtrTR_sub, substitution);
                thrust::exclusive_scan(thrust::device, d_cscColPtrTR_sub,
                                       d_cscColPtrTR_sub + blk_n[blk_count] + 1, d_cscColPtrTR_sub, 0);
                store_into_subtrimat_idxval<<<num_blocks, num_threads>>>(upbound, downbound, d_cscColPtrTR_new, d_cscRowIdxTR_new,
                                                                         d_cscValTR_new, d_cscColPtrTR_sub, d_cscRowIdxTR_sub,
                                                                         d_cscValTR_sub, substitution);

                int *d_csrRowPtrTR_sub;
                int *d_csrColIdxTR_sub;
                VALUE_TYPE *d_csrValTR_sub;
                cudaMalloc((void **)&d_csrRowPtrTR_sub, sizeof(int) * (blk_m[blk_count] + 1));
                cudaMalloc((void **)&d_csrColIdxTR_sub, sizeof(int) * blk_nnz[blk_count]);
                cudaMalloc((void **)&d_csrValTR_sub, sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

                // -------------------matrix_transposition-------------------
                matrix_transposition_cuda(blk_n[blk_count], blk_m[blk_count], blk_nnz[blk_count],
                                          d_cscColPtrTR_sub, d_cscRowIdxTR_sub, d_cscValTR_sub,
                                          d_csrColIdxTR_sub, d_csrRowPtrTR_sub, d_csrValTR_sub);
                // ----------------------------------------------------------
                int *d_nlv;
                cudaMalloc((void **)(&d_nlv), sizeof(int));
                cudaMemset(d_nlv, 0, sizeof(int));
                int *d_levelItem_local;
                int *d_levelPtr_local;
                cudaMalloc((void **)&d_levelItem_local, blk_m[blk_count] * sizeof(int));
                cudaMalloc((void **)&d_levelPtr_local, (blk_m[blk_count] + 1) * sizeof(int));
                int fasttrack = blk_m[blk_count] == blk_nnz[blk_count] ? 1 : 0;

                if (fasttrack)
                    cudaMemset(d_nlv, 1, sizeof(int));
                else
                    findlevel_cuda(d_cscColPtrTR_sub, d_cscRowIdxTR_sub, d_csrRowPtrTR_sub, blk_m[blk_count],
                                   d_nlv, d_levelPtr_local, d_levelItem_local);
                int nlv;
                cudaMemcpy(&nlv, d_nlv, sizeof(int), cudaMemcpyDeviceToHost);
                if (fasttrack)
                {
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)blk_m[blk_count] / (double)num_threads);
                    ((trsv_blk[trsv_count])).method = 0;
                    (trsv_blk[trsv_count]).num_threads = num_threads;
                    (trsv_blk[trsv_count]).num_blocks = num_blocks;
                    (trsv_blk[trsv_count]).m = blk_m[blk_count];
                    (trsv_blk[trsv_count]).substitution = substitution;

                    num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    num_blocks = ceil((double)blk_n[blk_count] / (double)num_threads);
                    pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_n[blk_count], d_cscColPtrTR_sub, d_recblock_Ptr + ptr_offset[blk_count] - 1);

                    thrust::exclusive_scan(thrust::device, d_recblock_Ptr + ptr_offset[blk_count] - 1,
                                           d_recblock_Ptr + ptr_offset[blk_count] + blk_n[blk_count], d_recblock_Ptr + ptr_offset[blk_count] - 1, recblock_nnz_ptr);

                    store_to_recblockdata<<<num_blocks, num_threads>>>(blk_n[blk_count], d_cscColPtrTR_sub, d_cscRowIdxTR_sub,
                                                                       d_cscValTR_sub, d_recblock_Index, d_recblock_Val, d_recblock_Ptr + ptr_offset[blk_count] - 1, recblock_nnz_ptr);
                }
                else
                {
                    int nnzr = blk_nnz[blk_count] / blk_m[blk_count];

                    if (nlv > 20000)
                    {
                        cusparseStatus_t status;
                        (trsv_blk[trsv_count]).handle = 0;
                        status = cusparseCreate(&(trsv_blk[trsv_count].handle));
                        // http://docs.nvidia.com/cuda/cusparse/#cusparse-lt-t-gt-csrsv2_solve
                        // Suppose that L is m x m sparse matrix represented by CSR format,
                        // L is lower triangular with unit diagonal.
                        // Assumption:
                        // - dimension of matrix L is m,
                        // - matrix L has nnz number zero elements,
                        // - handle is already created by cusparseCreate(),
                        // - (d_csrRowPtr, d_csrColInd, d_csrVal) is CSR of L on device memory,
                        // - d_b is right hand side vector on device memory,
                        // - d_x is solution vector on device memory.
                        (trsv_blk[trsv_count]).descr = 0;
                        (trsv_blk[trsv_count]).info = 0;
                        int pBufferSize;
                        (trsv_blk[trsv_count]).pBuffer = 0;
                        int structural_zero;
                        int numerical_zero;
                        (trsv_blk[trsv_count]).alpha_double = 1.;
                        (trsv_blk[trsv_count]).alpha_float = 1.;
                        (trsv_blk[trsv_count]).policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
                        (trsv_blk[trsv_count]).trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

                        // step 1: create a descriptor which contains
                        // - matrix L is base-0
                        // - matrix L is lower triangular
                        // - matrix L has unit diagonal, specified by parameter CUSPARSE_DIAG_TYPE_UNIT
                        //   (L may not have all diagonal elements.)
                        cusparseCreateMatDescr(&(trsv_blk[trsv_count].descr));
                        cusparseSetMatIndexBase((trsv_blk[trsv_count]).descr, CUSPARSE_INDEX_BASE_ZERO);

                        if (substitution == SUBSTITUTION_FORWARD)
                            cusparseSetMatFillMode((trsv_blk[trsv_count]).descr, CUSPARSE_FILL_MODE_LOWER);
                        else if (substitution == SUBSTITUTION_BACKWARD)
                            cusparseSetMatFillMode((trsv_blk[trsv_count]).descr, CUSPARSE_FILL_MODE_UPPER);

                        cusparseSetMatDiagType((trsv_blk[trsv_count]).descr, CUSPARSE_DIAG_TYPE_UNIT);
                        // step 2: create a empty info structure
                        cusparseCreateCsrsv2Info(&(trsv_blk[trsv_count].info));

                        // step 3: query how much memory used in csrsv2, and allocate the buffer
                        if (sizeof(VALUE_TYPE) == 8)
                            cusparseDcsrsv2_bufferSize((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                       (double *)d_csrValTR_sub, d_csrRowPtrTR_sub, d_csrColIdxTR_sub, (trsv_blk[trsv_count]).info, &pBufferSize);
                        else if (sizeof(VALUE_TYPE) == 4)
                            cusparseScsrsv2_bufferSize((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                       (float *)d_csrValTR_sub, d_csrRowPtrTR_sub, d_csrColIdxTR_sub, (trsv_blk[trsv_count]).info, &pBufferSize);
                        // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
                        cudaMalloc((void **)&(trsv_blk[trsv_count].pBuffer), pBufferSize);
                        if (sizeof(VALUE_TYPE) == 8)
                            cusparseDcsrsv2_analysis((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                     (double *)d_csrValTR_sub, d_csrRowPtrTR_sub, d_csrColIdxTR_sub,
                                                     (trsv_blk[trsv_count]).info, (trsv_blk[trsv_count]).policy, (trsv_blk[trsv_count]).pBuffer);
                        else if (sizeof(VALUE_TYPE) == 4)
                            cusparseScsrsv2_analysis((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                     (float *)d_csrValTR_sub, d_csrRowPtrTR_sub, d_csrColIdxTR_sub,
                                                     (trsv_blk[trsv_count]).info, (trsv_blk[trsv_count]).policy, (trsv_blk[trsv_count]).pBuffer);

                        // L has unit diagonal, so no structural zero is reported.
                        status = cusparseXcsrsv2_zeroPivot((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).info, &structural_zero);
                        if (CUSPARSE_STATUS_ZERO_PIVOT == status)
                        {
                            printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
                        }

                        (trsv_blk[trsv_count]).method = 1;
                        (trsv_blk[trsv_count]).m = blk_m[blk_count];
                        (trsv_blk[trsv_count]).nnzTR = blk_nnz[blk_count];

                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)blk_m[blk_count] / (double)num_threads);
                        pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_m[blk_count], d_csrRowPtrTR_sub, d_recblock_Ptr + ptr_offset[blk_count]);

                        thrust::exclusive_scan(thrust::device, d_recblock_Ptr + ptr_offset[blk_count],
                                               d_recblock_Ptr + ptr_offset[blk_count] + blk_m[blk_count] + 1, d_recblock_Ptr + ptr_offset[blk_count], 0);

                        store_to_recblockdata<<<num_blocks, num_threads>>>(blk_m[blk_count], d_csrRowPtrTR_sub, d_csrColIdxTR_sub,
                                                                           d_csrValTR_sub, d_recblock_Index, d_recblock_Val, d_recblock_Ptr + ptr_offset[blk_count], recblock_nnz_ptr);
                        cu_flag = 1;
                    }
                    else if ((nnzr <= 15 && nlv <= 20) || (nnzr == 1 && nlv <= 100))
                    {
                        (trsv_blk[trsv_count]).method = 2;
                        (trsv_blk[trsv_count]).m = blk_m[blk_count];
                        (trsv_blk[trsv_count]).substitution = substitution;
                        (trsv_blk[trsv_count]).nlv = nlv;
                        (trsv_blk[trsv_count]).nnz_lv_array = (int *)malloc(sizeof(int) * nlv);
                        (trsv_blk[trsv_count]).m_lv_array = (int *)malloc(sizeof(int) * nlv);
                        (trsv_blk[trsv_count]).offset_array = (int *)malloc(sizeof(int) * nlv);
                        int *levelPtr_local = (int *)malloc((blk_m[blk_count] + 1) * sizeof(int));
                        int *levelItem_local = (int *)malloc(blk_m[blk_count] * sizeof(int));
                        cudaMemcpy(levelPtr_local, d_levelPtr_local, (blk_m[blk_count] + 1) * sizeof(int), cudaMemcpyDeviceToHost);
                        cudaMemcpy(levelItem_local, d_levelItem_local, (blk_m[blk_count]) * sizeof(int), cudaMemcpyDeviceToHost);
                        for (int li = 0; li < nlv; li++)
                        {
                            (trsv_blk[trsv_count]).m_lv_array[li] = levelPtr_local[li + 1] - levelPtr_local[li];
                            (trsv_blk[trsv_count]).offset_array[li] = levelPtr_local[li];
                        }

                        int *d_nnz_lv_array;
                        cudaMalloc((void **)&d_nnz_lv_array, (nlv + 1) * sizeof(int));
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)nlv / (double)num_threads);
                        find_level_nnz<<<num_blocks, num_threads>>>(d_csrRowPtrTR_sub, d_levelPtr_local, d_nnz_lv_array, nlv);

                        cudaMemcpy((trsv_blk[trsv_count]).nnz_lv_array, d_nnz_lv_array, sizeof(int) * nlv, cudaMemcpyDeviceToHost);

                        cudaFree(d_nnz_lv_array);

                        num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        num_blocks = ceil((double)blk_m[blk_count] / (double)num_threads);
                        pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_n[blk_count], d_csrRowPtrTR_sub, d_recblock_Ptr + ptr_offset[blk_count] - 1);

                        thrust::exclusive_scan(thrust::device, d_recblock_Ptr + ptr_offset[blk_count] - 1,
                                               d_recblock_Ptr + ptr_offset[blk_count] + blk_n[blk_count], d_recblock_Ptr + ptr_offset[blk_count] - 1, recblock_nnz_ptr);

                        store_to_recblockdata<<<num_blocks, num_threads>>>(blk_n[blk_count], d_csrRowPtrTR_sub, d_csrColIdxTR_sub,
                                                                           d_csrValTR_sub, d_recblock_Index, d_recblock_Val, d_recblock_Ptr + ptr_offset[blk_count] - 1, recblock_nnz_ptr);

                        free(levelPtr_local);
                        free(levelItem_local);
                    }
                    else
                    {
                        cudaMalloc((void **)&(trsv_blk[trsv_count]).d_levelItem, blk_m[blk_count] * sizeof(int));
                        cudaMemcpy((trsv_blk[trsv_count]).d_levelItem, d_levelItem_local, blk_m[blk_count] * sizeof(int), cudaMemcpyDeviceToDevice);

                        cudaMalloc((void **)&(trsv_blk[trsv_count]).d_graphInDegree, blk_m[blk_count] * sizeof(int));
                        cudaMemset((trsv_blk[trsv_count]).d_graphInDegree, 0, blk_m[blk_count] * sizeof(int));

                        cudaMalloc((void **)&(trsv_blk[trsv_count]).d_id_extractor, sizeof(int));
                        cudaMemset((trsv_blk[trsv_count]).d_id_extractor, 0, sizeof(int));

                        int num_threads = 128;
                        int num_blocks = ceil((double)blk_nnz[blk_count] / (double)num_threads);
                        sptrsv_syncfree_csc_cuda_analyser<<<num_blocks, num_threads>>>(d_cscRowIdxTR_sub, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).d_graphInDegree);
                        cudaDeviceSynchronize();
                        cudaMalloc((void **)&(trsv_blk[trsv_count]).d_left_sum, sizeof(VALUE_TYPE) * blk_m[blk_count]);
                        cudaMemset((trsv_blk[trsv_count]).d_left_sum, 0, sizeof(VALUE_TYPE) * blk_m[blk_count]);

                        num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        num_blocks = ceil((double)blk_m[blk_count] / (double)(num_threads / WARP_SIZE));
                        (trsv_blk[trsv_count]).method = 3;
                        (trsv_blk[trsv_count]).num_threads = num_threads;
                        (trsv_blk[trsv_count]).num_blocks = num_blocks;
                        (trsv_blk[trsv_count]).m = blk_m[blk_count];
                        (trsv_blk[trsv_count]).substitution = substitution;
                        
                        num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        num_blocks = ceil((double)blk_n[blk_count] / (double)num_threads);
                        pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_n[blk_count], d_cscColPtrTR_sub, d_recblock_Ptr + ptr_offset[blk_count] - 1);
                        thrust::exclusive_scan(thrust::device, d_recblock_Ptr + ptr_offset[blk_count] - 1,
                                               d_recblock_Ptr + ptr_offset[blk_count] + blk_n[blk_count], d_recblock_Ptr + ptr_offset[blk_count] - 1, recblock_nnz_ptr);
                        store_to_recblockdata<<<num_blocks, num_threads>>>(blk_n[blk_count], d_cscColPtrTR_sub, d_cscRowIdxTR_sub,
                                                                           d_cscValTR_sub, d_recblock_Index, d_recblock_Val, d_recblock_Ptr + ptr_offset[blk_count] - 1, recblock_nnz_ptr);
                    }
                }

                cudaFree(d_levelPtr_local);
                cudaFree(d_levelItem_local);

                if (cu_flag == 0)
                    ptr_offset[blk_count + 1] = ptr_offset[blk_count] + blk_m[blk_count];
                else
                    ptr_offset[blk_count + 1] = ptr_offset[blk_count] + blk_m[blk_count] + 1;
                recblock_nnz_ptr += blk_nnz[blk_count];
                index_offset[blk_count + 1] = recblock_nnz_ptr;
                dcsrindex_offset[blk_count + 1] = dcsrindex_offset[blk_count];

                cudaFree(d_csrRowPtrTR_sub);
                cudaFree(d_csrValTR_sub);
                cudaFree(d_csrColIdxTR_sub);
                cudaFree(d_cscColPtrTR_sub);
                cudaFree(d_cscRowIdxTR_sub);
                cudaFree(d_cscValTR_sub);

                trsv_count++;

                cudaDeviceSynchronize();
            }
            else
            {
                int *d_cscColPtrTR_sub;
                int *d_cscRowIdxTR_sub;
                VALUE_TYPE *d_cscValTR_sub;
                cudaMalloc((void **)&d_cscColPtrTR_sub, sizeof(int) * (blk_n[blk_count] + 1));
                cudaMalloc((void **)&d_cscRowIdxTR_sub, sizeof(int) * blk_nnz[blk_count]);
                cudaMalloc((void **)&d_cscValTR_sub, sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

                int *d_csrRowPtrTR_sub;
                int *d_csrColIdxTR_sub;
                VALUE_TYPE *d_csrValTR_sub;
                cudaMalloc((void **)&d_csrRowPtrTR_sub, sizeof(int) * (blk_m[blk_count] + 1));
                cudaMalloc((void **)&d_csrColIdxTR_sub, sizeof(int) * blk_nnz[blk_count]);
                cudaMalloc((void **)&d_csrValTR_sub, sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

                int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                int num_blocks = ceil((double)blk_m[blk_count] / (double)(num_threads));
                int upbound = subrec_upbound[blk_count];
                int downbound = subrec_downbound[blk_count];
                int leftbound = subrec_leftbound[blk_count];
                int rightbound = subrec_rightbound[blk_count];
                store_into_subrecmat_ptr<<<num_blocks, num_threads>>>(upbound, downbound, leftbound, rightbound,
                                                                      d_cscColPtrTR_new, d_cscRowIdxTR_new, d_cscColPtrTR_sub);

                thrust::exclusive_scan(thrust::device, d_cscColPtrTR_sub,
                                       d_cscColPtrTR_sub + blk_n[blk_count] + 1, d_cscColPtrTR_sub, 0);

                store_into_subrecmat_idxval<<<num_blocks, num_threads>>>(upbound, downbound, leftbound, rightbound, d_cscColPtrTR_new, d_cscRowIdxTR_new,
                                                                         d_cscValTR_new, d_cscColPtrTR_sub, d_cscRowIdxTR_sub, d_cscValTR_sub);

                matrix_transposition_cuda(blk_n[blk_count], blk_m[blk_count], blk_nnz[blk_count],
                                          d_cscColPtrTR_sub, d_cscRowIdxTR_sub, d_cscValTR_sub,
                                          d_csrColIdxTR_sub, d_csrRowPtrTR_sub, d_csrValTR_sub);

                int *idx_offset;
                cudaMalloc((void **)(&idx_offset), sizeof(int) * (blk_m[blk_count] + 1));
                num_threads = WARP_PER_BLOCK * WARP_SIZE;
                num_blocks = ceil((double)blk_m[blk_count] / (double)num_threads);

                pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_m[blk_count], d_csrRowPtrTR_sub, idx_offset);

                thrust::exclusive_scan(thrust::device, idx_offset,
                                       idx_offset + blk_m[blk_count] + 1, idx_offset, recblock_nnz_ptr);

                store_to_recblockdata<<<num_blocks, num_threads>>>(blk_m[blk_count], d_csrRowPtrTR_sub, d_csrColIdxTR_sub,
                                                                   d_csrValTR_sub, d_recblock_Index, d_recblock_Val, idx_offset, recblock_nnz_ptr);
                cudaFree(idx_offset);

                int *d_i_new;
                int *d_lenmax;
                int *d_longrow;
                int *d_longlen;
                int *d_longrow_idx;
                cudaMalloc((void **)(&d_i_new), sizeof(int));
                cudaMalloc((void **)(&d_lenmax), sizeof(int));
                cudaMalloc((void **)(&d_longrow), sizeof(int));
                cudaMalloc((void **)(&d_longlen), sizeof(int));
                cudaMalloc((void **)(&d_longrow_idx), blk_m[blk_count] * sizeof(int));

                cal_longrow<<<1, 1>>>(d_i_new, d_lenmax, d_longrow, d_longlen,
                                      d_longrow_idx, d_csrRowPtrTR_sub, blk_m[blk_count]);

                int m_new;
                cudaMemcpy(&m_new, d_i_new, sizeof(int), cudaMemcpyDeviceToHost);
                m_new--;
                int lenmax;
                cudaMemcpy(&lenmax, d_lenmax, sizeof(int), cudaMemcpyDeviceToHost);
                int longrow;
                cudaMemcpy(&longrow, d_longrow, sizeof(int), cudaMemcpyDeviceToHost);
                int longlen;
                cudaMemcpy(&longlen, d_longlen, sizeof(int), cudaMemcpyDeviceToHost);
                int nnzr = (blk_nnz[blk_count] - longlen) / m_new;
                double empty_ratio = 100 * (double)(blk_m[blk_count] - m_new) / (double)blk_m[blk_count];
                int dcsr_i = 0;
                int real_i = 0;
                if (blk_nnz[blk_count] != 0)
                {
                    int m = blk_m[blk_count];
                    int nnz = blk_nnz[blk_count];
                    cudaMalloc((void **)&((mv_blk[mv_count]).d_y), m * sizeof(VALUE_TYPE));
                    cudaMemset((mv_blk[mv_count].d_y), 0, m * sizeof(VALUE_TYPE));
                    if (nnzr <= 12 && empty_ratio <= 50)
                    {
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)m / (double)num_threads);
                        (mv_blk[mv_count]).method = 0;
                        (mv_blk[mv_count]).num_threads = num_threads;
                        (mv_blk[mv_count]).num_blocks = num_blocks;
                        (mv_blk[mv_count]).m = m;

                        int init_val;
                        cudaMemcpy(&init_val, d_recblock_Ptr + ptr_offset[blk_count] - 1, sizeof(int), cudaMemcpyDeviceToHost);

                        pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_m[blk_count], d_csrRowPtrTR_sub, d_recblock_Ptr + ptr_offset[blk_count] - 1);
                        thrust::exclusive_scan(thrust::device, d_recblock_Ptr + ptr_offset[blk_count] - 1,
                                               d_recblock_Ptr + ptr_offset[blk_count] + blk_m[blk_count], d_recblock_Ptr + ptr_offset[blk_count] - 1, init_val);
                        real_i = m;
                    }
                    else if (nnzr <= 12 && empty_ratio > 50)
                    {
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)m_new / (double)num_threads);
                        (mv_blk[mv_count]).method = 1;
                        (mv_blk[mv_count]).num_threads = num_threads;
                        (mv_blk[mv_count]).num_blocks = num_blocks;
                        (mv_blk[mv_count]).m_new = m_new;

                        int *d_dcsr_i;
                        cudaMalloc((void **)(&d_dcsr_i), sizeof(int));
                        cudaMemcpy(d_dcsr_i, &dcsr_i, sizeof(int), cudaMemcpyHostToDevice);
                        dcsr_recblockdata_ptr<<<1, 1>>>(m, d_csrRowPtrTR_sub, ptr_offset[blk_count],
                                                        dcsrindex_offset[blk_count], d_dcsr_i, d_recblock_Ptr,
                                                        d_recblock_dcsr_rowidx);
                        cudaMemcpy(&dcsr_i, d_dcsr_i, sizeof(int), cudaMemcpyDeviceToHost);
                        cudaFree(d_dcsr_i);
                        real_i = dcsr_i;
                    }
                    else if (nnzr > 12 && empty_ratio <= 15)
                    {
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)m / (double)(num_threads / WARP_SIZE));
                        (mv_blk[mv_count]).method = 2;
                        (mv_blk[mv_count]).num_threads = num_threads;
                        (mv_blk[mv_count]).num_blocks = num_blocks;
                        (mv_blk[mv_count]).m = m;

                        int init_val;
                        cudaMemcpy(&init_val, d_recblock_Ptr + ptr_offset[blk_count] - 1, sizeof(int), cudaMemcpyDeviceToHost);

                        pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_m[blk_count], d_csrRowPtrTR_sub, d_recblock_Ptr + ptr_offset[blk_count] - 1);
                        thrust::exclusive_scan(thrust::device, d_recblock_Ptr + ptr_offset[blk_count] - 1,
                                               d_recblock_Ptr + ptr_offset[blk_count] + blk_m[blk_count], d_recblock_Ptr + ptr_offset[blk_count] - 1, init_val);
                        real_i = blk_m[blk_count];
                    }
                    else
                    {
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)m_new / (double)(num_threads / WARP_SIZE));

                        (mv_blk[mv_count]).method = 3;
                        (mv_blk[mv_count]).num_threads = num_threads;
                        (mv_blk[mv_count]).num_blocks = num_blocks;
                        (mv_blk[mv_count]).m_new = m_new;

                        int *d_dcsr_i;
                        cudaMalloc((void **)(&d_dcsr_i), sizeof(int));
                        cudaMemcpy(d_dcsr_i, &dcsr_i, sizeof(int), cudaMemcpyHostToDevice);
                        dcsr_recblockdata_ptr<<<1, 1>>>(m, d_csrRowPtrTR_sub, ptr_offset[blk_count],
                                                        dcsrindex_offset[blk_count], d_dcsr_i, d_recblock_Ptr,
                                                        d_recblock_dcsr_rowidx);
                        cudaMemcpy(&dcsr_i, d_dcsr_i, sizeof(int), cudaMemcpyDeviceToHost);
                        cudaFree(d_dcsr_i);
                        real_i = dcsr_i;
                    }

                    (mv_blk[mv_count]).longrow = longrow;
                    if (longrow != 0)
                    {
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)lenmax / (double)num_threads);
                        (mv_blk[mv_count]).num_threads_l = num_threads;
                        (mv_blk[mv_count]).num_blocks_l = num_blocks;
                        cudaMalloc((void **)&((mv_blk[mv_count]).d_csrRowPtr_l), (m + 1) * sizeof(int));
                        cudaMalloc((void **)&((mv_blk[mv_count]).d_csrColIdx_l), nnz * sizeof(int));
                        cudaMalloc((void **)&((mv_blk[mv_count]).d_csrVal_l), nnz * sizeof(VALUE_TYPE));
                        cudaMemcpy((mv_blk[mv_count]).d_csrRowPtr_l, d_csrRowPtrTR_sub, (m + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
                        cudaMemcpy((mv_blk[mv_count]).d_csrColIdx_l, d_csrColIdxTR_sub, nnz * sizeof(int), cudaMemcpyDeviceToDevice);
                        cudaMemcpy((mv_blk[mv_count]).d_csrVal_l, d_csrValTR_sub, nnz * sizeof(VALUE_TYPE), cudaMemcpyDeviceToDevice);

                        cudaMalloc((void **)&((mv_blk[mv_count]).d_longrow_idx), longrow * sizeof(int));
                        cudaMemcpy((mv_blk[mv_count]).d_longrow_idx, d_longrow_idx, longrow * sizeof(int), cudaMemcpyDeviceToDevice);
                    }
                }

                ptr_offset[blk_count + 1] = ptr_offset[blk_count] + real_i;
                recblock_nnz_ptr += blk_nnz[blk_count];
                index_offset[blk_count + 1] = recblock_nnz_ptr;
                dcsrindex_offset[blk_count + 1] = dcsrindex_offset[blk_count] + dcsr_i;

                cudaFree(d_csrRowPtrTR_sub);
                cudaFree(d_csrColIdxTR_sub);
                cudaFree(d_csrValTR_sub);
                cudaFree(d_cscRowIdxTR_sub);
                cudaFree(d_cscColPtrTR_sub);
                cudaFree(d_cscValTR_sub);

                mv_count++;
            }
        }

        gettimeofday(&t2, NULL);
        *preprocess_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        VALUE_TYPE *d_b_perm;
        cudaMalloc((void **)&d_b_perm, m * sizeof(VALUE_TYPE));
        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
        int num_blocks = ceil((double)m / (double)num_threads);
        levelset_reordering_vecb_cuda<<<num_blocks, num_threads>>>(d_b, d_b_perm, d_levelItem, m);

        cudaDeviceSynchronize();
        int rhs = 1;
        L_calculate(mv_blk, trsv_blk, sum_block, blk_m, blk_n, loc_off, tmp_off,
                    m, rhs, d_x, d_b, d_b_perm, d_recblock_Ptr, d_recblock_Index, d_recblock_dcsr_rowidx,
                    d_recblock_Val, ptr_offset, index_offset, dcsrindex_offset, cal_time);

        VALUE_TYPE *d_x_perm;
        cudaMalloc((void **)(&d_x_perm), sizeof(VALUE_TYPE) * n);
        cudaMemcpy(d_x_perm, d_x, rhs * n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToDevice);
        num_threads = WARP_PER_BLOCK * WARP_SIZE;
        num_blocks = ceil((double)n / (double)num_threads);
        levelset_reordering_vecx_cuda<<<num_blocks, num_threads>>>(d_x_perm, d_x, d_levelItem, n);

        free(ptr_offset);
        free(index_offset);
        free(dcsrindex_offset);
        cudaFree(d_b_perm);
        cudaFree(d_x_perm);
        cudaFree(d_recblock_Ptr);
        cudaFree(d_recblock_Index);
        cudaFree(d_recblock_dcsr_rowidx);
        cudaFree(d_recblock_Val);
    }
    else
    {
        int *d_cscColPtrTR_new;
        int *d_cscRowIdxTR_new;
        VALUE_TYPE *d_cscValTR_new;
        cudaMalloc((void **)&d_cscColPtrTR_new, (n + 1) * sizeof(int));
        cudaMalloc((void **)&d_cscRowIdxTR_new, nnzTR * sizeof(int));
        cudaMalloc((void **)&d_cscValTR_new, nnzTR * sizeof(VALUE_TYPE));

        // ----------------levelset_reordering_colrow_csc-----------------
        int *d_levelItem;
        cudaMalloc((void **)&d_levelItem, m * sizeof(int));
        levelset_reordering_colrow_csc_cuda(d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
                                            d_cscColPtrTR_new, d_cscRowIdxTR_new, d_cscValTR_new,
                                            d_levelItem, m, n, nnzTR, substitution);

        // ---------------------reorder end----------------------
        mat_preprocessing_cuda(d_cscColPtrTR_new, d_cscRowIdxTR_new, d_cscValTR_new, m, n,
                               lv, loc_off, tmp_off, blk_m, blk_n, d_blk_nnz, subtri_upbound,
                               subtri_downbound, subrec_upbound, subrec_downbound, subrec_rightbound,
                               subrec_leftbound, substitution);
        cudaMemcpy(blk_nnz, d_blk_nnz, sizeof(int) * (squ_block + tri_block), cudaMemcpyDeviceToHost);

        for (int i = 0; i < sum_block; i++)
        {
            if (i % 2 == 0)
            {
                ptr_size += blk_n[i];
                idx_size += blk_nnz[i];
                ptr_size += 1;
            }
            else
            {
                ptr_size += blk_m[i];
                idx_size += blk_nnz[i];
                dcsr_size += blk_m[i];
            }
        }
        // ---------------------get_recblock_size end--------------------------
        cudaMalloc((void **)&d_recblock_Ptr, ptr_size * sizeof(int));
        cudaMemset(d_recblock_Ptr, 0, sizeof(int));
        cudaMalloc((void **)&d_recblock_Index, idx_size * sizeof(int));
        cudaMalloc((void **)&d_recblock_dcsr_rowidx, dcsr_size * sizeof(int));
        cudaMalloc((void **)&d_recblock_Val, idx_size * sizeof(double));
        ptr_offset = (int *)malloc(sizeof(int) * (sum_block + 1));
        index_offset = (int *)malloc(sizeof(int) * (sum_block + 1));
        dcsrindex_offset = (int *)malloc(sizeof(int) * (sum_block + 1));
        dcsrindex_offset[0] = 0;
        ptr_offset[0] = 1;
        index_offset[0] = 0;

        int blk_count = 0;
        // store sub-matrix into device
        int trsv_count = 0;
        int mv_count = 0;
        int recblock_nnz_ptr = 0;
        for (blk_count = 0; blk_count < sum_block; blk_count++)
        {
            if (blk_count % 2 == 0)
            {
                int cu_flag = 0;
                int *d_cscColPtrTR_sub;
                int *d_cscRowIdxTR_sub;
                VALUE_TYPE *d_cscValTR_sub;
                cudaMalloc((void **)&d_cscColPtrTR_sub, sizeof(int) * (blk_n[blk_count] + 1));
                cudaMalloc((void **)&d_cscRowIdxTR_sub, sizeof(int) * blk_nnz[blk_count]);
                cudaMalloc((void **)&d_cscValTR_sub, sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

                int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                int num_blocks = ceil((double)(blk_m[blk_count]) / (double)num_threads);
                int upbound = subtri_upbound[blk_count];
                int downbound = subtri_downbound[blk_count];
                store_into_subtrimat_ptr<<<num_blocks, num_threads>>>(upbound, downbound, d_cscColPtrTR_new,
                                                                      d_cscRowIdxTR_new, d_cscColPtrTR_sub, substitution);
                thrust::exclusive_scan(thrust::device, d_cscColPtrTR_sub,
                                       d_cscColPtrTR_sub + blk_n[blk_count] + 1, d_cscColPtrTR_sub, 0);
                store_into_subtrimat_idxval<<<num_blocks, num_threads>>>(upbound, downbound, d_cscColPtrTR_new, d_cscRowIdxTR_new,
                                                                         d_cscValTR_new, d_cscColPtrTR_sub, d_cscRowIdxTR_sub,
                                                                         d_cscValTR_sub, substitution);

                int *d_csrRowPtrTR_sub;
                int *d_csrColIdxTR_sub;
                VALUE_TYPE *d_csrValTR_sub;
                cudaMalloc((void **)&d_csrRowPtrTR_sub, sizeof(int) * (blk_m[blk_count] + 1));
                cudaMalloc((void **)&d_csrColIdxTR_sub, sizeof(int) * blk_nnz[blk_count]);
                cudaMalloc((void **)&d_csrValTR_sub, sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

                // -------------------matrix_transposition-------------------
                matrix_transposition_cuda(blk_n[blk_count], blk_m[blk_count], blk_nnz[blk_count],
                                          d_cscColPtrTR_sub, d_cscRowIdxTR_sub, d_cscValTR_sub,
                                          d_csrColIdxTR_sub, d_csrRowPtrTR_sub, d_csrValTR_sub);
                // ----------------------------------------------------------
                int *d_nlv;
                cudaMalloc((void **)(&d_nlv), sizeof(int));
                cudaMemset(d_nlv, 0, sizeof(int));
                int *d_levelItem_local;
                int *d_levelPtr_local;
                cudaMalloc((void **)&d_levelItem_local, blk_m[blk_count] * sizeof(int));
                cudaMalloc((void **)&d_levelPtr_local, (blk_m[blk_count] + 1) * sizeof(int));
                int fasttrack = blk_m[blk_count] == blk_nnz[blk_count] ? 1 : 0;

                if (fasttrack)
                    cudaMemset(d_nlv, 1, sizeof(int));
                else
                    findlevel_cuda(d_cscColPtrTR_sub, d_cscRowIdxTR_sub, d_csrRowPtrTR_sub, blk_m[blk_count],
                                   d_nlv, d_levelPtr_local, d_levelItem_local);
                int nlv;
                cudaMemcpy(&nlv, d_nlv, sizeof(int), cudaMemcpyDeviceToHost);

                if (fasttrack)
                {
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)blk_m[blk_count] / (double)num_threads);
                    ((trsv_blk[trsv_count])).method = 0;
                    (trsv_blk[trsv_count]).num_threads = num_threads;
                    (trsv_blk[trsv_count]).num_blocks = num_blocks;
                    (trsv_blk[trsv_count]).m = blk_m[blk_count];
                    (trsv_blk[trsv_count]).substitution = substitution;

                    num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    num_blocks = ceil((double)blk_n[blk_count] / (double)num_threads);
                    pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_n[blk_count], d_cscColPtrTR_sub, d_recblock_Ptr + ptr_offset[blk_count] - 1);

                    thrust::exclusive_scan(thrust::device, d_recblock_Ptr + ptr_offset[blk_count] - 1,
                                           d_recblock_Ptr + ptr_offset[blk_count] + blk_n[blk_count], d_recblock_Ptr + ptr_offset[blk_count] - 1, recblock_nnz_ptr);

                    store_to_recblockdata<<<num_blocks, num_threads>>>(blk_n[blk_count], d_cscColPtrTR_sub, d_cscRowIdxTR_sub,
                                                                       d_cscValTR_sub, d_recblock_Index, d_recblock_Val, d_recblock_Ptr + ptr_offset[blk_count] - 1, recblock_nnz_ptr);
                }
                else
                {
                    int nnzr = blk_nnz[blk_count] / blk_m[blk_count];

                    if (nlv > 20000)
                    {
                        printf("trsv method = 1\n");
                        cusparseStatus_t status;
                        (trsv_blk[trsv_count]).handle = 0;
                        status = cusparseCreate(&(trsv_blk[trsv_count].handle));
                        // http://docs.nvidia.com/cuda/cusparse/#cusparse-lt-t-gt-csrsv2_solve
                        // Suppose that L is m x m sparse matrix represented by CSR format,
                        // L is lower triangular with unit diagonal.
                        // Assumption:
                        // - dimension of matrix L is m,
                        // - matrix L has nnz number zero elements,
                        // - handle is already created by cusparseCreate(),
                        // - (d_csrRowPtr, d_csrColInd, d_csrVal) is CSR of L on device memory,
                        // - d_b is right hand side vector on device memory,
                        // - d_x is solution vector on device memory.
                        (trsv_blk[trsv_count]).descr = 0;
                        (trsv_blk[trsv_count]).info = 0;
                        int pBufferSize;
                        (trsv_blk[trsv_count]).pBuffer = 0;
                        int structural_zero;
                        int numerical_zero;
                        (trsv_blk[trsv_count]).alpha_double = 1.;
                        (trsv_blk[trsv_count]).alpha_float = 1.;
                        (trsv_blk[trsv_count]).policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
                        (trsv_blk[trsv_count]).trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

                        // step 1: create a descriptor which contains
                        // - matrix L is base-0
                        // - matrix L is lower triangular
                        // - matrix L has unit diagonal, specified by parameter CUSPARSE_DIAG_TYPE_UNIT
                        //   (L may not have all diagonal elements.)
                        cusparseCreateMatDescr(&(trsv_blk[trsv_count].descr));
                        cusparseSetMatIndexBase((trsv_blk[trsv_count]).descr, CUSPARSE_INDEX_BASE_ZERO);

                        if (substitution == SUBSTITUTION_FORWARD)
                            cusparseSetMatFillMode((trsv_blk[trsv_count]).descr, CUSPARSE_FILL_MODE_LOWER);
                        else if (substitution == SUBSTITUTION_BACKWARD)
                            cusparseSetMatFillMode((trsv_blk[trsv_count]).descr, CUSPARSE_FILL_MODE_UPPER);

                        cusparseSetMatDiagType((trsv_blk[trsv_count]).descr, CUSPARSE_DIAG_TYPE_UNIT);
                        // step 2: create a empty info structure
                        cusparseCreateCsrsv2Info(&(trsv_blk[trsv_count].info));

                        // step 3: query how much memory used in csrsv2, and allocate the buffer
                        if (sizeof(VALUE_TYPE) == 8)
                            cusparseDcsrsv2_bufferSize((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                       (double *)d_csrValTR_sub, d_csrRowPtrTR_sub, d_csrColIdxTR_sub, (trsv_blk[trsv_count]).info, &pBufferSize);
                        else if (sizeof(VALUE_TYPE) == 4)
                            cusparseScsrsv2_bufferSize((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                       (float *)d_csrValTR_sub, d_csrRowPtrTR_sub, d_csrColIdxTR_sub, (trsv_blk[trsv_count]).info, &pBufferSize);
                        // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
                        cudaMalloc((void **)&(trsv_blk[trsv_count].pBuffer), pBufferSize);
                        if (sizeof(VALUE_TYPE) == 8)
                            cusparseDcsrsv2_analysis((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                     (double *)d_csrValTR_sub, d_csrRowPtrTR_sub, d_csrColIdxTR_sub,
                                                     (trsv_blk[trsv_count]).info, (trsv_blk[trsv_count]).policy, (trsv_blk[trsv_count]).pBuffer);
                        else if (sizeof(VALUE_TYPE) == 4)
                            cusparseScsrsv2_analysis((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                     (float *)d_csrValTR_sub, d_csrRowPtrTR_sub, d_csrColIdxTR_sub,
                                                     (trsv_blk[trsv_count]).info, (trsv_blk[trsv_count]).policy, (trsv_blk[trsv_count]).pBuffer);

                        // L has unit diagonal, so no structural zero is reported.
                        status = cusparseXcsrsv2_zeroPivot((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).info, &structural_zero);
                        if (CUSPARSE_STATUS_ZERO_PIVOT == status)
                        {
                            printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
                        }

                        (trsv_blk[trsv_count]).method = 1;
                        (trsv_blk[trsv_count]).m = blk_m[blk_count];
                        (trsv_blk[trsv_count]).nnzTR = blk_nnz[blk_count];

                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)blk_m[blk_count] / (double)num_threads);
                        pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_m[blk_count], d_csrRowPtrTR_sub, d_recblock_Ptr + ptr_offset[blk_count]);

                        thrust::exclusive_scan(thrust::device, d_recblock_Ptr + ptr_offset[blk_count],
                                               d_recblock_Ptr + ptr_offset[blk_count] + blk_m[blk_count] + 1, d_recblock_Ptr + ptr_offset[blk_count], 0);

                        store_to_recblockdata<<<num_blocks, num_threads>>>(blk_m[blk_count], d_csrRowPtrTR_sub, d_csrColIdxTR_sub,
                                                                           d_csrValTR_sub, d_recblock_Index, d_recblock_Val, d_recblock_Ptr + ptr_offset[blk_count], recblock_nnz_ptr);
                        cu_flag = 1;
                    }
                    else if ((nnzr <= 15 && nlv <= 20) || (nnzr == 1 && nlv <= 100))
                    {
                        (trsv_blk[trsv_count]).method = 2;
                        (trsv_blk[trsv_count]).m = blk_m[blk_count];
                        (trsv_blk[trsv_count]).substitution = substitution;
                        (trsv_blk[trsv_count]).nlv = nlv;
                        (trsv_blk[trsv_count]).nnz_lv_array = (int *)malloc(sizeof(int) * nlv);
                        (trsv_blk[trsv_count]).m_lv_array = (int *)malloc(sizeof(int) * nlv);
                        (trsv_blk[trsv_count]).offset_array = (int *)malloc(sizeof(int) * nlv);
                        int *levelPtr_local = (int *)malloc((blk_m[blk_count] + 1) * sizeof(int));
                        int *levelItem_local = (int *)malloc(blk_m[blk_count] * sizeof(int));
                        cudaMemcpy(levelPtr_local, d_levelPtr_local, (blk_m[blk_count] + 1) * sizeof(int), cudaMemcpyDeviceToHost);
                        cudaMemcpy(levelItem_local, d_levelItem_local, (blk_m[blk_count]) * sizeof(int), cudaMemcpyDeviceToHost);
                        for (int li = 0; li < nlv; li++)
                        {
                            (trsv_blk[trsv_count]).m_lv_array[li] = levelPtr_local[li + 1] - levelPtr_local[li];
                            (trsv_blk[trsv_count]).offset_array[li] = levelPtr_local[li];
                        }

                        int *d_nnz_lv_array;
                        cudaMalloc((void **)&d_nnz_lv_array, (nlv + 1) * sizeof(int));
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)nlv / (double)num_threads);
                        find_level_nnz<<<num_blocks, num_threads>>>(d_csrRowPtrTR_sub, d_levelPtr_local, d_nnz_lv_array, nlv);

                        cudaMemcpy((trsv_blk[trsv_count]).nnz_lv_array, d_nnz_lv_array, sizeof(int) * nlv, cudaMemcpyDeviceToHost);

                        cudaFree(d_nnz_lv_array);

                        num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        num_blocks = ceil((double)blk_m[blk_count] / (double)num_threads);
                        pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_n[blk_count], d_csrRowPtrTR_sub, d_recblock_Ptr + ptr_offset[blk_count] - 1);

                        thrust::exclusive_scan(thrust::device, d_recblock_Ptr + ptr_offset[blk_count] - 1,
                                               d_recblock_Ptr + ptr_offset[blk_count] + blk_n[blk_count], d_recblock_Ptr + ptr_offset[blk_count] - 1, recblock_nnz_ptr);

                        store_to_recblockdata<<<num_blocks, num_threads>>>(blk_n[blk_count], d_csrRowPtrTR_sub, d_csrColIdxTR_sub,
                                                                           d_csrValTR_sub, d_recblock_Index, d_recblock_Val, d_recblock_Ptr + ptr_offset[blk_count] - 1, recblock_nnz_ptr);

                        free(levelPtr_local);
                        free(levelItem_local);
                    }
                    else
                    {
                        cudaMalloc((void **)&(trsv_blk[trsv_count]).d_levelItem, blk_m[blk_count] * sizeof(int));
                        cudaMemcpy((trsv_blk[trsv_count]).d_levelItem, d_levelItem_local, blk_m[blk_count] * sizeof(int), cudaMemcpyDeviceToDevice);

                        cudaMalloc((void **)&(trsv_blk[trsv_count]).d_graphInDegree, blk_m[blk_count] * sizeof(int));
                        cudaMemset((trsv_blk[trsv_count]).d_graphInDegree, 0, blk_m[blk_count] * sizeof(int));

                        cudaMalloc((void **)&(trsv_blk[trsv_count]).d_id_extractor, sizeof(int));
                        cudaMemset((trsv_blk[trsv_count]).d_id_extractor, 0, sizeof(int));

                        int num_threads = 128;
                        int num_blocks = ceil((double)blk_nnz[blk_count] / (double)num_threads);
                        sptrsv_syncfree_csc_cuda_analyser<<<num_blocks, num_threads>>>(d_cscRowIdxTR_sub, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).d_graphInDegree);
                        cudaDeviceSynchronize();
                        cudaMalloc((void **)&(trsv_blk[trsv_count]).d_left_sum, sizeof(VALUE_TYPE) * blk_m[blk_count]);
                        cudaMemset((trsv_blk[trsv_count]).d_left_sum, 0, sizeof(VALUE_TYPE) * blk_m[blk_count]);

                        num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        num_blocks = ceil((double)blk_m[blk_count] / (double)(num_threads / WARP_SIZE));
                        (trsv_blk[trsv_count]).method = 3;
                        (trsv_blk[trsv_count]).num_threads = num_threads;
                        (trsv_blk[trsv_count]).num_blocks = num_blocks;
                        (trsv_blk[trsv_count]).m = blk_m[blk_count];
                        (trsv_blk[trsv_count]).substitution = substitution;

                        num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        num_blocks = ceil((double)blk_n[blk_count] / (double)num_threads);
                        pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_n[blk_count], d_cscColPtrTR_sub, d_recblock_Ptr + ptr_offset[blk_count] - 1);
                        thrust::exclusive_scan(thrust::device, d_recblock_Ptr + ptr_offset[blk_count] - 1,
                                               d_recblock_Ptr + ptr_offset[blk_count] + blk_n[blk_count], d_recblock_Ptr + ptr_offset[blk_count] - 1, recblock_nnz_ptr);
                        store_to_recblockdata<<<num_blocks, num_threads>>>(blk_n[blk_count], d_cscColPtrTR_sub, d_cscRowIdxTR_sub,
                                                                           d_cscValTR_sub, d_recblock_Index, d_recblock_Val, d_recblock_Ptr + ptr_offset[blk_count] - 1, recblock_nnz_ptr);
                    }
                }

                cudaFree(d_levelPtr_local);
                cudaFree(d_levelItem_local);

                if (cu_flag == 0)
                    ptr_offset[blk_count + 1] = ptr_offset[blk_count] + blk_m[blk_count];
                else
                    ptr_offset[blk_count + 1] = ptr_offset[blk_count] + blk_m[blk_count] + 1;
                recblock_nnz_ptr += blk_nnz[blk_count];
                index_offset[blk_count + 1] = recblock_nnz_ptr;
                dcsrindex_offset[blk_count + 1] = dcsrindex_offset[blk_count];

                cudaFree(d_csrRowPtrTR_sub);
                cudaFree(d_csrValTR_sub);
                cudaFree(d_csrColIdxTR_sub);
                cudaFree(d_cscColPtrTR_sub);
                cudaFree(d_cscRowIdxTR_sub);
                cudaFree(d_cscValTR_sub);

                trsv_count++;

                cudaDeviceSynchronize();
            }
            else
            {
                int *d_cscColPtrTR_sub;
                int *d_cscRowIdxTR_sub;
                VALUE_TYPE *d_cscValTR_sub;
                cudaMalloc((void **)&d_cscColPtrTR_sub, sizeof(int) * (blk_n[blk_count] + 1));
                cudaMalloc((void **)&d_cscRowIdxTR_sub, sizeof(int) * blk_nnz[blk_count]);
                cudaMalloc((void **)&d_cscValTR_sub, sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

                int *d_csrRowPtrTR_sub;
                int *d_csrColIdxTR_sub;
                VALUE_TYPE *d_csrValTR_sub;
                cudaMalloc((void **)&d_csrRowPtrTR_sub, sizeof(int) * (blk_m[blk_count] + 1));
                cudaMalloc((void **)&d_csrColIdxTR_sub, sizeof(int) * blk_nnz[blk_count]);
                cudaMalloc((void **)&d_csrValTR_sub, sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

                int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                int num_blocks = ceil((double)blk_m[blk_count] / (double)(num_threads));
                int upbound = subrec_upbound[blk_count];
                int downbound = subrec_downbound[blk_count];
                int leftbound = subrec_leftbound[blk_count];
                int rightbound = subrec_rightbound[blk_count];
                store_into_subrecmat_ptr<<<num_blocks, num_threads>>>(upbound, downbound, leftbound, rightbound,
                                                                      d_cscColPtrTR_new, d_cscRowIdxTR_new, d_cscColPtrTR_sub);

                thrust::exclusive_scan(thrust::device, d_cscColPtrTR_sub,
                                       d_cscColPtrTR_sub + blk_n[blk_count] + 1, d_cscColPtrTR_sub, 0);

                store_into_subrecmat_idxval<<<num_blocks, num_threads>>>(upbound, downbound, leftbound, rightbound, d_cscColPtrTR_new, d_cscRowIdxTR_new,
                                                                         d_cscValTR_new, d_cscColPtrTR_sub, d_cscRowIdxTR_sub, d_cscValTR_sub);

                matrix_transposition_cuda(blk_n[blk_count], blk_m[blk_count], blk_nnz[blk_count],
                                          d_cscColPtrTR_sub, d_cscRowIdxTR_sub, d_cscValTR_sub,
                                          d_csrColIdxTR_sub, d_csrRowPtrTR_sub, d_csrValTR_sub);

                int *idx_offset;
                cudaMalloc((void **)(&idx_offset), sizeof(int) * (blk_m[blk_count] + 1));
                num_threads = WARP_PER_BLOCK * WARP_SIZE;
                num_blocks = ceil((double)blk_m[blk_count] / (double)num_threads);

                pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_m[blk_count], d_csrRowPtrTR_sub, idx_offset);

                thrust::exclusive_scan(thrust::device, idx_offset,
                                       idx_offset + blk_m[blk_count] + 1, idx_offset, recblock_nnz_ptr);

                store_to_recblockdata<<<num_blocks, num_threads>>>(blk_m[blk_count], d_csrRowPtrTR_sub, d_csrColIdxTR_sub,
                                                                   d_csrValTR_sub, d_recblock_Index, d_recblock_Val, idx_offset, recblock_nnz_ptr);
                cudaFree(idx_offset);

                int *d_i_new;
                int *d_lenmax;
                int *d_longrow;
                int *d_longlen;
                int *d_longrow_idx;
                cudaMalloc((void **)(&d_i_new), sizeof(int));
                cudaMalloc((void **)(&d_lenmax), sizeof(int));
                cudaMalloc((void **)(&d_longrow), sizeof(int));
                cudaMalloc((void **)(&d_longlen), sizeof(int));
                cudaMalloc((void **)(&d_longrow_idx), blk_m[blk_count] * sizeof(int));

                cal_longrow<<<1, 1>>>(d_i_new, d_lenmax, d_longrow, d_longlen,
                                      d_longrow_idx, d_csrRowPtrTR_sub, blk_m[blk_count]);

                int m_new;
                cudaMemcpy(&m_new, d_i_new, sizeof(int), cudaMemcpyDeviceToHost);
                m_new--;
                int lenmax;
                cudaMemcpy(&lenmax, d_lenmax, sizeof(int), cudaMemcpyDeviceToHost);
                int longrow;
                cudaMemcpy(&longrow, d_longrow, sizeof(int), cudaMemcpyDeviceToHost);
                int longlen;
                cudaMemcpy(&longlen, d_longlen, sizeof(int), cudaMemcpyDeviceToHost);
                int nnzr = (blk_nnz[blk_count] - longlen) / m_new;
                double empty_ratio = 100 * (double)(blk_m[blk_count] - m_new) / (double)blk_m[blk_count];
                int dcsr_i = 0;
                int real_i = 0;
                if (blk_nnz[blk_count] != 0)
                {
                    int m = blk_m[blk_count];
                    int nnz = blk_nnz[blk_count];
                    cudaMalloc((void **)&((mv_blk[mv_count]).d_y), m * sizeof(VALUE_TYPE));
                    cudaMemset((mv_blk[mv_count].d_y), 0, m * sizeof(VALUE_TYPE));
                    if (nnzr <= 12 && empty_ratio <= 50)
                    {
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)m / (double)num_threads);
                        (mv_blk[mv_count]).method = 0;
                        (mv_blk[mv_count]).num_threads = num_threads;
                        (mv_blk[mv_count]).num_blocks = num_blocks;
                        (mv_blk[mv_count]).m = m;

                        int init_val;
                        cudaMemcpy(&init_val, d_recblock_Ptr + ptr_offset[blk_count] - 1, sizeof(int), cudaMemcpyDeviceToHost);

                        pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_m[blk_count], d_csrRowPtrTR_sub, d_recblock_Ptr + ptr_offset[blk_count] - 1);
                        thrust::exclusive_scan(thrust::device, d_recblock_Ptr + ptr_offset[blk_count] - 1,
                                               d_recblock_Ptr + ptr_offset[blk_count] + blk_m[blk_count], d_recblock_Ptr + ptr_offset[blk_count] - 1, init_val);
                        real_i = m;
                    }
                    else if (nnzr <= 12 && empty_ratio > 50)
                    {
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)m_new / (double)num_threads);
                        (mv_blk[mv_count]).method = 1;
                        (mv_blk[mv_count]).num_threads = num_threads;
                        (mv_blk[mv_count]).num_blocks = num_blocks;
                        (mv_blk[mv_count]).m_new = m_new;

                        int *d_dcsr_i;
                        cudaMalloc((void **)(&d_dcsr_i), sizeof(int));
                        cudaMemcpy(d_dcsr_i, &dcsr_i, sizeof(int), cudaMemcpyHostToDevice);
                        dcsr_recblockdata_ptr<<<1, 1>>>(m, d_csrRowPtrTR_sub, ptr_offset[blk_count],
                                                        dcsrindex_offset[blk_count], d_dcsr_i, d_recblock_Ptr,
                                                        d_recblock_dcsr_rowidx);
                        cudaMemcpy(&dcsr_i, d_dcsr_i, sizeof(int), cudaMemcpyDeviceToHost);
                        cudaFree(d_dcsr_i);
                        real_i = dcsr_i;
                    }
                    else if (nnzr > 12 && empty_ratio <= 15)
                    {
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)m / (double)(num_threads / WARP_SIZE));
                        (mv_blk[mv_count]).method = 2;
                        (mv_blk[mv_count]).num_threads = num_threads;
                        (mv_blk[mv_count]).num_blocks = num_blocks;
                        (mv_blk[mv_count]).m = m;

                        int init_val;
                        cudaMemcpy(&init_val, d_recblock_Ptr + ptr_offset[blk_count] - 1, sizeof(int), cudaMemcpyDeviceToHost);

                        pre_store_to_recblockdata<<<num_blocks, num_threads>>>(blk_m[blk_count], d_csrRowPtrTR_sub, d_recblock_Ptr + ptr_offset[blk_count] - 1);
                        thrust::exclusive_scan(thrust::device, d_recblock_Ptr + ptr_offset[blk_count] - 1,
                                               d_recblock_Ptr + ptr_offset[blk_count] + blk_m[blk_count], d_recblock_Ptr + ptr_offset[blk_count] - 1, init_val);
                        real_i = blk_m[blk_count];
                    }
                    else
                    {
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)m_new / (double)(num_threads / WARP_SIZE));

                        (mv_blk[mv_count]).method = 3;
                        (mv_blk[mv_count]).num_threads = num_threads;
                        (mv_blk[mv_count]).num_blocks = num_blocks;
                        (mv_blk[mv_count]).m_new = m_new;

                        int *d_dcsr_i;
                        cudaMalloc((void **)(&d_dcsr_i), sizeof(int));
                        cudaMemcpy(d_dcsr_i, &dcsr_i, sizeof(int), cudaMemcpyHostToDevice);
                        dcsr_recblockdata_ptr<<<1, 1>>>(m, d_csrRowPtrTR_sub, ptr_offset[blk_count],
                                                        dcsrindex_offset[blk_count], d_dcsr_i, d_recblock_Ptr,
                                                        d_recblock_dcsr_rowidx);
                        cudaMemcpy(&dcsr_i, d_dcsr_i, sizeof(int), cudaMemcpyDeviceToHost);
                        cudaFree(d_dcsr_i);
                        real_i = dcsr_i;
                    }

                    (mv_blk[mv_count]).longrow = longrow;
                    if (longrow != 0)
                    {
                        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                        int num_blocks = ceil((double)lenmax / (double)num_threads);
                        (mv_blk[mv_count]).num_threads_l = num_threads;
                        (mv_blk[mv_count]).num_blocks_l = num_blocks;
                        cudaMalloc((void **)&((mv_blk[mv_count]).d_csrRowPtr_l), (m + 1) * sizeof(int));
                        cudaMalloc((void **)&((mv_blk[mv_count]).d_csrColIdx_l), nnz * sizeof(int));
                        cudaMalloc((void **)&((mv_blk[mv_count]).d_csrVal_l), nnz * sizeof(VALUE_TYPE));
                        cudaMemcpy((mv_blk[mv_count]).d_csrRowPtr_l, d_csrRowPtrTR_sub, (m + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
                        cudaMemcpy((mv_blk[mv_count]).d_csrColIdx_l, d_csrColIdxTR_sub, nnz * sizeof(int), cudaMemcpyDeviceToDevice);
                        cudaMemcpy((mv_blk[mv_count]).d_csrVal_l, d_csrValTR_sub, nnz * sizeof(VALUE_TYPE), cudaMemcpyDeviceToDevice);

                        cudaMalloc((void **)&((mv_blk[mv_count]).d_longrow_idx), longrow * sizeof(int));
                        cudaMemcpy((mv_blk[mv_count]).d_longrow_idx, d_longrow_idx, longrow * sizeof(int), cudaMemcpyDeviceToDevice);
                    }
                }

                ptr_offset[blk_count + 1] = ptr_offset[blk_count] + real_i;
                recblock_nnz_ptr += blk_nnz[blk_count];
                index_offset[blk_count + 1] = recblock_nnz_ptr;
                dcsrindex_offset[blk_count + 1] = dcsrindex_offset[blk_count] + dcsr_i;

                cudaFree(d_csrRowPtrTR_sub);
                cudaFree(d_csrColIdxTR_sub);
                cudaFree(d_csrValTR_sub);
                cudaFree(d_cscRowIdxTR_sub);
                cudaFree(d_cscColPtrTR_sub);
                cudaFree(d_cscValTR_sub);

                mv_count++;
            }
        }

        VALUE_TYPE *d_b_perm;
        cudaMalloc((void **)&d_b_perm, m * sizeof(VALUE_TYPE));
        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
        int num_blocks = ceil((double)m / (double)num_threads);
        levelset_reordering_vecb_cuda<<<num_blocks, num_threads>>>(d_b, d_b_perm, d_levelItem, m);

        cudaDeviceSynchronize();
        int rhs = 1;

        U_calculate(mv_blk, trsv_blk, sum_block, blk_m, blk_n, loc_off, tmp_off,
                    m, nnzTR, rhs, d_x, d_b, d_b_perm, d_recblock_Ptr, d_recblock_Index, d_recblock_dcsr_rowidx,
                    d_recblock_Val, ptr_offset, index_offset, dcsrindex_offset, cal_time);

        VALUE_TYPE *d_x_perm;
        cudaMalloc((void **)(&d_x_perm), sizeof(VALUE_TYPE) * n);
        cudaMemcpy(d_x_perm, d_x, rhs * n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToDevice);
        num_threads = WARP_PER_BLOCK * WARP_SIZE;
        num_blocks = ceil((double)n / (double)num_threads);
        levelset_reordering_vecx_cuda<<<num_blocks, num_threads>>>(d_x_perm, d_x, d_levelItem, n);

        free(ptr_offset);
        free(index_offset);
        free(dcsrindex_offset);
        cudaFree(d_b_perm);
        cudaFree(d_x_perm);

        cudaFree(d_recblock_Ptr);
        cudaFree(d_recblock_Index);
        cudaFree(d_recblock_dcsr_rowidx);
        cudaFree(d_recblock_Val);
    }
    cudaDeviceSynchronize();
}

#endif
