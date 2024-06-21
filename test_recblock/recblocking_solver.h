#ifndef __RECBLOCKING_SOLVER__
#define __RECBLOCKING_SOLVER__
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

void recblocking_solver(int *cscColPtr,
                        int *cscRowIdx,
                        VALUE_TYPE *cscVal,
                        int m,
                        int n,
                        int nnz,
                        VALUE_TYPE *b,
                        VALUE_TYPE *x,
                        VALUE_TYPE *x_ref,
                        int rhs,
                        int lv,
                        int substitution,
                        double *cal_time)
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
    int *loc_off = (int *)malloc(sizeof(int) * (squ_block + tri_block));
    memset(loc_off, 0, sizeof(int) * (squ_block + tri_block));
    int *tmp_off = (int *)malloc(sizeof(int) * (squ_block + tri_block));
    memset(tmp_off, 0, sizeof(int) * (squ_block + tri_block));
    int *levelItem = (int *)malloc(m * sizeof(int));
    int *subtri_upbound = (int *)malloc(sizeof(int) * sum_block);
    int *subtri_downbound = (int *)malloc(sizeof(int) * sum_block);
    int *subrec_upbound = (int *)malloc(sizeof(int) * sum_block);
    int *subrec_downbound = (int *)malloc(sizeof(int) * sum_block);
    int *subrec_rightbound = (int *)malloc(sizeof(int) * sum_block);
    int *subrec_leftbound = (int *)malloc(sizeof(int) * sum_block);

    int *recblock_Ptr;
    int *recblock_Index;
    int *recblock_dcsr_rowidx;
    double *recblock_Val;
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
        
        int *cscColPtrTR_new = (int *)malloc((n + 1) * sizeof(int));
        int *cscRowIdxTR_new = (int *)malloc(nnz * sizeof(int));
        VALUE_TYPE *cscValTR_new = (VALUE_TYPE *)malloc(nnz * sizeof(VALUE_TYPE));

        get_recblock_size(cscRowIdx, cscColPtr, cscVal, cscRowIdxTR_new, cscColPtrTR_new, cscValTR_new,
                          nnz, m, n, levelItem, substitution, lv, loc_off, tmp_off, blk_m, blk_n, blk_nnz,
                          subtri_upbound, subtri_downbound, subrec_upbound, subrec_downbound, subrec_rightbound,
                          subrec_leftbound, &ptr_size, &idx_size, &dcsr_size);


        recblock_Ptr = (int *)malloc(sizeof(int) * ptr_size);
        recblock_Ptr[0] = 0;
        recblock_Index = (int *)malloc(sizeof(int) * idx_size);
        recblock_dcsr_rowidx = (int *)malloc(sizeof(int) * dcsr_size);
        recblock_Val = (double *)malloc(sizeof(double) * idx_size);
        ptr_offset = (int *)malloc(sizeof(int) * (sum_block + 1));
        index_offset = (int *)malloc(sizeof(int) * (sum_block + 1));
        dcsrindex_offset = (int *)malloc(sizeof(int) * (sum_block + 1));
        dcsrindex_offset[0] = 0;
        ptr_offset[0] = 1;
        index_offset[0] = 0;

        // preprocess L matrix
        L_preprocessing(cscRowIdxTR_new, cscColPtrTR_new, cscValTR_new, nnz, m, n,
                        substitution, lv, blk_m, blk_n, blk_nnz,
                        subtri_upbound, subtri_downbound, subrec_upbound,
                        subrec_downbound, subrec_rightbound, subrec_leftbound,
                        mv_blk, trsv_blk, recblock_Ptr, recblock_Index, recblock_dcsr_rowidx,
                        recblock_Val, ptr_offset, index_offset, dcsrindex_offset,
                        ptr_size, idx_size, dcsr_size);

        free(cscColPtrTR_new);
        free(cscRowIdxTR_new);
        free(cscValTR_new);

        cudaMalloc((void **)&d_recblock_Ptr, ptr_size * sizeof(int));
        cudaMalloc((void **)&d_recblock_Index, idx_size * sizeof(int));
        cudaMalloc((void **)&d_recblock_dcsr_rowidx, dcsr_size * sizeof(int));
        cudaMalloc((void **)&d_recblock_Val, idx_size * sizeof(double));
        cudaMemcpy(d_recblock_Ptr, recblock_Ptr, ptr_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_recblock_Index, recblock_Index, idx_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_recblock_dcsr_rowidx, recblock_dcsr_rowidx, dcsr_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_recblock_Val, recblock_Val, idx_size * sizeof(double), cudaMemcpyHostToDevice);

        // for (int i = 0; i < ptr_size; i++)
        //     printf("%d ", recblock_Ptr[i]);
        // printf("\n\n");
        
        // for (int i = 0; i < nnz; i++)
        //     printf("%d ", recblock_Index[i]);
        // printf("\n\n");
        
        free(recblock_Ptr);
        free(recblock_Index);
        free(recblock_dcsr_rowidx);
        free(recblock_Val);

        VALUE_TYPE *b_perm = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m * rhs);
        levelset_reordering_vecb(b, b_perm, levelItem, m);
        VALUE_TYPE *b_perm_d;
        cudaMalloc((void **)&b_perm_d, rhs * m * sizeof(VALUE_TYPE));
        cudaMemcpy(b_perm_d, b_perm, sizeof(VALUE_TYPE) * m * rhs, cudaMemcpyHostToDevice);

        VALUE_TYPE *x_d;
        cudaMalloc((void **)&x_d, rhs * n * sizeof(VALUE_TYPE));

        VALUE_TYPE *b_d;
        cudaMalloc((void **)&b_d, rhs * m * sizeof(VALUE_TYPE));

        L_calculate(mv_blk, trsv_blk, sum_block, blk_m, blk_n, loc_off, tmp_off,
                    m, rhs, x_d, b_d, b_perm_d, d_recblock_Ptr, d_recblock_Index, d_recblock_dcsr_rowidx,
                    d_recblock_Val, ptr_offset, index_offset, dcsrindex_offset, cal_time);

        VALUE_TYPE *x_perm = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n * rhs);
        cudaMemcpy(x_perm, x_d, rhs * n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
        levelset_reordering_vecx(x_perm, x, levelItem, n);

        free(ptr_offset);
        free(index_offset);
        free(dcsrindex_offset);
        free(b_perm);
        free(x_perm);
        cudaFree(x_d);
        cudaFree(b_d);
        cudaFree(d_recblock_Ptr);
        cudaFree(d_recblock_Index);
        cudaFree(d_recblock_dcsr_rowidx);
        cudaFree(d_recblock_Val);

        // gettimeofday(&t2, NULL);
        // double preprocess_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        // printf("preprocess time = %.3lf ms\n", preprocess_time);

        // FILE *fouttime = fopen("recblocking_211122.csv", "a");
        // fprintf(fouttime, "%i, %f, %f, %f, %f, %f, %f, %f, %i", );
        // fclose(fouttime);
    }
    else if (substitution == SUBSTITUTION_BACKWARD)
    {
        int *cscColPtrTR_new = (int *)malloc((n + 1) * sizeof(int));
        int *cscRowIdxTR_new = (int *)malloc(nnz * sizeof(int));
        VALUE_TYPE *cscValTR_new = (VALUE_TYPE *)malloc(nnz * sizeof(VALUE_TYPE));

        get_recblock_size(cscRowIdx, cscColPtr, cscVal, cscRowIdxTR_new, cscColPtrTR_new, cscValTR_new,
                          nnz, m, n, levelItem, substitution, lv, loc_off, tmp_off, blk_m, blk_n, blk_nnz,
                          subtri_upbound, subtri_downbound, subrec_upbound, subrec_downbound, subrec_rightbound,
                          subrec_leftbound, &ptr_size, &idx_size, &dcsr_size);

        recblock_Ptr = (int *)malloc(sizeof(int) * ptr_size);
        recblock_Ptr[0] = 0;
        recblock_Index = (int *)malloc(sizeof(int) * idx_size);
        recblock_dcsr_rowidx = (int *)malloc(sizeof(int) * dcsr_size);
        recblock_Val = (double *)malloc(sizeof(double) * idx_size);
        ptr_offset = (int *)malloc(sizeof(int) * (sum_block + 1));
        index_offset = (int *)malloc(sizeof(int) * (sum_block + 1));
        dcsrindex_offset = (int *)malloc(sizeof(int) * (sum_block + 1));
        dcsrindex_offset[0] = 0;
        ptr_offset[0] = 1;
        index_offset[0] = 0;

        // preprocess U matrix
        U_preprocessing(cscRowIdxTR_new, cscColPtrTR_new, cscValTR_new, nnz, m, n,
                        substitution, lv, blk_m, blk_n, blk_nnz,
                        subtri_upbound, subtri_downbound, subrec_upbound,
                        subrec_downbound, subrec_rightbound, subrec_leftbound,
                        mv_blk, trsv_blk, recblock_Ptr, recblock_Index, recblock_dcsr_rowidx,
                        recblock_Val, ptr_offset, index_offset, dcsrindex_offset,
                        ptr_size, idx_size, dcsr_size);

        free(cscColPtrTR_new);
        free(cscRowIdxTR_new);
        free(cscValTR_new);

        cudaMalloc((void **)&d_recblock_Ptr, ptr_size * sizeof(int));
        cudaMalloc((void **)&d_recblock_Index, idx_size * sizeof(int));
        cudaMalloc((void **)&d_recblock_dcsr_rowidx, dcsr_size * sizeof(int));
        cudaMalloc((void **)&d_recblock_Val, idx_size * sizeof(double));
        cudaMemcpy(d_recblock_Ptr, recblock_Ptr, ptr_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_recblock_Index, recblock_Index, idx_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_recblock_dcsr_rowidx, recblock_dcsr_rowidx, dcsr_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_recblock_Val, recblock_Val, idx_size * sizeof(double), cudaMemcpyHostToDevice);

        free(recblock_Ptr);
        free(recblock_Index);
        free(recblock_dcsr_rowidx);
        free(recblock_Val);

        VALUE_TYPE *b_perm = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m * rhs);
        levelset_reordering_vecb(b, b_perm, levelItem, m);

        VALUE_TYPE *x_d;
        cudaMalloc((void **)&x_d, rhs * n * sizeof(VALUE_TYPE));

        VALUE_TYPE *b_d;
        cudaMalloc((void **)&b_d, rhs * m * sizeof(VALUE_TYPE));

        U_calculate(mv_blk, trsv_blk, sum_block, blk_m, blk_n, loc_off, tmp_off,
                    m, nnz, rhs, x_d, b_d, b_perm, d_recblock_Ptr, d_recblock_Index, d_recblock_dcsr_rowidx,
                    d_recblock_Val, ptr_offset, index_offset, dcsrindex_offset, cal_time);

        VALUE_TYPE *x_perm = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n * rhs);
        cudaMemcpy(x_perm, x_d, rhs * n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
        levelset_reordering_vecx(x_perm, x, levelItem, n);

        free(ptr_offset);
        free(index_offset);
        free(dcsrindex_offset);
        free(b_perm);
        free(x_perm);
        cudaFree(x_d);
        cudaFree(b_d);
        cudaFree(d_recblock_Ptr);
        cudaFree(d_recblock_Index);
        cudaFree(d_recblock_dcsr_rowidx);
        cudaFree(d_recblock_Val);
    }

    device_memfree(mv_blk, trsv_blk, tri_block, squ_block);
    free(blk_m);
    free(blk_n);
    free(loc_off);
    free(tmp_off);
    free(levelItem);
    free(subtri_upbound);
    free(subtri_downbound);
    free(subrec_upbound);
    free(subrec_downbound);
    free(subrec_rightbound);
    free(subrec_leftbound);
}

#endif