#ifndef __RECBLOCKING_PREPROCESS__
#define __RECBLOCKING_PREPROCESS__
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include "findlevel.h"
#include <cuda_runtime.h>
#include "cusparse.h"
#include "tranpose.h"
#include "utils.h"
#include "utils_reordering.h"
#include "utils_sptrsv_cuda.h"
#include "utils_spmv_cuda.h"

void mat_preprocessing(const int *cscColPtrTR,
                       const int *cscRowIdxTR,
                       const VALUE_TYPE *cscValTR,
                       const int m,
                       const int n,
                       const int nlevel,
                       int *loc_off,
                       int *tmp_off,
                       int *blk_m,
                       int *blk_n,
                       int *blk_nnz,
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

                // printf("here:\n");
                // for (int i = 0; i < 108; i++)
                //     printf("%d ", cscRowIdxTR[i]);
                // printf("\n\n");
                // printf("tri_up = %d     tri_down = %d\n", tri_up, tri_down);

                int tri_nnz = 0;
                int flag = 0;
                for (int j = tri_up; j < tri_down; j++)
                {
                    for (int k = cscColPtrTR[j]; k < cscColPtrTR[j + 1]; k++)
                    {
                        if (cscRowIdxTR[k] < tri_down)
                        {
                            tri_nnz++;
                        }
                    }
                }
                blk_nnz[i] = tri_nnz;

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

                int sqr_nnz = 0;
                for (int j = rec_left; j < rec_right; j++)
                {
                    for (int k = cscColPtrTR[j]; k < cscColPtrTR[j + 1]; k++)
                    {
                        if (cscRowIdxTR[k] >= rec_up && cscRowIdxTR[k] < rec_down)
                        {
                            sqr_nnz++;
                        }
                    }
                }

                blk_nnz[i] = sqr_nnz;
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

                int tri_nnz = 0;
                for (int j = tri_up; j < tri_down; j++)
                {
                    for (int k = cscColPtrTR[j]; k < cscColPtrTR[j + 1]; k++)
                    {
                        if (cscRowIdxTR[k] >= tri_up)
                        {
                            tri_nnz++;
                        }
                    }
                }
                blk_nnz[i] = tri_nnz;

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

                int sqr_nnz = 0;
                for (int j = rec_left; j < rec_right; j++)
                {
                    for (int k = cscColPtrTR[j]; k < cscColPtrTR[j + 1]; k++)
                    {
                        if (cscRowIdxTR[k] >= rec_up && cscRowIdxTR[k] < rec_down)
                        {
                            sqr_nnz++;
                        }
                    }
                }

                blk_nnz[i] = sqr_nnz;
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

void get_recblock_size(int *cscRowIdxTR,
                       int *cscColPtrTR,
                       VALUE_TYPE *cscValTR,
                       int *cscRowIdxTR_new,
                       int *cscColPtrTR_new,
                       VALUE_TYPE *cscValTR_new,
                       int nnzTR,
                       int m,
                       int n,
                       int *levelItem,
                       int substitution,
                       int nlevel,
                       int *loc_off,
                       int *tmp_off,
                       int *blk_m,
                       int *blk_n,
                       int *blk_nnz,
                       int *subtri_upbound,
                       int *subtri_downbound,
                       int *subrec_upbound,
                       int *subrec_downbound,
                       int *subrec_rightbound,
                       int *subrec_leftbound,
                       int *ptr_size,
                       int *idx_size,
                       int *dcsr_size)
{

    // for (int i = 0; i < n + 1; i++)
    //     printf("%d ", cscColPtrTR[i]);
    // printf("\n\n");

    // for (int i = 0; i < nnzTR; i++)
    //     printf("%d ", cscRowIdxTR[i]);
    // printf("\n\n");
    
    // reorder input CSC according to level-set order
    levelset_reordering_colrow_csc(cscColPtrTR, cscRowIdxTR, cscValTR,
                                   cscColPtrTR_new, cscRowIdxTR_new, cscValTR_new,
                                   levelItem, m, n, nnzTR, substitution);

    // get auxiliary arrary for our datastruct
    int tri_block = pow(2, nlevel);
    int squ_block = tri_block - 1;
    int sum_block = tri_block + squ_block;

    int blk_count = 0;

    mat_preprocessing(cscColPtrTR_new, cscRowIdxTR_new, cscValTR_new, m, n,
                      nlevel, loc_off, tmp_off, blk_m, blk_n, blk_nnz, subtri_upbound,
                      subtri_downbound, subrec_upbound, subrec_downbound, subrec_rightbound, subrec_leftbound, substitution);

    // for (int i = 0; i < n + 1; i++)
    //     printf("%d ", cscColPtrTR_new[i]);
    // printf("\n\n");

    // for (int i = 0; i < nnzTR; i++)
    //     printf("%d ", cscRowIdxTR_new[i]);
    // printf("\n\n");

    // for (int i = 0; i < sum_block; i++)
    //     printf("nnz real = %d\n", blk_nnz[i]);
    // printf("\n");

    // for (int i = 0; i < sum_block; i++)
    // {
    //     if (i % 2 == 0)
    //         printf("up = %d         down = %d\n", subtri_upbound[i], subtri_downbound[i]);
    //     else    
    //         printf("up = %d         down = %d       left = %d       right = %d\n", subrec_upbound[i], subrec_downbound[i], subrec_leftbound[i], subrec_rightbound[i]);
    // }

    for (int i = 0; i < sum_block; i++)
    {
        if (i % 2 == 0)
        {
            *ptr_size += blk_n[i];
            *idx_size += blk_nnz[i];
            *ptr_size += 1;
        }
        else
        {
            *ptr_size += blk_m[i];
            *idx_size += blk_nnz[i];
            *dcsr_size += blk_m[i];
        }
    }
}

void L_preprocessing(int *cscRowIdxTR_new,
                     int *cscColPtrTR_new,
                     VALUE_TYPE *cscValTR_new,
                     int nnzTR,
                     int m,
                     int n,
                     int substitution,
                     int nlevel,
                     int *blk_m,
                     int *blk_n,
                     int *blk_nnz,
                     int *subtri_upbound,
                     int *subtri_downbound,
                     int *subrec_upbound,
                     int *subrec_downbound,
                     int *subrec_rightbound,
                     int *subrec_leftbound,
                     SpMV_block *mv_blk,
                     SpTRSV_block *trsv_blk,
                     int *recblock_Ptr,
                     int *recblock_Index,
                     int *recblock_dcsr_rowidx,
                     double *recblock_Val,
                     int *ptr_offset,
                     int *index_offset,
                     int *dcsrindex_offset,
                     int ptr_size,
                     int idx_size,
                     int dcsr_size)
{
    // get auxiliary arrary for our datastruct
    int tri_block = pow(2, nlevel);
    int squ_block = tri_block - 1;
    int sum_block = tri_block + squ_block;

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
            int *cscColPtrTR_sub = (int *)malloc(sizeof(int) * (blk_n[blk_count] + 1));
            cscColPtrTR_sub[0] = 0;
            int *cscRowIdxTR_sub = (int *)malloc(sizeof(int) * blk_nnz[blk_count]);
            VALUE_TYPE *cscValTR_sub = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

            int nnz_ptr = 0;
            for (int i = subtri_upbound[blk_count]; i < subtri_downbound[blk_count]; i++)
            {
                for (int j = cscColPtrTR_new[i]; j < cscColPtrTR_new[i + 1]; j++)
                {
                    if (cscRowIdxTR_new[j] < subtri_downbound[blk_count])
                    {
                        cscRowIdxTR_sub[nnz_ptr] = cscRowIdxTR_new[j] - subtri_upbound[blk_count];
                        cscValTR_sub[nnz_ptr] = cscValTR_new[j];
                        nnz_ptr++;
                    }
                }
                cscColPtrTR_sub[i - subtri_upbound[blk_count] + 1] = nnz_ptr;
            }

            // for (int i = 0; i < blk_nnz[blk_count]; i++)
            //     printf("%d ", cscRowIdxTR_sub[i]);
            // printf("\n\n");

            int *csrRowPtrTR_sub = (int *)malloc(sizeof(int) * (blk_m[blk_count] + 1));
            csrRowPtrTR_sub[0] = 0;
            int *csrColIdxTR_sub = (int *)malloc(sizeof(int) * blk_nnz[blk_count]);
            VALUE_TYPE *csrValTR_sub = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * blk_nnz[blk_count]);
            matrix_transposition(blk_n[blk_count], blk_m[blk_count], blk_nnz[blk_count],
                                 cscColPtrTR_sub, cscRowIdxTR_sub, cscValTR_sub,
                                 csrColIdxTR_sub, csrRowPtrTR_sub, csrValTR_sub);

            // for (int i = 0; i < blk_nnz[blk_count]; i++)
            //     printf("%d ", cscRowIdxTR_sub[i]);
            // printf("\n\n");

            int nlv = 0;
            int *levelItem_local = (int *)malloc(blk_m[blk_count] * sizeof(int));
            int *levelPtr_local = (int *)malloc((blk_m[blk_count] + 1) * sizeof(int));
            int fasttrack = blk_m[blk_count] == blk_nnz[blk_count] ? 1 : 0;

            // for (int i = 0; i < blk_nnz[blk_count]+1; i++)
            //     printf("%d ", csrColIdxTR_sub[i]);
            // printf("\n\n");
            
            if (fasttrack)
                nlv = 1;
            else
            {
                findlevel(cscColPtrTR_sub, cscRowIdxTR_sub, csrRowPtrTR_sub, blk_m[blk_count],
                          &nlv, levelPtr_local, levelItem_local);
            }
            // fasttrack = 1;
            // nlv = 30000;
            if (fasttrack)
            {
                printf("trsv method = 0\n");
                int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                int num_blocks = ceil((double)blk_m[blk_count] / (double)num_threads);
                ((trsv_blk[trsv_count])).method = 0;
                (trsv_blk[trsv_count]).num_threads = num_threads;
                (trsv_blk[trsv_count]).num_blocks = num_blocks;
                (trsv_blk[trsv_count]).m = blk_m[blk_count];
                (trsv_blk[trsv_count]).substitution = substitution;

                for (int i = 0; i < blk_n[blk_count]; i++)
                {
                    for (int j = cscColPtrTR_sub[i]; j < cscColPtrTR_sub[i + 1]; j++)
                    {
                        recblock_Index[recblock_nnz_ptr] = cscRowIdxTR_sub[j];
                        recblock_Val[recblock_nnz_ptr] = cscValTR_sub[j];
                        recblock_nnz_ptr++;
                    }
                    int index = ptr_offset[blk_count] + i;
                    recblock_Ptr[index] = recblock_Ptr[index - 1] + cscColPtrTR_sub[i + 1] - cscColPtrTR_sub[i];
                }
                // printf("here\n");
                // for (int i = 0; i < blk_nnz[blk_count]; i++)
                //     printf("%d ", recblock_Index[index_offset[blk_count] + i]);
                // printf("\n");
            }
            else
            {
                int nnzr = blk_nnz[blk_count] / blk_m[blk_count];
                // printf("nnzr = %d       nlv = %d\n", nnzr, nlv);
                if (nlv > 20000)
                {
                    printf("trsv method = 1\n");
                    int *d_csrRowPtrTR = NULL;
                    int *d_csrColIdxTR = NULL;
                    VALUE_TYPE *d_csrValTR = NULL;

                    cudaMalloc((void **)&d_csrRowPtrTR, (blk_m[blk_count] + 1) * sizeof(int));
                    cudaMalloc((void **)&d_csrColIdxTR, blk_nnz[blk_count] * sizeof(int));
                    cudaMalloc((void **)&d_csrValTR, blk_nnz[blk_count] * sizeof(VALUE_TYPE));

                    cudaMemcpy(d_csrRowPtrTR, csrRowPtrTR_sub, (blk_m[blk_count] + 1) * sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_csrColIdxTR, csrColIdxTR_sub, blk_nnz[blk_count] * sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_csrValTR, csrValTR_sub, blk_nnz[blk_count] * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

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
                                                   (double *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR, (trsv_blk[trsv_count]).info, &pBufferSize);
                    else if (sizeof(VALUE_TYPE) == 4)
                        cusparseScsrsv2_bufferSize((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                   (float *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR, (trsv_blk[trsv_count]).info, &pBufferSize);
                    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
                    cudaMalloc((void **)&(trsv_blk[trsv_count].pBuffer), pBufferSize);
                    if (sizeof(VALUE_TYPE) == 8)
                        cusparseDcsrsv2_analysis((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                 (double *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR,
                                                 (trsv_blk[trsv_count]).info, (trsv_blk[trsv_count]).policy, (trsv_blk[trsv_count]).pBuffer);
                    else if (sizeof(VALUE_TYPE) == 4)
                        cusparseScsrsv2_analysis((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                 (float *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR,
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

                    recblock_Ptr[ptr_offset[blk_count]] = 0;
                    int nnz_ptr = 0;
                    for (int i = 0; i < blk_m[blk_count]; i++)
                    {
                        for (int j = csrRowPtrTR_sub[i]; j < csrRowPtrTR_sub[i + 1]; j++)
                        {
                            recblock_Index[recblock_nnz_ptr] = csrColIdxTR_sub[j];
                            recblock_Val[recblock_nnz_ptr] = csrValTR_sub[j];
                            recblock_nnz_ptr++;
                            nnz_ptr++;
                        }
                        int index = ptr_offset[blk_count] + i;
                        recblock_Ptr[index + 1] = nnz_ptr;
                    }
                    cu_flag = 1;
                    
                    // printf("here\n");
                    // for (int i = 0; i < blk_nnz[blk_count]; i++)
                    //     printf("%d ", recblock_Index[index_offset[blk_count] + i]);
                    // printf("\n");
                }
                else if ((nnzr <= 15 && nlv <= 20) || (nnzr == 1 && nlv <= 100))
                {
                    printf("trsv method = 2\n");
                    // printf("YYY\n");
                    (trsv_blk[trsv_count]).method = 2;
                    (trsv_blk[trsv_count]).m = blk_m[blk_count];
                    (trsv_blk[trsv_count]).substitution = substitution;
                    (trsv_blk[trsv_count]).nlv = nlv;
                    (trsv_blk[trsv_count]).nnz_lv_array = (int *)malloc(sizeof(int) * nlv);
                    (trsv_blk[trsv_count]).m_lv_array = (int *)malloc(sizeof(int) * nlv);
                    (trsv_blk[trsv_count]).offset_array = (int *)malloc(sizeof(int) * nlv);
                    for (int li = 0; li < nlv; li++)
                    {
                        (trsv_blk[trsv_count]).m_lv_array[li] = levelPtr_local[li + 1] - levelPtr_local[li];
                        (trsv_blk[trsv_count]).offset_array[li] = levelPtr_local[li];
                        int nnz_lv = 0;
                        for (int lvi = levelPtr_local[li]; lvi < levelPtr_local[li + 1]; lvi++)
                        {
                            nnz_lv += csrRowPtrTR_sub[lvi + 1] - csrRowPtrTR_sub[lvi];
                        }
                        (trsv_blk[trsv_count]).nnz_lv_array[li] = nnz_lv;
                    }

                    // printf("here:\n");
                    // for (int i = 0; i < nlv; i++)
                    //     printf("%d ", (trsv_blk[trsv_count]).nnz_lv_array[i]);
                    // printf("\n");

                    for (int i = 0; i < blk_m[blk_count]; i++)
                    {
                        for (int j = csrRowPtrTR_sub[i]; j < csrRowPtrTR_sub[i + 1]; j++)
                        {
                            recblock_Index[recblock_nnz_ptr] = csrColIdxTR_sub[j];
                            recblock_Val[recblock_nnz_ptr] = csrValTR_sub[j];
                            recblock_nnz_ptr++;
                        }
                        int index = ptr_offset[blk_count] + i;
                        recblock_Ptr[index] = recblock_Ptr[index - 1] + csrRowPtrTR_sub[i + 1] - csrRowPtrTR_sub[i];
                    }
                }
                else
                {
                    printf("trsv method = 3\n");
                    int *d_cscRowIdxTR;
                    cudaMalloc((void **)&d_cscRowIdxTR, blk_nnz[blk_count] * sizeof(int));
                    cudaMemcpy(d_cscRowIdxTR, cscRowIdxTR_sub, blk_nnz[blk_count] * sizeof(int), cudaMemcpyHostToDevice);

                    cudaMalloc((void **)&(trsv_blk[trsv_count]).d_levelItem, blk_m[blk_count] * sizeof(int));
                    cudaMemcpy((trsv_blk[trsv_count]).d_levelItem, levelItem_local, blk_m[blk_count] * sizeof(int), cudaMemcpyHostToDevice);

                    cudaMalloc((void **)&(trsv_blk[trsv_count]).d_graphInDegree, blk_m[blk_count] * sizeof(int));
                    cudaMemset((trsv_blk[trsv_count]).d_graphInDegree, 0, blk_m[blk_count] * sizeof(int));

                    cudaMalloc((void **)&(trsv_blk[trsv_count]).d_id_extractor, sizeof(int));

                    int num_threads = 128;
                    int num_blocks = ceil((double)blk_nnz[blk_count] / (double)num_threads);
                    sptrsv_syncfree_csc_cuda_analyser<<<num_blocks, num_threads>>>(d_cscRowIdxTR, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).d_graphInDegree);
                    cudaDeviceSynchronize();
                    cudaMalloc((void **)&(trsv_blk[trsv_count]).d_left_sum, sizeof(VALUE_TYPE) * blk_m[blk_count]);
                    cudaMemset((trsv_blk[trsv_count]).d_left_sum, 0, sizeof(VALUE_TYPE) * blk_m[blk_count]);
                    cudaMemset((trsv_blk[trsv_count]).d_id_extractor, 0, sizeof(int));
                    cudaFree(d_cscRowIdxTR);

                    num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    num_blocks = ceil((double)blk_m[blk_count] / (double)(num_threads / WARP_SIZE));
                    (trsv_blk[trsv_count]).method = 3;
                    (trsv_blk[trsv_count]).num_threads = num_threads;
                    (trsv_blk[trsv_count]).num_blocks = num_blocks;
                    (trsv_blk[trsv_count]).m = blk_m[blk_count];
                    (trsv_blk[trsv_count]).substitution = substitution;

                    // printf("offset = %d     n = %d\n", ptr_offset[blk_count], blk_n[blk_count]);
                    
                    for (int i = 0; i < blk_n[blk_count]; i++)
                    {
                        for (int j = cscColPtrTR_sub[i]; j < cscColPtrTR_sub[i + 1]; j++)
                        {
                            recblock_Index[recblock_nnz_ptr] = cscRowIdxTR_sub[j];
                            recblock_Val[recblock_nnz_ptr] = cscValTR_sub[j];
                            recblock_nnz_ptr++;
                        }
                        int index = ptr_offset[blk_count] + i;
                        recblock_Ptr[index] = recblock_Ptr[index - 1] + cscColPtrTR_sub[i + 1] - cscColPtrTR_sub[i];
                    }
                }
            }

            // for (int i = 0; i < blk_nnz[blk_count]; i++)
            //     printf("%d ", recblock_Index[index_offset[blk_count]+i]);
            // printf("\n\n");

            // for (int i = 0; i < blk_nnz[blk_count]; i++)
            //     printf("%d ", recblock_Index[index_offset[blk_count]+i]);
            // printf("\n\n");

            free(levelPtr_local);
            free(levelItem_local);

                // printf("ptr offset = %d\n", ptr_offset[blk_count]);
                // printf("idx offset = %d\n", index_offset[blk_count]);
                
            if (cu_flag == 0)
                ptr_offset[blk_count + 1] = ptr_offset[blk_count] + blk_m[blk_count];
            else
                ptr_offset[blk_count + 1] = ptr_offset[blk_count] + blk_m[blk_count] + 1;
            index_offset[blk_count + 1] = recblock_nnz_ptr;
            dcsrindex_offset[blk_count + 1] = dcsrindex_offset[blk_count];

            free(csrRowPtrTR_sub);
            free(csrValTR_sub);
            free(csrColIdxTR_sub);

            free(cscColPtrTR_sub);
            free(cscRowIdxTR_sub);
            free(cscValTR_sub);

            trsv_count++;



            cudaDeviceSynchronize();
        }
        else
        {
            int *cscColPtr_sqr = (int *)malloc(sizeof(int) * (blk_n[blk_count] + 1));
            cscColPtr_sqr[0] = 0;
            int *cscRowIdx_sqr = (int *)malloc(sizeof(int) * blk_nnz[blk_count]);
            VALUE_TYPE *cscVal_sqr = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

            int *csrRowPtr_sqr = (int *)malloc(sizeof(int) * (blk_m[blk_count] + 1));
            csrRowPtr_sqr[0] = 0;
            int *csrColIdx_sqr = (int *)malloc(sizeof(int) * blk_nnz[blk_count]);
            VALUE_TYPE *csrVal_sqr = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

            int nnz_ptr = 0;
            for (int i = subrec_leftbound[blk_count]; i < subrec_rightbound[blk_count]; i++)
            {
                for (int j = cscColPtrTR_new[i]; j < cscColPtrTR_new[i + 1]; j++)
                {
                    if (cscRowIdxTR_new[j] >= subrec_upbound[blk_count] && cscRowIdxTR_new[j] < subrec_downbound[blk_count])
                    {
                        cscRowIdx_sqr[nnz_ptr] = cscRowIdxTR_new[j] - subrec_upbound[blk_count];
                        cscVal_sqr[nnz_ptr] = cscValTR_new[j];
                        nnz_ptr++;
                    }
                }
                cscColPtr_sqr[i - subrec_leftbound[blk_count] + 1] = nnz_ptr;
            }

            matrix_transposition(blk_n[blk_count], blk_m[blk_count], blk_nnz[blk_count],
                                 cscColPtr_sqr, cscRowIdx_sqr, cscVal_sqr,
                                 csrColIdx_sqr, csrRowPtr_sqr, csrVal_sqr);

            // for (int i = 0; i < blk_n[blk_count]; i++)
            //     printf("%d ", csrRowPtr_sqr[i]);
            // printf("\n\n");
            
            // printf("rec real = %d\n", recblock_nnz_ptr);
            for (int i = 0; i < blk_m[blk_count]; i++)
            {
                // printf("%d ", recblock_nnz_ptr);
                for (int j = csrRowPtr_sqr[i]; j < csrRowPtr_sqr[i + 1]; j++)
                {
                    recblock_Index[recblock_nnz_ptr] = csrColIdx_sqr[j];
                    // printf("%d ", recblock_Index[recblock_nnz_ptr]);
                    recblock_Val[recblock_nnz_ptr] = csrVal_sqr[j];
                    recblock_nnz_ptr++;
                }
            }
            // printf("\n");
            // printf("rec real = %d\n", recblock_nnz_ptr);
            // for (int i = 0; i < blk_nnz[blk_count]; i++)
            //     printf("%d ", recblock_Index[index_offset[blk_count]+i]);
            // printf("\n\n");

            int i_new = 1;
            int lenmax = csrRowPtr_sqr[1] - csrRowPtr_sqr[0];
            int longrow = 0;
            int longlen = 0;
            int *longrow_idx = (int *)malloc(blk_m[blk_count] * sizeof(int));
            for (int i = 1; i <= blk_m[blk_count]; i++)
            {
                int len = csrRowPtr_sqr[i] - csrRowPtr_sqr[i - 1];
                lenmax = len > lenmax ? len : lenmax;

                if (csrRowPtr_sqr[i] != csrRowPtr_sqr[i - 1])
                {
                    if (len > LONGROW_THRESHOLD)
                    {
                        longrow_idx[longrow] = i - 1;
                        longrow++;
                        longlen += len;
                    }
                    i_new++;
                }
            }

            // printf("i_new = %d\n", i_new);

            int m_new = i_new - 1;
            int nnzr = (blk_nnz[blk_count] - longlen) / m_new;
            double empty_ratio = 100 * (double)(blk_m[blk_count] - m_new) / (double)blk_m[blk_count];
            int dcsr_i = 0;
            int real_i = 0;
            if (blk_nnz[blk_count] != 0)
            {
                int m = blk_m[blk_count];
                int nnz = blk_nnz[blk_count];

                cudaMalloc((void **)&((mv_blk[mv_count]).d_y), m * sizeof(VALUE_TYPE));
                if (nnzr <= 12 && empty_ratio <= 50)
                {
                    printf("mv method = 0\n");
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)m / (double)num_threads);
                    (mv_blk[mv_count]).method = 0;
                    (mv_blk[mv_count]).num_threads = num_threads;
                    (mv_blk[mv_count]).num_blocks = num_blocks;
                    (mv_blk[mv_count]).m = m;
                    for (int i = 0; i < blk_m[blk_count]; i++)
                    {
                        int row_nnz = csrRowPtr_sqr[i + 1] - csrRowPtr_sqr[i];
                        int index = ptr_offset[blk_count] + i;
                        recblock_Ptr[index] = recblock_Ptr[index - 1] + row_nnz;
                    }
                    real_i = blk_m[blk_count];
                    
                    // for (int i = 0; i < m+1; i++)
                    //     printf("%d ", recblock_Ptr[ptr_offset[blk_count]+i-1]);
                    // printf("\n\n");
                }
                else if (nnzr <= 12 && empty_ratio > 50)
                {
                    printf("mv method = 1\n");
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)m_new / (double)num_threads);
                    (mv_blk[mv_count]).method = 1;
                    (mv_blk[mv_count]).num_threads = num_threads;
                    (mv_blk[mv_count]).num_blocks = num_blocks;
                    (mv_blk[mv_count]).m_new = m_new;
                    for (int i = 0; i < blk_m[blk_count]; i++)
                    {
                        if (csrRowPtr_sqr[i + 1] != csrRowPtr_sqr[i])
                        {
                            int row_nnz = csrRowPtr_sqr[i + 1] - csrRowPtr_sqr[i];
                            int index = ptr_offset[blk_count] + dcsr_i;
                            int index_dcsr = dcsrindex_offset[blk_count] + dcsr_i;
                            recblock_Ptr[index] = recblock_Ptr[index - 1] + row_nnz;
                            recblock_dcsr_rowidx[index_dcsr] = i;
                            dcsr_i++;
                        }
                    }
                    real_i = dcsr_i;
                    // printf("real = %d\n", real_i);
                }
                else if (nnzr > 12 && empty_ratio <= 15)
                {
                    printf("mv method = 2\n");
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)m / (double)(num_threads / WARP_SIZE));

                    (mv_blk[mv_count]).method = 2;
                    (mv_blk[mv_count]).num_threads = num_threads;
                    (mv_blk[mv_count]).num_blocks = num_blocks;
                    (mv_blk[mv_count]).m = m;
                    for (int i = 0; i < blk_m[blk_count]; i++)
                    {
                        int row_nnz = csrRowPtr_sqr[i + 1] - csrRowPtr_sqr[i];
                        int index = ptr_offset[blk_count] + i;
                        recblock_Ptr[index] = recblock_Ptr[index - 1] + row_nnz;
                    }
                    real_i = blk_m[blk_count];
                }
                else
                {
                    printf("mv method = 3\n");
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)m_new / (double)(num_threads / WARP_SIZE));

                    (mv_blk[mv_count]).method = 3;
                    (mv_blk[mv_count]).num_threads = num_threads;
                    (mv_blk[mv_count]).num_blocks = num_blocks;
                    (mv_blk[mv_count]).m_new = m_new;

                    for (int i = 0; i < blk_m[blk_count]; i++)
                    {
                        if (csrRowPtr_sqr[i + 1] != csrRowPtr_sqr[i])
                        {
                            int row_nnz = csrRowPtr_sqr[i + 1] - csrRowPtr_sqr[i];
                            int index = ptr_offset[blk_count] + dcsr_i;
                            int index_dcsr = dcsrindex_offset[blk_count] + dcsr_i;
                            recblock_Ptr[index] = recblock_Ptr[index - 1] + row_nnz;
                            recblock_dcsr_rowidx[index_dcsr] = i;
                            dcsr_i++;
                        }
                    }
                    real_i = dcsr_i;
                }

                (mv_blk[mv_count]).longrow = longrow;
                //process long rows
                if (longrow != 0)
                {
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)lenmax / (double)num_threads);
                    (mv_blk[mv_count]).num_threads_l = num_threads;
                    (mv_blk[mv_count]).num_blocks_l = num_blocks;
                    cudaMalloc((void **)&((mv_blk[mv_count]).d_csrRowPtr_l), (m + 1) * sizeof(int));
                    cudaMalloc((void **)&((mv_blk[mv_count]).d_csrColIdx_l), nnz * sizeof(int));
                    cudaMalloc((void **)&((mv_blk[mv_count]).d_csrVal_l), nnz * sizeof(VALUE_TYPE));
                    cudaMemcpy((mv_blk[mv_count]).d_csrRowPtr_l, csrRowPtr_sqr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy((mv_blk[mv_count]).d_csrColIdx_l, csrColIdx_sqr, nnz * sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy((mv_blk[mv_count]).d_csrVal_l, csrVal_sqr, nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

                    cudaMalloc((void **)&((mv_blk[mv_count]).d_longrow_idx), longrow * sizeof(int));
                    cudaMemcpy((mv_blk[mv_count]).d_longrow_idx, longrow_idx, longrow * sizeof(int), cudaMemcpyHostToDevice);
                }
            }

            // for (int i = 0; i < blk_nnz[blk_count]; i++)
            //     printf("%d ", recblock_Index[index_offset[blk_count]+i]);
            // printf("\n\n");

                // printf("ptr offset = %d\n", ptr_offset[blk_count]);
                // printf("idx offset = %d\n", index_offset[blk_count]);
                // printf("dcsr offset = %d\n", dcsrindex_offset[blk_count]);
            
            ptr_offset[blk_count + 1] = ptr_offset[blk_count] + real_i;
            index_offset[blk_count + 1] = recblock_nnz_ptr;
            dcsrindex_offset[blk_count + 1] = dcsrindex_offset[blk_count] + dcsr_i;

            free(csrRowPtr_sqr);
            free(csrColIdx_sqr);
            free(csrVal_sqr);

            free(cscRowIdx_sqr);
            free(cscColPtr_sqr);
            free(cscVal_sqr);

            mv_count++;
        }
    }
}

void U_preprocessing(int *cscRowIdxTR_new,
                     int *cscColPtrTR_new,
                     VALUE_TYPE *cscValTR_new,
                     int nnzTR,
                     int m,
                     int n,
                     int substitution,
                     int nlevel,
                     int *blk_m,
                     int *blk_n,
                     int *blk_nnz,
                     int *subtri_upbound,
                     int *subtri_downbound,
                     int *subrec_upbound,
                     int *subrec_downbound,
                     int *subrec_rightbound,
                     int *subrec_leftbound,
                     SpMV_block *mv_blk,
                     SpTRSV_block *trsv_blk,
                     int *recblock_Ptr,
                     int *recblock_Index,
                     int *recblock_dcsr_rowidx,
                     double *recblock_Val,
                     int *ptr_offset,
                     int *index_offset,
                     int *dcsrindex_offset,
                     int ptr_size,
                     int idx_size,
                     int dcsr_size)
{
    // get auxiliary arrary for our datastruct
    int tri_block = pow(2, nlevel);
    int squ_block = tri_block - 1;
    int sum_block = tri_block + squ_block;

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
            int *cscColPtrTR_sub = (int *)malloc(sizeof(int) * (blk_n[blk_count] + 1));
            cscColPtrTR_sub[0] = 0;
            int *cscRowIdxTR_sub = (int *)malloc(sizeof(int) * blk_nnz[blk_count]);
            VALUE_TYPE *cscValTR_sub = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

            int nnz_ptr = 0;
            for (int i = subtri_upbound[blk_count]; i < subtri_downbound[blk_count]; i++)
            {
                for (int j = cscColPtrTR_new[i]; j < cscColPtrTR_new[i + 1]; j++)
                {
                    if (cscRowIdxTR_new[j] >= subtri_upbound[blk_count])
                    {
                        cscRowIdxTR_sub[nnz_ptr] = cscRowIdxTR_new[j] - subtri_upbound[blk_count];
                        cscValTR_sub[nnz_ptr] = cscValTR_new[j];
                        nnz_ptr++;
                    }
                }
                cscColPtrTR_sub[i - subtri_upbound[blk_count] + 1] = nnz_ptr;
            }

            int *csrRowPtrTR_sub = (int *)malloc(sizeof(int) * (blk_m[blk_count] + 1));
            csrRowPtrTR_sub[0] = 0;
            int *csrColIdxTR_sub = (int *)malloc(sizeof(int) * blk_nnz[blk_count]);
            VALUE_TYPE *csrValTR_sub = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * blk_nnz[blk_count]);
            matrix_transposition(blk_n[blk_count], blk_m[blk_count], blk_nnz[blk_count],
                                 cscColPtrTR_sub, cscRowIdxTR_sub, cscValTR_sub,
                                 csrColIdxTR_sub, csrRowPtrTR_sub, csrValTR_sub);

            int nlv = 0;
            int *levelItem_local = (int *)malloc(blk_m[blk_count] * sizeof(int));
            int *levelPtr_local = (int *)malloc((blk_m[blk_count] + 1) * sizeof(int));
            int fasttrack = blk_m[blk_count] == blk_nnz[blk_count] ? 1 : 0;

            for (int i = 0; i <= blk_n[blk_count]; i++)
                printf("%d ", cscColPtrTR_sub[i]);
            printf("\n\n");
        
            if (fasttrack)
                nlv = 1;
            else
            {
                findlevel(cscColPtrTR_sub, cscRowIdxTR_sub, csrRowPtrTR_sub, blk_m[blk_count],
                          &nlv, levelPtr_local, levelItem_local);
            }

            if (fasttrack)
            {
                int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                int num_blocks = ceil((double)blk_m[blk_count] / (double)num_threads);
                ((trsv_blk[trsv_count])).method = 0;
                (trsv_blk[trsv_count]).num_threads = num_threads;
                (trsv_blk[trsv_count]).num_blocks = num_blocks;
                (trsv_blk[trsv_count]).m = blk_m[blk_count];
                (trsv_blk[trsv_count]).substitution = substitution;

                for (int i = 0; i < blk_n[blk_count]; i++)
                {
                    for (int j = cscColPtrTR_sub[i]; j < cscColPtrTR_sub[i + 1]; j++)
                    {
                        recblock_Index[recblock_nnz_ptr] = cscRowIdxTR_sub[j];
                        recblock_Val[recblock_nnz_ptr] = cscValTR_sub[j];
                        recblock_nnz_ptr++;
                    }
                    int index = ptr_offset[blk_count] + i;
                    recblock_Ptr[index] = recblock_Ptr[index - 1] + cscColPtrTR_sub[i + 1] - cscColPtrTR_sub[i];
                }
            }
            else
            {
                int nnzr = blk_nnz[blk_count] / blk_m[blk_count];
                printf("nnzr = %d       nlv = %d\n", nnzr, nlv);
                if (nlv > 20000)
                {
                    int *d_csrRowPtrTR = NULL;
                    int *d_csrColIdxTR = NULL;
                    VALUE_TYPE *d_csrValTR = NULL;

                    cudaMalloc((void **)&d_csrRowPtrTR, (blk_m[blk_count] + 1) * sizeof(int));
                    cudaMalloc((void **)&d_csrColIdxTR, blk_nnz[blk_count] * sizeof(int));
                    cudaMalloc((void **)&d_csrValTR, blk_nnz[blk_count] * sizeof(VALUE_TYPE));

                    cudaMemcpy(d_csrRowPtrTR, csrRowPtrTR_sub, (blk_m[blk_count] + 1) * sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_csrColIdxTR, csrColIdxTR_sub, blk_nnz[blk_count] * sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_csrValTR, csrValTR_sub, blk_nnz[blk_count] * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

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
                                                   (double *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR, (trsv_blk[trsv_count]).info, &pBufferSize);
                    else if (sizeof(VALUE_TYPE) == 4)
                        cusparseScsrsv2_bufferSize((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                   (float *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR, (trsv_blk[trsv_count]).info, &pBufferSize);
                    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
                    cudaMalloc((void **)&(trsv_blk[trsv_count].pBuffer), pBufferSize);
                    if (sizeof(VALUE_TYPE) == 8)
                        cusparseDcsrsv2_analysis((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                 (double *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR,
                                                 (trsv_blk[trsv_count]).info, (trsv_blk[trsv_count]).policy, (trsv_blk[trsv_count]).pBuffer);
                    else if (sizeof(VALUE_TYPE) == 4)
                        cusparseScsrsv2_analysis((trsv_blk[trsv_count]).handle, (trsv_blk[trsv_count]).trans, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).descr,
                                                 (float *)d_csrValTR, d_csrRowPtrTR, d_csrColIdxTR,
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

                    recblock_Ptr[ptr_offset[blk_count]] = 0;
                    int nnz_ptr = 0;
                    for (int i = 0; i < blk_m[blk_count]; i++)
                    {
                        for (int j = csrRowPtrTR_sub[i]; j < csrRowPtrTR_sub[i + 1]; j++)
                        {
                            recblock_Index[recblock_nnz_ptr] = csrColIdxTR_sub[j];
                            recblock_Val[recblock_nnz_ptr] = csrValTR_sub[j];
                            recblock_nnz_ptr++;
                            nnz_ptr++;
                        }
                        int index = ptr_offset[blk_count] + i;
                        recblock_Ptr[index + 1] = nnz_ptr;
                    }
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
                    for (int li = 0; li < nlv; li++)
                    {
                        (trsv_blk[trsv_count]).m_lv_array[li] = levelPtr_local[li + 1] - levelPtr_local[li];
                        (trsv_blk[trsv_count]).offset_array[li] = levelPtr_local[li];
                        int nnz_lv = 0;
                        for (int lvi = levelPtr_local[li]; lvi < levelPtr_local[li + 1]; lvi++)
                        {
                            nnz_lv += csrRowPtrTR_sub[lvi + 1] - csrRowPtrTR_sub[lvi];
                        }
                        (trsv_blk[trsv_count]).nnz_lv_array[li] = nnz_lv;
                    }

                    for (int i = 0; i < blk_m[blk_count]; i++)
                    {
                        for (int j = csrRowPtrTR_sub[i]; j < csrRowPtrTR_sub[i + 1]; j++)
                        {
                            recblock_Index[recblock_nnz_ptr] = csrColIdxTR_sub[j];
                            recblock_Val[recblock_nnz_ptr] = csrValTR_sub[j];
                            recblock_nnz_ptr++;
                        }
                        int index = ptr_offset[blk_count] + i;
                        recblock_Ptr[index] = recblock_Ptr[index - 1] + csrRowPtrTR_sub[i + 1] - csrRowPtrTR_sub[i];
                    }
                }
                else
                {
                    int *d_cscRowIdxTR;
                    cudaMalloc((void **)&d_cscRowIdxTR, blk_nnz[blk_count] * sizeof(int));
                    cudaMemcpy(d_cscRowIdxTR, cscRowIdxTR_sub, blk_nnz[blk_count] * sizeof(int), cudaMemcpyHostToDevice);

                    cudaMalloc((void **)&(trsv_blk[trsv_count]).d_levelItem, blk_m[blk_count] * sizeof(int));
                    cudaMemcpy((trsv_blk[trsv_count]).d_levelItem, levelItem_local, blk_m[blk_count] * sizeof(int), cudaMemcpyHostToDevice);

                    cudaMalloc((void **)&(trsv_blk[trsv_count]).d_graphInDegree, blk_m[blk_count] * sizeof(int));
                    cudaMemset((trsv_blk[trsv_count]).d_graphInDegree, 0, blk_m[blk_count] * sizeof(int));

                    cudaMalloc((void **)&(trsv_blk[trsv_count]).d_id_extractor, sizeof(int));

                    int num_threads = 128;
                    int num_blocks = ceil((double)blk_nnz[blk_count] / (double)num_threads);
                    sptrsv_syncfree_csc_cuda_analyser<<<num_blocks, num_threads>>>(d_cscRowIdxTR, blk_m[blk_count], blk_nnz[blk_count], (trsv_blk[trsv_count]).d_graphInDegree);
                    cudaDeviceSynchronize();
                    cudaMalloc((void **)&(trsv_blk[trsv_count]).d_left_sum, sizeof(VALUE_TYPE) * blk_m[blk_count]);
                    cudaMemset((trsv_blk[trsv_count]).d_left_sum, 0, sizeof(VALUE_TYPE) * blk_m[blk_count]);
                    cudaMemset((trsv_blk[trsv_count]).d_id_extractor, 0, sizeof(int));
                    cudaFree(d_cscRowIdxTR);

                    num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    num_blocks = ceil((double)blk_m[blk_count] / (double)(num_threads / WARP_SIZE));
                    (trsv_blk[trsv_count]).method = 3;
                    (trsv_blk[trsv_count]).num_threads = num_threads;
                    (trsv_blk[trsv_count]).num_blocks = num_blocks;
                    (trsv_blk[trsv_count]).m = blk_m[blk_count];
                    (trsv_blk[trsv_count]).substitution = substitution;

                    for (int i = 0; i < blk_n[blk_count]; i++)
                    {
                        for (int j = cscColPtrTR_sub[i]; j < cscColPtrTR_sub[i + 1]; j++)
                        {
                            recblock_Index[recblock_nnz_ptr] = cscRowIdxTR_sub[j];
                            recblock_Val[recblock_nnz_ptr] = cscValTR_sub[j];
                            recblock_nnz_ptr++;
                        }
                        int index = ptr_offset[blk_count] + i;
                        recblock_Ptr[index] = recblock_Ptr[index - 1] + cscColPtrTR_sub[i + 1] - cscColPtrTR_sub[i];
                    }
                }
            }

            free(levelPtr_local);
            free(levelItem_local);

            if (cu_flag == 0)
                ptr_offset[blk_count + 1] = ptr_offset[blk_count] + blk_m[blk_count];
            else
                ptr_offset[blk_count + 1] = ptr_offset[blk_count] + blk_m[blk_count] + 1;
            index_offset[blk_count + 1] = recblock_nnz_ptr;
            dcsrindex_offset[blk_count + 1] = dcsrindex_offset[blk_count];

            free(csrRowPtrTR_sub);
            free(csrValTR_sub);
            free(csrColIdxTR_sub);

            free(cscColPtrTR_sub);
            free(cscRowIdxTR_sub);
            free(cscValTR_sub);

            trsv_count++;
        }
        else
        {
            int *cscColPtr_sqr = (int *)malloc(sizeof(int) * (blk_n[blk_count] + 1));
            cscColPtr_sqr[0] = 0;
            int *cscRowIdx_sqr = (int *)malloc(sizeof(int) * blk_nnz[blk_count]);
            VALUE_TYPE *cscVal_sqr = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

            int *csrRowPtr_sqr = (int *)malloc(sizeof(int) * (blk_m[blk_count] + 1));
            csrRowPtr_sqr[0] = 0;
            int *csrColIdx_sqr = (int *)malloc(sizeof(int) * blk_nnz[blk_count]);
            VALUE_TYPE *csrVal_sqr = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * blk_nnz[blk_count]);

            int nnz_ptr = 0;
            for (int i = subrec_leftbound[blk_count]; i < subrec_rightbound[blk_count]; i++)
            {
                for (int j = cscColPtrTR_new[i]; j < cscColPtrTR_new[i + 1]; j++)
                {
                    if (cscRowIdxTR_new[j] >= subrec_upbound[blk_count] && cscRowIdxTR_new[j] < subrec_downbound[blk_count])
                    {
                        cscRowIdx_sqr[nnz_ptr] = cscRowIdxTR_new[j] - subrec_upbound[blk_count];
                        cscVal_sqr[nnz_ptr] = cscValTR_new[j];
                        nnz_ptr++;
                    }
                }
                cscColPtr_sqr[i - subrec_leftbound[blk_count] + 1] = nnz_ptr;
            }

            matrix_transposition(blk_n[blk_count], blk_m[blk_count], blk_nnz[blk_count],
                                 cscColPtr_sqr, cscRowIdx_sqr, cscVal_sqr,
                                 csrColIdx_sqr, csrRowPtr_sqr, csrVal_sqr);

            for (int i = 0; i < blk_m[blk_count]; i++)
            {
                for (int j = csrRowPtr_sqr[i]; j < csrRowPtr_sqr[i + 1]; j++)
                {
                    recblock_Index[recblock_nnz_ptr] = csrColIdx_sqr[j];
                    recblock_Val[recblock_nnz_ptr] = csrVal_sqr[j];
                    recblock_nnz_ptr++;
                }
            }
            int i_new = 1;
            int lenmax = csrRowPtr_sqr[1] - csrRowPtr_sqr[0];
            int longrow = 0;
            int longlen = 0;
            int *longrow_idx = (int *)malloc(blk_m[blk_count] * sizeof(int));
            for (int i = 1; i <= blk_m[blk_count]; i++)
            {
                int len = csrRowPtr_sqr[i] - csrRowPtr_sqr[i - 1];
                lenmax = len > lenmax ? len : lenmax;

                if (csrRowPtr_sqr[i] != csrRowPtr_sqr[i - 1])
                {
                    if (len > LONGROW_THRESHOLD)
                    {
                        longrow_idx[longrow] = i - 1;
                        longrow++;
                        longlen += len;
                    }
                    i_new++;
                }
            }
            int m_new = i_new - 1;
            int nnzr = (blk_nnz[blk_count] - longlen) / m_new;
            double empty_ratio = 100 * (double)(blk_m[blk_count] - m_new) / (double)blk_m[blk_count];

            int dcsr_i = 0;
            int real_i = 0;
            if (blk_nnz[blk_count] != 0)
            {
                int m = blk_m[blk_count];
                int nnz = blk_nnz[blk_count];

                cudaMalloc((void **)&((mv_blk[mv_count]).d_y), m * sizeof(VALUE_TYPE));
                if (nnzr <= 12 && empty_ratio <= 50)
                {
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)m / (double)num_threads);
                    (mv_blk[mv_count]).method = 0;
                    (mv_blk[mv_count]).num_threads = num_threads;
                    (mv_blk[mv_count]).num_blocks = num_blocks;
                    (mv_blk[mv_count]).m = m;
                    for (int i = 0; i < blk_m[blk_count]; i++)
                    {
                        int row_nnz = csrRowPtr_sqr[i + 1] - csrRowPtr_sqr[i];
                        int index = ptr_offset[blk_count] + i;
                        recblock_Ptr[index] = recblock_Ptr[index - 1] + row_nnz;
                    }
                    real_i = blk_m[blk_count];
                }
                else if (nnzr <= 12 && empty_ratio > 50)
                {
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)m_new / (double)num_threads);
                    (mv_blk[mv_count]).method = 1;
                    (mv_blk[mv_count]).num_threads = num_threads;
                    (mv_blk[mv_count]).num_blocks = num_blocks;
                    (mv_blk[mv_count]).m_new = m_new;
                    for (int i = 0; i < blk_m[blk_count]; i++)
                    {
                        if (csrRowPtr_sqr[i + 1] != csrRowPtr_sqr[i])
                        {
                            int row_nnz = csrRowPtr_sqr[i + 1] - csrRowPtr_sqr[i];
                            int index = ptr_offset[blk_count] + dcsr_i;
                            int index_dcsr = dcsrindex_offset[blk_count] + dcsr_i;
                            recblock_Ptr[index] = recblock_Ptr[index - 1] + row_nnz;
                            recblock_dcsr_rowidx[index_dcsr] = i;
                            dcsr_i++;
                        }
                    }
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
                    for (int i = 0; i < blk_m[blk_count]; i++)
                    {
                        int row_nnz = csrRowPtr_sqr[i + 1] - csrRowPtr_sqr[i];
                        int index = ptr_offset[blk_count] + i;
                        recblock_Ptr[index] = recblock_Ptr[index - 1] + row_nnz;
                    }
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

                    for (int i = 0; i < blk_m[blk_count]; i++)
                    {
                        if (csrRowPtr_sqr[i + 1] != csrRowPtr_sqr[i])
                        {
                            int row_nnz = csrRowPtr_sqr[i + 1] - csrRowPtr_sqr[i];
                            int index = ptr_offset[blk_count] + dcsr_i;
                            int index_dcsr = dcsrindex_offset[blk_count] + dcsr_i;
                            recblock_Ptr[index] = recblock_Ptr[index - 1] + row_nnz;
                            recblock_dcsr_rowidx[index_dcsr] = i;
                            dcsr_i++;
                        }
                    }
                    real_i = dcsr_i;
                }
                (mv_blk[mv_count]).longrow = longrow;
                //process long rows
                if (longrow != 0)
                {
                    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                    int num_blocks = ceil((double)lenmax / (double)num_threads);
                    (mv_blk[mv_count]).num_threads_l = num_threads;
                    (mv_blk[mv_count]).num_blocks_l = num_blocks;
                    cudaMalloc((void **)&((mv_blk[mv_count]).d_csrRowPtr_l), (m + 1) * sizeof(int));
                    cudaMalloc((void **)&((mv_blk[mv_count]).d_csrColIdx_l), nnz * sizeof(int));
                    cudaMalloc((void **)&((mv_blk[mv_count]).d_csrVal_l), nnz * sizeof(VALUE_TYPE));
                    cudaMemcpy((mv_blk[mv_count]).d_csrRowPtr_l, csrRowPtr_sqr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy((mv_blk[mv_count]).d_csrColIdx_l, csrColIdx_sqr, nnz * sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy((mv_blk[mv_count]).d_csrVal_l, csrVal_sqr, nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

                    cudaMalloc((void **)&((mv_blk[mv_count]).d_longrow_idx), longrow * sizeof(int));
                    cudaMemcpy((mv_blk[mv_count]).d_longrow_idx, longrow_idx, longrow * sizeof(int), cudaMemcpyHostToDevice);
                }
            }


            ptr_offset[blk_count + 1] = ptr_offset[blk_count] + real_i;
            index_offset[blk_count + 1] = recblock_nnz_ptr;
            dcsrindex_offset[blk_count + 1] = dcsrindex_offset[blk_count] + dcsr_i;

            free(csrRowPtr_sqr);
            free(csrColIdx_sqr);
            free(csrVal_sqr);

            free(cscRowIdx_sqr);
            free(cscColPtr_sqr);
            free(cscVal_sqr);

            mv_count++;
        }
    }
}

#endif
