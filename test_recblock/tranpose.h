#ifndef _TRANS_
#define _TRANS_

#include "common.h"
#include "utils.h"

void matrix_transposition(const int m,
                          const int n,
                          const int nnz,
                          const int *csrRowPtr,
                          const int *csrColIdx,
                          const VALUE_TYPE *csrVal,
                          int *cscRowIdx,
                          int *cscColPtr,
                          VALUE_TYPE *cscVal)
{
    // histogram in column pointer
    memset(cscColPtr, 0, sizeof(int) * (n + 1));
    for (int i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);

    // for (int i = 0; i < m + 1; i++)
    //     printf("%d ", cscColPtr[i]);
    // printf("\n\n");

    int *cscColIncr = (int *)malloc(sizeof(int) * (n + 1));
    memcpy(cscColIncr, cscColPtr, sizeof(int) * (n + 1));

    // for (int i = 0; i < n + 1; i++)
    //     printf("%d ", cscColIncr[i]);
    // printf("\n\n");

    // insert nnz to csc
    for (int row = 0; row < m; row++)
    {
        for (int j = csrRowPtr[row]; j < csrRowPtr[row + 1]; j++)
        {
            int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            // printf("%d ", cscRowIdx[cscColIncr[col]]);
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
        // printf("\n\n");
    }

    free(cscColIncr);
}

void matrix_transposition_lite(const int m,
                               const int n,
                               const int nnz,
                               const int *csrRowPtr,
                               const int *csrColIdx,
                               int *cscRowIdx,
                               int *cscColPtr)
{
    // histogram in column pointer
    memset(cscColPtr, 0, sizeof(int) * (n + 1));
    for (int i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);

    int *cscColIncr = (int *)malloc(sizeof(int) * (n + 1));
    memcpy(cscColIncr, cscColPtr, sizeof(int) * (n + 1));

    // insert nnz to csc
    for (int row = 0; row < m; row++)
    {
        for (int j = csrRowPtr[row]; j < csrRowPtr[row + 1]; j++)
        {
            int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            cscColIncr[col]++;
        }
    }

    free(cscColIncr);
}

void matrix_transposition_litelite(const int m,
                                   const int n,
                                   const int nnz,
                                   const int *csrRowPtr,
                                   const int *csrColIdx,
                                   int *cscColPtr)
{
    // histogram in column pointer
    memset(cscColPtr, 0, sizeof(int) * (n + 1));
    for (int i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);
}

#endif
