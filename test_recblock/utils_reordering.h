#ifndef _UTILS_REORDERING_
#define _UTILS_REORDERING_

#include "common.h"
#include "utils.h"

// code for for reordering columns of CSC according to level-set execution order
void levelset_reordering_col_csc(const int *cscColPtrTR,
                                 const int *cscRowIdxTR,
                                 const VALUE_TYPE *cscValTR,
                                 int *cscColPtrTR_new,
                                 int *cscRowIdxTR_new,
                                 VALUE_TYPE *cscValTR_new,
                                 int *levelPtr,
                                 int *levelItem,
                                 int *nlv,
                                 int m,
                                 int n,
                                 int nnzTR,
                                 int substitution)
{
    // transpose to have csr data
    int *csrRowPtrTR = (int *)malloc((m + 1) * sizeof(int));

    // transpose from csc to csr
    matrix_transposition_litelite(m, n, nnzTR,
                                  cscColPtrTR, cscRowIdxTR, csrRowPtrTR);

    *nlv = 0;
    findlevel(cscColPtrTR, cscRowIdxTR, csrRowPtrTR, m, nlv, levelPtr, levelItem);

    // reorder columns
    cscColPtrTR_new[0] = 0;

    for (int i = 0; i < n; i++)
    {
        int idx = substitution == SUBSTITUTION_FORWARD ? levelItem[i] : levelItem[n - i - 1];
        int nnzr = cscColPtrTR[idx + 1] - cscColPtrTR[idx];
        cscColPtrTR_new[i + 1] = cscColPtrTR_new[i] + nnzr;

        for (int j = 0; j < nnzr; j++)
        {
            int off = cscColPtrTR[idx] + j;
            int off_new = cscColPtrTR_new[i] + j;
            cscRowIdxTR_new[off_new] = cscRowIdxTR[off];
            cscValTR_new[off_new] = cscValTR[off];
        }
    }

    free(csrRowPtrTR);
}

// code for for reordering rows of CSR according to level-set execution order
void levelset_reordering_row_csr(const int *csrRowPtrTR,
                                 const int *csrColIdxTR,
                                 const VALUE_TYPE *csrValTR,
                                 int *csrRowPtrTR_new,
                                 int *csrColIdxTR_new,
                                 VALUE_TYPE *csrValTR_new,
                                 int *levelPtr,
                                 int *levelItem,
                                 int *nlv,
                                 int m,
                                 int n,
                                 int nnzTR,
                                 int substitution)
{
    // transpose to have csc data
    int *cscColPtrTR = (int *)malloc((n + 1) * sizeof(int));
    int *cscRowIdxTR = (int *)malloc(nnzTR * sizeof(int));

    // transpose from csr to csc
    matrix_transposition_lite(m, n, nnzTR, csrRowPtrTR, csrColIdxTR,
                              cscRowIdxTR, cscColPtrTR);

    *nlv = 0;
    findlevel(cscColPtrTR, cscRowIdxTR, csrRowPtrTR, m, nlv, levelPtr, levelItem);

    // reorder columns
    csrRowPtrTR_new[0] = 0;

    for (int i = 0; i < n; i++)
    {
        int idx = substitution == SUBSTITUTION_FORWARD ? levelItem[i] : levelItem[n - i - 1];
        int nnzr = csrRowPtrTR[idx + 1] - csrRowPtrTR[idx];
        csrRowPtrTR_new[i + 1] = csrRowPtrTR_new[i] + nnzr;

        for (int j = 0; j < nnzr; j++)
        {
            int off = csrRowPtrTR[idx] + j;
            int off_new = csrRowPtrTR_new[i] + j;
            csrColIdxTR_new[off_new] = csrColIdxTR[off];
            csrValTR_new[off_new] = csrValTR[off];
        }
    }

    free(cscColPtrTR);
    free(cscRowIdxTR);
}

// code for for reordering columns and rows of CSC according to level-set execution order
void levelset_reordering_colrow_csc(const int *cscColPtrTR,
                                    const int *cscRowIdxTR,
                                    const VALUE_TYPE *cscValTR,
                                    int *cscColPtrTR_new,
                                    int *cscRowIdxTR_new,
                                    VALUE_TYPE *cscValTR_new,
                                    int *levelItem,
                                    int m,
                                    int n,
                                    int nnzTR,
                                    int substitution)
{
    // transpose to have csr data
    int *csrRowPtrTR = (int *)malloc((m + 1) * sizeof(int));

    // transpose from csc to csr
    matrix_transposition_litelite(m, n, nnzTR,
                                  cscColPtrTR, cscRowIdxTR, csrRowPtrTR);
    
    // for (int i = 0; i < m + 1; i++)
    //     printf("%d ", csrRowPtrTR[i]);
    // printf("\n");

    int *levelPtr = (int *)malloc((m + 1) * sizeof(int));

    int nlv = 0;
    findlevel(cscColPtrTR, cscRowIdxTR, csrRowPtrTR, m, &nlv, levelPtr, levelItem);

    int *levelItem_tmp = (int *)malloc(m * sizeof(int));
    memcpy(levelItem_tmp, levelItem, m * sizeof(int));
    int *levelperm = (int *)malloc(m * sizeof(int));

    if (substitution == SUBSTITUTION_FORWARD)
        for (int i = 0; i < m; i++)
            levelperm[i] = i;
    else
        for (int i = m - 1; i >= 0; i--)
            levelperm[i] = i;

    // printf("perm ori:\n");
    // for (int i = 0; i < m; i++)
    //     printf("%d ", levelItem_tmp[i]);
    // printf("\n");

    quicksort_keyval<int, int>(levelItem_tmp, levelperm, 0, m - 1);
    
    // printf("perm:\n");
    // for (int i = 0; i < m; i++)
    //     printf("%d ", levelperm[i]);
    // printf("\n");

    if (substitution == SUBSTITUTION_BACKWARD)
    {
        for (int i = 0; i < m; i++)
        {
            levelperm[levelItem[i]] = m - i - 1;
        }
    }

    // for (int i = 0; i < m; i++)
    //     printf("%d ", levelItem[i]);
    // printf("\n");

    // reorder columns
    cscColPtrTR_new[0] = 0;

    for (int i = 0; i < n; i++)
    {
        int idx = substitution == SUBSTITUTION_FORWARD ? levelItem[i] : levelItem[n - i - 1];

        int nnzr = cscColPtrTR[idx + 1] - cscColPtrTR[idx];
        cscColPtrTR_new[i + 1] = cscColPtrTR_new[i] + nnzr;
        for (int j = 0; j < nnzr; j++)
        {
            int off = cscColPtrTR[idx] + j;
            int off_new = cscColPtrTR_new[i] + j;
            cscRowIdxTR_new[off_new] = cscRowIdxTR[off];
            cscValTR_new[off_new] = cscValTR[off];
        }
    }

    // reorder row ids in each column
    for (int i = 0; i < nnzTR; i++)
    {
        cscRowIdxTR_new[i] = levelperm[cscRowIdxTR_new[i]];
    }

    if (substitution == SUBSTITUTION_BACKWARD)
    {
        int *levelItem_tmp = (int *)malloc(sizeof(int) * m);
        for (int i = 0; i < m; i++)
        {
            levelItem_tmp[m - i - 1] = levelItem[i];
        }
        memcpy(levelItem, levelItem_tmp, sizeof(int) * m);
    }

    // for (int i = 0; i < nnzTR; i++)
    //     printf("%d ", cscRowIdxTR_new[i]);
    // printf("\n\n");


    free(csrRowPtrTR);
    free(levelPtr);
    free(levelperm);
    free(levelItem_tmp);

    return;
}

void levelset_reordering_vecb(const VALUE_TYPE *b,
                              VALUE_TYPE *b_perm,
                              const int *levelItem,
                              const int m)
{
    int *levelItem_tmp = (int *)malloc(m * sizeof(int));
    memcpy(levelItem_tmp, levelItem, m * sizeof(int));
    int *levelperm = (int *)malloc(m * sizeof(int));
    for (int i = 0; i < m; i++)
        levelperm[i] = i;

    //quick_sort_key_val_pair<int, int>(levelItem_tmp, levelperm, m);
    quicksort_keyval<int, int>(levelItem_tmp, levelperm, 0, m - 1);

    for (int i = 0; i < m; i++)
    {
        b_perm[i] = b[levelItem[i]];
    }

    free(levelperm);
    free(levelItem_tmp);

    return;
}

void levelset_reordering_vecx(const VALUE_TYPE *x_perm,
                              VALUE_TYPE *x,
                              const int *levelItem,
                              const int n)
{
    for (int i = 0; i < n; i++)
    {
        x[levelItem[i]] = x_perm[i];
    }
    return;
}

#endif
