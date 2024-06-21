#ifndef _SPTRSV_LEVELPARALLELISM_
#define _SPTRSV_LEVELPARALLELISM_

#include "common.h"
#include "tranpose.h"

int findlevel(const int *cscColPtr,
              const int *cscRowIdx,
              const int *csrRowPtr,
              const int m,
              int *nlevel,
              int *levelPtr,
              int *levelItem)
{
    // prepare arrays for level-sets of size maximum m or n
    int *indegree = (int *)malloc(m * sizeof(int));

    // prepare in-degree
    for (int i = 0; i < m; i++)
    {
        indegree[i] = csrRowPtr[i + 1] - csrRowPtr[i];
    }
    // for (int i = 0; i < m; i++)
    //     printf("%d ", indegree[i]);
    // for (int i = 0; i < m; i++)
    // printf("%d ", indegree[i]);
    // printf("\n");


    // find root items
    int lv = 0;
    int ptr = 0;

    levelPtr[0] = 0;

    for (int i = 0; i < m; i++)
    {
        if (indegree[i] == 1)
        {
            levelItem[ptr] = i;
            ptr++;
        }
    }

    // for (int i = 0; i < ptr; i++)
    // printf("%d ", levelItem[i]);
    // printf("\n");
    
    // #items in the 1st level
    levelPtr[1] = ptr;
    // printf("shoule lvptr = %d\n\n\n", levelPtr[1]);

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
                    levelItem[ptr] = visit_node;
                    ptr++;
                }
            }
        }
        lvi++;
        levelPtr[lvi] = ptr;
    }

    *nlevel = lvi;
        // printf("lvi = %d\n", lvi);
    // printf("levelPtr:\n");
    // for (int i = 0; i < m; i++)
    //     printf("%d ", levelPtr[i]);

    free(indegree);

    return 0;
}

int findlevel_csc(const int *cscColPtr,
                  const int *cscRowIdx,
                  const VALUE_TYPE *cscVal,
                  const int m,
                  const int n,
                  const int nnz,
                  int *nlevel,
                  int *parallelism_min,
                  int *parallelism_avg,
                  int *parallelism_max)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    // transpose to have csr data
    int *csrRowPtr = (int *)malloc((m + 1) * sizeof(int));

    // transpose from csc to csr
    matrix_transposition_litelite(m, n, nnz,
                                  cscColPtr, cscRowIdx, csrRowPtr);

    int *levelPtr = (int *)malloc((m + 1) * sizeof(int));
    int *levelItem = (int *)malloc(m * sizeof(int));

    int nlv = 0;
    findlevel(cscColPtr, cscRowIdx, csrRowPtr, m, &nlv, levelPtr, levelItem);

    // print col_size distribution
    int lenmax = cscColPtr[1] - cscColPtr[0];
    for (int i = 1; i < m; i++)
    {
        int len = cscColPtr[i + 1] - cscColPtr[i];
        if (len > lenmax)
            lenmax = len;
    }
    int *dist = (int *)malloc((lenmax + 1) * sizeof(int));
    for (int i = 0; i < lenmax + 1; i++)
        dist[i] = 0;

    for (int i = 0; i < m; i++)
    {
        int len = cscColPtr[i + 1] - cscColPtr[i];
        dist[len]++;
    }

    free(dist);

    // calculate min, avg, and max #item in levels
    int min = levelPtr[1] - levelPtr[0];
    int max = levelPtr[1] - levelPtr[0];
    for (int i = 1; i < nlv; i++)
    {
        int nitem = levelPtr[i + 1] - levelPtr[i];
        if (nitem > max)
            max = nitem;
        else if (nitem < min)
            min = nitem;
    }
    *parallelism_min = min;
    *parallelism_max = max;
    *parallelism_avg = m / nlv;

    *nlevel = nlv;

    free(levelPtr);
    free(levelItem);
    free(csrRowPtr);

    return 0;
}

int findlevel_csr(const int *csrRowPtr,
                  const int *csrColIdx,
                  const VALUE_TYPE *csrVal,
                  const int m,
                  const int n,
                  const int nnz,
                  int *nlevel,
                  int *parallelism_min,
                  int *parallelism_avg,
                  int *parallelism_max)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    // transpose to have csr data
    int *cscColPtr = (int *)malloc((n + 1) * sizeof(int));
    int *cscRowIdx = (int *)malloc(nnz * sizeof(int));
    VALUE_TYPE *cscVal = (VALUE_TYPE *)malloc(nnz * sizeof(VALUE_TYPE));

    // transpose from csc to csr
    matrix_transposition(m, n, nnz,
                         csrRowPtr, csrColIdx, csrVal,
                         cscRowIdx, cscColPtr, cscVal);

    int *levelPtr = (int *)malloc((m + 1) * sizeof(int));
    int *levelItem = (int *)malloc(m * sizeof(int));

    int nlv = 0;
    findlevel(cscColPtr, cscRowIdx, csrRowPtr, m, &nlv, levelPtr, levelItem);

    // calculate min, avg, and max #item in levels
    int min = levelPtr[1] - levelPtr[0];
    int max = levelPtr[1] - levelPtr[0];
    for (int i = 1; i < nlv; i++)
    {
        int nitem = levelPtr[i + 1] - levelPtr[i];
        if (nitem > max)
            max = nitem;
        else if (nitem < min)
            min = nitem;
    }
    *parallelism_min = min;
    *parallelism_max = max;
    *parallelism_avg = m / nlv;

    *nlevel = nlv;

    free(levelPtr);
    free(levelItem);
    free(cscColPtr);
    free(cscRowIdx);
    free(cscVal);

    return 0;
}

#endif
