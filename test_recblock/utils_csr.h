#ifndef _UTILS_CSR_
#define _UTILS_CSR_

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

template <typename T>
int read_tri(char *filename, int *m, int *nnz, 
        int **csrRowPtr, int **csrColIdx, T **csrVal)
{
    FILE *f;
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    fscanf(f, "%d%d\n", m, nnz);
    *csrRowPtr = (int*)malloc((*m + 1) * sizeof(int));
    *csrColIdx = (int*)malloc(*nnz * sizeof(int));
    *csrVal = (T*)malloc(*nnz * sizeof(T));

    for (int i = 0; i < *m; i++)
    {
        fscanf(f, "%d", *csrRowPtr + i);
    }
    (*csrRowPtr)[*m] = *nnz;
    for (int i = 0; i < *nnz; i++)
    {
        fscanf(f, "%d", *csrColIdx + i);
    }

    if (sizeof(T) == sizeof(float))
    {
        for (int i = 0; i < *nnz; i++)
        {
            fscanf(f, "%f", *csrVal + i);
        }
    }
    else
    {
        for (int i = 0; i < *nnz; i++)
        {
            fscanf(f, "%lf", *csrVal + i);
        }
    }
}

#endif
