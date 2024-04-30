#include "utils.h"
#include "mmio.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

using namespace std;

template<typename T>
int read_mtx_t(char *filename, int* m_add, int *n_add,
        int *nnzA_add, int **csrRowPtrA_add, int **csrColIdxA_add, T **csrValA_add)
{
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    T *csrValA;
    // read matrix from mtx file
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    
    //printf("00000\n");
    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;
    
    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;
    
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }
    
    if ( mm_is_complex( matcode ) )
    {
        printf("Sorry, data type 'COMPLEX' is not supported.\n");
        return -3;
    }
    
    //printf("111111\n");
    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }
    //printf("222222\n");
    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    //printf("3333333,%d\n",ret_code);
    
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }
    
    int *csrRowPtrA_counter = (int *)malloc((m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));
    
    int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    T *csrValA_tmp = (T*)malloc(nnzA_mtx_report * sizeof(T));
    
    if(csrRowIdxA_tmp==NULL || csrColIdxA_tmp==NULL || csrValA_tmp==NULL)
        return -2;
    
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    
    //printf("222222\n");
    int i;
    for (i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval;
        int ival;
        int returnvalue;
        
        if (isReal)
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }
        
        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        
        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }
    
    if (f != stdin)
        fclose(f);
    
    if (isSymmetric)
    {
        for (i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }
    
    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;
    
    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i-1];
        old_val = new_val;
    }
    
    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));
    
    csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    csrValA    = (T *)malloc(nnzA * sizeof(T));
    
    if (isSymmetric)
    {
        for ( i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
                
                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for (i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }
    
    // free tmp space
    free(csrColIdxA_tmp);
    free(csrValA_tmp);
    free(csrRowIdxA_tmp);
    free(csrRowPtrA_counter);
    
    //printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);
    *m_add=m;
    *n_add=n;
    *nnzA_add=nnzA;
    *csrColIdxA_add=csrColIdxA;
    *csrValA_add=csrValA;
    *csrRowPtrA_add=csrRowPtrA;
    
    return 0;
}

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

int write_tri(char *filename, int m, int nnz,
        int *csrRowPtr, int *csrColIdx, float *csrVal)
{
    FILE *f;
    if ((f = fopen(filename, "w")) == NULL)
        return -1;

    fprintf(f, "%d %d\n", m, nnz);

    for (int i = 0; i < m; i++)
    {
        fprintf(f, "%d ", csrRowPtr[i]);
    }
    fprintf(f, "\n");
    for (int i = 0; i < nnz; i++)
    {
        fprintf(f, "%d ", csrColIdx[i]);
    }
    fprintf(f, "\n");
    for (int i = 0; i < nnz; i++)
    {
        fprintf(f, "%f ", csrVal[i]);
    }
    fprintf(f, "\n");

    return 0;
}

template <typename T>
void change2trian_t(int m, int nnzA,int *csrRowPtrA, int *csrColIdxA, 
        T *csrValA, int *nnzL_add, int **csrRowPtrL_tmp_add, 
        int **csrColIdxL_tmp_add, T **csrValL_tmp_add)
{
    int nnzL = 0;
    int *csrRowPtrL_tmp = (int *)malloc((m+1) * sizeof(int));
    int *csrColIdxL_tmp = (int *)malloc(nnzA * sizeof(int));
    T *csrValL_tmp    = (T *)malloc(nnzA * sizeof(T));
    
    int i,j,k;
    int tmp_col;
    T tmp_value;
    
    int nnz_pointer = 0;
    csrRowPtrL_tmp[0] = 0;

    srand((unsigned int)time(NULL));
    
    for (i = 0; i < m; i++)
    {
        for (j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
        {
            tmp_col=csrColIdxA[j];
            tmp_value=csrValA[j];
            for(k=j+1;k<csrRowPtrA[i+1];k++)
            {
                if(csrColIdxA[k]<tmp_col)
                {
                    csrColIdxA[j]=csrColIdxA[k];
                    csrValA[j]=csrValA[k];
                    csrColIdxA[k]=tmp_col;
                    csrValA[k]=tmp_value;
                    tmp_col=csrColIdxA[j];
                    tmp_value=csrValA[j];
                }
            }
            
            if (csrColIdxA[j] < i)
            {
                csrColIdxL_tmp[nnz_pointer] = csrColIdxA[j];
                csrValL_tmp[nnz_pointer] = 1.0 * (double)rand() / (double)RAND_MAX;
                nnz_pointer++;
            }
            else
            {
                break;
            }
        }
        
        csrColIdxL_tmp[nnz_pointer] = i;
        csrRowPtrL_tmp[i+1] = nnz_pointer + 1;
        // printf("??%.2f\n", )
        csrValL_tmp[nnz_pointer] = (1.0 + 1.0 * (double)rand() / (double)RAND_MAX) * 
        (csrRowPtrL_tmp[i+1] - csrRowPtrL_tmp[i]);
        // printf("row %d len %d %.2f %.2f\n", i, csrRowPtrL_tmp[i+1] - csrRowPtrL_tmp[i], csrValL_tmp[nnz_pointer],
        // (1.0 + 1.0 * (double)rand() / (double)RAND_MAX) * (csrRowPtrL_tmp[i+1] - csrRowPtrL_tmp[i]));
        nnz_pointer++;

    }
    
    nnzL = csrRowPtrL_tmp[m];
    
    csrColIdxL_tmp = (int *)realloc(csrColIdxL_tmp, sizeof(int) * nnzL);
    csrValL_tmp = (T *)realloc(csrValL_tmp, sizeof(T) * nnzL);
    
    *nnzL_add=nnzL;
    *csrRowPtrL_tmp_add=csrRowPtrL_tmp;
    *csrColIdxL_tmp_add=csrColIdxL_tmp;
    *csrValL_tmp_add=csrValL_tmp;
}

template <typename T>
void csr2csc(int m, int nnz, int *csrRowPtr, int *csrColIdx, T *csrVal,
        int **cscColPtr, int **cscRowIdx, T **cscVal)
{
    *cscColPtr = (int*)malloc((m + 1) * sizeof(int));
    *cscRowIdx = (int*)malloc(nnz * sizeof(int));
    *cscVal = (T*)malloc(nnz * sizeof(T));

    vector<int> Row_buf[m];
    vector<T> Val_buf[m];

    for (int row = 0; row < m; row++)
    {
        for (int idx = csrRowPtr[row]; idx < csrRowPtr[row + 1]; idx++)
        {
            int col = csrColIdx[idx];
            T val = csrVal[idx];
            Row_buf[col].push_back(row);
            Val_buf[col].push_back(val);
        }
    }

    int csc_count = 0;
    (*cscColPtr)[0] = 0;
    for (int col = 0; col < m; col++)
    {
        (*cscColPtr)[col + 1] = (*cscColPtr)[col] + Row_buf[col].size();
        for (int row = 0; row < Row_buf[col].size(); row++)
        {
            (*cscRowIdx)[csc_count] = Row_buf[col][row];
            (*cscVal)[csc_count] = Val_buf[col][row];
            csc_count++;
        }
    }
}

template <typename T>
void get_x_b_t(int m, const int * csrRowPtrA, const int *csrColIdxA, 
        const T *csrValA, const T *x_add, T *b_add)
{
    // run spmv to get b
    for (int i = 0; i < m; i++)
    {
        b_add[i] = 0;
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
            b_add[i] += csrValA[j] * x_add[csrColIdxA[j]];
    }
}

//instance
int read_mtx(char *filename, int* m_add, int *n_add,
        int *nnzA_add, int **csrRowPtrA_add, int **csrColIdxA_add, float **csrValA_add)
{
    return read_mtx_t<float>(filename, m_add, n_add, 
            nnzA_add, csrRowPtrA_add, csrColIdxA_add, csrValA_add);
}

void change2trian(int m, int nnzA, int *csrRowPtrA, int *csrColIdxA, 
        float *csrValA, int *nnzL_add, int **csrRowPtrL_tmp_add, 
        int **csrColIdxL_tmp_add, float **csrValL_tmp_add)
{
    change2trian_t<float>(m, nnzA, csrRowPtrA, csrColIdxA, 
            csrValA, nnzL_add, csrRowPtrL_tmp_add, 
            csrColIdxL_tmp_add, csrValL_tmp_add);
}

void get_x_b(int m, const int * csrRowPtrA, const int *csrColIdxA, 
        const float *csrValA, const float *x_add, float *b_add)
{
    get_x_b_t<float>(m, csrRowPtrA, csrColIdxA, 
            csrValA, x_add, b_add);
}

void get_x_b(int m, const int * csrRowPtrA, const int *csrColIdxA, 
        const double *csrValA, const double *x_add, double *b_add)
{
    get_x_b_t<double>(m, csrRowPtrA, csrColIdxA, 
            csrValA, x_add, b_add);
}

template int read_tri<float>(char *filename, int *m, int *nnz, 
        int **csrRowPtr, int **csrColIdx, float **csrVal);
template int read_tri<double>(char *filename, int *m, int *nnz, 
        int **csrRowPtr, int **csrColIdx, double **csrVal);

template void csr2csc<float>(int m, int nnz, int *csrRowPtr, int *csrColIdx, float *csrVal,
        int **cscColPtr, int **cscRowIdx, float **cscVal);
template void csr2csc<double>(int m, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal,
        int **cscColPtr, int **cscRowIdx, double **cscVal);
