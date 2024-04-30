#ifndef READ_MTX_H__
#define READ_MTX_H__

// read .mtx file
int read_mtx(char *filename, int* m_add, int *n_add,
        int *nnzA_add, int **csrRowPtrA_add, int **csrColIdxA_add, float **csrValA_add);

// read triangular matrix file in CSR format
template <typename T>
int read_tri(char *filename, int *m, int *nnz, 
        int **csrRowPtr, int **csrColIdx, T **csrVal);

// write triangular to file in CSR format
int write_tri(char *filename, int m, int nnz,
        int *csrRowPtr, int *csrColIdx, float *csrVal);

// change from original matrix to triangular matrix
void change2trian(int m, int nnzA, int *csrRowPtrA, int *csrColIdxA, 
        float *csrValA, int *nnzL_add, int **csrRowPtrL_tmp_add, 
        int **csrColIdxL_tmp_add, float **csrValL_tmp_add);

// change from csr format to csc format
template <typename T>
void csr2csc(int m, int nnz, int *csrRowPtr, int *csrColIdx, T *csrVal,
        int **cscColPtr, int **cscColIdx, T **cscVal);

// calculate b = Lx
void get_x_b(int m, const int * csrRowPtrA, const int *csrColIdxA, 
        const float *csrValA, const float *x_add, float *b_add);
void get_x_b(int m, const int * csrRowPtrA, const int *csrColIdxA, 
        const double *csrValA, const double *x_add, double *b_add);

#endif