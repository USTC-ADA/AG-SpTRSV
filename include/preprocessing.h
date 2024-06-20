#ifndef PREPROCESSING__
#define PREPROCESSING__

#include "common.h"
#include <cuda.h>

ptr_handler SpTRSV_preprocessing(const int m,
                const int nnz,
                const int *csrRowPtr,
                const int *csrColIdx,
                PREPROCESSING_STRATEGY strategy,
                int row_block);

void graph_reorder_with_level(ptr_handler handler);

// ptr_graph generate_graph(const int m,
//                 const int nnz,
//                 const int *csrRowPtr,
//                 const int *csrColIdx);

void write_graph(const char* file_name, ptr_handler handler, unsigned int max_depthm,
int layer, float parallelism);

void show_graph_layer(ptr_handler handler);

void get_matrix_info(const int    m,
                const int         nnz,
                const int        *csrRowPtr,
                const int        *csrColIdx,
                float            *avg_rnnz,
                float            *cov_rnnz,
                float            *avg_lnnz,
                float            *cov_lnnz,
                float            *dep_dist,
                float            *reverse_level);

void write_matrix_info(const char* file_name,
                const char*       matrix_name,
                const int         m,
                const int         nnz,
                const int        *csrRowPtr,
                const int        *csrColIdx);

// TACO: for learning model
void get_matrix_info2(const int    m,
                const int         nnz,
                const int        *csrRowPtr,
                const int        *csrColIdx,
                float *out_args,
                int size);

void write_matrix_info2(const char* file_name,
                const char*       matrix_name,
                const int         m,
                const int         nnz,
                const int        *csrRowPtr,
                const int        *csrColIdx);

void SpTRSV_preprocessing_new(int m, int nnz, int *csrRowPtr, int *csrColIdx, ptr_anainfo ana, anaparas paras);
int get_matrix_level(const int m, const int nnz, const int *csrRowPtr, const int *csrColIdx, ptr_anainfo info);
void get_matrix_partition(int m, int nnz, int *csrRowPtr, int *csrColIdx, ptr_anainfo ana, anaparas paras);
void get_matrix_schedule(int m, int nnz, int *csrRowPtr, int *csrColIdx, ptr_anainfo ana, anaparas paras);

#endif