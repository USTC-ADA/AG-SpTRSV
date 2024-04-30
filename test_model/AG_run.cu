#include "AG-SpTRSV.h"
#include "utils.h"
#include "YYSpTRSV.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <string>
#include <unistd.h>

#define VALUE_TYPE double
#define VALUE_SIZE 8

#define ERROR_THRESH 1e-4

#define REPEAT_TIME 10

#define duration(a, b) (1.0 * (b.tv_usec - a.tv_usec + (b.tv_sec - a.tv_sec) * 1.0e6))

int m;
int nnzL;
int *csrRowPtrL;
int *csrColIdxL;
VALUE_TYPE *csrValL;
VALUE_TYPE *x, *b;

PREPROCESSING_STRATEGY ps_in;
SCHEDULE_STRATEGY ss_in;
int rb_in;

ptr_handler handler;

ptr_anainfo ana;
anaparas paras;

extern void AG_read(char *input_name)
{
    cudaSetDevice(0);

    read_tri<VALUE_TYPE>(input_name, &m, &nnzL, &csrRowPtrL, &csrColIdxL, &csrValL);

    // x & randomized b
    x = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));
    b = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));
    srand(0);

    for (int i = 0; i < m; i++)
    {
        b[i] = rand() * 1.0 / RAND_MAX;
    }
}

extern void AG_matrix_info(float *args, int size)
{
    struct timeval tv_begin, tv_end;

    gettimeofday(&tv_begin, NULL);
    get_matrix_info2(m, nnzL, csrRowPtrL, csrColIdxL, args, size);
    gettimeofday(&tv_end, NULL);
}

extern void AG_matrix_info(int *m_in, int *nnz_in, float *avg_rnnz, float *cov_rnnz, float *avg_lnnz, float *cov_lnnz, float *dep_dist, float *reverse)
{
    *m_in = m;
    *nnz_in = nnzL;
    get_matrix_info(m, nnzL, csrRowPtrL, csrColIdxL, avg_rnnz, cov_rnnz, avg_lnnz, cov_lnnz, dep_dist, reverse);

}

extern void AG_matrix_output(char* matrix_info_name, char* matrix_name)
{
    write_matrix_info(matrix_info_name, matrix_name, m, nnzL, csrRowPtrL, csrColIdxL);

}

extern void AG_new_ana()
{
    ana = new anainfo(m);
}

extern void AG_preprocessing2(int *scheme)
{
    struct timeval tv_begin, tv_end;

    gettimeofday(&tv_begin, NULL);

    paras = anaparas(scheme[0], scheme[1], (PREPROCESSING_STRATEGY)scheme[2], 
    scheme[3], (LEVEL_PART_STRATEGY)scheme[4], (LEVEL_SCHED_STRATEGY)scheme[5], 
    (SCHEDULE_STRATEGY)scheme[6], (ROW_GROUP_SCHED_STRATEGY)scheme[7]);

    SpTRSV_preprocessing_new(m, nnzL, csrRowPtrL, csrColIdxL, ana, paras);

    gettimeofday(&tv_end, NULL);
    
}

extern void AG_preprocessing(int ps, int ss, int rb)
{

    ps_in = PREPROCESSING_STRATEGY(ps);
    ss_in = SCHEDULE_STRATEGY(ss);
    rb_in = rb;

    handler = SpTRSV_preprocessing(m, nnzL, csrRowPtrL, csrColIdxL, ps_in, rb_in);

    sptrsv_schedule(handler, ss_in);

}

extern void AG_matrix_reorder()
{
    int permutation[m];

    handler = SpTRSV_preprocessing(m, nnzL, csrRowPtrL, csrColIdxL,
            ROW_BLOCK, 1);
    
    graph_reorder_with_level(handler);

    matrix_reorder<VALUE_TYPE>(handler, permutation, csrRowPtrL, csrColIdxL, csrValL);

    graph_finalize(handler);

}

extern void AG_finalize()
{
    schedule_finalize(handler);

    graph_finalize(handler);

    delete handler;
}

extern void AG_execute(const char *matrix_name, const char *csv_name)
{

    struct timeval tv_begin, tv_end;

    int flag;
    cudaDeviceSynchronize();

    // copy matrix and vector from CPU to GPU memory
    int *csrRowPtr_d, *csrColIdx_d;
    VALUE_TYPE *csrValue_d, *b_d, *x_d;
    cudaMalloc(&csrRowPtr_d, (m + 1) * sizeof(int));
    cudaMemcpy(csrRowPtr_d, csrRowPtrL, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&csrColIdx_d, nnzL * sizeof(int));
    cudaMemcpy(csrColIdx_d, csrColIdxL, nnzL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&csrValue_d, nnzL * sizeof(VALUE_TYPE));
    cudaMemcpy(csrValue_d, csrValL, nnzL * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    cudaMalloc(&b_d, m * sizeof(VALUE_TYPE));
    cudaMemcpy(b_d, b, m * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    cudaMalloc(&x_d, m * sizeof(VALUE_TYPE));

    float sptrsv_time = 0;

    for (int i = 0; i < REPEAT_TIME; i++)
    {
        cudaMemset(x_d, 0, sizeof(int) * m);
        cudaMemset(ana->get_value, 0, sizeof(int) * m);

        cudaDeviceSynchronize();

        gettimeofday(&tv_begin, NULL);

        SpTRSV_executor_variant(ana, paras, csrRowPtr_d, csrColIdx_d, csrValue_d, b_d, x_d);
        cudaDeviceSynchronize();

        gettimeofday(&tv_end, NULL);

        sptrsv_time += duration(tv_begin, tv_end);
    }

    sptrsv_time /= REPEAT_TIME;

    printf("Running time %.2f us\n", sptrsv_time);

    cudaMemcpy(x, x_d, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    VALUE_TYPE *b_base;
    b_base = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));

    get_x_b(m, csrRowPtrL, csrColIdxL, csrValL, x, b_base);

    for (int i = 0; i < m; i++)
    {
        if (fabs(b[i] - b_base[i]) > ERROR_THRESH)
        {
            flag = 0;
            printf("Error at index %d, b = %.5f, b_base = %.5f!\n", i, b[i], b_base[i]);
            break;
        }
    }
    if (flag) printf("AG-SpTRSV correct!\n");

    if (csv_name != NULL)
    {
        FILE *fp = fopen(csv_name, "a");
        // out to csv
        if (fp) fprintf(fp, "%s,%d,%d,%d,%f\n", matrix_name, ps_in, ss_in, rb_in, sptrsv_time);
        fclose(fp);
    }

    cudaFree(csrRowPtr_d);
    cudaFree(csrColIdx_d);
    cudaFree(csrValue_d);
    cudaFree(b_d);
    cudaFree(x_d);

    cudaDeviceSynchronize();
}