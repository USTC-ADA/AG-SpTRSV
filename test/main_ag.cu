#include "AG-SpTRSV.h"
#include "utils.h"
#include "YYSpTRSV.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <unistd.h>

#define VALUE_TYPE double
#define VALUE_SIZE 8

#define ERROR_THRESH 1e-4

// #define REPEAT_TIME 100
// #define WARM_UP 10

#define REPEAT_TIME 1
#define WARM_UP 0

#define CU_TEST false

#define duration(a, b) (1.0 * (b.tv_usec - a.tv_usec + (b.tv_sec - a.tv_sec) * 1.0e6))

int error_detect(VALUE_TYPE *x, VALUE_TYPE *x_base, int m)
{
    VALUE_TYPE max_error = 0;
    int maxi = -1;
    for (int i = 0; i < m; i++)
    {
        if (fabs(x[i] - x_base[i]) > max_error)
        {
            max_error = fabs(x[i] - x_base[i]);
            maxi = i;
            return maxi;
        }
    }
    return maxi;
}

int main(int argc, char* argv[])
{
    cudaSetDevice(1);

    struct timeval tv_begin, tv_end;

    int ch;

    int input_flag = 0, graph_flag = 0, outcsv_flag = 0;
    char *input_name, *graph_name, *outcsv_name;

    while ((ch = getopt(argc, argv, "g:o:i:")) != -1)
    {
        switch (ch)
        {
            case 'g':
                graph_flag = 1;
                graph_name = optarg;
                break;
            
            case 'o':
                outcsv_flag = 1;
                outcsv_name = optarg;
                break;

            case 'i':
                input_flag = 1;
                input_name = optarg;
                break;
        }
    }

    if (input_flag == 0)
    {
        printf("[Usage]: ./main_batch -i {input_filename}\n");
        exit(1);
    }

    // Original matrix A;
    int m;
    // int n;
    // int nnzA;
    // int *csrRowPtrA;
    // int *csrColIdxA;
    // VALUE_TYPE *csrValA;

    //read_mtx(argv[1], &m, &n, &nnzA, &csrRowPtrA, &csrColIdxA, &csrValA);

    // Triangular matrix L;
    int nnzL;
    int *csrRowPtrL;
    int *csrColIdxL;
    VALUE_TYPE *csrValL;

    read_tri<VALUE_TYPE>(input_name, &m, &nnzL, &csrRowPtrL, &csrColIdxL, &csrValL);

    int layer;
    double parallelism;
    int max_row_nnz;
    matrix_layer2<VALUE_TYPE>(m, m, nnzL, csrRowPtrL, csrColIdxL, &layer, &parallelism, &max_row_nnz);

    // x & randomized b
    VALUE_TYPE *x, *b;
    x = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));
    b = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));
    srand(0);
    for (int i = 0; i < m; i++)
    {
        b[i] = rand() * 1.0 / RAND_MAX;
    }

    printf("matrix information: location %s\n"
        "m %d nnz %d layer %d parallelism %.2f max_row_nnz %d\n", 
        input_name, m, nnzL, layer, parallelism, max_row_nnz);

    gettimeofday(&tv_begin, NULL);

    PREPROCESSING_STRATEGY ps = ROW_BLOCK;
    SCHEDULE_STRATEGY strategy = SIMPLE;
    int rb = 1;

    int graph_reorder = 0;

    ptr_handler handler;

    if (graph_reorder)
    {
        printf("Begin reordering\n");

        handler = SpTRSV_preprocessing(m, nnzL, csrRowPtrL, csrColIdxL,
        ROW_BLOCK, 1);

        graph_reorder_with_level(handler);

        int permutation[m];

        matrix_reorder(handler, permutation, csrRowPtrL, csrColIdxL, csrValL);

        graph_finalize(handler);
    }

    int flag;
    float sptrsv_time = 0;

    flag = 1;

    // matrix examples
    // atmosmodd
    //anaparas paras = anaparas(1024, 4, ROW_BLOCK, 1, LEVEL_WISE, ONE_LEVEL, SIMPLE, RG_SIMPLE);

    // Wiki-Talk
    //anaparas paras = anaparas(64, 1, ROW_BLOCK, 1, LEVEL_WISE, ONE_LEVEL, WORKLOAD_BALANCE, RG_SIMPLE);

    // cant
    anaparas paras = anaparas(64, 2, ROW_BLOCK, 1, LEVEL_WISE, ONE_LEVEL, SIMPLE2, RG_SIMPLE);

    // delaunay_n23
    //anaparas paras = anaparas(1024, 4, ROW_BLOCK_AVG, 2, LEVEL_WISE, THRESH_LEVEL, SIMPLE, RG_SIMPLE);

    // delaunay_n13
    //anaparas paras = anaparas(64, 4, ROW_BLOCK_AVG, 1, LEVEL_WISE, THRESH_LEVEL, WORKLOAD_BALANCE, RG_SIMPLE);

    // webbase-1M
    // anaparas paras = anaparas(1024, 1, ROW_BLOCK_THRESH, 32, LEVEL_WISE, ONE_LEVEL, SIMPLE, RG_SIMPLE);

    ptr_anainfo ana = new anainfo(m);
    SpTRSV_preprocessing_new(m, nnzL, csrRowPtrL, csrColIdxL, ana, paras);

    gettimeofday(&tv_end, NULL);

    printf("Preprocessing time: %.2f us\n", duration(tv_begin, tv_end));

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
    cudaMemset(x_d, 0, sizeof(VALUE_TYPE) * m);

    for (int i = 0; i < REPEAT_TIME; i++)
    {
        cudaMemset(ana->get_value, 0, sizeof(int) * m);
        cudaMemset(x_d, 0, sizeof(VALUE_TYPE) * m);

        cudaDeviceSynchronize();

        gettimeofday(&tv_begin, NULL);
        
        SpTRSV_executor_variant(ana, paras, csrRowPtr_d, csrColIdx_d, csrValue_d, b_d, x_d);
        cudaDeviceSynchronize();

        gettimeofday(&tv_end, NULL);

        if (i >= WARM_UP) sptrsv_time += duration(tv_begin, tv_end);
    }

    sptrsv_time /= (REPEAT_TIME - WARM_UP);

    cudaMemcpy(x, x_d, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    VALUE_TYPE *b_base;
    b_base = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));

    get_x_b(m, csrRowPtrL, csrColIdxL, csrValL, x, b_base);

    printf("Solve time: %.2f us\n", sptrsv_time);

    int maxi = error_detect(b, b_base, m);
    VALUE_TYPE error_max = fabs(b[maxi] - b_base[maxi]);
    if (error_max >= ERROR_THRESH)
        printf("Backward max error at index %d, b = %.8f, b_base = %.8f!\n", maxi, b[maxi], b_base[maxi]);
    else
        printf("AG-SpTRSV correct!\n");

    VALUE_TYPE *x_base;
    x_base = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));

    #define G (1024 * 1024 * 1024)
    #define M (1024 * 1024)

    float gflops = 1.0 * (2 * nnzL + m) / G;
    // csrValue + ColIdx + x + b + RowPtr
    float gmems = 1.0 * (nnzL * (sizeof(int) + sizeof(VALUE_TYPE)) + 
    2 * m * sizeof(VALUE_TYPE) + m * sizeof(int)) / G;

    if (outcsv_flag)
    {
        // Write to batch log
        int table_head = 0;
        if (access(outcsv_name, F_OK)) table_head = 1;

        FILE *fp_out;
        fp_out = fopen(outcsv_name, "a");

        if (table_head) fprintf(fp_out, "matrix,m,nnz,layer,parallelism,"
        "sptrsv time(us),sptrsv gflops,sptrsv memory\n");
        fprintf(fp_out, "%s,%d,%d,%d,%.2f,", input_name, m, nnzL, layer, parallelism);

        fprintf(fp_out, "%.2f,%.2f,%.2f\n", sptrsv_time, gflops / sptrsv_time * M, gmems / sptrsv_time * M);
    }

    printf("Gflops: %.4f \nBwidth: %.4f\n", gflops / sptrsv_time * M, gmems / sptrsv_time * M);

    #undef G
    #undef M

    // Finalize
    cudaFree(csrRowPtr_d);
    cudaFree(csrColIdx_d);
    cudaFree(csrValue_d);
    cudaFree(x_d);
    cudaFree(b_d);

}
