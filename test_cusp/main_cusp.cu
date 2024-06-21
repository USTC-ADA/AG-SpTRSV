#include "AG-SpTRSV.h"
#include "utils.h"
#include "YYSpTRSV.h"
#include "spts_syncfree_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <unistd.h>

#define VALUE_TYPE double
#define VALUE_SIZE 8

#define ERROR_THRESH 1e-4

#define REPEAT_TIME 10
#define WARM_UP 1

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
    cudaSetDevice(0);

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

    int m;
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

    VALUE_TYPE *x_base = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));
    VALUE_TYPE *b_base = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));

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

    // cuSparse
    gettimeofday(&tv_begin, NULL);

    cusparseHandle_t cusparse_handler;
    cusparseStatus_t ErrorStatus;
    ErrorStatus = cusparseCreate(&cusparse_handler);

    cusparseMatDescr_t desc;
    ErrorStatus = cusparseCreateMatDescr(&desc);
    cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(desc, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(desc, CUSPARSE_DIAG_TYPE_NON_UNIT);

    bsrsv2Info_t cusparse_info;
    cusparseCreateBsrsv2Info(&cusparse_info);

    int buffer_size;
#if (VALUE_SIZE == 4)
    ErrorStatus = cusparseSbsrsv2_bufferSize(cusparse_handler,
    CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
    desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info, &buffer_size);
#else
    ErrorStatus = cusparseDbsrsv2_bufferSize(cusparse_handler,
    CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
    desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info, &buffer_size);
#endif

    if (ErrorStatus != CUSPARSE_STATUS_SUCCESS)
    {
        printf("Error in buffersize stage!\n");
        exit(1);
    }

    VALUE_TYPE alpha = 1.0;

    VALUE_TYPE *x_cusparse_d;
    cudaMalloc(&x_cusparse_d, m * sizeof(VALUE_TYPE));

    void *cusparse_buffer;
    cudaMalloc((void **)&cusparse_buffer, buffer_size);

#if (VALUE_SIZE == 4)
    ErrorStatus = cusparseSbsrsv2_analysis(cusparse_handler,
    CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
    desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info, 
    CUSPARSE_SOLVE_POLICY_USE_LEVEL, cusparse_buffer);
#else
    ErrorStatus = cusparseDbsrsv2_analysis(cusparse_handler,
    CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
    desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info, 
    CUSPARSE_SOLVE_POLICY_USE_LEVEL, cusparse_buffer);
#endif
    if (ErrorStatus != CUSPARSE_STATUS_SUCCESS)
    {
        printf("Error in analysis stage!\n");
        printf("%s\n", cusparseGetErrorString(ErrorStatus));
        exit(1);
    }
    int structural_zero;
    ErrorStatus = cusparseXbsrsv2_zeroPivot(cusparse_handler, cusparse_info, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == ErrorStatus)
        printf("L(%d,%d) is missing\n", structural_zero, structural_zero);

    gettimeofday(&tv_end, NULL);

    printf("cuSPARSE preprocessing time: %.2f us\n", duration(tv_begin, tv_end));

    float cusparse_time_use_level = 0;
    float cusparse_time_no_level = 0;

    for (int i = 0; i < REPEAT_TIME; i++)
    {
        gettimeofday(&tv_begin, NULL);

#if (VALUE_SIZE == 4)
        ErrorStatus = cusparseSbsrsv2_solve(cusparse_handler,
        CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
        &alpha, desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info,
        b_d, x_cusparse_d, CUSPARSE_SOLVE_POLICY_USE_LEVEL, cusparse_buffer);
#else
        ErrorStatus = cusparseDbsrsv2_solve(cusparse_handler,
        CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
        &alpha, desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info,
        b_d, x_cusparse_d, CUSPARSE_SOLVE_POLICY_USE_LEVEL, cusparse_buffer);
#endif
        cudaDeviceSynchronize();

        gettimeofday(&tv_end, NULL);
        
        if (i >= WARM_UP) cusparse_time_use_level += duration(tv_begin, tv_end);

        if (ErrorStatus != CUSPARSE_STATUS_SUCCESS)
        {
            printf("Error in solve stage!\n");
            exit(1);
        }
    }

    cusparse_time_use_level /= (REPEAT_TIME - WARM_UP);

    cudaMemcpy(x, x_cusparse_d, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    get_x_b(m, csrRowPtrL, csrColIdxL, csrValL, x, b_base);

    int maxi = error_detect(b, b_base, m);
    VALUE_TYPE error_max = fabs(b[maxi] - b_base[maxi]);
    if (error_max >= ERROR_THRESH)
        printf("Backward max error at index %d, b = %.8f, b_base = %.8f!\n", maxi, b[maxi], b_base[maxi]);
    else
        printf("cusparse no level correct!\n");

#if (VALUE_SIZE == 4)
    ErrorStatus = cusparseSbsrsv2_analysis(cusparse_handler,
    CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
    desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info, 
    CUSPARSE_SOLVE_POLICY_USE_LEVEL, cusparse_buffer);
#else
    ErrorStatus = cusparseDbsrsv2_analysis(cusparse_handler,
    CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
    desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info, 
    CUSPARSE_SOLVE_POLICY_USE_LEVEL, cusparse_buffer);
#endif
    if (ErrorStatus != CUSPARSE_STATUS_SUCCESS)
    {
        printf("Error in analysis stage!\n");
        printf("%s\n", cusparseGetErrorString(ErrorStatus));
        exit(1);
    }

    ErrorStatus = cusparseXbsrsv2_zeroPivot(cusparse_handler, cusparse_info, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == ErrorStatus)
        printf("L(%d,%d) is missing\n", structural_zero, structural_zero);

    for (int i = 0; i < REPEAT_TIME; i++)
    {
        gettimeofday(&tv_begin, NULL);

#if (VALUE_SIZE == 4)
        ErrorStatus = cusparseSbsrsv2_solve(cusparse_handler,
        CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
        &alpha, desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info,
        b_d, x_cusparse_d, CUSPARSE_SOLVE_POLICY_NO_LEVEL, cusparse_buffer);
#else
        ErrorStatus = cusparseDbsrsv2_solve(cusparse_handler,
        CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
        &alpha, desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info,
        b_d, x_cusparse_d, CUSPARSE_SOLVE_POLICY_NO_LEVEL, cusparse_buffer);
#endif
        cudaDeviceSynchronize();

        gettimeofday(&tv_end, NULL);
        
        if (i >= WARM_UP) cusparse_time_no_level += duration(tv_begin, tv_end);

        if (ErrorStatus != CUSPARSE_STATUS_SUCCESS)
        {
            printf("Error in solve stage!\n");
            exit(1);
        }
    }

    cudaMemcpy(x, x_cusparse_d, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    get_x_b(m, csrRowPtrL, csrColIdxL, csrValL, x, b_base);

    maxi = error_detect(b, b_base, m);
    error_max = fabs(b[maxi] - b_base[maxi]);
    if (error_max >= ERROR_THRESH)
        printf("Backward max error at index %d, b = %.8f, b_base = %.8f!\n", maxi, b[maxi], b_base[maxi]);
    else
        printf("cusparse use level correct!\n");

    cusparse_time_no_level /= (REPEAT_TIME - WARM_UP);

    float cusparse_time = cusparse_time_use_level;
    if (cusparse_time_no_level < cusparse_time) cusparse_time = cusparse_time_no_level;
    printf("Cusparse use level %.2f us, no level %.2f us\n", cusparse_time_use_level, cusparse_time_no_level);
    printf("Cusparse solve time: %.2f us\n", cusparse_time);

    // L has unit diagonal, so no numerical zero is reported.
    int numerical_zero;
    ErrorStatus = cusparseXbsrsv2_zeroPivot(cusparse_handler, cusparse_info, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == ErrorStatus){
        printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    cudaFree(cusparse_buffer);
    cudaFree(x_cusparse_d);

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

        if (table_head) fprintf(fp_out, 
        "cusparse time(us),cusparse gflops,cusparse memory\n");
        fprintf(fp_out, "%s,%d,%d,%d,%.2f,", input_name, m, nnzL, layer, parallelism);

        fprintf(fp_out, "%.2f,%.2f,%.2f,", cusparse_time,
        gflops / cusparse_time * M, gmems / cusparse_time * M);

    }

    printf("gflops: %.4f Gflops: %.4f \ngmems:  %.4f Bwidth: %.4f\n", gflops, gflops / cusparse_time * M, gmems, gmems / cusparse_time * M);

    #undef G
    #undef M

    // Finalize
    cudaFree(csrRowPtr_d);
    cudaFree(csrColIdx_d);
    cudaFree(csrValue_d);
    cudaFree(x_d);
    cudaFree(b_d);

}
