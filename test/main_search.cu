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

#define REPEAT_TIME 1
#define WARM_UP 0

#define PRINT_LOG true

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
        }
    }
    return maxi;
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    struct timeval tv_begin, tv_end;
    struct timeval prep_begin, prep_end;

    int ch;

    int input_flag = 0, graph_flag = 0, outcsv_flag = 0, full_flag = 0, prep_flag = 0, reorder_flag = 0;
    char *input_name, *graph_name, *outcsv_name, *full_name, *prep_name;

    while ((ch = getopt(argc, argv, "o:i:f:p:r")) != -1)
    {
        switch (ch)
        {
            case 'o':
                outcsv_flag = 1;
                outcsv_name = optarg;
                break;

            case 'i':
                input_flag = 1;
                input_name = optarg;
                break;

            case 'f':
                full_flag = 1;
                full_name = optarg;
                break;
            
            case 'p':
                prep_flag = 1;
                prep_name = optarg;
                break;
            
            case 'r':
                reorder_flag = 1;
        }
    }

    if (input_flag == 0)
    {
        printf("[Usage]: ./main_batch -i {input_filename}\n");
        exit(1);
    }

    // Original matrix A;
    int m;

    // Triangular matrix L;
    int nnzL;
    int *csrRowPtrL;
    int *csrColIdxL;
    VALUE_TYPE *csrValL;

    read_tri<VALUE_TYPE>(input_name, &m, &nnzL, &csrRowPtrL, &csrColIdxL, &csrValL);

    if (reorder_flag)
    {
        printf("Matrix reordering...\n");

        ptr_handler handler = SpTRSV_preprocessing(m, nnzL, csrRowPtrL, csrColIdxL,
        ROW_BLOCK, 1);

        graph_reorder_with_level(handler);

        int permutation[m];

        matrix_reorder(handler, permutation, csrRowPtrL, csrColIdxL, csrValL);

        graph_finalize(handler);

        printf("Reordering done!\n");
    }

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

    int flag;
    float sptrsv_time = -1;
    float agprep = 0.0, yyprep = 0.0, cuprep = 0.0;

    flag = 1;

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

    VALUE_TYPE *b_base;
    b_base = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));

    FILE *full_out = NULL;
    if (full_flag)
    {
        int full_head = 0;
        if (access(full_name, F_OK)) full_head = 1;

        full_out = fopen(full_name, "a");

        if (full_head) fprintf(full_out, "matrix,tbs,sws,rp,alpha,lp,rg,ws,ls,time\n");
    }

    int maxi;
    VALUE_TYPE error_max;

    printf("Search begin\n");

    anaspace space;
    anaparas paras;
    anaparas best_paras;
    ptr_anainfo ana = new anainfo(m);

    gettimeofday(&prep_begin, NULL);

    get_matrix_level(m, nnzL, csrRowPtrL, csrColIdxL, ana);

    gettimeofday(&prep_end, NULL);
    agprep += duration(prep_begin, prep_end);

    int flag1 = 0;
    int flag2 = 0;

    while (flag1 == 0)
    {
        gettimeofday(&prep_begin, NULL);

        space.get_next_partition(paras);

        get_matrix_partition(m, nnzL, csrRowPtrL, csrColIdxL, ana, paras);

        gettimeofday(&prep_end, NULL);
        agprep += duration(prep_begin, prep_end);

        while (flag2 == 0)
        {
            gettimeofday(&prep_begin, NULL);

            space.get_next_schedule(paras);

            get_matrix_schedule(m, nnzL, csrRowPtrL, csrColIdxL, ana, paras);

            gettimeofday(&prep_end, NULL);

#if (PRINT_LOG == true)
            printf("----------\n");
            show_paras(paras);
#endif

            agprep += duration(prep_begin, prep_end);
            
            float tmp_sptrsv_time = 0;
            memset(x, 0, sizeof(VALUE_TYPE) * m);

            for (int i = 0; i < REPEAT_TIME; i++)
            {
                cudaMemset(ana->get_value, 0, sizeof(int) * m);
                cudaMemset(x_d, 0, sizeof(VALUE_TYPE) * m);

                cudaDeviceSynchronize();

                gettimeofday(&tv_begin, NULL);
                
                SpTRSV_executor_variant(ana, paras, csrRowPtr_d, csrColIdx_d, csrValue_d, b_d, x_d);
                cudaDeviceSynchronize();

                gettimeofday(&tv_end, NULL);

                if (i >= WARM_UP) tmp_sptrsv_time += duration(tv_begin, tv_end);
            }

            tmp_sptrsv_time /= (REPEAT_TIME - WARM_UP);

#if (PRINT_LOG == true)
            printf("Solve time: %.2f us\n", tmp_sptrsv_time);
#endif

            cudaMemcpy(x, x_d, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

            get_x_b(m, csrRowPtrL, csrColIdxL, csrValL, x, b_base);

            if (sptrsv_time == -1 || tmp_sptrsv_time < sptrsv_time)
            {
                best_paras = paras;
                sptrsv_time = tmp_sptrsv_time;
            }

#if (PRINT_LOG == true)
            printf("Solve time: %.2f us\n", tmp_sptrsv_time);
            printf("Current best: ");
            show_paras(best_paras);
            printf("Best time: %.2f us\n", sptrsv_time);
            printf("\n");
#endif

            if (full_flag) print_paras(full_out, input_name, paras, tmp_sptrsv_time);

            schedule_finalize(ana, paras);

            flag2 = space.schedule_incr();
        }

        partition_finalize(ana);

        flag1 = space.partition_incr();
        flag2 = 0;
    }

    matrix_level_finalize(ana);

    #define G (1024 * 1024 * 1024)
    #define M (1024 * 1024)
    #define get_gflops(x) (gflops / (x) * M)
    #define get_gmems(x) (gmems / (x) * M)

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

        fprintf(fp_out, "%.2f,%.2f,%.2f\n", sptrsv_time, get_gflops(sptrsv_time), get_gmems(sptrsv_time));
    }

    printf("Best run time: %.4f\n", sptrsv_time);
    printf("Best paras:\n");
    show_paras(best_paras);
    printf("gflops: %.4f Gflops: %.4f \ngmems:  %.4f Bwidth: %.4f\n", gflops, gflops / sptrsv_time * M, gmems, gmems / sptrsv_time * M);

    #undef G
    #undef M

    // Finalize
    cudaFree(csrRowPtr_d);
    cudaFree(csrColIdx_d);
    cudaFree(csrValue_d);
    cudaFree(x_d);
    cudaFree(b_d);

}
