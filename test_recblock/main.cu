#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "common.h"
#include "mmio.h"
#include "utils_csr.h"
#include "mmio_highlevel.h"
#include "recblocking_solver.h"
#include "recblocking_solver_cuda.h"

// "Usage: ``./sptrsv-double -d 0 -rhs 1 -lv -1 -forward/-backward -mtx A.mtx'' for Ax=b on device 0"
int main(int argc,  char ** argv)
{
    // report precision of floating-point
    printf("---------------------------------------------------------------------------------------------\n");
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char *)"32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char *)"64-bit Double Precision";
    }
    else
    {
        printf("Wrong precision. Program exit!\n");
        return 0;
    }

    printf("PRECISION = %s\n", precision);
    printf("Benchmark REPEAT = %i\n", BENCH_REPEAT);
    printf("---------------------------------------------------------------------------------------------\n");

    int m, n, nnzA, isSymmetricA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

    int nnzTR;
    int *csrRowPtrTR;
    int *csrColIdxTR;
    VALUE_TYPE *csrValTR;

    int device_id = 0;
    int rhs = 0;
    int lv = 0;
    int substitution;

    int argi = 1;

    // load device id
    char *devstr;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }

    if (strcmp(devstr, "-d") != 0) return 0;

    if(argc > argi)
    {
        device_id = atoi(argv[argi]);
        argi++;
    }
    printf("device_id = %i\n", device_id);

    // load the number of right-hand-side
    char *rhsstr;
    if(argc > argi)
    {
        rhsstr = argv[argi];
        argi++;
    }

    if (strcmp(rhsstr, "-rhs") != 0) return 0;

    if(argc > argi)
    {
        rhs = atoi(argv[argi]);
        argi++;
    }
    printf("rhs = %i\n", rhs);

    // load the number of recursive levels
    char *lvstr;
    if(argc > argi)
    {
        lvstr = argv[argi];
        argi++;
    }

    if (strcmp(lvstr, "-lv") != 0) return 0;

    if(argc > argi)
    {
        lv = atoi(argv[argi]);
        argi++;
    }

    // load substitution, forward or backward
    char *substitutionstr;
    if(argc > argi)
    {
        substitutionstr = argv[argi];
        argi++;
    }

    if (strcmp(substitutionstr, "-forward") == 0)
        substitution = SUBSTITUTION_FORWARD;
    else if (strcmp(substitutionstr, "-backward") == 0)
        substitution = SUBSTITUTION_BACKWARD;
    printf("substitutionstr = %s\n", substitutionstr);
    printf("substitution = %i\n", substitution);

    // load matrix file type, mtx, cscl, or cscu
    char *matstr;
    if(argc > argi)
    {
        matstr = argv[argi];
        argi++;
    }
    printf("matstr = %s\n", matstr);

    // load matrix data from file
    char *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }

    char *out_enable = NULL;
    if(argc > argi)
    {
        out_enable = argv[argi];
        argi++;
    }
    
    char *out_csv = NULL;
    // write performance data to csv file
    if (out_enable != NULL && strcmp(out_enable, "-csv") == 0)
    {
        if(argc > argi)
        {
            out_csv = argv[argi];
            argi++;
        }
        else
        {
            printf("Wrong csv_name\n");
            return 1;
        }
    }

    printf("-------------- %s --------------\n", filename);

    srand(time(NULL));

    // load mtx data to the csr format
    read_tri<VALUE_TYPE>(filename, &m, &nnzTR, &csrRowPtrTR, &csrColIdxTR, &csrValTR);
    n = m;
    
    // perm b and y (Ly=b and Ux=y)
    VALUE_TYPE *x_ref  =  (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n * rhs);
    VALUE_TYPE *x  =  (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n * rhs);
    for (int i = 0; i < n * rhs; i++)
    {
        x_ref[i] = rand() % 10 + 1;
    }

    VALUE_TYPE *b  =  (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m * rhs);
    memset(b, 0, sizeof(VALUE_TYPE) * m * rhs);
    for (int r = 0; r < rhs; r++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = csrRowPtrTR[i]; j < csrRowPtrTR[i + 1]; j++)
            {
                b[r * m + i] += csrValTR[j] * x_ref[r * n + csrColIdxTR[j]];
            }
        }
    }

    // transpose CSR of U and L to CSC
    int *cscColPtrTR = (int *)malloc(sizeof(int) * (n + 1));
    cscColPtrTR[0] = 0;
    int *cscRowIdxTR = (int *)malloc(sizeof(int) * nnzTR);
    VALUE_TYPE *cscValTR = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * nnzTR);
    matrix_transposition(m, n, nnzTR,
                             csrRowPtrTR, csrColIdxTR, csrValTR,
                             cscRowIdxTR, cscColPtrTR, cscValTR);


    free(csrRowPtrTR);
    free(csrColIdxTR);
    free(csrValTR);
        
    if (lv == -1)
    {
        int li = 1;
        for (li = 1; li <= 100; li++)
        {
            if (m / pow(2, (li+1)) < (device_id == 0 ? 92160 : 58880)) // 92160 (4608x20) is titan rtx, 58880 (2944x20) is rtx 2080
                break;
        }
        lv = li;
    }
    
    int *d_cscColPtrTR;
    int *d_cscRowIdxTR;
    VALUE_TYPE *d_cscValTR;
    cudaMalloc((void **)&d_cscColPtrTR, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_cscRowIdxTR, nnzTR * sizeof(int));
    cudaMalloc((void **)&d_cscValTR, nnzTR * sizeof(VALUE_TYPE));
    
    cudaMemcpy(d_cscColPtrTR, cscColPtrTR, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIdxTR, cscRowIdxTR, sizeof(int) * nnzTR, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscValTR, cscValTR, sizeof(VALUE_TYPE) * nnzTR, cudaMemcpyHostToDevice);


    VALUE_TYPE *d_x;
    VALUE_TYPE *d_b;
    cudaMalloc((void **)&d_x, m * sizeof(VALUE_TYPE));
    cudaMalloc((void **)&d_b, m * sizeof(VALUE_TYPE));
    
    cudaMemcpy(d_x, x, sizeof(VALUE_TYPE) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(VALUE_TYPE) * m, cudaMemcpyHostToDevice);
    
    double cal_time = 0;
    double preprocess_time = 0;
    recblocking_solver_cuda(d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
                            m, n, nnzTR, d_x, d_b, substitution, lv, &cal_time, &preprocess_time);
    cudaMemcpy(x, d_x, sizeof(VALUE_TYPE) * m, cudaMemcpyDeviceToHost);
    
    printf("computation usetime = %.3lf ms\n", cal_time);
    printf("preprocessing time  = %.3lf ms\n", preprocess_time);
    printf("Performance = %.3lf gflops\n", (2 * nnzTR) / (cal_time * 1e6));

    cudaFree(d_cscColPtrTR);
    cudaFree(d_cscRowIdxTR);
    cudaFree(d_cscValTR);
    cudaFree(d_b);
    cudaFree(d_x);
    free(cscColPtrTR);
    free(cscRowIdxTR);
    free(cscValTR);
    free(b);
    
    
    // validate x
    int flag = 0;
    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n * rhs ; i++)
    {
        ref += abs(x_ref[i]);
        res += abs(x[i] - x_ref[i]);
    }
    printf("\n");

    res = ref == 0 ? res : res / ref;
    
    if (res < accuracy && (res >= 0))
    {
        printf("x check passed! |x-xref|/|xref| = %8.2e\n", res);
        flag = 1;
    }
    else
        printf(" x check _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

    free(x);
    free(x_ref);

    if (out_csv != NULL)
    {
        // Write to batch log
        int table_head = 0;
        if (access(out_csv, F_OK)) table_head = 1;

        FILE *fp_out;
        fp_out = fopen(out_csv, "a");

        if (table_head) fprintf(fp_out, "matrix,rec_prep,rec_exec\n");

        fprintf(fp_out, "%s,%.2f,%.2f\n", filename, cal_time * 1000, preprocess_time * 1000);
    }
    
    return 0;
}