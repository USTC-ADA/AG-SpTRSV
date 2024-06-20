#ifndef _SPTS_SYNCFREE_CUDA_
#define _SPTS_SYNCFREE_CUDA_

#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef BENCH_REPEAT
#define BENCH_REPEAT 10
#endif

// #ifndef WARP_SIZE
// #define WARP_SIZE   32
// #endif

// #ifndef WARP_PER_BLOCK
// #define WARP_PER_BLOCK 32
// #endif

#define duration(a, b) (1.0 * (b.tv_usec - a.tv_usec + (b.tv_sec - a.tv_sec) * 1.0e6))

// print 1D array
void print_1darray(int *input, int length)
{
    for (int i = 0; i < length; i++)
        printf(", %i", input[i]);
    printf("\n");
}


// in-place exclusive scan
void exclusive_scan(int *input, int length)
{
    if(length == 0 || length == 1)
        return;

    int old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}

/*
__forceinline__ __device__
static double atomicAdd(double *addr, double val)
{
    double old = *addr, assumed;
    do
    {
        assumed = old;
        old = __longlong_as_double(
                    atomicCAS((unsigned long long int*)addr,
                              __double_as_longlong(assumed),
                              __double_as_longlong(val+assumed)));

    }while(assumed != old);

    return old;
}
*/

template<typename vT>
__forceinline__ __device__
vT sum_32_shfl(vT sum)
{
#pragma unroll
    for(int mask = WARP_SIZE / 2 ; mask > 0 ; mask >>= 1)
        sum += __shfl_xor(sum, mask);
    
    return sum;
}


/*
// segmented sum
template<typename vT, typename bT>
void segmented_sum(vT *input, bT *bit_flag, int length)
{
    if(length == 0 || length == 1)
        return;

    for (int i = 0; i < length; i++)
    {
        if (bit_flag[i])
        {
            int j = i + 1;
            while (!bit_flag[j] && j < length)
            {
                input[i] += input[j];
                j++;
            }
        }
    }
}

// reduce sum
template<typename T>
T reduce_sum(T *input, int length)
{
    if(length == 0)
        return 0;

    T sum = 0;

    for (int i = 0; i < length; i++)
    {
        sum += input[i];
    }

    return sum;
}*/

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

__global__
void spts_syncfree_cuda_analyser(const int   *d_cscRowIdx,
                                 const int    m,
                                 const int    nnz,
                                       int   *d_csrRowHisto)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (global_id < nnz)
    {
        atomicAdd(&d_csrRowHisto[d_cscRowIdx[global_id]], 1);
    }
}

__global__
void spts_syncfree_cuda_executor_pre(const int   *d_csrRowPtrL,
                                     const int    m,
                                           int   *d_csrRowHisto)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (global_id < m)
    {
        d_csrRowHisto[global_id] = d_csrRowPtrL[global_id+1] - d_csrRowPtrL[global_id];
    }
}

void matrix_transposition(const int         m,
                          const int         n,
                          const int         nnz,
                          const int        *csrRowPtr,
                          const int        *csrColIdx,
                          const VALUE_TYPE *csrVal,
                                int        *cscRowIdx,
                                int        *cscColPtr,
                                VALUE_TYPE *cscVal)
{
    // histogram in column pointer
    memset (cscColPtr, 0, sizeof(int) * (n+1));
    for (int i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);

    int *cscColIncr = (int *)malloc(sizeof(int) * (n+1));
    memcpy (cscColIncr, cscColPtr, sizeof(int) * (n+1));

    // insert nnz to csc
    for (int row = 0; row < m; row++)
    {
        for (int j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
    }

    free (cscColIncr);
}

__global__
void spts_syncfree_cuda_executor(const int* __restrict__        d_cscColPtr,
                                 const int* __restrict__        d_cscRowIdx,
                                 const VALUE_TYPE* __restrict__ d_cscVal,
                                 const int* __restrict__        d_csrRowPtr,
                                 int*                           d_csrRowHisto,
                                 VALUE_TYPE*                    d_left_sum,
                                 VALUE_TYPE*                    d_partial_sum,
                                 const int                      m,
                                 const int                      nnz,
                                 const VALUE_TYPE* __restrict__ d_b,
                                 VALUE_TYPE*                    d_x,
                                 int*                           d_while_profiler)

{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_x_id = global_id / WARP_SIZE;
    volatile __shared__ int s_csrRowHisto[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK];

    if (global_x_id >= m) return;
    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int starting_x = (global_id / (WARP_PER_BLOCK * WARP_SIZE)) * WARP_PER_BLOCK;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    // Prefetch
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[d_cscColPtr[global_x_id]];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    if (threadIdx.x < WARP_PER_BLOCK) { s_csrRowHisto[threadIdx.x] = 1; s_left_sum[threadIdx.x] = 0; }
    __syncthreads();

    //clock_t start;
    // Consumer
    do {
        //start = clock();
        __threadfence();
    }
    while (s_csrRowHisto[local_warp_id] != d_csrRowHisto[global_x_id]);
  
    //// Consumer
    //int graphInDegree;
    //do {
    //    //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
    //    asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_csrRowHisto[global_x_id]) :: "memory"); 
    //}
    //while (s_csrRowHisto[local_warp_id] != graphInDegree );

    VALUE_TYPE xi = d_left_sum[global_x_id] + s_left_sum[local_warp_id]; 
    xi = (d_b[global_x_id] - xi) * coef;

    // Producer
    for (int j = d_cscColPtr[global_x_id] + 1 + lane_id; j < d_cscColPtr[global_x_id+1]; j += WARP_SIZE) {   
        int rowIdx = d_cscRowIdx[j];
        if (rowIdx < starting_x + WARP_PER_BLOCK) {
            atomicAdd((VALUE_TYPE *)&s_left_sum[rowIdx - starting_x], xi * d_cscVal[j]);
            atomicAdd((int *)&s_csrRowHisto[rowIdx - starting_x], 1);
        }
        else {
            atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
            atomicSub(&d_csrRowHisto[rowIdx], 1);
        }
    }
    // Finish
    if (!lane_id) d_x[global_x_id] = xi;
}

float spts_syncfree_cuda_test(const int           *csrRowPtrL_host,
                          const int           *csrColIdxL_host,
                          const VALUE_TYPE    *csrValL_host,
                          VALUE_TYPE          *d_x,
                          const VALUE_TYPE    *d_b,
                          const int            m,
                          const int            n,
                          const int            nnzL,
                          float                &sf_time,
                          float                &sfprep,
                          const int            test_time = BENCH_REPEAT)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    // VALUE_TYPE *x_ref = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);
    // for ( int i = 0; i < n; i++)
    //     x_ref[i] = 1;

    // VALUE_TYPE *b = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m);

    // for (int i = 0; i < m; i++)
    // {
    //     b[i] = 0;
    //     for (int j = csrRowPtrL_tmp[i]; j < csrRowPtrL_tmp[i+1]; j++)
    //         b[i] += csrValL_tmp[j] * x_ref[csrColIdxL_tmp[j]];
    // }

    //VALUE_TYPE *x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);

    // transpose from csr to csc first
    int *cscRowIdxL = (int *)malloc(nnzL * sizeof(int));
    int *cscColPtrL = (int *)malloc((n+1) * sizeof(int));
    memset(cscColPtrL, 0, (n+1) * sizeof(int));
    VALUE_TYPE *cscValL    = (VALUE_TYPE *)malloc(nnzL * sizeof(VALUE_TYPE));

    matrix_transposition(m, n, nnzL,
                         csrRowPtrL_host, csrColIdxL_host, csrValL_host,
                         cscRowIdxL, cscColPtrL, cscValL);

    // transfer host mem to device mem
    int *d_cscColPtrL;
    int *d_cscRowIdxL;
    VALUE_TYPE *d_cscValL;

    // Matrix L
    cudaMalloc((void **)&d_cscColPtrL, (n+1) * sizeof(int));
    cudaMalloc((void **)&d_cscRowIdxL, nnzL  * sizeof(int));
    cudaMalloc((void **)&d_cscValL,    nnzL  * sizeof(VALUE_TYPE));

    cudaMemcpy(d_cscColPtrL, cscColPtrL, (n+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIdxL, cscRowIdxL, nnzL  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscValL,    cscValL,    nnzL  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice);

    sfprep = 0;

    //  - cuda syncfree SpTS analysis start!
    printf(" - cuda syncfree SpTS analysis start!\n");

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    // malloc tmp memory to simulate atomic operations
    int *d_csrRowHisto;
    cudaMalloc((void **)&d_csrRowHisto, sizeof(int) * (m+1));

    // generate row pointer by partial transposition
    int *d_csrRowPtrL;
    cudaMalloc((void **)&d_csrRowPtrL, (m+1) * sizeof(int));
    thrust::device_ptr<int> d_csrRowPtrL_thrust = thrust::device_pointer_cast(d_csrRowPtrL);
    thrust::device_ptr<int> d_csrRowHisto_thrust = thrust::device_pointer_cast(d_csrRowHisto);

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *d_left_sum;
    cudaMalloc((void **)&d_left_sum, sizeof(VALUE_TYPE) * m);

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *d_partial_sum;
    cudaMalloc((void **)&d_partial_sum, sizeof(VALUE_TYPE) * nnzL);
    //cudaMemset(d_partial_sum, 0, sizeof(VALUE_TYPE) * nnzL);

    int num_threads = 256;
    int num_blocks = ceil ((double)nnzL / (double)num_threads);

    for (int i = 0; i < test_time; i++)
    {
        cudaMemset(d_csrRowHisto, 0, (m+1) * sizeof(int));
        spts_syncfree_cuda_analyser<<< num_blocks, num_threads >>>
                                      (d_cscRowIdxL, m, nnzL, d_csrRowHisto);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);
    double time_cuda_analysis = duration(t1, t2);
    time_cuda_analysis /= test_time;

    thrust::exclusive_scan(d_csrRowHisto_thrust, d_csrRowHisto_thrust + m+1, d_csrRowPtrL_thrust);

    printf("cuda syncfree SpTS analysis on L used %4.2f us\n", time_cuda_analysis);
    sfprep = time_cuda_analysis;

    // validate csrRowPtrL
    int *csrRowPtrL = (int *)malloc((m+1) * sizeof(int));
    cudaMemcpy(csrRowPtrL, d_csrRowPtrL, (m+1) * sizeof(int), cudaMemcpyDeviceToHost);

    // int err_counter = 0;
    // for (int i = 0; i <= m; i++)
    // {
    //     //printf("[%i]: csrRowPtrL = %i, csrRowPtrL_tmp = %i\n", i, csrRowPtrL[i], csrRowPtrL_tmp[i]);
    //     if (csrRowPtrL[i] != csrRowPtrL_host[i])
    //         err_counter++;
    // }

    // free(csrRowPtrL);

    // if (err_counter)
    //     printf("cuda syncfree SpTS analyser on L failed!\n");

    //  - cuda syncfree SpTS solve start!
    //printf(" - cuda syncfree SpTS solve start!\n");

    int *d_while_profiler;
    cudaMalloc((void **)&d_while_profiler, sizeof(int) * n);
    cudaMemset(d_while_profiler, 0, sizeof(int) * n);
    int *while_profiler = (int *)malloc(sizeof(int) * n);

    // step 5: solve L*y = x
    sf_time = 0;

    for (int i = 0; i < test_time; i++)
    {
        num_threads = 1024;
        num_blocks = ceil ((double)m / (double)(num_threads));
        spts_syncfree_cuda_executor_pre<<< num_blocks, num_threads >>>
                                          (d_csrRowPtrL, m, d_csrRowHisto);
    

        cudaMemset(d_left_sum, 0, sizeof(VALUE_TYPE) * m);

        num_threads = WARP_PER_BLOCK * WARP_SIZE;
        num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));

        gettimeofday(&t1, NULL);
        spts_syncfree_cuda_executor<<< num_blocks, num_threads >>>
                                   (d_cscColPtrL, d_cscRowIdxL, d_cscValL, 
                                    d_csrRowPtrL, d_csrRowHisto, 
                                    d_left_sum, d_partial_sum,
                                    m, nnzL, d_b, d_x, d_while_profiler);
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        sf_time += duration(t1, t2);

    }
    cudaDeviceSynchronize();

    sf_time /= test_time;

    printf("cuda syncfree SpTS solve used %4.2f us, throughput is %4.2f gflops\n",
           sf_time, 2*nnzL/(1e3*sf_time));

    //cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    // validate x
    // err_counter = 0;
    // for (int i = 0; i < n; i++)
    // {
    //     if (abs(x_ref[i] - x[i]) > 0.01 * abs(x_ref[i]))
    //     {
    //         printf("x_ref %.4f x %.4f\n", x_ref[i], x[i]);
    //         err_counter++;
    //     }
    // }

    // if (err_counter)
    //     printf("cuda syncfree SpTS on L failed!\n");

    cudaMemcpy(while_profiler, d_while_profiler, n * sizeof(int), cudaMemcpyDeviceToHost);
    long long unsigned int while_count = 0;
    for (int i = 0; i < n; i++)
    {
        while_count += while_profiler[i];
        //printf("while_profiler[%i] = %i\n", i, while_profiler[i]);
    }
    //printf("\nwhile_count= %llu in total, %llu per row/column\n", while_count, while_count/m);

    // step 6: free resources
    free(while_profiler);

    cudaFree(d_csrRowHisto);
    cudaFree(d_left_sum);
    cudaFree(d_partial_sum);
    cudaFree(d_csrRowPtrL);
    cudaFree(d_while_profiler);

    cudaFree(d_cscColPtrL);
    cudaFree(d_cscRowIdxL);
    cudaFree(d_cscValL);
    // cudaFree(d_b);
    // cudaFree(d_x);

    return sf_time;
}

// #undef WARP_SIZE
// #undef WARP_PER_BLOCK
// #undef VALUE_TYPE
// #undef duration(a, b)

#endif



