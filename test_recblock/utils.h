#ifndef _UTILS_
#define _UTILS_

#include "common.h"
#include "cusparse.h"

// print 1D array
template <typename T>
void print_1darray(T *input, int length)
{
    for (int i = 0; i < length; i++)
        printf("%i, ", input[i]);
    printf("\n");
}

__forceinline__ __device__
    VALUE_TYPE
    sum_32_shfl(VALUE_TYPE sum)
{
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
}

template <typename vT>
__forceinline__ __device__
    vT
    sum_32_shfl(vT sum)
{
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
}

void check_cusparse_kernel(cusparseStatus_t cudaerr)
{
    if (cudaerr != CUSPARSE_STATUS_SUCCESS)
        printf("cuda kernel fail, err = %s\n", cudaerr);
}

template <typename T>
void swap(T *a, T *b)
{
    T tmp = *a;
    *a = *b;
    *b = tmp;
}

int choose_pivot(int i, int j)
{
    return (i + j) / 2;
}

template <typename iT, typename vT>
void quicksort_keyval(iT *key, vT *val, int start, int end)
{
    iT pivot;
    int i, j, k;

    if (start < end)
    {
        k = choose_pivot(start, end);
        swap<iT>(&key[start], &key[k]);
        swap<vT>(&val[start], &val[k]);
        pivot = key[start];

        i = start + 1;
        j = end;
        while (i <= j)
        {
            while ((i <= end) && (key[i] <= pivot))
                i++;
            while ((j >= start) && (key[j] > pivot))
                j--;
            if (i < j)
            {
                swap<iT>(&key[i], &key[j]);
                swap<vT>(&val[i], &val[j]);
            }
        }

        // swap two elements
        swap<iT>(&key[start], &key[j]);
        swap<vT>(&val[start], &val[j]);

        // recursively sort the lesser key
        quicksort_keyval<iT, vT>(key, val, start, j - 1);
        quicksort_keyval<iT, vT>(key, val, j + 1, end);
    }
}

// in-place exclusive scan
template <typename T>
void exclusive_scan(T *input, int length)
{
    if (length == 0 || length == 1)
        return;

    T old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

// segmented sum
template <typename vT, typename bT>
void segmented_sum(vT *input, bT *bit_flag, int length)
{
    if (length == 0 || length == 1)
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
template <typename T>
T reduce_sum(T *input, int length)
{
    if (length == 0)
        return 0;

    T sum = 0;

    for (int i = 0; i < length; i++)
    {
        sum += input[i];
    }

    return sum;
}

#endif
