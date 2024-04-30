#include "../include/AG-SpTRSV.h"
#include "core/SpTRSV_variant.h"
#include <cuda_runtime.h>

template <typename T, int subwarp_size>
void subwarp_variant(ptr_anainfo ana, anaparas paras, 
            const int *csrRowPtr_d, const int *csrColIdx_d, const T* csrValue_d,
            const T* b_d, T* x_d)
{
    int tb_size = paras.tb_size;

    int shared_size = BUF_SIZE * tb_size * sizeof(int) + 
    BUF_SIZE * tb_size * sizeof(T);

    if (ana->winfo_d)
    {
        cudaFuncSetAttribute(SpTRSV_code_subwarp<T, subwarp_size>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);
        cudaFuncSetAttribute(SpTRSV_code_subwarp_nosync<T, subwarp_size>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

        for (int i = 0; i < ana->winfo_n.size() - 1; i++)
        {
            int warp_num = ana->winfo_n[i + 1] - ana->winfo_n[i];

            int block_num = (warp_num * WARP_SIZE + tb_size - 1) / tb_size;
            //block_num = 80;
            if (ana->winfo_multilevel[i])
                SpTRSV_code_subwarp<T, subwarp_size><<<block_num, ALIGN_THREAD_PER_BLOCK, shared_size>>>(ana->winfo_n[i], ana->winfo_n[i + 1],
                (subwarpinfo<subwarp_size>*)ana->winfo_d, csrRowPtr_d, csrColIdx_d, csrValue_d, b_d, x_d, ana->get_value, tb_size, block_num);
            else
            {
                SpTRSV_code_subwarp_nosync<T, subwarp_size><<<block_num, ALIGN_THREAD_PER_BLOCK, shared_size>>>(ana->winfo_n[i], ana->winfo_n[i + 1],
                (subwarpinfo<subwarp_size>*)ana->winfo_d, csrRowPtr_d, csrColIdx_d, csrValue_d, b_d, x_d, ana->get_value, tb_size, block_num);
            }
            cudaDeviceSynchronize();
        }
    }
    else
    {
        cudaFuncSetAttribute(SpTRSV_code_subwarp_determ<T, subwarp_size>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

        //printf("???\n");

        int block_num = BLOCK_NUM;

        SpTRSV_code_subwarp_determ<T, subwarp_size><<<block_num, ALIGN_THREAD_PER_BLOCK, shared_size>>>(
        (subwarpinfo<subwarp_size>**)ana->winfo_d2, ana->winfo_n2, 
        csrRowPtr_d, csrColIdx_d, csrValue_d, b_d, x_d, ana->get_value, tb_size);
    }
}

// template void subwarp_variant<float, subwarpinfo<2>>(ptr_anainfo ana, anaparas paras, 
//             const int *csrRowPtr_d, const int *csrColIdx_d, const float* csrValue_d,
//             const float* b_d, float* x_d);

// template void subwarp_variant<double, subwarpinfo<2>>(ptr_anainfo ana, anaparas paras, 
//             const int *csrRowPtr_d, const int *csrColIdx_d, const double* csrValue_d,
//             const double* b_d, double* x_d);

template <typename T>
void SpTRSV_executor_variant(ptr_anainfo ana, anaparas paras,
            const int *csrRowPtr_d, const int *csrColIdx_d, const T* csrValue_d,
            const T* b_d, T* x_d)
{
    if (paras.subwarp_size == 1)
        subwarp_variant<T, 1>(ana, paras, csrRowPtr_d, csrColIdx_d, csrValue_d, b_d, x_d);
    if (paras.subwarp_size == 2)
        subwarp_variant<T, 2>(ana, paras, csrRowPtr_d, csrColIdx_d, csrValue_d, b_d, x_d);
    if (paras.subwarp_size == 4)
        subwarp_variant<T, 4>(ana, paras, csrRowPtr_d, csrColIdx_d, csrValue_d, b_d, x_d);
    if (paras.subwarp_size == 8)
        subwarp_variant<T, 8>(ana, paras, csrRowPtr_d, csrColIdx_d, csrValue_d, b_d, x_d);
    if (paras.subwarp_size == 16)
        subwarp_variant<T, 16>(ana, paras, csrRowPtr_d, csrColIdx_d, csrValue_d, b_d, x_d);
    if (paras.subwarp_size == 32)
        subwarp_variant<T, 32>(ana, paras, csrRowPtr_d, csrColIdx_d, csrValue_d, b_d, x_d);
}

// instance
template void SpTRSV_executor_variant<float>(ptr_anainfo ana, anaparas paras,
            const int *csrRowPtr, const int *csrColIdx, const float* csrValue,
            const float* b, float* x);

template void SpTRSV_executor_variant<double>(ptr_anainfo ana, anaparas paras,
            const int *csrRowPtr, const int *csrColIdx, const double* csrValue,
            const double* b, double* x);
