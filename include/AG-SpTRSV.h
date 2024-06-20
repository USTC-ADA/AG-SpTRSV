#ifndef SPTRSV__
#define SPTRSV__

#include "GPU_setup.h"
#include "preprocessing.h"
#include "schedule.h"
#include "common.h"
#include "specialization.h"
#include "finalize.h"
#include "format_def.h"
#include "transformation.h"
#include "elimination.h"
#include "subwarp.h"
#include "search.h"

template <typename T>
void SpTRSV_executor(ptr_handler handler, 
            const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
            const T* b, T* x);

// template <typename T>
// void SpTRSV_executor(ptr_handler handler, int *get_value,
//             const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
//             const T* b, T* x);

template <typename T>
void SpTRSV_executor_hybrid(ptr_handler handler, 
            const int *RowPtr_d, const int *ValPtr_d,
            const int *idx_d, const T* value_d,
            const T* b_d, T* x_d);

template <typename T>
void SpTRSV_executor_variant(ptr_handler handler, 
            const int *csrRowPtr_d, const int *csrColIdx_d, const T* csrValue_d,
            const T* b_d, T* x_d);

template <typename T>
void SpTRSV_executor_variant(ptr_anainfo ana, anaparas paras,
            const int *csrRowPtr_d, const int *csrColIdx_d, const T* csrValue_d,
            const T* b_d, T* x_d);

#endif