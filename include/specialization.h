#ifndef SPECIALIZATION_
#define SPECIALIZATION_

#include "common.h"
#include "GPU_setup.h"
#include <string.h>
#include <stdio.h>
#include <typeinfo>

//SpTRSV(int *level, node_info **info, 
//       const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
//       const T* b, T* x, int *get_value)

#define SPE_CSR_IDX "csrColIdx"
#define SPE_CSR_VALUE "csrValue"
#define SPE_X_VALUE "x"
#define SPE_B_VALUE "b"
#define SPE_GET_VALUE "get_value"

#define SPE_VTYPE typeid(T)

template <typename T>
void SpTRSV_specialization(ptr_handler handler, const char* filename, 
        int *csrRowPtr, int *csrColIdx);

#endif