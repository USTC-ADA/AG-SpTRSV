#ifndef SPTRSV_TEMPLATE__
#define SPTRSV_TEMPLATE__

#include "../../include/AG-SpTRSV.h"

// Read flags
#define READ_FLAG (get_value[dep_row] == 1)
#define READ_FLAG_INNER_LEVEL (dep_row < row_st || get_value[dep_row] == 1)
#define NO_READ_FLAG true

// Read fence
#define READ_FENCE __threadfence()
#define NO_READ_FENCE

// Write flags

#define NO_WRITE_FLAG

#define WRITE_FLAG { \
    __threadfence(); \
    get_value[row] = 1; \
}

#define WRITE_FLAG_NOFENCE { \
    get_value[row] = 1; \
}

#endif