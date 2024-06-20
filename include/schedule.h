#ifndef SCHEDULE__
#define SCHEDULE__

#include "common.h"
#include "GPU_setup.h"
#include "elimination.h"
#include <stdlib.h>
#include <queue>
#include <vector>
#include <cuda.h>
#include <algorithm>

#define SHOW_USER_IMBALANCE true

void sptrsv_schedule(ptr_handler handler, SCHEDULE_STRATEGY strategy);

void sptrsv_schedule_new(ptr_anainfo ana, SCHEDULE_STRATEGY strategy);

// void simple_schedule(ptr_handler handler);
// void simple_schedule_workload_balance(ptr_handler handler);
// void simple_schedule_warp_locality(ptr_handler handler);
// void simple_schedule_workload_locality(ptr_handler handler);

#endif