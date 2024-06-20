#ifndef FINALIZE__
#define FINALIZE__

#include "common.h"

void graph_finalize(ptr_handler handler);

void schedule_finalize(ptr_handler handler);

void SpTRSV_finalize(ptr_handler handler);

void partition_finalize(ptr_anainfo ana);

void schedule_finalize(ptr_anainfo ana, anaparas paras);

void matrix_level_finalize(ptr_anainfo ana);

void SpTRSV_finalize(ptr_anainfo ana, anaparas paras);

#endif