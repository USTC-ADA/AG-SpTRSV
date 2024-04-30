#include "../include/common.h"

void level_wise_partition(ptr_anainfo ana)
{
    ana->partition_levels = ana->level_num;
    
    for (int i = 0; i < ana->level_num; i++)
    {
        ana->level_partition_map[i] = i;
    }
}

void row_wise_partition(ptr_anainfo ana)
{
    ana->partition_levels = 1;
    
    for (int i = 0; i < ana->level_num; i++)
    {
        ana->level_partition_map[i] = 0;
    }
}

void level_partition(ptr_anainfo ana, anaparas paras)
{
    ana->level_partition_map = (int*)malloc(ana->level_num * sizeof(int));

    if (paras.level_ps == LEVEL_WISE)
    {
        level_wise_partition(ana);
    }
    else if (paras.level_ps == ROW_WISE)
    {
        row_wise_partition(ana);
    }
    else
    {
        printf("Not implemented!\n");
    }
}
