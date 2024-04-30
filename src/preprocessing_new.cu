#include <stdlib.h>
#include <string>
#include <queue>
#include <algorithm>
#include <math.h>
#include <unistd.h>
#include <vector>
#include <sys/time.h>
#include "../include/preprocessing.h"
#include "../include/common.h"

extern void level_partition(ptr_anainfo ana, anaparas paras);
extern void row_partition(int m, int nnz, int *RowPtr, int *ColIdx, ptr_anainfo ana, anaparas paras);
//extern void heuristic_schedule(ptr_anainfo ana, anaparas paras);

extern void row_group_schedule(ptr_anainfo ana, anaparas paras);
extern void warp_schedule(ptr_anainfo ana, anaparas paras, int *csrRowPtr);
extern void level_schedule(ptr_anainfo ana, anaparas paras);

int get_matrix_level(const int m, const int nnz, const int *csrRowPtr, const int *csrColIdx, ptr_anainfo info)
{
    info->level_num = 0;

    info->row_level = (int*)malloc(m * sizeof(int));
    if (info->row_level == NULL)
    {
        printf("row_level malloc error!\n");
        return -1;
    }
    memset(info->row_level, 0, sizeof(int) * m);

    info->level_rownum = (int*)malloc(m * sizeof(int));
    if (info->level_rownum == NULL)
    {
        printf("level_rownum malloc error!\n");
        return -1;
    }
    memset(info->level_rownum, 0, sizeof(int) * m);

    // count layer
    for (int row = 0; row < m; row++)
    {
        int max_level = 0;
        // Assume orderd CSR format
        for (int j = csrRowPtr[row]; j < csrRowPtr[row+1] - 1; j++)
        {
            int col = csrColIdx[j];
            if (info->row_level[col] + 1 > max_level)
                max_level = info->row_level[col] + 1;
        }

        if (info->max_row_nnz < csrRowPtr[row+1] - csrRowPtr[row])
            info->max_row_nnz = csrRowPtr[row+1] - csrRowPtr[row];

        info->row_level[row] = max_level;
        // level is calculated in row partition
        //info->row_level[row] = 0;

        info->level_rownum[max_level]++;
        if (max_level + 1 > info->level_num)
            info->level_num = max_level + 1;
    }

    int max = 0;
    unsigned int min = -1;

    // printf("layer_num: ");
    // for(int j = 0; j < info->level_num; j++)
    // {
    //     if(max < info->level_rownum[j])
    //         max = info->level_rownum[j];
    //     if(min > info->level_rownum[j])
    //         min = info->level_rownum[j];
    //     printf("%d ", info->level_rownum[j]);
    // }
    // printf("\n");

    info->avg_parallelism = (double) m / info->level_num;
    info->max_parallelism = max;

    info->level_num = 0;
    memset(info->row_level, 0, sizeof(int) * m);

    //cudaMalloc(&info->get_value, m * sizeof(int));

    return info->level_num;
}

void get_matrix_partition(int m, int nnz, int *csrRowPtr, int *csrColIdx, ptr_anainfo ana, anaparas paras)
{
    // Partition
    // level_partition(ana, paras);
    // printf("row partition begin\n");
    row_partition(m, nnz, csrRowPtr, csrColIdx, ana, paras);
    level_partition(ana, paras);
}

void get_matrix_schedule(int m, int nnz, int *csrRowPtr, int *csrColIdx, ptr_anainfo ana, anaparas paras)
{
    // Schedule
    row_group_schedule(ana, paras);
    warp_schedule(ana, paras, csrRowPtr);
    level_schedule(ana, paras);
}

#define TIME_COUNT false
#define duration(a, b) (1.0 * (b.tv_usec - a.tv_usec + (b.tv_sec - a.tv_sec) * 1.0e6))

// When subwarp_size > 1, LEVEL_WISE schedule must be used
void SpTRSV_preprocessing_new(int m, int nnz, int *csrRowPtr, int *csrColIdx, ptr_anainfo ana, anaparas paras)
{
    //ptr_anainfo ana = new anainfo();

#if (TIME_COUNT == true)
    struct timeval tv_begin, tv_end;
    gettimeofday(&tv_begin, NULL);
#endif

    // Get level
    get_matrix_level(m, nnz, csrRowPtr, csrColIdx, ana);

#if (TIME_COUNT == true)
    gettimeofday(&tv_end, NULL);
    printf("Get matrix level time: %.2f us\n", duration(tv_begin, tv_end));
    gettimeofday(&tv_begin, NULL);
#endif

    // Partition
    get_matrix_partition(m, nnz, csrRowPtr, csrColIdx, ana, paras);
    //printf("level_num %d level_partition %d\n", ana->level_num, ana->partition_levels);

#if (TIME_COUNT == true)
    gettimeofday(&tv_end, NULL);
    printf("Get matrix partition time: %.2f us\n", duration(tv_begin, tv_end));
    gettimeofday(&tv_begin, NULL);
#endif

    // Schedule
    get_matrix_schedule(m, nnz, csrRowPtr, csrColIdx, ana, paras);

#if (TIME_COUNT == true)
    gettimeofday(&tv_end, NULL);
    printf("Get matrix schedule time: %.2f us\n", duration(tv_begin, tv_end));
#endif
    
    //return ana;
}

#undef TIME_COUNT