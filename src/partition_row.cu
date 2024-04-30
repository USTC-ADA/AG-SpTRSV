#include "../include/common.h"

inline int get_winfo(int row_st, int row_ed, int *RowPtr, int *ColIdx, ptr_anainfo ana, anaparas paras, warpinfo &tmp)
{
    int info = 0;
    int level_max = 0;
    int row_max = 0;
    int multi_level = 0;
    //int last_level = ana->row_level[row_st];
    int same_level = 1;
    for (int row = row_st; row < row_ed; row++)
    {
        // Assume diagnoal is stored
        for (int idx = RowPtr[row]; idx < RowPtr[row + 1] - 1; idx++)
        {
            int col = ColIdx[idx];
            if (col >= row_st)
            {
                same_level = 0;
                multi_level = 1;
                continue;
            }
            if (ana->row_level[col] + 1 > level_max)
            {
                level_max = ana->row_level[col] + 1;
            }
        }
        // if (last_level != ana->row_level[row]) same_level = 0;
        if (ana->row_level[row] > level_max)
        {
            if (level_max != -1) multi_level = 1;
            level_max = ana->row_level[row];
        }
        if (RowPtr[row + 1] - RowPtr[row] > row_max)
            row_max = RowPtr[row + 1] - RowPtr[row];
    }

    for (int row = row_st; row < row_ed; row++)
    {
        ana->row_level[row] = level_max;
    }

    // if there are multi levels in the row group, we must use separative process
    if (multi_level) info = info | 0x1;
    // if sufficient rows, use separative process
    else
    {
        if (row_ed - row_st >= 8) info = info | 0x1;
        if (row_ed - row_st < 8) info = info & 0xfffffffe;
        multi_level = 1;
    }

    if (RowPtr[row_ed] - RowPtr[row_st] > BUF_SIZE * WARP_SIZE / paras.subwarp_size)
        info = info | 0x2;

    if (same_level) info = info | 0x100;

    tmp.copy(row_st, row_ed, info, RowPtr[row_ed] - RowPtr[row_st]);

    return level_max;

}

#define row_group_append(row_st, row_ed) \
{ \
    warpinfo tmp; \
    int tmp_level = get_winfo(row_st, row_ed, RowPtr, ColIdx, ana, paras, tmp); \
    if (tmp_level > level_max) level_max = tmp_level; \
    ana->row_group_vec.push_back(tmp); \
}

void row_block_partition(int m, int nnz, int *RowPtr, int *ColIdx, ptr_anainfo ana, anaparas paras)
{
    //ana->winfo_vec.resize(ana->partition_levels);

    int count = 0;
    int count2 = 0;

    int level_max = 0;

    int row_block = paras.row_alpha;
    for (int i = 0; i < m; i += row_block)
    {
        int row_st = i;
        int row_ed = ag_min(i + row_block, m);

        row_group_append(row_st, row_ed);

        // warpinfo tmp;
        // int tmp_level = get_winfo(row_st, row_ed, RowPtr, ColIdx, ana, tmp);

        // //printf("row partition %d %d\n", row_st, row_ed);

        // if (tmp_level > level_max) level_max = tmp_level;
        
        // ana->row_group_vec.push_back(tmp);
        //ana->winfo_vec[ana->level_partition_map[tmp_level]].push_back(tmp);

        count++;
    }

    ana->level_num = level_max + 1;

    // printf("total %d one level count %d\n", count, count2);
}

void row_thresh_partition(int m, int nnz, int *RowPtr, int *ColIdx, ptr_anainfo ana, anaparas paras)
{
    int level_max = 0;

    int row_thresh = paras.row_alpha;

    int row = 0;
    int node_count = 0;

    while (row < m)
    {
        int row_st = row;
        if (RowPtr[row + 1] - RowPtr[row] > row_thresh)
        {
            row++;
        }
        else
        {
            //while (row < m && csrRowPtr[row + 1] - csrRowPtr[row] <= thresh)
            while (row < m && row - row_st < WARP_SIZE && RowPtr[row + 1] - RowPtr[row] <= row_thresh)
            {
                row++;
            }
        }

        row_group_append(row_st, row);

    }

    ana->level_num = level_max + 1;

}

void row_avg_partition(int m, int nnz, int *RowPtr, int *ColIdx, ptr_anainfo ana, anaparas paras)
{
    int count = 0;
    int count2 = 0;

    int level_max = 0;

    int row_avg_thresh = paras.row_alpha;
    for (int i = 0; i < m; i += 32)
    {
        int row_st = i;
        int row_ed = ag_min(i + 32, m);
        float avg_nnz = (RowPtr[row_ed] - RowPtr[row_st]) / (row_ed - row_st);

        if (avg_nnz < row_avg_thresh)
        {
            row_group_append(row_st, row_ed);
        }
        else
        {
            for (int j = row_st; j < row_ed; j++)
            {
                row_group_append(j, j + 1);
            }
        }

    }

    ana->level_num = level_max + 1;

}

void init_row_level(int m, ptr_anainfo ana)
{
    ana->level_num = 0;
    //ana->row_level = (int*)malloc(sizeof(int) * m);
    memset(ana->row_level, 0, sizeof(int) * m);
}

// void show_winfo_vec(ptr_anainfo ana)
// {
//     printf("winfo_vec: \n");
//     for (int i = 0; i < ana->partition_levels; i++)
//     {
//         printf("%d ", ana->row_group_vec[i].size());
//     }
//     printf("\n");
// }

template <typename info_type>
void show_swinfo_vec(ptr_anainfo ana, vector<vector<info_type>> &winfo_vec)
{
    int subwarp_size = sizeof(info_type);
    printf("swinfo_vec: \n");
    for (int i = 0; i < ana->partition_levels; i++)
    {
        printf("%d ", winfo_vec[i].size());
    }
    printf("\n");
}

template<typename type_info>
void get_partition_num(ptr_anainfo ana, vector<vector<type_info>> &tgt_vec)
{
    for (int i = 0; i < ana->partition_levels; i++)
        ana->partition_num.push_back(tgt_vec[i].size());
}

void row_partition(int m, int nnz, int *RowPtr, int *ColIdx, ptr_anainfo ana, anaparas paras)
{
    // ordering by level
    //ana->winfo_vec.resize(m);

    // ana->winfo_vec = (*(vector<warpinfo>))malloc(ana->level_num * sizeof(vector<warpinfo>));

    init_row_level(m, ana);

    if (paras.row_s == ROW_BLOCK)
    {
        row_block_partition(m, nnz, RowPtr, ColIdx, ana, paras);
    }
    else if (paras.row_s == ROW_BLOCK_THRESH)
    {
        row_thresh_partition(m, nnz, RowPtr, ColIdx, ana, paras);
    }
    else if (paras.row_s == ROW_BLOCK_AVG)
    {
        row_avg_partition(m, nnz, RowPtr, ColIdx, ana, paras);
    }
    else
    {
        printf("Not implemented!\n");
    }

}