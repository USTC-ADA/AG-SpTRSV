#include "../include/schedule.h"

// #define AG_SCHEDULE(FUNC_NAME) { \
// if (paras.subwarp_size == 1) \
//     FUNC_NAME<1>(ana, ana->swinfo_vec1); \
// if (paras.subwarp_size == 2) \
//     FUNC_NAME<2>(ana, ana->swinfo_vec2); \
// if (paras.subwarp_size == 4) \
//     FUNC_NAME<4>(ana, ana->swinfo_vec4); \
// if (paras.subwarp_size == 8) \
//     FUNC_NAME<8>(ana, ana->swinfo_vec8); \
// if (paras.subwarp_size == 16) \
//     FUNC_NAME<16>(ana, ana->swinfo_vec16); \
// if (paras.subwarp_size == 32) \
//     FUNC_NAME<32>(ana, ana->swinfo_vec32); \
// }

void one_level_schedule(ptr_anainfo ana, anaparas paras)
{
    int count = 0;
    for (int i = 0; i < ana->partition_levels; i++)
    {
        count += ana->partition_num[i];
    }
    ana->winfo_n.push_back(count);
    if (ana->level_num > 1)
        ana->winfo_multilevel.push_back(true);
    else
        ana->winfo_multilevel.push_back(false);
}

void thresh_level_schedule(ptr_anainfo ana, anaparas paras)
{
    int count = 0;
    bool multi_level = false;
    //int level_alpha = paras.level_alpha;
    int level_alpha = 1024;
    for (int i = 0; i < ana->partition_levels; i++)
    {
        count += ana->partition_num[i];
        if (ana->partition_num[i] < SM_NUM * paras.tb_size)
        {
            multi_level = true;
        }
        else
        {
            ana->winfo_n.push_back(count);
            ana->winfo_multilevel.push_back(multi_level);
            multi_level = false;
        }
    }
    if (ana->winfo_n.back() != count)
    {
        ana->winfo_n.push_back(count);
        ana->winfo_multilevel.push_back(multi_level);
    }
}

template <int subwarp_size>
void refine_winfo_multi_level(ptr_anainfo ana, vector<vector<subwarpinfo<subwarp_size>>> &info_vec)
{
    ana->winfo_multilevel.resize(ana->partition_levels);
    for (int i = 0; i < ana->partition_levels; i++)
    {
        for (int j = 0; j < info_vec[i].size(); j++)
            if (info_vec[i][j].info & 0x1)
            {
                ana->winfo_multilevel[i] = true;
                break;
            }
    }
}

void indep_level_schedule(ptr_anainfo ana, anaparas paras)
{
    // Nothing to do
    // printf("? ana->partition_levels %d\n", ana->partition_levels);
    int count = 0;
    for (int i = 0; i < ana->partition_levels; i++)
    {
        count += ana->partition_num[i];
        ana->winfo_n.push_back(count);
        ana->winfo_multilevel.push_back(false);
    }
}

void level_schedule(ptr_anainfo ana, anaparas paras)
{
    ana->winfo_n.push_back(0);
    if (paras.level_ss == INDEP_LEVEL)
    {
        indep_level_schedule(ana, paras);
    }
    else if (paras.level_ss == ONE_LEVEL)
    {
        one_level_schedule(ana, paras);
    }
    else if (paras.level_ss == THRESH_LEVEL)
    {
        thresh_level_schedule(ana, paras);
    }
    else
    {
        printf("Not implemented!\n");
    }

    AG_SCHEDULE(refine_winfo_multi_level, ana);
}

template <int subwarp_size>
void simple_schedule(ptr_anainfo ana, vector<vector<subwarpinfo<subwarp_size>>> &info_vec)
{
    //ana->winfo_n = (int*)malloc(sizeof(int) * (ana->partition_levels + 1));

    int total_warps = 0;
    for (int i = 0; i < ana->partition_levels; i++)
    {
        total_warps += info_vec[i].size();
    }

    subwarpinfo<subwarp_size> *winfo_h_tmp = 
    (subwarpinfo<subwarp_size>*)malloc(sizeof(subwarpinfo<subwarp_size>) * total_warps);

    int count = 0;
    for (int i = 0; i < ana->partition_levels; i++)
    {
        //ana->winfo_n[i] = count;
        for (auto j = info_vec[i].begin(); j != info_vec[i].end(); j++)
        {
            winfo_h_tmp[count] = *j;
            count++;
        }
    }
    //ana->winfo_n[ana->partition_levels] = count;

    cudaMalloc(&ana->winfo_d, total_warps * sizeof(subwarpinfo<subwarp_size>));
    cudaMemcpy(ana->winfo_d, winfo_h_tmp, total_warps * sizeof(subwarpinfo<subwarp_size>), cudaMemcpyHostToDevice);

    free(winfo_h_tmp);

}

template <int subwarp_size>
void schedule_info_hosttodevice2(ptr_anainfo ana, vector<vector<subwarpinfo<subwarp_size>>> &tmp_vec)
{
    int warp_num = ana->warp_num;

    subwarpinfo<subwarp_size> *info_h[warp_num];
    int winfo_n2_h[warp_num];

    cudaMalloc(&ana->winfo_n2, warp_num * sizeof(int));
    cudaMalloc(&ana->winfo_d2, warp_num * sizeof(subwarpinfo<subwarp_size>*));

    for (int i = 0; i < warp_num; i++)
    {
        winfo_n2_h[i] = tmp_vec[i].size();
        // if (tmp_vec[i].size() > 0)
        //     printf("warp_num %d size %d\n", i, tmp_vec[i].size());
        //     printf("hha %d %d %d %d\n", i, tmp_vec[i].size(),
        //     tmp_vec[i][0].row_st[0], tmp_vec[i][0].row_ed[0]);
    }

    cudaMemcpy(ana->winfo_n2, winfo_n2_h, warp_num * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < warp_num; i++)
    {
        int level = winfo_n2_h[i];
        if (level)
        {
            cudaMalloc(&info_h[i], level * sizeof(subwarpinfo<subwarp_size>));
            cudaMemcpy(info_h[i], &tmp_vec[i][0], level * sizeof(subwarpinfo<subwarp_size>), cudaMemcpyHostToDevice);
        }
        else
        {
            info_h[i] = NULL;
        }
    }

    cudaMemcpy(ana->winfo_d2, info_h, warp_num * sizeof(subwarpinfo<subwarp_size>*), cudaMemcpyHostToDevice);
}

template <int subwarp_size>
void simple2_schedule(ptr_anainfo ana, vector<vector<subwarpinfo<subwarp_size>>> &info_vec)
{
    vector<vector<subwarpinfo<subwarp_size>>> tmp_vec;
    tmp_vec.resize(ana->warp_num);

    int current_warp = 0;

    for (int i = 0; i < ana->partition_levels; i++)
    {
        for (auto j = info_vec[i].begin(); j != info_vec[i].end(); j++)
        {
            tmp_vec[current_warp].push_back(*j);

            current_warp = (current_warp + 1) % ana->warp_num;
        }
    }

    schedule_info_hosttodevice2<subwarp_size>(ana, tmp_vec);
}

struct warp_load_struct
{
    int warp_id;
    int load = 0;
    warp_load_struct(int warp_id_in, int load_in): warp_id(warp_id_in), load(load_in) {};

    bool operator > (const warp_load_struct &a) const
    {
        return load > a.load || load == a.load && warp_id > a.warp_id;
    };
};

template <int subwarp_size>
void balance_schedule(ptr_anainfo ana, int *csrRowPtr, vector<vector<subwarpinfo<subwarp_size>>> &info_vec)
{
    vector<vector<subwarpinfo<subwarp_size>>> tmp_vec;
    tmp_vec.resize(ana->warp_num);

    priority_queue<warp_load_struct, vector<warp_load_struct>, greater<warp_load_struct>> q;
    for (int i = 0; i < ana->warp_num; i++)
    {
        q.push(warp_load_struct(i, 0));
    }

    //printf("??? %d\n", q.top().warp_id);

    for (int i = 0; i < ana->partition_levels; i++)
    {
        for (auto j = info_vec[i].begin(); j != info_vec[i].end(); j++)
        {
            int current_warp = q.top().warp_id;
            int current_load = q.top().load;

            tmp_vec[current_warp].push_back(*j);

            //current_warp = (current_warp + 1) % ana->warp_num;

            int tmp_load = 0;
            for (int k = 0; k < subwarp_size; k++)
            {
                if ((*j).row_st[k] > 0)
                    tmp_load += csrRowPtr[(*j).row_ed[k]] - csrRowPtr[(*j).row_st[k]];
            }

            //printf("%d %d %d\n", current_warp, tmp_load, current_load);

            q.pop();
            q.push(warp_load_struct(current_warp, current_load + tmp_load));
        }
    }

    // while (!q.empty())
    // {
    //     q.pop();
    // }

    // q.clear();
    // q.shrink_to_size();

    schedule_info_hosttodevice2<subwarp_size>(ana, tmp_vec);
}

void warp_schedule(ptr_anainfo ana, anaparas paras, int *csrRowPtr)
{
    ana->warp_num = BLOCK_NUM * paras.tb_size / WARP_SIZE;

    if (paras.schedule_s == SIMPLE)
    {
        AG_SCHEDULE(simple_schedule, ana);
    }
    else if (paras.schedule_s == SIMPLE2)
    {
        AG_SCHEDULE(simple2_schedule, ana);
    }
    else if (paras.schedule_s == WORKLOAD_BALANCE)
    {
        AG_SCHEDULE(balance_schedule, ana, csrRowPtr);
    }
    else
    {
        printf("Warp scheduling strategy not implemented!\n");
    }
}

template <int subwarp_size>
void row_group_schedule_simple(ptr_anainfo ana, vector<vector<subwarpinfo<subwarp_size>>> &tgt_vec)
{
    tgt_vec.resize(ana->partition_levels);

    int pos[ana->partition_levels];
    for (int i = 0; i < ana->partition_levels; i++)
    {
        pos[i] = 0;
    }

    for (int i = 0; i < ana->row_group_vec.size(); i++)
    {
        int row_st = ana->row_group_vec[i].row_st;
        int row_ed = ana->row_group_vec[i].row_ed;
        unsigned int info = ana->row_group_vec[i].info;

        int current_level = ana->level_partition_map[ana->row_level[row_st]];
        int current_pos = pos[current_level];
        
        if (current_pos == 0)
        {
            subwarpinfo<subwarp_size> tmp_new = subwarpinfo<subwarp_size>();
            tmp_new.append(row_st, row_ed, info, 0);
            tgt_vec[current_level].push_back(tmp_new);
        }
        else
        {
            tgt_vec[current_level].back().append(row_st, row_ed, info, current_pos);
        }

        pos[current_level] = (pos[current_level] + 1) % subwarp_size;
    }
}

bool cmp_swinfo(warpinfo &a, warpinfo &b) {
    return a.nnz > b.nnz;
}

template <int subwarp_size>
void row_group_schedule_balance(ptr_anainfo ana, vector<vector<subwarpinfo<subwarp_size>>> &tgt_vec)
{
    tgt_vec.resize(ana->partition_levels);

    int pos[ana->partition_levels];
    for (int i = 0; i < ana->partition_levels; i++)
    {
        pos[i] = 0;
    }

    vector<vector<warpinfo>> tmp_row_group_list;
    tmp_row_group_list.resize(ana->partition_levels);

    for (int i = 0; i < ana->row_group_vec.size(); i++)
    {
        int row_st = ana->row_group_vec[i].row_st;
        int current_level = ana->level_partition_map[ana->row_level[row_st]];
        tmp_row_group_list[current_level].push_back(ana->row_group_vec[i]);
    }

    for (int i = 0; i < ana->partition_levels; i++)
    {
        std::sort(tmp_row_group_list[i].begin(), tmp_row_group_list[i].end(), cmp_swinfo);
    }

    for (int i = 0; i < ana->partition_levels; i++)
    {
        int current_pos = 0;

        for (int j = 0; j < tmp_row_group_list[i].size(); j++)
        {
            int row_st = tmp_row_group_list[i][j].row_st;
            int row_ed = tmp_row_group_list[i][j].row_ed;
            unsigned int info = tmp_row_group_list[i][j].info;

            if (current_pos == 0)
            {
                subwarpinfo<subwarp_size> tmp_new = subwarpinfo<subwarp_size>();
                tmp_new.append(row_st, row_ed, info, 0);
                tgt_vec[i].push_back(tmp_new);
            }
            else
            {
                tgt_vec[i].back().append(row_st, row_ed, info, current_pos);
            }

            current_pos = (current_pos + 1) % subwarp_size;
        }
    }
    //printf("???\n");
}

template <int subwarp_size>
void get_partition_num(ptr_anainfo ana, vector<vector<subwarpinfo<subwarp_size>>> &tgt_vec, bool show = false)
{
    ana->partition_num.resize(ana->partition_levels);
    if (show) printf("partition num: ");
    for (int i = 0; i < ana->partition_levels; i++)
    {
        ana->partition_num[i] = tgt_vec[i].size();
        if (show) printf("%d ", ana->partition_num[i]);
    }
    if (show) printf("\n");
}

void row_group_schedule(ptr_anainfo ana, anaparas paras)
{
    if (paras.rg_ss == RG_SIMPLE)
    {
        AG_SCHEDULE(row_group_schedule_simple, ana);
    }
    else if (paras.rg_ss == RG_BALANCE)
    {
        AG_SCHEDULE(row_group_schedule_balance, ana);
    }
    else
    {
        printf("Row group scheduling strategy not implemented!\n");
    }

    AG_SCHEDULE(get_partition_num, ana);
}
