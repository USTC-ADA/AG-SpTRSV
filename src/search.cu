#include "../include/search.h"

int anaspace::partition_incr()
{
    //printf("before: %d %d %d %d %d\n", lp_pos, alpha_pos, rp_pos, sws_pos, tbs_pos);
    lp_pos++;
    while (lp_list[lp_pos] == ROW_WISE && sws_list[sws_pos] > 1)
    {
        lp_pos++;
    }
    //printf("mid: %d %d %d %d %d\n", lp_pos, alpha_pos, rp_pos, sws_pos, tbs_pos);
    if (lp_pos == lp_list.size())
    {
        lp_pos = 0;
        alpha_pos++;
    }
    if (alpha_pos == alpha_list.size())
    {
        alpha_pos = 0;
        rp_pos++;
    }
    if (rp_pos == rp_list.size())
    {
        rp_pos = 0;
        sws_pos++;
    }
    if (sws_pos == sws_list.size())
    {
        sws_pos = 0;
        tbs_pos++;
    }
    if (tbs_pos >= tbs_list.size())
    {
        return -1;
    }
    //printf("after: %d %d %d %d %d\n", lp_pos, alpha_pos, rp_pos, sws_pos, tbs_pos);
    // system("pause");
    return 0;
}

int anaspace::schedule_incr()
{
    ls_pos++;
    if (ls_pos == ls_list.size())
    {
        ls_pos = 0;
        ws_pos++;
    }
    if (ws_pos == ws_list.size())
    {
        ws_pos = 0;
        rg_pos++;
        if (lp_list[lp_pos] == ROW_WISE && rg_list[rg_pos] == RG_BALANCE) rg_pos++;
    }
    if (rg_pos >= rg_list.size())
    {
        ls_pos = ws_pos = rg_pos = 0;
        //partition_incr();
        return -1;
    }

    return 0;
}

int anaspace::get_next_partition(anaparas &paras)
{
    if (tbs_pos >= tbs_list.size()) return -1;
    paras.tb_size = tbs_list[tbs_pos];
    paras.subwarp_size = sws_list[sws_pos];
    paras.row_s = rp_list[rp_pos];
    paras.row_alpha = alpha_list[alpha_pos];
    paras.level_ps = lp_list[lp_pos];

    //partition_incr();

    return 0;
}

int anaspace::get_next_schedule(anaparas &paras)
{
    // printf("Reading: %d %d %d %d sched %d %d %d\n", lp_pos, alpha_pos, rp_pos, sws_pos,
    // ls_pos, ws_pos, rg_pos);

    paras.level_ss = ls_list[ls_pos];
    paras.schedule_s = ws_list[ws_pos];
    paras.rg_ss = rg_list[rg_pos];

    //schedule_incr();

    return 0;
}

void print_paras(FILE* full_out, char *input_name, anaparas &paras, float parastime = 0)
{
    fprintf(full_out, "%s,", input_name);
    fprintf(full_out, "%d,%d,%d,%d,%d,%d,%d,%d,%.2f",
    paras.tb_size, paras.subwarp_size, paras.row_s, paras.row_alpha,
    paras.level_ps, paras.level_ss, paras.schedule_s, paras.rg_ss, parastime);
    fprintf(full_out, "\n");
}