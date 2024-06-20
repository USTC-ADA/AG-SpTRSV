#ifndef SEARCH__
#define SEARCH__

#include "common.h"
#include <cuda.h>

// Some parameters and strategies are eliminated for quicker seach
struct anaspace
{
    vector<int> tbs_list = {64, 256, 1024};
    vector<int> sws_list = {1, 4, 8};

    vector<PREPROCESSING_STRATEGY> rp_list = 
    {ROW_BLOCK, ROW_BLOCK_THRESH, ROW_BLOCK_AVG};
    vector<int> alpha_list = {1, 4, 8, 16, 32};
    vector<LEVEL_PART_STRATEGY> lp_list = 
    {LEVEL_WISE, ROW_WISE};

    vector<LEVEL_SCHED_STRATEGY> ls_list =
    {ONE_LEVEL};
    vector<SCHEDULE_STRATEGY> ws_list =
    {SIMPLE, SIMPLE2, WORKLOAD_BALANCE};
    vector<ROW_GROUP_SCHED_STRATEGY> rg_list =
    {RG_SIMPLE, RG_BALANCE};

    int tbs_pos = 0;
    int sws_pos = 0;
    int rp_pos = 0;
    int alpha_pos = 0;
    int lp_pos = 0;

    int rg_pos = 0;
    int ws_pos = 0;
    int ls_pos = 0;

    anaspace() {};

    int partition_incr();
    int schedule_incr();

    int get_next_partition(anaparas &paras);
    int get_next_schedule(anaparas &paras);
};

void print_paras(FILE* full_out, char *input_name, anaparas &paras, float parastime);

#endif

