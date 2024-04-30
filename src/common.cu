#include "../include/common.h"

node_info_ :: node_info_(int start_row_input, int end_row_input, int using_shared_mem_input):
            start_row(start_row_input), end_row(end_row_input)
{
    format = CSR;
    elim = NO_ELIM;
    using_shared_mem = using_shared_mem_input;
    //start_idx = end_idx = -1;
}

node_info_ :: node_info_()
{}

// node_info_ :: node_info_(int start_row_input, int end_row_input,
//             int start_idx_input, int end_idx_input):
//             start_row(start_row_input), end_row(end_row_input),
//             start_idx(start_idx_input), end_idx(end_idx_input) {}

subwarp_info_ :: subwarp_info_()
{
    subwarp_size = 0;
    elim = NO_ELIM;
}

void subwarp_info_ :: add(int start_row_input, int end_row_input)
{
    start_row[subwarp_size] = start_row_input;
    end_row[subwarp_size] = end_row_input;
    subwarp_size = subwarp_size + 1;
}

void subwarp_info_ :: clear()
{
    subwarp_size = 0;
}

node :: node(int nid, int start_row_input, int end_row_input, int nnz_input, int using_shared_mem_input):
            info(start_row_input, end_row_input, using_shared_mem_input)
{
    id = nid;
    topo_level = 0;
    num_nnz = nnz_input;
    in_degree = out_degree = in_degree_tmp = 0;
    warp_sche_level = 0;
    ori_start = start_row_input;
    locality_node = NULL;
};

SpTRSV_handler :: ~SpTRSV_handler()
{
    // Actual destructor is implemented in finalize.cpp
}

anainfo :: anainfo(int m)
{
    cudaMalloc(&get_value, m * sizeof(int));
}

void show_paras(anaparas paras)
{
    printf("tbs %d sws %d ps %d %d %d ss %d %d %d\n",
    paras.tb_size, paras.subwarp_size,
    paras.row_s, paras.row_alpha, paras.level_ps,
    paras.rg_ss, paras.schedule_s, paras.level_ss);
}