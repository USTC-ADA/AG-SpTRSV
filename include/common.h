#ifndef COMMON__
#define COMMON__

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "GPU_setup.h"
#include "subwarp.h"
#include <string>
using namespace std;

enum PREPROCESSING_STRATEGY
{
    ROW_BLOCK,
    ROW_BLOCK_THRESH,
    ROW_BLOCK_AVG,
    ROW_LEVEL,
    ROW_SUBWARP,
    ROW_BLOCK_NODEP,
    SUPERNODE_BLOCK,
};

enum SCHEDULE_STRATEGY
{
    SIMPLE,
    SIMPLE2,
    WORKLOAD_BALANCE,
    SIMPLE_INTERLEAVED,
    WARP_LOCALITY,
    BALANCE_AND_LOCALITY,
    SEQUENTIAL,
    STRUCTURED,
    WINDOW,
    SEQUENTIAL2,
    SUBWARP,
};

enum LEVEL_PART_STRATEGY
{
    LEVEL_WISE,
    ROW_WISE
};

enum LEVEL_SCHED_STRATEGY
{
    INDEP_LEVEL,
    ONE_LEVEL,
    THRESH_LEVEL
};

enum ROW_GROUP_SCHED_STRATEGY
{
    RG_SIMPLE,
    RG_BALANCE
};

enum LOCAL_FORMAT
{
    CSR,
    TELL
};

enum SYNC_ELIM
{
    NO_ELIM,
    NO_WRITE_FENCE,
    WRITE_FENCE_BLOCK
};

typedef struct node_info_
{
    // node row ID
    int start_row;
    int end_row;

    // node format
    LOCAL_FORMAT format;
    SYNC_ELIM elim;

    // whether or not to use shared memory
    int using_shared_mem;

    node_info_(int start_row_input, int end_row_input, int using_shared_mem = 1);
    node_info_();

} node_info;

typedef struct subwarp_info_
{
    int start_row[MAX_SUBWARP];
    int end_row[MAX_SUBWARP];
    int subwarp_size;
    SYNC_ELIM elim;

    subwarp_info_();
    void add(int start_row_input, int end_row_input);
    void clear();
} subwarp_info;

typedef struct node* ptr_node;
struct node
{
    node_info info;

    // node id
    int id;

    // in & out degree, used in topological sort
    int in_degree;
    int out_degree;
    int in_degree_tmp;

    // number of nnz in the row_block
    int num_nnz;

    // row start position in original matrix (in case of reordering)
    int ori_start;

    // topological level
    int topo_level;

    // ID of the warp the node scheduled to, and schedule level
    int warp_id;

    // schedule level of current warp
    int warp_sche_level;

    // child nodes of data dependency edges
    vector<ptr_node> child;

    // parent nodes of data dependency edges
    vector<ptr_node> parent;

    // locality edge
    ptr_node locality_node;

    node(int nid, int start_row_input, int end_row_input, int nnz_input, int using_shared_mem_input = 1);
};

typedef struct graph* ptr_graph;
struct graph
{
    // total number of nodes
    int global_node;

    // total number of edges
    int global_edge;

    // List all nodes with no parent node
    vector<ptr_node> start_nodes;

    graph() 
    {
        global_node = global_edge = 0;
    }
};

typedef struct SpTRSV_handler* ptr_handler;
struct SpTRSV_handler
{
    // Number of rows
    int m;
    int nnz;
    int row_block;

    // Dependency graph of the matrix
    ptr_graph graph;

    // Schedule vector
    vector<ptr_node> warp_schedule[WARP_NUM];

    int *schedule_level;
    // Schedule info for each warp
    node_info **schedule_info;
    // Schedule info for each subwarp
    subwarp_info **schedule_subwarp_info;

    // Schedule info on device memory
    // Currently, preprocessing is implemented on CPU, 
    // considering transferring this stage to GPU in the future
    int *schedule_level_d;
    node_info **schedule_info_d;
    subwarp_info **subwarp_info_d;

    // When no schedule strategy is enabled, using the hardware scheduler
    node_info *no_schedule_info;
    node_info *no_schedule_info_d;

    // Array for data dependency
    int *get_value;
    int *warp_runtime;

    // Schedule strategy
    SCHEDULE_STRATEGY sched_s;
    int subwarp_size;

    ~SpTRSV_handler();

};

struct warpinfo
{
    int row_st;
    int row_ed;
    // bit-wise information
    // bit 0: whether the rows have multiple levels
    // bit 1: whether use shared memory
    // bit 2: whether all rows are in the same level
    unsigned int info;
    int nnz = 0;

    warpinfo() { nnz = 0; };
    void copy(int row_st_in, int row_ed_in, unsigned int info_in, int nnz_in)
    {
        row_st = row_st_in; row_ed = row_ed_in; info = info_in;
        nnz = nnz_in;
    }
};

typedef struct anainfo* ptr_anainfo;
struct anainfo
{
    // level info (on host)
    int partition_levels;
    int *level_partition_map = NULL;
    vector<int> partition_num;

    // warp info reordering vector (on host)
    vector<vector<warpinfo>> winfo_vec;

    // subwarp info vector (on host)
    vector<vector<subwarpinfo<1>>> swinfo_vec1;
    vector<vector<subwarpinfo<2>>> swinfo_vec2;
    vector<vector<subwarpinfo<4>>> swinfo_vec4;
    vector<vector<subwarpinfo<8>>> swinfo_vec8;
    vector<vector<subwarpinfo<16>>> swinfo_vec16;
    vector<vector<subwarpinfo<32>>> swinfo_vec32;

    // warp info reordering tmp vector (on host)
    // to store row groups
    vector<warpinfo> row_group_vec;

    // scheduled warp info (_h on host, _d on device)
    // ** indicates deterministic schedule, * indicates auto schedule
    void *winfo_d = NULL;
    void **winfo_d2 = NULL;
    int *winfo_n2 = NULL;
    //void *winfo_h = NULL;
    vector<int> winfo_n;
    vector<bool> winfo_multilevel;

    // partition information
    int *row_level = NULL;

    // matrix information
    int level_num;
    int *level_rownum = NULL;
    int max_row_nnz;
    double avg_parallelism;
    int max_parallelism;

    // for deterministic scheduling
    int warp_num;

    // get value
    int *get_value = NULL;

    anainfo(int m);

};

struct anaparas
{
    int tb_size;
    int subwarp_size;

    PREPROCESSING_STRATEGY row_s;
    int row_alpha;

    LEVEL_PART_STRATEGY level_ps;
    LEVEL_SCHED_STRATEGY level_ss;
    int level_alpha;

    SCHEDULE_STRATEGY schedule_s;
    ROW_GROUP_SCHED_STRATEGY rg_ss;

    anaparas() {};
    anaparas(int tb_size_in, int subwarp_size_in, PREPROCESSING_STRATEGY row_s_in, int row_alpha_in, LEVEL_PART_STRATEGY level_ps_in, LEVEL_SCHED_STRATEGY level_ss_in,
    SCHEDULE_STRATEGY schedule_s_in, ROW_GROUP_SCHED_STRATEGY rg_ss_in):
    tb_size(tb_size_in), subwarp_size(subwarp_size_in), 
    row_s(row_s_in), row_alpha(row_alpha_in), level_ps(level_ps_in), 
    level_ss(level_ss_in), schedule_s(schedule_s_in), rg_ss(rg_ss_in) { level_alpha = 1024; };
};

void show_paras(anaparas paras);

#define ag_max(a, b) (((a) >= (b))? (a): (b))
#define ag_min(a, b) (((a) < (b))? (a): (b))

#endif