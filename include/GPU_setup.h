#ifndef GPU_SETUP__
#define GPU_SETUP__

// cuda parallelization setup

// Please set SM_NUM as the number of stream-multiprocessors on GPU
// Due to register constraint, set BLOCK_NUM = SM_NUM on Ampere
#define SM_NUM 80
#define BLOCK_NUM 80

#define THREAD_NUM_PER_BLOCK 128
#define ALIGN_THREAD_PER_BLOCK 1024
#define WARP_SIZE 32
#define WARP_NUM_PER_BLOCK (THREAD_NUM_PER_BLOCK / WARP_SIZE)
#define WARP_NUM_PER_SM (WARP_NUM_PER_BLOCK * BLOCK_NUM / SM_NUM)
#define WARP_NUM (BLOCK_NUM * WARP_NUM_PER_BLOCK)
#define BLOCK_PER_SM (BLOCK_NUM / SM_NUM)

// #define TOTAL_BLOCK_BUF_SIZE 8192
// #define BUF_SIZE (TOTAL_BLOCK_BUF_SIZE / THREAD_NUM_PER_BLOCK)
#define BUF_SIZE 8
#define MAX_SUBWARP 4

#define get_block_id(x) ((x) / WARP_NUM_PER_BLOCK)
#define get_sm_id(x) ((x) / WARP_NUM_PER_SM)
#define get_warp_id(sm_id, local_warp_id) ((sm_id) * WARP_NUM_PER_SM + (local_warp_id))

int next_nearwarp(int warp_id);
int next_nonearwarp(int warp_id);

int is_nearwarp(int warp1, int warp2);

#endif