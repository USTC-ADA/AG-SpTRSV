#include "../include/GPU_setup.h"

int next_nearwarp(int warp_id)
{
    if (get_sm_id(warp_id) == get_sm_id(warp_id + 1))
        return warp_id + 1;
    else
        return warp_id + 1 - WARP_NUM_PER_SM;
}

int next_nonearwarp(int warp_id)
{
    return (warp_id + WARP_NUM_PER_SM) % WARP_NUM;
}

int is_nearwarp(int warp1, int warp2)
{
    if (get_sm_id(warp1) == get_sm_id(warp2))
        return 1;
    else
        return 0;
}