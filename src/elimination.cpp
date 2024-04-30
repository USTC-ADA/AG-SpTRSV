#include "../include/elimination.h"
#include <stdio.h>

// 3 levels
// return 0: no elimination 
// return 1: only read sync inside block
// return 2: no sync for read
int rule_read_sync(ptr_node t)
{
    int flag = 2;
    for (auto i = t->parent.begin(); i != t->parent.end(); i++)
    {
        if (t->info.start_row >= 550400 && t->info.start_row <= 550460)
        printf("node %d parent %d %d %d flag %d\n", t->info.start_row,
        (*i)->info.start_row, (*i)->warp_id, get_sm_id((*i)->warp_id), flag);
        if ((*i)->warp_id != t->warp_id && flag == 2)
        {
            flag = 1;
        }
        if (!is_nearwarp((*i)->warp_id, t->warp_id) && flag == 1)
        {
            flag = 0;
            break;
        }
    }
    return flag;
}

// 3 levels
// return 0: no elimination 
// return 1: only write sync inside block 
// return 2: no sync for read
int rule_write_sync(ptr_node t)
{
    int flag = 2;
    for (auto i = t->child.begin(); i != t->child.end(); i++)
    {
        if ((*i)->warp_id != t->warp_id && flag == 2)
        {
            flag = 1;
        }
        if (!is_nearwarp((*i)->warp_id, t->warp_id) && flag == 1)
        {
            flag = 0;
            break;
        }
        if (t->info.start_row == 0) printf("dep %d %d %d flag %d\n", (*i)->info.start_row, (*i)->warp_id, get_sm_id((*i)->warp_id), flag);
    }
    return flag;
}

void sync_elimination(ptr_handler handler)
{
    int count = 0, count2 = 0;
    int k = 0;
    int row_flag[handler->m];
    for (int i = 0; i < WARP_NUM; i++)
    {
        for (int j = 0; j < handler->m; j++)
            row_flag[j] = 0;
        for (auto j = handler->warp_schedule[i].begin(); j != handler->warp_schedule[i].end(); j++)
        {
            count++;
            k++;
            ptr_node current_node = *j;
            // int read_flag = rule_read_sync(current_node);
            int write_flag = rule_write_sync(current_node);
            if (write_flag == 2) current_node->info.elim = NO_WRITE_FENCE;
            if (write_flag == 1) current_node->info.elim = WRITE_FENCE_BLOCK;
            if (write_flag >= 1) count2++; 

            // if (i == 0) printf("smi %d warpj %d id %d row %d %d %d\n", get_sm_id(i), k, current_node->warp_id, current_node->info.start_row, read_flag, write_flag);

            // if (read_flag >= 1 && write_flag >= 1)
            // {
            //     current_node->info.elim = READ_WRITE_BLOCK;
            //     //printf("current_node %d %d %d\n", current_node->topo_level, current_node->info.start_row, current_node->warp_id);
            //     count2++;
            // }
            // else
            // {
            //     current_node->info.elim = NO_ELIM;
            // }

            // auto next_node = std::next(j);
            // if (next_node != handler->warp_schedule[i].end())
            // {
            //     if ((*next_node)->topo_level == (*j)->topo_level)
            //     {
            //         current_node->info.elim = READ_WRITE_BLOCK;
            //         count++;
            //         //if (i == 0) printf("(*next_node)->topo_level %d (*j)->topo_level %d\n", (*next_node)->topo_level, (*j)->topo_level);
            //     }
            //     else
            //     {
            //         current_node->info.elim = NO_ELIM;
            //         count2++;
            //     }
            // }

            // no read wait
            // int read_wait_flag = 1;
            // for (auto parent_node = current_node->parent.begin(); parent_node != current_node->parent.end(); parent_node++)
            // {
            //     for (int parent_row = (*parent_node)->info.start_row; parent_row < (*parent_node)->info.end_row; parent_row++)
            //     {
            //         if (!row_flag[parent_row])
            //         {
            //             row_flag[parent_row] = 1;
            //             read_wait_flag = 0;
            //         }
            //     }
            // }
            // if (read_wait_flag)
            // {
            //     current_node->info.elim = NO_READ_FENCE;
            //     count2++;
            // }
            //current_node->info.elim = NO_READ_FENCE;
            // if (current_node->child.size() == 0)
            // {
            //     count2++;
            //     current_node->info.elim = NO_WRITE_FENCE;
            // }
            // if (current_node->parent.size() == 0)
            // {
            //     count2++;
            //     current_node->info.elim = NO_FENCE;
            // }
            //current_node->info.elim = READ_WRITE_BLOCK;
            // count++;
        }
    }
    printf("count %d %d\n", count, count2);
}