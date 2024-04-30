#include "../include/schedule.h"

void schedule_node(vector<ptr_node> &schedule_vec, ptr_node current_node, int current_warp)
{
    schedule_vec.push_back(current_node);
    current_node->warp_id = current_warp;
    current_node->warp_sche_level = schedule_vec.size() - 1;
}

void schedule_info_hosttodevice(ptr_handler handler)
{
    node_info *info_h[WARP_NUM];

    cudaMalloc(&handler->schedule_level_d, WARP_NUM * sizeof(int));
    cudaMalloc(&handler->schedule_info_d, WARP_NUM * sizeof(node_info*));

    cudaMemcpy(handler->schedule_level_d, handler->schedule_level, WARP_NUM * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < WARP_NUM; i++)
    {
        int level = handler->schedule_level[i];
        if (level)
        {
            cudaMalloc(&info_h[i], level * sizeof(node_info));
            cudaMemcpy(info_h[i], handler->schedule_info[i], level * sizeof(node_info), cudaMemcpyHostToDevice);
        }
        else
        {
            info_h[i] = NULL;
        }
    }

    cudaMemcpy(handler->schedule_info_d, info_h, WARP_NUM * sizeof(node_info*), cudaMemcpyHostToDevice);
}

void schedule_info_hosttodevice_subwarp(ptr_handler handler)
{
    subwarp_info *info_h[WARP_NUM];

    cudaMalloc(&handler->schedule_level_d, WARP_NUM * sizeof(int));
    cudaMalloc(&handler->subwarp_info_d, WARP_NUM * sizeof(subwarp_info*));

    cudaMemcpy(handler->schedule_level_d, handler->schedule_level, WARP_NUM * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < WARP_NUM; i++)
    {
        int level = handler->schedule_level[i];
        if (level)
        {
            cudaMalloc(&info_h[i], level * sizeof(subwarp_info));
            cudaMemcpy(info_h[i], handler->schedule_subwarp_info[i], level * sizeof(subwarp_info), cudaMemcpyHostToDevice);
        }
        else
        {
            info_h[i] = NULL;
        }
    }

    cudaMemcpy(handler->subwarp_info_d, info_h, WARP_NUM * sizeof(subwarp_info*), cudaMemcpyHostToDevice);
}

void transfer_to_2D(ptr_handler handler)
{
    // transform from vector to 2D array
    handler->schedule_level = (int*)malloc(WARP_NUM * sizeof(int));
    handler->schedule_info = (node_info**)malloc(WARP_NUM * sizeof(node_info*));
    for (int i = 0; i < WARP_NUM; i++)
    {
        int schedule_level = handler->warp_schedule[i].size();
        handler->schedule_level[i] = schedule_level;

        if (schedule_level == 0)
        {
            handler->schedule_info[i] = NULL;
            continue;
        }

        handler->schedule_info[i] = (node_info*)malloc(schedule_level * sizeof(node_info));

        for (int j = 0; j < schedule_level; j++)
        {
            handler->schedule_info[i][j] = handler->warp_schedule[i][j]->info;
            // if (i == 0 || i == 1) printf("warp i %d num j %d row %d end %d\n", 
            // i, j, handler->warp_schedule[i][j]->info.start_row, handler->warp_schedule[i][j]->info.end_row);
        }
    }
}

void transfer_to_2D_subwarp(ptr_handler handler, int subwarp_size)
{
    // transform from vector to 2D array
    handler->schedule_level = (int*)malloc(WARP_NUM * sizeof(int));
    handler->schedule_subwarp_info = (subwarp_info**)malloc(WARP_NUM * sizeof(subwarp_info*));

    for (int i = 0; i < WARP_NUM; i++)
    {
        if (handler->warp_schedule[i].empty())
        {
            handler->schedule_subwarp_info[i] = NULL;
            handler->schedule_level[i] = 0;
            continue;
        }
        //printf("%d %d i\n", i, handler->warp_schedule[i].size());

        vector<subwarp_info> subwarp_list;
        int j = 0;
        int current_warp = 0;
        int last_level = handler->warp_schedule[i][0]->topo_level;
        subwarp_info tmp;

        //printf("last level %d\n", last_level);

        while (j < handler->warp_schedule[i].size())
        {
            ptr_node current_node = handler->warp_schedule[i][j];
            if (current_warp < subwarp_size && current_node->topo_level == last_level)
            {
                tmp.add(current_node->info.start_row, current_node->info.end_row);
                current_warp++;
            }
            else
            {
                subwarp_list.push_back(tmp);
                tmp.clear();
                tmp.add(current_node->info.start_row, current_node->info.end_row);
                last_level = current_node->topo_level;
                current_warp = 1;
            }
            j++;
        }

        if (tmp.subwarp_size > 0) subwarp_list.push_back(tmp);
        tmp.clear();

        int schedule_level = subwarp_list.size();
        handler->schedule_level[i] = schedule_level;

        handler->schedule_subwarp_info[i] = (subwarp_info*)malloc(schedule_level * sizeof(subwarp_info));

        for (int j = 0; j < schedule_level; j++)
        {
            handler->schedule_subwarp_info[i][j] = subwarp_list[j];
            // if (i < 2 && j == 0) printf("warp i %d num j %d size %d row2 %d end2 %d row3 %d end3 %d\n", 
            // i, j, handler->schedule_subwarp_info[i][j].subwarp_size, handler->schedule_subwarp_info[i][j].start_row[2], handler->schedule_subwarp_info[i][j].end_row[2],
            // handler->schedule_subwarp_info[i][j].start_row[3], handler->schedule_subwarp_info[i][j].end_row[3]);
        }
    }

}

void show_imbalance(ptr_handler handler)
{
    // print load imbalance
    int max_nnz, min_nnz;
    int sum = 0;
    for (int j = 0; j < handler->warp_schedule[0].size(); j++)
    {
        sum += handler->warp_schedule[0][j]->num_nnz;
    }
    max_nnz = min_nnz = sum;
    for (int i = 1; i < WARP_NUM; i++)
    {
        sum = 0;
        for (int j = 0; j < handler->warp_schedule[i].size(); j++)
        {
            sum += handler->warp_schedule[i][j]->num_nnz;
        }
        if (sum > max_nnz) max_nnz = sum;
        if (sum < min_nnz) min_nnz = sum;
        //printf(" %d", sum);
    }
    //printf("warp schedule: max_nnz %d min_nnz %d\n", max_nnz, min_nnz);

    for (int j = 0; j < WARP_NUM_PER_SM; j++)
    {
        for (int k = 0; k < handler->warp_schedule[j].size(); k++)
        {
            sum += handler->warp_schedule[j][k]->num_nnz;
        }
    }
    max_nnz = min_nnz = sum;
    for (int i = 0; i < SM_NUM; i++)
    {
        sum = 0;
        for (int j = 0; j < WARP_NUM_PER_SM; j++)
        {
            for (int k = 0; k < handler->warp_schedule[i * WARP_NUM_PER_SM + j].size(); k++)
            {
                sum += handler->warp_schedule[i * WARP_NUM_PER_SM + j][k]->num_nnz;
            }
        }
        if (sum > max_nnz) max_nnz = sum;
        if (sum < min_nnz) min_nnz = sum;
        //printf(" %d", sum);
    }
    //printf("\nsm   schedule: max_nnz %d min_nnz %d\n", max_nnz, min_nnz);
}

void simple_schedule(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
        //(*iter)->info.elim = READ_WRITE_BLOCK;
    }

    int count = 0;

    //vector<ptr_node> warp_schedule[WARP_NUM];

    int old_level = 0;

    int current_warp = 0;
    // topological sort
    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();
        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            (*iter)->in_degree_tmp++;
            if ((*iter)->in_degree_tmp == (*iter)->in_degree)
            {
                topo_queue.push(*iter);
                (*iter)->in_degree_tmp = 0;
                // (*iter)->info.elim = READ_WRITE_BLOCK;
                // if (!is_nearwarp((*iter)->warp_id, current_node->warp_id))
                // {
                //     //printf("No elim\n");
                //     (*iter)->info.elim = NO_ELIM;
                //     current_node->info.elim = NO_ELIM;
                //     count++;
                // }
            }
        }

        // if (current_node->topo_level > old_level)
        // {
        //     old_level = current_node->topo_level;
        //     current_warp = 0;
        // }

        // static schedule in turn
        schedule_node(handler->warp_schedule[current_warp], current_node, current_warp);
        current_warp = (current_warp + 1) % WARP_NUM;

        //if (current_node->info.start_row == 0) printf("%d %d\n", current_warp)

        topo_queue.pop();

    }
    //printf("count %d\n", count);

}

void simple_schedule_interleaved(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
    }

    int count = 0;

    int old_level = 0;

    int current_warp = 0;
    // topological sort
    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();
        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            (*iter)->in_degree_tmp++;
            if ((*iter)->in_degree_tmp == (*iter)->in_degree)
            {
                topo_queue.push(*iter);
                (*iter)->in_degree_tmp = 0;
            }
        }

        // static schedule in turn
        schedule_node(handler->warp_schedule[current_warp], current_node, current_warp);
        current_warp = (current_warp + WARP_NUM_PER_BLOCK) % WARP_NUM;
        if (current_warp < WARP_NUM_PER_BLOCK) current_warp = (current_warp + 1) % WARP_NUM_PER_BLOCK;

        topo_queue.pop();

    }

}

int approximate_workload(ptr_node current_node)
{
    // for (int i = 0; i < ; i++)
    // {

    // }
    return (current_node->num_nnz + WARP_SIZE - 1) / WARP_SIZE;
    return current_node->num_nnz;
}

// schedule to warp with least workload
void simple_schedule_workload_balance(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
    }

    //vector<ptr_node> warp_schedule[WARP_NUM];
    int workload[WARP_NUM];

    for (int i = 0; i < WARP_NUM; i++) workload[i] = 0;

    // topological sort
    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();
        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            // if (!current_node->id) 
            // {
            //     printf("queue node %d child %d\n", current_node->id, (*iter)->id);
            //     printf("in_degree_tmp %d in_degree %d\n", (*iter)->in_degree_tmp, 
            //     (*iter)->in_degree);
            // }

            (*iter)->in_degree_tmp++;
            if ((*iter)->in_degree_tmp == (*iter)->in_degree)
            {
                topo_queue.push(*iter);
                (*iter)->in_degree_tmp = 0;
            }
        }
        // choose the warp with least workload
        int least_nnz = workload[0], least_i = 0;
        for (int i = 1; i < WARP_NUM; i++)
        {
            if (workload[i] < least_nnz)
            {
                least_nnz = workload[i];
                least_i = i;
            }
        }
        schedule_node(handler->warp_schedule[least_i], current_node, least_i);

        // approximate the workload
        // if (current_node->)

        workload[least_i] += approximate_workload(current_node);
        // warp_schedule[least_i].push_back(current_node);
        // current_node->warp_id = least_i;
        // current_node->warp_sche_level = warp_schedule[least_i].size() - 1;

        topo_queue.pop();
    }

}

// schedule to warp with least workload: optimized
void simple_schedule_workload_balance2(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
    }

    int workload[SM_NUM];
    int sm_pos[SM_NUM];

    for (int i = 0; i < SM_NUM; i++)
    {
        workload[i] = 0;
        sm_pos[i] = 0;
    }

    // topological sort
    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();
        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            (*iter)->in_degree_tmp++;
            if ((*iter)->in_degree_tmp == (*iter)->in_degree)
            {
                topo_queue.push(*iter);
                (*iter)->in_degree_tmp = 0;
            }
        }

        // choose sm with least workload
        int least_nnz = workload[0], least_i = 0;
        for (int i = 1; i < SM_NUM; i++)
        {
            if (workload[i] < least_nnz)
            {
                least_nnz = workload[i];
                least_i = i;
            }
        }

        int dest_warp = get_warp_id(least_i, sm_pos[least_i]);
        schedule_node(handler->warp_schedule[dest_warp], current_node, dest_warp);
        sm_pos[least_i] = (sm_pos[least_i] + 1) % WARP_NUM_PER_SM;
        workload[least_i] += current_node->num_nnz;

        topo_queue.pop();
    }
}

// schedule child node to the warp of its last parent node
void simple_schedule_warp_locality(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    //vector<ptr_node> warp_schedule[WARP_NUM];

    float workload[WARP_NUM];

    for (int i = 0; i < WARP_NUM; i++)
        workload[i] = 0;
    
    int current_warp = 0;
    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
        // static schedule in turn
        schedule_node(handler->warp_schedule[current_warp], (*iter), current_warp);
        workload[current_warp] += (*iter)->num_nnz;
        // warp_schedule[current_warp].push_back(*iter);
        // (*iter)->warp_id = current_warp;
        // current_node->warp_sche_level = warp_schedule[current_warp].size() - 1;
        current_warp = (current_warp + 1) % WARP_NUM;
    }

    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();
        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            (*iter)->in_degree_tmp++;
            if ((*iter)->in_degree_tmp == (*iter)->in_degree)
            {
                topo_queue.push(*iter);
                (*iter)->in_degree_tmp = 0;
                int parent_block = current_node->warp_id / WARP_NUM_PER_BLOCK;
                int start_warp = parent_block * WARP_NUM_PER_BLOCK;

                int min_load = workload[start_warp];
                current_warp = start_warp;

                for (int i = start_warp + 1; i < start_warp + WARP_NUM_PER_BLOCK; i++)
                    if (workload[i] < min_load)
                    {
                        min_load = workload[i];
                        current_warp = i;
                    }
                schedule_node(handler->warp_schedule[current_warp], (*iter), current_warp);
                workload[current_warp] += current_node->num_nnz;
                // warp_schedule[current_warp].push_back(*iter);
                // (*iter)->warp_id = current_node->warp_id;
            }
        }

        topo_queue.pop();
    }

}

void simple_schedule_warp_locality2(ptr_handler handler)
{
    int window = 5;

    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
        //(*iter)->info.elim = READ_WRITE_BLOCK;
    }

    int schedule_pos[WARP_NUM];
    for (int i = 0; i < WARP_NUM; i++)
        schedule_pos[i] = 0;

    int count = 0;

    int old_level = 0;
    int old_m = -1;

    int current_warp = 0;
    int current_sm = 0;
    // topological sort
    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();
        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            (*iter)->in_degree_tmp++;
            if ((*iter)->in_degree_tmp == (*iter)->in_degree)
            {
                topo_queue.push(*iter);
                (*iter)->in_degree_tmp = 0;
            }
        }

        // static schedule in turn
        schedule_node(handler->warp_schedule[current_warp], current_node, current_warp);

        if (current_node->topo_level != old_level)
        {
            current_sm = 0;
            old_m = current_node->info.start_row;
            old_level = current_node->topo_level;
        }

        if (current_node->info.start_row - old_m > window)
            current_sm++;

        old_m = current_node->info.start_row;

        current_warp = schedule_pos[current_sm];
        schedule_node(handler->warp_schedule[current_warp], current_node, current_warp);
        schedule_pos[current_sm] = (schedule_pos[current_sm] + 1) % WARP_NUM_PER_SM;

        topo_queue.pop();
    }
    
}

// considering workload-balance and data-locality at the same time
void simple_schedule_workload_locality(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    //vector<ptr_node> warp_schedule[WARP_NUM];

    float workload[WARP_NUM];

    for (int i = 0; i < WARP_NUM; i++)
        workload[i] = 0;

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
    }

    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();
        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            (*iter)->in_degree_tmp++;
            if ((*iter)->in_degree_tmp == (*iter)->in_degree)
            {
                topo_queue.push(*iter);
                (*iter)->in_degree_tmp = 0;
            }
        }

        // record the number of parent nodes in each block
        int block_parent[BLOCK_NUM];
        int block_parent_sum = 0;
        for (int i = 0; i < BLOCK_NUM; i++) block_parent[i] = 0;
        for (auto iter = current_node->parent.begin(); iter != current_node->parent.end(); iter++)
        {
            int block_id = (*iter)->warp_id / WARP_NUM_PER_BLOCK;
            block_parent[block_id]++;
        }
        for (int i = 0; i < BLOCK_NUM; i++) block_parent_sum += block_parent[i];

        // choose the warp with the minimum total workload
        // workload = comm_alpha * inter_block_comm + calc_alpha * num_nnz;
        float comm_alpha = 1.0;
        float calc_alpha = 1.0;
        float min_total_workload = -1;
        int least_i = -1;
        for (int i = 0; i < WARP_NUM; i++)
        {
            int current_block_id = i / WARP_NUM_PER_BLOCK;
            int inter_block_comm = block_parent_sum - block_parent[current_block_id];

            float new_workload = comm_alpha * inter_block_comm + calc_alpha * current_node->num_nnz;
            
            if (workload[i] + new_workload < min_total_workload || min_total_workload == -1)
            {
                min_total_workload = workload[i] + new_workload;
                least_i = i;
            }
        }

        schedule_node(handler->warp_schedule[least_i], current_node, least_i);
        //warp_schedule[least_i].push_back(current_node);
        //current_node->warp_id = least_i;
        workload[least_i] = min_total_workload;

        topo_queue.pop();
    }
}

// considering workload-balance and data-locality at the same time: optimized
void simple_schedule_workload_locality2(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    //vector<ptr_node> warp_schedule[WARP_NUM];

    float workload[SM_NUM];
    int sm_pos[SM_NUM];
    int sm_parent[SM_NUM];

    for (int i = 0; i < SM_NUM; i++)
    {
        workload[i] = 0;
        sm_pos[i] = 0;
    }

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
    }

    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();
        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            (*iter)->in_degree_tmp++;
            if ((*iter)->in_degree_tmp == (*iter)->in_degree)
            {
                topo_queue.push(*iter);
                (*iter)->in_degree_tmp = 0;
            }
        }

        // record the number of parent nodes in each block
        int parent_sum = current_node->parent.size();
        for (int i = 0; i < SM_NUM; i++) sm_parent[i] = 0;
        for (auto iter = current_node->parent.begin(); iter != current_node->parent.end(); iter++)
        {
            int sm_id = (*iter)->warp_id / WARP_NUM_PER_SM;
            sm_parent[sm_id]++;
        }
        //for (int i = 0; i < SM_NUM; i++) block_parent_sum += sm_parent[i];

        // choose the warp with the minimum total workload
        // workload = comm_alpha * inter_block_comm + calc_alpha * num_nnz;
        float comm_alpha = 1.0;
        float calc_alpha = 1.0;
        float min_total_workload = -1;
        int least_i = -1;
        for (int i = 0; i < SM_NUM; i++)
        {
            int current_sm_id = i;
            int inter_sm_comm = parent_sum - sm_parent[current_sm_id];

            float new_workload = comm_alpha * inter_sm_comm + calc_alpha * current_node->num_nnz;
            
            if (workload[i] + new_workload < min_total_workload || min_total_workload == -1)
            {
                min_total_workload = workload[i] + new_workload;
                least_i = i;
            }
        }

        int dest_warp = get_warp_id(least_i, sm_pos[least_i]);
        schedule_node(handler->warp_schedule[dest_warp], current_node, dest_warp);

        sm_pos[least_i] = (sm_pos[least_i] + 1) % WARP_NUM_PER_SM;
        workload[least_i] = min_total_workload;

        topo_queue.pop();
    }

    //printf("Done\n");
}

void schedule_structured(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    //vector<ptr_node> warp_schedule[WARP_NUM];

    float sm_workload[SM_NUM];
    float warp_workload[SM_NUM][WARP_NUM_PER_SM];

    for (int i = 0; i < SM_NUM; i++)
    {
        sm_workload[i] = 0;
        for (int j = 0; j < WARP_NUM_PER_SM; j++)
            warp_workload[i][j] = 0;
    }

    int current_sm = 0;
    int current_warp = 0;
    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
        int swarp = get_warp_id(current_sm, current_warp);
        schedule_node(handler->warp_schedule[swarp], *iter, swarp);

        sm_workload[current_sm] += (*iter)->num_nnz;
        warp_workload[current_sm][current_warp] += (*iter)->num_nnz;

        current_sm++;
        if (current_sm >= SM_NUM)
        {
            current_warp = (current_warp + 1) % WARP_NUM_PER_SM;
            current_sm = 0;
        }

        printf("start row 0 %d\n", (*iter)->locality_node->info.start_row);

        //printf("row %d new_warp %d\n", (*iter)->info.start_row, swarp);
    }

    //printf("??\n");

    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();
        current_warp = current_node->warp_id;
        int current_sm = get_sm_id(current_node->warp_id);
        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            (*iter)->in_degree_tmp++;
            if ((*iter)->in_degree_tmp == (*iter)->in_degree)
            {
                //printf("node row %d\n", (*iter)->info.start_row);
                topo_queue.push(*iter);
                (*iter)->in_degree_tmp = 0;

                int flag = 0;
                ptr_node parent = NULL;
                for (auto i = (*iter)->parent.begin(); i != (*iter)->parent.end(); i++)
                {
                    if ((*i)->locality_node == (*iter))
                    {
                        flag = 1;
                        parent = *i;
                    }
                }

                if ((*iter)->info.start_row == 1) printf("start row 1 %d\n", flag);

                if (flag)
                {
                    //printf("start 1\n");

                    int new_warp = next_nearwarp(parent->warp_id);

                    //printf("current %d new_warp %d\n", current_warp, new_warp);
                    //printf("sm %d warp_id %d\n", get_sm_id(new_warp), new_warp % WARP_NUM_PER_SM);

                    schedule_node(handler->warp_schedule[new_warp], *iter, new_warp);

                    sm_workload[get_sm_id(new_warp)] += (*iter)->num_nnz;
                    warp_workload[get_sm_id(new_warp)][new_warp % WARP_NUM_PER_SM]
                    += (*iter)->num_nnz;

                    //printf("row %d new_warp %d\n", (*iter)->info.start_row, new_warp);
                }
                else
                {
                    //printf("start 2\n");

                    float min_load = -1;
                    int min_i = -1;
                    for (int i = 0; i < SM_NUM; i++)
                    {
                        if (i != current_sm && (sm_workload[i] < min_load || min_load == -1))
                        {
                            min_i = i;
                            min_load = sm_workload[i];
                        }
                        //printf("%f ", sm_workload[i]);
                    }
                    //printf("sm_workload\n");
                    min_load = warp_workload[min_i][0];
                    int min_j = 0;
                    for (int j = 1; j < WARP_NUM_PER_SM; j++)
                    {
                        if (warp_workload[min_i][j] < min_load)
                        {
                            min_j = j;
                            min_load = warp_workload[min_i][j];
                        }
                    }

                    //printf("%d %d %d\n", min_i, min_j, get_warp_id(min_i, min_j));
                    int new_warp = get_warp_id(min_i, min_j);
                    schedule_node(handler->warp_schedule[new_warp], *iter, new_warp);

                    sm_workload[min_i] += (*iter)->num_nnz;
                    warp_workload[min_i][min_j] += (*iter)->num_nnz;

                    //printf("row %d new_warp %d\n", (*iter)->info.start_row, new_warp);
                }
            }
        }
        topo_queue.pop();
    }

    for (int i = 0; i < handler->warp_schedule[0].size(); i++)
    {
        printf("%d ", handler->warp_schedule[0][i]->info.start_row);
    }

    // printf("??\n");

}

// schedule with row sequence
void schedule_structured2(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    int current_sm = 0;
    int current_warp = 0;
    //ptr_node last_node = NULL;
    int count1 = 0, count2 = 0;
    for (ptr_node i = graph->start_nodes[0]; i != NULL; i = i->locality_node)
    {
        int swarp = get_warp_id(current_sm, current_warp);
        // static schedule in turn
        schedule_node(handler->warp_schedule[swarp], i, swarp);
        if (i->locality_node && std::find(i->child.begin(), i->child.end(), i->locality_node) != i->child.end())
        {
            count1++;
            current_warp = (current_warp + 1) % WARP_NUM_PER_SM;
        }
        else
        {
            count2++;
            current_sm = (current_sm + 1) % SM_NUM;
            current_warp = 0;
        }
    }

}

void schedule_subwarp(ptr_handler handler, int subwarp_size)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
    }

    // int count = 0;

    int old_level = 0;

    int current_warp = 0;
    int current_subwarp_size = 0;

    // topological sort
    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();

        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            (*iter)->in_degree_tmp++;
            if ((*iter)->in_degree_tmp == (*iter)->in_degree)
            {
                topo_queue.push(*iter);
                (*iter)->in_degree_tmp = 0;
            }
        }

        //printf("current node %d level %d\n", current_node->info.start_row, current_node->topo_level);

        // static schedule in turn
        // subwarps must handle tasks with no inter-dependency
        if (current_subwarp_size < subwarp_size && old_level == current_node->topo_level)
        {
            schedule_node(handler->warp_schedule[current_warp], current_node, current_warp);
            current_subwarp_size++;
        }
        else
        {
            current_warp = (current_warp + 1) % WARP_NUM;
            schedule_node(handler->warp_schedule[current_warp], current_node, current_warp);
            current_subwarp_size = 1;
        }

        old_level = current_node->topo_level;

        topo_queue.pop();

    }

    // for (int i = 0; i <= current_warp; i++)
    // {
    //     printf("warp %d tasks %d:\n", i, handler->warp_schedule[i].size());
    //     for (int j = 0; j < handler->warp_schedule[i].size(); j++)
    //         printf("task %d start %d end %d\n", j,
    //         handler->warp_schedule[i][j]->info.start_row, handler->warp_schedule[i][j]->info.end_row);
    // }

}

void schedule_subwarp_balance(ptr_handler handler, int subwarp_size)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
    }

    // int count = 0;

    int old_level = 0;

    int current_warp = 0;
    int current_subwarp_size = 0;

    // topological sort
    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();

        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            (*iter)->in_degree_tmp++;
            if ((*iter)->in_degree_tmp == (*iter)->in_degree)
            {
                topo_queue.push(*iter);
                (*iter)->in_degree_tmp = 0;
            }
        }

        // static schedule in turn
        // subwarps must handle tasks with no inter-dependency
        if (current_subwarp_size < subwarp_size && old_level == current_node->topo_level)
        {
            schedule_node(handler->warp_schedule[current_warp], current_node, current_warp);
            current_subwarp_size++;
        }
        else
        {
            current_warp = (current_warp + 1) % WARP_NUM;
            schedule_node(handler->warp_schedule[current_warp], current_node, current_warp);
            current_subwarp_size = 1;
        }

        old_level = current_node->topo_level;

        topo_queue.pop();

    }

}

bool node_cmp(ptr_node a, ptr_node b)
{
    if (a->topo_level != b->topo_level)
        return a->topo_level < b->topo_level;
    else
        return a->info.start_row < b->info.start_row;
}

void inside_window_simple(ptr_handler handler, vector<ptr_node> &node_vec)
{
    static int local_warp[WARP_NUM];
    static int current_sm = 0;

    for (auto iter = node_vec.begin(); iter != node_vec.end(); iter++)
    {
        int current_warp = get_warp_id(current_sm, local_warp[current_sm]);
        schedule_node(handler->warp_schedule[current_warp], *iter, current_warp);
        // current_warp = (current_warp + 1) % WARP_NUM;

        local_warp[current_sm] = (local_warp[current_sm] + 1) % WARP_NUM_PER_BLOCK;
        if (local_warp[current_sm] % 8 == 0)
        {
            current_sm = (current_sm + 1) % SM_NUM;
        }
    }
}

void simple_schedule_window(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
    }

    int window_size = 20480;

    int count = 0; 

    vector<ptr_node> window_nodes;
    for (ptr_node i = graph->start_nodes[0]; i != NULL; i = i->locality_node)
    {
        window_nodes.push_back(i);
        count++;

        if (count == window_size)
        {
            std::sort(window_nodes.begin(), window_nodes.end(), node_cmp);
            inside_window_simple(handler, window_nodes);
            window_nodes.clear();
            count = 0;
        }
    }
    if (count > 0)
    {
        sort(window_nodes.begin(), window_nodes.end(), node_cmp);
        inside_window_simple(handler, window_nodes);
    }
}

void simple_schedule_sequential(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    int node_num = 0;

    int current_warp = 0;
    for (ptr_node i = graph->start_nodes[0]; i != NULL; i = i->locality_node)
    {
        // static schedule in turn
        schedule_node(handler->warp_schedule[current_warp], i, current_warp);
        current_warp = (current_warp + 1) % WARP_NUM;

        // printf("%d %d %d %d\n", 
        // node_num, i->info.start_row, i->info.end_row, i->info.elim);
        // node_num++;
    }

    //printf("node_num %d nnz_num %d\n", node_num, graph->global_node);

}

void no_schedule_sequential(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    handler->no_schedule_info = (node_info*)malloc(graph->global_node * sizeof(node_info));
    
    int count = 0;
    for (ptr_node i = graph->start_nodes[0]; i != NULL; i = i->locality_node)
    {
        handler->no_schedule_info[count] = i->info;
        count++;
    }

    cudaMalloc(&handler->no_schedule_info_d, graph->global_node * sizeof(node_info));
    cudaMemcpy(handler->no_schedule_info_d, handler->no_schedule_info,
    graph->global_node * sizeof(node_info), cudaMemcpyHostToDevice);
}

void sptrsv_schedule(ptr_handler handler, SCHEDULE_STRATEGY strategy)
{
    handler->sched_s = strategy;

    int subwarp_size = MAX_SUBWARP;
    handler->subwarp_size = subwarp_size;

    if (strategy == SIMPLE)
        simple_schedule(handler);
    else if (strategy == SIMPLE_INTERLEAVED)
        simple_schedule_interleaved(handler);
    else if (strategy == WORKLOAD_BALANCE)
        simple_schedule_workload_balance2(handler);
    else if (strategy == WARP_LOCALITY)
        simple_schedule_warp_locality2(handler);
    else if (strategy == BALANCE_AND_LOCALITY)
        simple_schedule_workload_locality2(handler);
    else if (strategy == SEQUENTIAL)
        simple_schedule_sequential(handler);
    else if (strategy == STRUCTURED)
        schedule_structured(handler);
    else if (strategy == WINDOW)
        simple_schedule_window(handler);
    else if (strategy == SEQUENTIAL2)
    {
        no_schedule_sequential(handler);
        return;
    }
    else if (strategy == SUBWARP)
        schedule_subwarp(handler, subwarp_size);
    else
        printf("Scheduling strategy error!\n");

    if (SHOW_USER_IMBALANCE) show_imbalance(handler);

    //if (strategy != SUBWARP) sync_elimination(handler);
    
    if (strategy != SUBWARP)
        transfer_to_2D(handler);
    else
        transfer_to_2D_subwarp(handler, subwarp_size);

    if (strategy != SUBWARP)
        schedule_info_hosttodevice(handler);
    else
        schedule_info_hosttodevice_subwarp(handler);

}