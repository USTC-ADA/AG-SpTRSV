#include "../include/finalize.h"
#include <queue>
#include <cuda.h>

void graph_finalize(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

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
            (*iter)->in_degree--;
            if ((*iter)->in_degree == 0)
            {
                topo_queue.push(*iter);
            }
        }
        topo_queue.pop();
        delete current_node;
    }

}

void schedule_finalize(ptr_handler handler)
{
    if (handler->sched_s != SEQUENTIAL2)
    {
        if (handler->schedule_level != NULL)
        {
            free(handler->schedule_level);
            handler->schedule_level = NULL;
        }
        if (handler->schedule_info != NULL)
        {
            for (int i = 0; i < WARP_NUM; i++)
            {
                free(handler->schedule_info[i]);
            }
            free(handler->schedule_info);
            handler->schedule_info = NULL;
        }
        if (handler->schedule_subwarp_info != NULL)
        {
            for (int i = 0; i < WARP_NUM; i++)
            {
                free(handler->schedule_subwarp_info[i]);
            }
            free(handler->schedule_subwarp_info);
            handler->schedule_subwarp_info = NULL;
        }

        node_info* tmp_schedule_info[WARP_NUM];
        if (handler->schedule_info_d)
        {
            cudaMemcpy(tmp_schedule_info, handler->schedule_info_d, WARP_NUM * sizeof(node_info*), cudaMemcpyDeviceToHost);
            for (int i = 0; i < WARP_NUM; i++)
            {
                cudaFree(tmp_schedule_info[i]);
                handler->warp_schedule[i].clear();
            }
            handler->schedule_info_d = NULL;
        }

        subwarp_info* tmp_subwarp_info[WARP_NUM];
        if (handler->subwarp_info_d)
        {
            cudaMemcpy(tmp_subwarp_info, handler->subwarp_info_d, WARP_NUM * sizeof(subwarp_info*), cudaMemcpyDeviceToHost);
            for (int i = 0; i < WARP_NUM; i++)
            {
                cudaFree(tmp_subwarp_info[i]);
                handler->warp_schedule[i].clear();
            }
            handler->subwarp_info_d = NULL;
        }

        cudaFree(handler->schedule_level_d);
        cudaFree(handler->schedule_info_d);
        cudaFree(handler->subwarp_info_d);
    }
    else
    {
        if (handler->no_schedule_info != NULL)
            free(handler->no_schedule_info);
        cudaFree(handler->no_schedule_info_d);
    }

    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

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
        topo_queue.pop();

        current_node->in_degree_tmp = 0;
        //current_node->topo_level = 0;
    }

}

void SpTRSV_finalize(ptr_handler handler)
{
    schedule_finalize(handler);

    //free the graph, together with the nodes
    if (handler->graph != NULL)
        graph_finalize(handler);

}

void partition_finalize(ptr_anainfo ana)
{
    if (ana->level_partition_map) free(ana->level_partition_map);

    ana->partition_num.clear();
    ana->partition_num.shrink_to_fit();

    //free_info_vec<warpinfo>(ana->winfo_vec);

    ana->row_group_vec.clear();
    ana->row_group_vec.shrink_to_fit();
}

template <typename infotype> 
void free_info_vec(vector<vector<infotype>> &vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        vec[i].clear();
        vec[i].shrink_to_fit();
    }
    vec.clear();
    vec.shrink_to_fit();
}

template <int subwarp_size>
void free_info_d2(void **info_d2, int warp_num)
{
    subwarpinfo<subwarp_size> *tmp_info[warp_num];

    cudaMemcpy(tmp_info, info_d2, warp_num * sizeof(subwarpinfo<subwarp_size>*), cudaMemcpyDeviceToHost);
    for (int i = 0; i < warp_num; i++)
    {
        cudaFree(tmp_info[i]);
    }
    
    cudaFree(info_d2);
}

void schedule_finalize(ptr_anainfo ana, anaparas paras)
{
    free_info_vec<subwarpinfo<1>>(ana->swinfo_vec1);
    free_info_vec<subwarpinfo<2>>(ana->swinfo_vec2);
    free_info_vec<subwarpinfo<4>>(ana->swinfo_vec4);
    free_info_vec<subwarpinfo<8>>(ana->swinfo_vec8);
    free_info_vec<subwarpinfo<16>>(ana->swinfo_vec16);
    free_info_vec<subwarpinfo<32>>(ana->swinfo_vec32);

    if (ana->winfo_d) cudaFree(ana->winfo_d);

    if (ana->winfo_d2)
    {
        int warp_num = ana->warp_num;
        if (paras.subwarp_size == 1) free_info_d2<1>(ana->winfo_d2, warp_num);
        if (paras.subwarp_size == 2) free_info_d2<2>(ana->winfo_d2, warp_num);
        if (paras.subwarp_size == 4) free_info_d2<4>(ana->winfo_d2, warp_num);
        if (paras.subwarp_size == 8) free_info_d2<8>(ana->winfo_d2, warp_num);
        if (paras.subwarp_size == 16) free_info_d2<16>(ana->winfo_d2, warp_num);
        if (paras.subwarp_size == 32) free_info_d2<32>(ana->winfo_d2, warp_num);
        cudaFree(ana->winfo_n2);
    }

    ana->winfo_d = NULL;
    ana->winfo_d2 = NULL;
    ana->winfo_n2 = NULL;

    //if (ana->winfo_h) free(ana->winfo_h);
    ana->winfo_n.clear();
    ana->winfo_n.shrink_to_fit();
    ana->winfo_multilevel.clear();
    ana->winfo_multilevel.shrink_to_fit();
}

void matrix_level_finalize(ptr_anainfo ana)
{
    if (ana->row_level) free(ana->row_level);
    if (ana->level_rownum) free(ana->level_rownum);
    if (ana->get_value) cudaFree(ana->get_value);
}

void SpTRSV_finalize(ptr_anainfo ana, anaparas paras)
{
    schedule_finalize(ana, paras);
    partition_finalize(ana);
    matrix_level_finalize(ana);
}