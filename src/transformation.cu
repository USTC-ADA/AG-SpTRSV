#include "../include/transformation.h"
#include <stdlib.h>
#include <string.h>
#include <queue>

template <typename T>
void matrix_reorder(ptr_handler handler, int *permutation,
            int *csrRowPtr, int *csrColIdx, T* csrValue)
{
    //printf("%d %d\n", handler->m, handler->nnz);
    int *csrRowPtr_tmp = (int*)malloc((handler->m + 1) * sizeof(int));
    int *csrColIdx_tmp = (int*)malloc(handler->nnz * sizeof(int));
    T *csrValue_tmp = (T*)malloc(handler->nnz * sizeof(T));

    memcpy(csrRowPtr_tmp, csrRowPtr, (handler->m + 1) * sizeof(int));
    memcpy(csrColIdx_tmp, csrColIdx, handler->nnz * sizeof(int));
    memcpy(csrValue_tmp, csrValue, handler->nnz * sizeof(T));

    ptr_graph g = handler->graph;

    int perm_reverse[handler->m];

    int ord_pos = 0;
    for (ptr_node i = g->start_nodes[0]; i != NULL; i = i->locality_node)
    {
        int node_len = i->info.end_row - i->info.start_row;
        
        // start & end pos in original matrix
        int ori_start = i->ori_start;
        int ori_end = i->ori_start + node_len;
        // start & end pos in ordered matrix
        int ord_start = i->info.start_row;

        int idx_len = csrRowPtr_tmp[ori_end] - csrRowPtr_tmp[ori_start];

        int diff = ord_pos - csrRowPtr_tmp[ori_start];

        for (int j = 0; j < node_len; j++)
        {
            permutation[ord_start + j] = ori_start + j;
            perm_reverse[ori_start + j] = ord_start + j;
            csrRowPtr[ord_start + j] = csrRowPtr_tmp[ori_start + j] + diff;
        }

        for (int j = 0; j < idx_len; j++)
        {
            csrColIdx[csrRowPtr[ord_start] + j] = perm_reverse[csrColIdx_tmp[csrRowPtr_tmp[ori_start] + j]];
        }

        //memcpy(csrColIdx + csrRowPtr[ord_start], csrColIdx_tmp + csrRowPtr_tmp[ori_start], idx_len * sizeof(int));
        memcpy(csrValue + csrRowPtr[ord_start], csrValue_tmp + csrRowPtr_tmp[ori_start], idx_len * sizeof(T));

        ord_pos += idx_len;

    }

    free(csrRowPtr_tmp);
    free(csrColIdx_tmp);
    free(csrValue_tmp);
}

template <typename T>
void local_format(ptr_handler handler, int *csrRowPtr, int *csrColIdx, T* csrValue,
int **hybrid_RowPtr, int **hybrid_ValPtr,
int &RowPtr_len, int **hybrid_idx,
int &ValPtr_len, T** hybrid_Value)
{
    int format[handler->m];

    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
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
        
        int start_row = current_node->info.start_row;
        int end_row = current_node->info.end_row;

        if (end_row == start_row + 1)
        {
            current_node->info.format = CSR;
        }
        else
        {
            current_node->info.format = TELL;
        }

        format[start_row] = current_node->info.format;
        for (int i = start_row + 1; i < end_row; i++)
        {
            format[i] = -((int)current_node->info.format);
        }

        topo_queue.pop();
    }

    // format[i] = 1;
    RowPtr_len = 0;
    ValPtr_len = 0;
    int i = 0;

    // calculate format length
    while (i < handler->m)
    {
        int start_row = i;

        basic_format<T> *current_format;
        if (format[i] == CSR)
        {
            current_format = new CSR_format<T>(csrRowPtr[i + 1] - csrRowPtr[i]);
            i++;
        }
        else if (format[i] == TELL)
        {
            i++;
            while (i < handler->m && format[i] == -(int)TELL) i++;
            current_format = new TELL_format<T>(start_row, i - start_row, csrRowPtr, csrColIdx);
        }
        else
        {
            printf("Format not implemented!\n");
            break;
        }
        RowPtr_len += current_format->RowPtr_size();
        ValPtr_len += current_format->ValPtr_size();

        delete current_format;
    }

    *hybrid_RowPtr = (int*)malloc(sizeof(int) * (handler->m + 1));
    *hybrid_idx = (int*)malloc(sizeof(int) * RowPtr_len);
    *hybrid_ValPtr = (int*)malloc(sizeof(int) * (handler->m + 1));
    *hybrid_Value = (T*)malloc(sizeof(T) * ValPtr_len);

    int RowPtr_index = 0;
    int ValPtr_index = 0;

    i = 0;
    while (i < handler->m)
    {
        basic_format<T>* current_format;

        int start_row = i;

        if (format[i] == CSR)
        {
            current_format = new CSR_format<T>(csrRowPtr[i + 1] - csrRowPtr[i]);
            i++;
        }
        else if (format[i] == TELL)
        {
            i++;
            while (i < handler->m && format[i] == -(int)TELL) i++;
            current_format = new TELL_format<T>(start_row, i - start_row, csrRowPtr, csrColIdx);
        }
        else
        {
            printf("Format not implemented!\n");
            break;
        }

        current_format->transform_format(*hybrid_RowPtr, *hybrid_ValPtr,
        *hybrid_idx, *hybrid_Value, RowPtr_index, ValPtr_index,
        start_row, csrRowPtr, csrColIdx, csrValue);

        delete current_format;
    }
}

template void local_format<float>(ptr_handler handler, int *csrRowPtr, int *csrColIdx, float* csrValue,
int **hybrid_RowPtr, int **hybrid_ValPtr,
int &RowPtr_len, int **hybrid_idx,
int &ColPtr_len, float** hybrid_Value);

template void local_format<double>(ptr_handler handler, int *csrRowPtr, int *csrColIdx, double* csrValue,
int **hybrid_RowPtr, int **hybrid_ValPtr,
int &RowPtr_len, int **hybrid_idx,
int &ColPtr_len, double** hybrid_Value);

template void matrix_reorder<float>(ptr_handler handler, int *permutation,
int *csrRowPtr, int *csrColIdx, float* csrValue);

template void matrix_reorder<double>(ptr_handler handler, int *permutation,
int *csrRowPtr, int *csrColIdx, double* csrValue);
