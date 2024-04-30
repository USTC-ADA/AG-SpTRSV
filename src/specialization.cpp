#include "../include/specialization.h"

void code_specialization_warp_level(ptr_node node, FILE* fp, int *csrRowPtr, int *csrColIdx, const char* value_type)
{
    int row = node->info.start_row;

    fprintf(fp, "   leftsum = 0;\n");

    //fprintf(fp, "   if (!lane_id) printf(\"leftsum %%.3f\\n\", leftsum);\n");
    //fprintf(fp, "   if (!lane_id) printf(\"global_wid %%d\\n\", global_wid);\n");
    
    for (int i = csrRowPtr[row]; i < csrRowPtr[row + 1] - 1; i+=WARP_SIZE)
    {
        
        if (i + WARP_SIZE >= csrRowPtr[row + 1] - 1)
            fprintf(fp, "   if (lane_id <= %d) {\n", csrRowPtr[row + 1] - i - 2);

        // wait for dependency
        fprintf(fp, "   dep_row = %s[%d + lane_id];\n", SPE_CSR_IDX, i);

        //fprintf(fp, "   if (!lane_id) printf(\"global_wid %%d dep_row %%d\\n\", global_wid, dep_row);\n");

        fprintf(fp, "   while (!%s[dep_row]) { __threadfence(); }\n", SPE_GET_VALUE);

        // calculate
        fprintf(fp, "   leftsum += %s[%d + lane_id] * %s[dep_row];\n",
        SPE_CSR_VALUE, i, SPE_X_VALUE);

        if (i + WARP_SIZE >= csrRowPtr[row + 1] - 1)
            fprintf(fp, "    }\n");
    }

    // warp-level reduce
    if (csrRowPtr[row] + 1 < csrRowPtr[row + 1])
        fprintf(fp, "   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)\n"
               "   leftsum += __shfl_down_sync(0xffffffff, leftsum, offset);\n");

    // write back

    //fprintf(fp, "   if (!lane_id) printf(\"x %%.3f b %%.3f\\n\", x[%d], b[%d]);\n", row, row);

    fprintf(fp, "   if (!lane_id) {\n"
            "   %s[%d] = (%s[%d] - leftsum) / %s[%d];\n"
            "   __threadfence();\n"
            "   %s[%d] = 1;}\n", SPE_X_VALUE, row, SPE_B_VALUE, row, SPE_CSR_VALUE, csrRowPtr[row + 1] - 1, SPE_GET_VALUE, row);
}

void code_specialization_warp_level_no_getvalue(ptr_node node, FILE* fp, int *csrRowPtr, int *csrColIdx, const char* value_type)
{
    int row = node->info.start_row;

    fprintf(fp, "   leftsum = 0;\n");

    for (int i = csrRowPtr[row]; i < csrRowPtr[row + 1] - 1; i+=WARP_SIZE)
    {
        
        if (i + WARP_SIZE >= csrRowPtr[row + 1] - 1)
            fprintf(fp, "   if (lane_id <= %d) {\n", csrRowPtr[row + 1] - i - 2);

        // wait for dependency
        fprintf(fp, "   dep_row = %s[%d + lane_id];\n", SPE_CSR_IDX, i);

        //fprintf(fp, "   if (!lane_id) printf(\"global_wid %%d dep_row %%d\\n\", global_wid, dep_row);\n");

        fprintf(fp, "   while (!%s[dep_row]) { __threadfence(); }\n", SPE_GET_VALUE);

        // calculate
        fprintf(fp, "   leftsum += %s[%d + lane_id] * %s[dep_row];\n",
        SPE_CSR_VALUE, i, SPE_X_VALUE);

        if (i + WARP_SIZE >= csrRowPtr[row + 1] - 1)
            fprintf(fp, "    }\n");
    }

    // warp-level reduce
    if (csrRowPtr[row] + 1 < csrRowPtr[row + 1])
        fprintf(fp, "   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)\n"
                "   leftsum += __shfl_down_sync(0xffffffff, leftsum, offset);\n");

    // write back

    //fprintf(fp, "   if (!lane_id) printf(\"x %%.3f b %%.3f\\n\", x[%d], b[%d]);\n", row, row);

    fprintf(fp, "   if (!lane_id) {\n"
            "   %s[%d] = (%s[%d] - leftsum) / %s[%d];\n"
            "   __threadfence();\n"
            "   %s[%d] = 1;}\n", SPE_X_VALUE, row, SPE_B_VALUE, row, SPE_CSR_VALUE, csrRowPtr[row + 1] - 1, SPE_GET_VALUE, row);
            
}

void code_specialization_thread_level(node_info &node_info, FILE* fp, int *csrRowPtr, int *csrColIdx, const char* value_type)
{
    
}

void warp_code_specialization(ptr_handler handler, int warp_id, FILE* fp,
        int *csrRowPtr, int *csrColIdx, const char* value_type)
{

    fprintf(fp, "   if (global_wid == %d) {\n", warp_id);

    for (int j = 0; j < handler->warp_schedule[warp_id].size(); j++)
    {
        //node_info current_info = handler->schedule_info[warp_id][j];
        ptr_node current_node = handler->warp_schedule[warp_id][j];

        if (current_node->info.start_row + 1 == current_node->info.end_row)
        {
            code_specialization_warp_level(current_node, fp, csrRowPtr, csrColIdx, value_type);
        }
        else
        {
            //code_specialization_thread_level();
        }
    }

    fprintf(fp, "   }\n");
}

// specialization SpTRSV code for ultimate 
template <typename T>
void SpTRSV_specialization(ptr_handler handler, const char* filename, 
        int *csrRowPtr, int *csrColIdx)
{
    FILE *fp = fopen(filename, "w");

    const char* value_type;
    if (typeid(T) == typeid(float)) value_type = string("float").c_str();

    // include
    fprintf(fp, "#include <cuda.h>\n");
    fprintf(fp, "#include <stdio.h>\n");

    fprintf(fp, "#define BLOCK_NUM %d\n"
            "#define THREAD_NUM_PER_BLOCK %d\n"
            "#define WARP_SIZE %d\n"
            "#define WARP_NUM_PER_BLOCK (THREAD_NUM_PER_BLOCK / WARP_SIZE)\n"
            "#define WARP_NUM (BLOCK_NUM * WARP_NUM_PER_BLOCK)\n", BLOCK_NUM, THREAD_NUM_PER_BLOCK, WARP_SIZE);

    //SpTRSV(int *level, node_info **info, 
    //       const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
    //       const T* b, T* x, int *get_value)
    fprintf(fp, "__global__ void SpTRSV(const int *csrRowPtr, const int *csrColIdx, const %s* csrValue,\n"
                "const %s* b, %s* x, int *get_value) {\n", value_type, value_type, value_type);

    fprintf(fp, "   int bid = blockIdx.x;\n"
            "   int tid = threadIdx.x;\n"
            "   int wid = tid / WARP_SIZE;\n"
            "   int global_wid = bid * WARP_NUM_PER_BLOCK + wid;\n"
            "   int lane_id = tid %% WARP_SIZE;\n");

    fprintf(fp, "   %s leftsum;\n", value_type);
    fprintf(fp, "   int dep_row;\n");

    for (int i = 0; i < WARP_NUM; i++)
    {
        warp_code_specialization(handler, i, fp, csrRowPtr, csrColIdx, value_type);
    }

    fprintf(fp, "}\n");

    // host code
    fprintf(fp, "void spe_SpTRSV_executor(const int *csrRowPtr, const int *csrColIdx,\n"
                "const %s* csrValue, const %s* b, %s* x, int *get_value) {\n", value_type, value_type, value_type);

    fprintf(fp, "SpTRSV<<<%d, %d>>>(csrRowPtr, csrColIdx, csrValue,\n"
                "b, x, get_value);\n", BLOCK_NUM, THREAD_NUM_PER_BLOCK);

    fprintf(fp, "}\n");

}


// instance
template void SpTRSV_specialization<float>(ptr_handler handler, const char* filename, 
        int *csrRowPtr, int *csrColIdx);