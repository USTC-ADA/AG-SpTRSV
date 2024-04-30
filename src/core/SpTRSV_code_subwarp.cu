// if (!multi_level)
// {
int row_st = current_info.row_st[sw_id];
int row_ed = current_info.row_ed[sw_id];

//if (row_st != -1) printf("wid %d row %d %d\n", global_wid, row_st, row_ed);

int multi_level = current_info.info & 0x1;
//int multi_level = 0;
int shared_mem  = (current_info.info & 0x2) == 0;

int idx_st;
int idx_ed;

if (!multi_level)
{
    if (row_st < 0) continue;
    //if (!sw_off) printf("row num %d nnz num %d %d\n", row_ed - row_st, csrRowPtr[row_ed] - csrRowPtr[row_st], sw_size);
    
    for (int row = row_st; row < row_ed; row++)
    {
        unsigned mask = __activemask();

        T rh_value = 0;
        T leftsum = 0;
        T diag_inv = 0;

        if (row != -1)
        {
            idx_st = csrRowPtr[row];
            idx_ed = csrRowPtr[row + 1];

            rh_value = b[row];
            diag_inv = 1.0 / csrValue[idx_ed - 1];
            
            for (int idx = idx_st + sw_off; idx < idx_ed - 1; idx += sw_size)
            {
                int dep_row = csrColIdx[idx];

                T dep_value = csrValue[idx];

                READ_FENCE_TEMPLATE;

                while (!READ_FLAG_TEMPLATE)
                {
                    READ_FENCE_TEMPLATE;
                }

                leftsum += dep_value * x[dep_row];
                //if (row_st == 0) printf("debug %.2f\n", leftsum);
            }
        }

        for (int offset = WARP_SIZE / (subwarp_size) / 2; offset > 0; offset /= 2)
            leftsum += __shfl_down_sync(mask, leftsum, offset);

        if (!sw_off && row != -1)
        {
            //rh_value = b[row];
            //diag_inv = 1.0 / csrValue[idx_ed - 1];
            x[row] = (rh_value - leftsum) * diag_inv;

            WRITE_FLAG_TEMPLATE;
        }
    }
}
else
{
    // if (!lane_id) printf("haha multi-level!\n");
    if (shared_mem)
    {
        // if (!lane_id) printf("haha shared memory!\n");
        for (int sw_iter = 0; sw_iter < sw_num; sw_iter++)
        {
            if (current_info.row_st[sw_iter] < 0) break;
            // if (!lane_id) printf("row_st %d row_ed %d\n", current_info.row_st[sw_iter],
            // current_info.row_ed[sw_iter]);

            int buf_st = wid * WARP_SIZE * BUF_SIZE + sw_iter * sw_size * BUF_SIZE;
            for (int idx = csrRowPtr[current_info.row_st[sw_iter]] + lane_id; idx < csrRowPtr[current_info.row_ed[sw_iter]]; idx += WARP_SIZE)
            {
                int buf_idx = buf_st + idx - csrRowPtr[current_info.row_st[sw_iter]];
                ColIdx_buf[buf_idx] = csrColIdx[idx];
                csrValue_buf[buf_idx] = csrValue[idx];
            }
        }

        if (row_st < 0) continue;

        int buf_st = wid * WARP_SIZE * BUF_SIZE + sw_id * sw_size * BUF_SIZE;
        int idx_st = csrRowPtr[current_info.row_st[sw_id]];

        int row = row_st + sw_off;
        
        while (row < row_ed)
        {
            int idx = csrRowPtr[row];
            T rh_value = b[row];
            //T inv_value = 1.0 / csrValue[csrRowPtr[row + 1] - 1];
            T inv_value = 1.0 / csrValue_buf[buf_st + csrRowPtr[row + 1] - 1 - idx_st];

            T dep_value;

            while (idx < csrRowPtr[row + 1])
            {
                int dep_row = ColIdx_buf[buf_st + idx - idx_st];

                dep_value = csrValue_buf[buf_st + idx - idx_st];

                //if (sw_off <= 5) printf("sw_off %d dep_row %d dep_value %.2f\n", sw_off, dep_row, dep_value);

                READ_FENCE_TEMPLATE;

                if (READ_FLAG_TEMPLATE)
                {
                    rh_value -= dep_value * x[dep_row];
                    idx++;
                    dep_row = csrColIdx[idx];
                }

                if (dep_row == row)
                {
                    x[row] = rh_value * inv_value;
                    WRITE_FLAG_TEMPLATE;
                    idx++;
                    continue;
                }
            }

            row += WARP_SIZE / subwarp_size;
        }

    }
    else
    {
        if (row_st < 0) continue;
        int row = row_st + sw_off;
        
        while (row < row_ed)
        {
            int idx = csrRowPtr[row];
            T rh_value = b[row];
            T inv_value = 1.0 / csrValue[csrRowPtr[row + 1] - 1];

            T dep_value;

            while (idx < csrRowPtr[row + 1])
            {
                int dep_row = csrColIdx[idx];

                dep_value = csrValue[idx];

                READ_FENCE_TEMPLATE;

                if (READ_FLAG_TEMPLATE)
                {
                    rh_value -= dep_value * x[dep_row];
                    idx++;
                    dep_row = csrColIdx[idx];
                }

                if (dep_row == row)
                {
                    x[row] = rh_value * inv_value;
                    WRITE_FLAG_TEMPLATE;
                    idx++;
                    continue;
                }
            }

            row += WARP_SIZE / subwarp_size;
        }
    }
}
// }
// else
// {
//     T leftsum = 0;
//     T rh_value = 0;
//     T diag_inv = 0;

//     if (row != -1)
//     {
//         idx_st = csrRowPtr[row];
//         idx_ed = csrRowPtr[row + 1];

//         for (int idx = idx_st + sw_off; idx < idx_ed - 1; idx += sw_size)
//         {
//             int dep_row = csrColIdx[idx];

//             T dep_value = csrValue[idx];

//             // skip the inter-dependent rows
//             if (dep_row >= row) break;

//             while (!READ_FLAG_TEMPLATE)
//             {
//                 READ_FENCE_TEMPLATE;
//             }

//             leftsum += dep_value * x[dep_row];
//             //if (row_st == 0) printf("debug %.2f\n", leftsum);
//         }
//     }

//     for (int offset = sw_size / 2; offset > 0; offset /= 2)
//         leftsum += __shfl_down_sync(0xffffffff, leftsum, offset);

//     if (!sw_off && row != -1)
//     {
//         __shared__ int 
//         while (a != )
//         {

//         }
//     }
// }
