if (row_st + 1 == row_ed)
{
    int idx_st = csrRowPtr[row_st];
    int idx_ed = csrRowPtr[row_st + 1];

    T diag_inv = 1.0 / csrValue[idx_ed - 1];
    T rh_value = b[row_st];

    T leftsum = 0;
    for (int idx = idx_st + lane_id; idx < idx_ed - 1; idx += WARP_SIZE)
    {
        int dep_row = csrColIdx[idx];

        T dep_value = csrValue[idx];

        READ_FENCE;

        while (!get_value[dep_row])
        {
            READ_FENCE;
        }
        
        leftsum += dep_value * x[dep_row];
        //leftsum = fma(dep_value, x[dep_row], leftsum);
    }

    //Reduce_buf[tid] = leftsum;

    // warp-level reduce
    // if (idx_ed - idx_st > WARP_REDUCE_THRESH)
    //int reduce_max = (idx_ed - idx_st - 1 > WARP_SIZE)? WARP_SIZE: idx_ed - idx_st - 1;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        if (offset <= idx_ed - idx_st - 1)
            leftsum += __shfl_down_sync(0xffffffff, leftsum, offset);
    }
    // else
    // {
    //     // leftsum = 0;
    //     // if (!lane_id)
    //     //     for (int t = 0; t < idx_ed - idx_st; t++)
    //     //         leftsum += Reduce_buf[wid + t];
    // }
    
    if (!lane_id)
    {
        x[row_st] = (rh_value - leftsum) * diag_inv;
        WRITE_FENCE;
        get_value[row_st] = 1;
    }
}
// else if (using_shared_mem)
// {
//     //for (int row_iter = row_st; row_iter < row_ed; row_iter += WARP_SIZE)
//     int row_iter = row_st;
//     {
//         int idx;
//         int iter_ed = (row_iter + WARP_SIZE < row_ed)? row_iter + WARP_SIZE: row_ed;
//         int idx_ed = csrRowPtr[iter_ed];
//         int buf_st = wid * WARP_SIZE * BUF_SIZE;

//         //if (!lane_id) printf("???\n");

//         for (idx = csrRowPtr[row_iter] + lane_id; idx < idx_ed; idx += WARP_SIZE)
//         {
//             int buf_idx = buf_st + idx - csrRowPtr[row_iter];
//             ColIdx_buf[buf_idx] = csrColIdx[idx];
//             csrValue_buf[buf_idx] = csrValue[idx];
//             //printf("%d %d %d\n", buf_idx, idx, ColIdx_buf[buf_idx]);
//         }

//         int row = row_iter + lane_id;
//         if (row >= row_ed) return;

//         idx = csrRowPtr[row];
//         T rh_value = b[row];
//         T inv_value = 1.0 / csrValue[csrRowPtr[row + 1] - 1];

//         T leftsum = 0;
//         T dep_value;

//         // if (csrRowPtr[row_iter + 1] - 32 > csrRowPtr[row_iter])
//         //     idx = csrRowPtr[row_iter + 1] - 32;

//         while (idx < csrRowPtr[row + 1])
//         {
//             //int dep_row = csrColIdx[idx];
//             int dep_row = ColIdx_buf[buf_st + idx - csrRowPtr[row_st]];

//             if (dep_row == row)
//             {
//                 //x[row_iter] = (b[row_iter] - leftsum) / csrValue[idx];
//                 x[row] = rh_value * inv_value;
//                 WRITE_FENCE;
//                 get_value[row] = 1;
//                 idx++;
//             }

//             //dep_value = csrValue[idx];
//             dep_value = csrValue_buf[buf_st + idx - csrRowPtr[row_st]];
//             //dep_value = 1.0;//

//             if (get_value[dep_row] == 1)
//             {
//                 rh_value -= dep_value * x[dep_row];
//                 idx++;
//                 //dep_row = csrColIdx[idx];
//                 dep_row = ColIdx_buf[buf_st + idx - csrRowPtr[row_st]];
//             }
//             else
//             {
//                 READ_FENCE;
//             }
//         }
//     }
// }
else
{
    for (int row_iter = row_st + lane_id; row_iter < row_ed; row_iter += WARP_SIZE)
    {
        int idx = csrRowPtr[row_iter];
        T rh_value = b[row_iter];
        T inv_value = 1.0 / csrValue[csrRowPtr[row_iter + 1] - 1];

        T leftsum = 0;
        T dep_value;

        // if (csrRowPtr[row_iter + 1] - 32 > csrRowPtr[row_iter])
        //     idx = csrRowPtr[row_iter + 1] - 32;

        while (idx < csrRowPtr[row_iter + 1])
        {
            int dep_row = csrColIdx[idx];

            if (dep_row == row_iter)
            {
                //x[row_iter] = (b[row_iter] - leftsum) / csrValue[idx];
                x[row_iter] = rh_value * inv_value;
                WRITE_FENCE;
                get_value[row_iter] = 1;
                idx++;
            }

            dep_value = csrValue[idx];

            if (get_value[dep_row] == 1)
            {
                rh_value -= dep_value * x[dep_row];
                idx++;
                dep_row = csrColIdx[idx];
            }
            else
            {
                READ_FENCE;
            }
        }
    }
}