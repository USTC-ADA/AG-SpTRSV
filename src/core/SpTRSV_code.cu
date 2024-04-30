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

        while (!READ_FLAG_TEMPLATE)
        {
            READ_FENCE_TEMPLATE;
        }
        
        leftsum += dep_value * x[dep_row];
    }


    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        if (offset <= idx_ed - idx_st - 1)
            leftsum += __shfl_down_sync(0xffffffff, leftsum, offset);
    }

    int row = row_st;
    if (!lane_id)
    {
        x[row_st] = (rh_value - leftsum) * diag_inv;
        WRITE_FLAG_TEMPLATE;
    }
}
else if (using_shared_mem)
{
    // printf("???\n");

    int row_iter = row_st;
    {
        int idx;
        int iter_ed = (row_iter + WARP_SIZE < row_ed)? row_iter + WARP_SIZE: row_ed;
        int idx_ed = csrRowPtr[iter_ed];
        int buf_st = wid * WARP_SIZE * BUF_SIZE;

        //if (!lane_id) printf("hahaha row %d %d %d\n", row_st + lane_id, csrRowPtr[row_iter], idx_ed);

        for (idx = csrRowPtr[row_iter] + lane_id; idx < idx_ed; idx += WARP_SIZE)
        {
            int buf_idx = buf_st + idx - csrRowPtr[row_iter];
            ColIdx_buf[buf_idx] = csrColIdx[idx];
            csrValue_buf[buf_idx] = csrValue[idx];
        }

        int row = row_iter + lane_id;
        //if (!lane_id) printf("hahaha row %d %d\n", row, row_ed);
        if (row >= row_ed) return;

        idx = csrRowPtr[row];
        T rh_value = b[row];
        T inv_value = 1.0 / csrValue[csrRowPtr[row + 1] - 1];

        // if (!lane_id) printf("calc %f %f %d %f\n", 
        // rh_value, csrValue[0], csrRowPtr[row + 1] - 1, csrValue[csrRowPtr[row + 1] - 1]);

        T leftsum = 0;
        T dep_value;

        while (idx < csrRowPtr[row + 1])
        {
            int dep_row = ColIdx_buf[buf_st + idx - csrRowPtr[row_st]];

            dep_value = csrValue_buf[buf_st + idx - csrRowPtr[row_st]];

            READ_FENCE_TEMPLATE;

            if (READ_FLAG_TEMPLATE)
            {
                rh_value -= dep_value * x[dep_row];
                idx++;
                dep_row = ColIdx_buf[buf_st + idx - csrRowPtr[row_st]];
            }

            if (dep_row == row)
            {
                x[row] = rh_value * inv_value;
                WRITE_FLAG_TEMPLATE;
                idx++;
                continue;
            }
        }
    }
    //if (!lane_id) printf("warp_id %d row_st %d row_ed %d done!\n", global_wid, row_st, row_ed);
}
else
{
    //for (int row_iter = row_st + lane_id; row_iter < row_ed; row_iter += WARP_SIZE)
    int row_iter = row_st + lane_id;
    if (row_iter >= row_ed) return;
    {
        int row = row_iter;
        int idx = csrRowPtr[row_iter];
        T rh_value = b[row_iter];
        T inv_value = 1.0 / csrValue[csrRowPtr[row_iter + 1] - 1];

        //T leftsum = 0;
        T dep_value;

        while (idx < csrRowPtr[row_iter + 1])
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
            // else
            // {
            //     READ_FENCE_TEMPLATE;
            // }

            if (dep_row == row_iter)
            {
                x[row_iter] = rh_value * inv_value;
                WRITE_FLAG_TEMPLATE;
                idx++;
                continue;
            }
        }
    }
}