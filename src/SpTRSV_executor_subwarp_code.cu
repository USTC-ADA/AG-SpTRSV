// if (row_st + 1 == row_ed)
// {
    T leftsum = 0;
    T rh_value = 0;
    T diag_inv = 0;
    //if(!global_wid && !sw_id && !sw_off && row_st == 0) printf("%d %d %d\n", row_st, sw_id, current_info.subwarp_size);
    if (sw_id < current_info.subwarp_size)
    {
        int idx_st = csrRowPtr[row_st];
        int idx_ed = csrRowPtr[row_ed];
        //if (idx_ed > idx_st + 100) idx_ed = idx_st + 100;

        rh_value = b[row_st];
        diag_inv = 1.0 / csrValue[idx_ed - 1];

        for (int idx = idx_st + sw_off; idx < idx_ed - 1; idx += sw_size)
        {
            int dep_row = csrColIdx[idx];

            T dep_value = csrValue[idx];

            READ_FENCE;

            while (!get_value[dep_row])
            {
                READ_FENCE;
            }

            leftsum += dep_value * x[dep_row];
            if (row_st == 0) printf("debug %.2f\n", leftsum);
        }
    }

    for (int offset = sw_size / 2; offset > 0; offset /= 2)
        leftsum += __shfl_down_sync(0xffffffff, leftsum, offset);
    
    //if (!sw_off && row_st == 5) printf("row_st %d get_value %d\n", row_st, get_value[row_st]);

    if (sw_id < current_info.subwarp_size && !sw_off)
    {
        
        x[row_st] = (rh_value - leftsum) * diag_inv;
        //__threadfence_block();
        WRITE_FENCE;
        get_value[row_st] = 1;

        //printf("%d x %.3f b %.3f\n", row_st, x[row_st], b[row_st]);
    }

// }
// else
// {
//     // Not implemented
//     printf("Not implemented!\n");
// }