#include "../include/format_def.h"

template <typename T>
int basic_format<T> :: RowPtr_size()
{
    return this->RowPtr_len;
}

template <typename T>
int basic_format<T> :: ValPtr_size()
{
    return this->ValPtr_len;
}

template <typename T>
CSR_format<T> :: CSR_format(int nnz)
{
    this->RowPtr_len = nnz;
    this->ValPtr_len = nnz;
}

template <typename T>
void CSR_format<T> :: transform_format(int *RowPtr, int *ValPtr,
    int *idx, T *value, int &RowPtr_index, int &ValPtr_index,
    int m, int *csrRowPtr, int *csrColIdx, T *csrValue)
{
    RowPtr[m] = RowPtr_index;
    RowPtr[m + 1] = RowPtr_index + this->RowPtr_len;

    ValPtr[m] = ValPtr_index;
    ValPtr[m + 1] = ValPtr_index + this->ValPtr_len;
    
    for (int i = csrRowPtr[m]; i < csrRowPtr[m + 1]; i++)
    {
        idx[RowPtr_index++] = csrColIdx[i];
    }
    for (int i = csrRowPtr[m]; i < csrRowPtr[m + 1]; i++)
    {
        value[ValPtr_index++] = csrValue[i];
    }

}

template <typename T>
TELL_format<T> :: TELL_format(int m, int num_row_input, int *csrRowPtr, int *csrColIdx)
{
    num_row = num_row_input;
    max_row_nnz = 0;

    for (int i = m; i < m + num_row; i++)
    {
        int current_row_nnz = csrRowPtr[i + 1] - csrRowPtr[i] - 1;
        if (current_row_nnz > max_row_nnz)
            max_row_nnz = current_row_nnz;
    }
    this->RowPtr_len = max_row_nnz * num_row;
    this->ValPtr_len = (max_row_nnz + 1) * num_row;
}

template <typename T>
void TELL_format<T> :: transform_format(int *RowPtr, int *ValPtr,
    int *idx, T *value, int &RowPtr_index, int &ValPtr_index,
    int m, int *csrRowPtr, int *csrColIdx, T *csrValue)
{
    RowPtr[m] = RowPtr_index;
    ValPtr[m] = ValPtr_index;
    for (int i = m + 1; i < m + num_row; i++)
    {
        RowPtr[i] = -1;
        ValPtr[i] = -1;
    }
    for (int i = 0; i < num_row; i++)
    {
        int row = m + i;
        for (int j = 0; j < csrRowPtr[row + 1] - csrRowPtr[row] - 1; j++)
        {
            idx[RowPtr_index + i + j * num_row] = csrColIdx[csrRowPtr[row] + j];
            value[ValPtr_index + i + j * num_row] = csrValue[csrRowPtr[row] + j];
        }
        for (int j = csrRowPtr[row + 1] - csrRowPtr[row] - 1; j < max_row_nnz; j++)
        {
            idx[RowPtr_index + i + j * num_row] = 0;
            value[ValPtr_index + i + j * num_row] = 0;
        }
        //idx[RowPtr_index + i + max_row_nnz * num_row] = csrColIdx[csrRowPtr[i + 1] - 1];
        //printf("m %d row %d csrValue %.2f index %d\n", m, row, csrValue[csrRowPtr[row + 1] - 1], ValPtr_index + i + max_row_nnz * num_row);
        value[ValPtr_index + i + max_row_nnz * num_row] = csrValue[csrRowPtr[row + 1] - 1];
    }

    RowPtr_index += max_row_nnz * num_row;
    ValPtr_index += (max_row_nnz + 1) * num_row;

    RowPtr[m + num_row] = RowPtr_index;
    ValPtr[m + num_row] = ValPtr_index;
    
}
