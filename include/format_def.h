#ifndef FORMAT_DEF__
#define FORMAT_DEF__
#include <stdio.h>

template <typename T>
class basic_format
{
public:
    int RowPtr_size();
    int ValPtr_size();

    basic_format() {};

    virtual void transform_format(int *RowPtr, int *ValPtr,
    int *idx, T *value, int &RowPtr_index, int &ValPtr_index,
    int m, int *csrRowPtr, int *csrColIdx, T *csrValue) {};

protected:
    int RowPtr_len;
    int ValPtr_len;
};

template <typename T>
class CSR_format: public basic_format<T>
{
public:
    CSR_format(int nnz);

    virtual void transform_format(int *RowPtr, int *ValPtr,
    int *idx, T *value, int &RowPtr_index, int &ValPtr_index,
    int m, int *csrRowPtr, int *csrColIdx, T *csrValue);
};

template <typename T>
class TELL_format: public basic_format<T>
{
    int num_row;
    // max number of nnz in a row, except for diagonal elements
    int max_row_nnz;

public:
    TELL_format(int m, int num_row_input, int *csrRowPtr, int *csrColIdx);

    virtual void transform_format(int *RowPtr, int *ValPtr,
    int *idx, T *value, int &RowPtr_index, int &ValPtr_index,
    int m, int *csrRowPtr, int *csrColIdx, T *csrValue);
};

template class basic_format<float>;
template class basic_format<double>;

template class CSR_format<float>;
template class CSR_format<double>;

template class TELL_format<float>;
template class TELL_format<double>;

#endif