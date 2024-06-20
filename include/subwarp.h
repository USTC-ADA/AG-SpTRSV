#ifndef SPTSV_SUBWARP__
#define SPTSV_SUBWARP__

template <int SUBWARP_SIZE>
struct subwarpinfo
{
    int row_st[SUBWARP_SIZE];
    int row_ed[SUBWARP_SIZE];
    // bit-wise information
    // bit 0: whether the rows have multiple levels
    // bit 1: whether use shared memory (1 for cannot use)
    // bit 2: whether all rows are in the same level
    unsigned int info;
    // max number of rows of the subwarps
    // int row_max;

    subwarpinfo()
    {
        for (int i = 0; i < SUBWARP_SIZE; i++)
            row_st[i] = -1;
        info = 0;
        // row_max = 0;
    }

    // get current available position
    int get_pos()
    {
        for (int i = 0; i < SUBWARP_SIZE; i++)
            if (row_st[i] == -1)
            {
                return i;
            }
        return -1;
    }

    void append(int row_st_in, int pos)
    {
        // if (pos > 0 && pos < SUBWARP_SIZE) row_st[pos] = row_st_in;
        row_st[pos] = row_st_in;
    }

    void append(int row_st_in, int row_ed_in, unsigned int info_in, int pos)
    {
        // if (pos > 0 && pos < SUBWARP_SIZE) row_st[pos] = row_st_in;
        row_st[pos] = row_st_in;
        row_ed[pos] = row_ed_in;
        info = info | info_in;
        // if (row_ed_in - row_st_in > row_max) row_max = row_ed_in - row_st_in;
    }

};

#define AG_SCHEDULE(FUNC_NAME, ...) { \
if (paras.subwarp_size == 1) \
    FUNC_NAME<1>(__VA_ARGS__, ana->swinfo_vec1); \
if (paras.subwarp_size == 2) \
    FUNC_NAME<2>(__VA_ARGS__, ana->swinfo_vec2); \
if (paras.subwarp_size == 4) \
    FUNC_NAME<4>(__VA_ARGS__, ana->swinfo_vec4); \
if (paras.subwarp_size == 8) \
    FUNC_NAME<8>(__VA_ARGS__, ana->swinfo_vec8); \
if (paras.subwarp_size == 16) \
    FUNC_NAME<16>(__VA_ARGS__, ana->swinfo_vec16); \
if (paras.subwarp_size == 32) \
    FUNC_NAME<32>(__VA_ARGS__, ana->swinfo_vec32); \
}

// #include "common.h"
// #define SUBWARP_SIZE 8

// #define info_cont(infoname, SUBWARP_SIZE) struct infoname \
// { \
//     int row_st[SUBWARP_SIZE]; \
//     infoname() \
//     { \
//         for (int i = 0; i < SUBWARP_SIZE; i++) \
//             row_st[i] = -1; \
//     } \
//     void append(int row_st_in, int pos) \
//     { \
//         row_st[pos] = row_st_in; \
//     } \
// };

// // #define infoname subwarpinfo_2
// // #define SUBWARP_SIZE 2
// info_cont(subwarpinfo_2, 2)
// info_cont(subwarpinfo_4, 4)
// info_cont(subwarpinfo_8, 8)
// info_cont(subwarpinfo_16, 16)
// #undef infoname
// #undef SUBWARP_SIZE

// #define infoname subwarpinfo_4
// #define SUBWARP_SIZE 4
// info_cont(infoname, SUBWARP_SIZE)
// #undef infoname
// #undef SUBWARP_SIZE

// #define infoname subwarpinfo_8
// #define SUBWARP_SIZE 8
// info_cont(infoname, SUBWARP_SIZE)
// #undef infoname
// #undef SUBWARP_SIZE

// #define infoname subwarpinfo_16
// #define SUBWARP_SIZE 16
// info_cont(infoname, SUBWARP_SIZE)
// #undef infoname
// #undef SUBWARP_SIZE

// template <int SUBWARP_SIZE> class subwarpinfo<2>;
// template <4> class subwarpinfo;
// template <8> class subwarpinfo;
// template <16> class subwarpinfo;

#endif