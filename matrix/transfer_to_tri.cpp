#include "utils.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

#define VALUE_TYPE float

int main(int argc, char* argv[])
{

    if (argc != 3)
    {
        printf("[Usage]: ./transfer {input_matrix_file(.mtx)} {output_matrix_file(.csr)}\n");
        exit(1);
    }

    string input_filename = argv[1];

    printf("Reading: %s\n", input_filename.c_str());

    int error;

    // Original matrix A;
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

    error = read_mtx((char*)input_filename.c_str(), &m, &n, &nnzA, &csrRowPtrA, &csrColIdxA, &csrValA);
    if (error)
    {
        printf("Read error occurs!\n");
        exit(-1);
    }

    // Triangular matrix L;
    int nnzL;
    int *csrRowPtrL;
    int *csrColIdxL;
    VALUE_TYPE *csrValL;

    change2trian(m, nnzA, csrRowPtrA, csrColIdxA, csrValA, 
    &nnzL, &csrRowPtrL, &csrColIdxL, &csrValL);

    srand(time(NULL));
    for (int i = 0; i < nnzL; i++)
    {
        csrValL[i] = 1.0 * rand() / RAND_MAX + 1.0;
    }

    //output to file
    string output_filename = argv[2];

    printf("Writing: %s\n", output_filename.c_str());
    write_tri((char*)output_filename.c_str(), m, nnzL, csrRowPtrL, csrColIdxL, csrValL);

    if (error)
    {
        printf("Write error occurs!\n");
        exit(-1);
    }

    FILE *flist = fopen("matrix_tri_list", "a");
    fprintf(flist, "%s\n", argv[1]);

}