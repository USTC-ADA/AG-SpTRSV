#include "utils.h"
#include "AG-SpTRSV.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define VALUE_TYPE double
#define VALUE_SIZE 8

int main(int argc, char* argv[])
{
    int input_flag = 0, output_flag = 0;
    char *input_name, *output_name;

    int ch;

    while ((ch = getopt(argc, argv, "i:o:")) != -1)
    {
        switch (ch)
        {
            case 'o':
                output_flag = 1;
                output_name = optarg;
                break;

            case 'i':
                input_flag = 1;
                input_name = optarg;
                break;
        }
    }

    if (input_flag == 0 || output_flag == 0)
    {
        printf("[Usage]: ./info -i {input_filename} -o {output_filename}\n");
        exit(1);
    }

    int m;
    int nnzL;
    int *csrRowPtrL;
    int *csrColIdxL;
    VALUE_TYPE *csrValL;

    read_tri<VALUE_TYPE>(input_name, &m, &nnzL, &csrRowPtrL, &csrColIdxL, &csrValL);

    int layer;
    double parallelism;

    write_matrix_info(output_name, input_name, m, nnzL, csrRowPtrL, csrColIdxL);
}