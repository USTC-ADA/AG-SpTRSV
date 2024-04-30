#!/bin/bash

if [ $# -ne 1 ]; then
    echo "[Usage]: run_recblock.sh {matrix_file_name}"
    exit 1
fi
matrix=$1

cd test_recblock
make test_ag

cd ..

test_recblock/sptrsv-double -d 0 -rhs 1 -lv -1 -forward -mtx $1