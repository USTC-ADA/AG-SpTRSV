#!/bin/bash

if [ $# -ne 1 ]; then
    echo "[Usage]: run_cusp.sh {matrix_file_name}"
    exit 1
fi
matrix=$1

cd test
make test_ag

cd ..

test/test_ag -i ${matrix}
