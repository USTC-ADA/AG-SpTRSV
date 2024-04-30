#!/bin/bash

if [ $# -ne 2 ]; then
    echo "[Usage]: run_recblock.sh {matrix_list} {out_csv}"
    exit 1
fi
matrix_list=$1
out_csv=$2

cd test_recblock
make

cd ..

matrix_dir=/data/SuiteSparse_triangular_new2/

cat ${matrix_list} | xargs -n1 -I {} \
sh -c "test_recblock/sptrsv-double -d 0 -rhs 1 -lv -1 -forward -mtx ${matrix_dir}{}.csr -csv ${out_csv}"