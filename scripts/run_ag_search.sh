#!/bin/bash

if [ $# -ne 1 ] && [ $# -ne 2 ]; then
    echo "[Usage]: run.sh {matrix_file_name} {output_csv_name | optional}"
    exit 1
fi

matrix=$1

cd test
make test_ag_search

cd ..

if [ $# = 1 ]; then
    ./test/test_ag_search -i ${matrix}
else
    outcsv=$2
    ./test/test_ag_search -i ${matrix} -f ${outcsv}
fi
