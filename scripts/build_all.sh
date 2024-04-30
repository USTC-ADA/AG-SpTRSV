#!/bin/bash

rm -r build
if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake ..
make --quiet
cd ..

cd test
make clean
make all --quiet
cd ..

cd test_cusp
make clean
make all --quiet
cd ..

cd test_recblock
make clean
make --quiet
cd ..

cd matrix
make clean
make --quiet
cd ..
