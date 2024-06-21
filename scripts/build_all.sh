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
make clean --quiet
make all --quiet
cd ..

cd test_cusp
make clean --quiet
make all --quiet
cd ..

cd test_recblock
make clean --quiet
make --quiet
cd ..

cd matrix
make clean --quiet
make --quiet
cd ..
