NVCC=nvcc
CPP=g++

CFLAG= -Wno-unused-result -Wsign-compare -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -ffunction-sections -pipe -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall
PY_CFLAG=-I/root/anaconda3/envs/pytorch/include/python3.9
PY_LDFLAG=-L/root/anaconda3/envs/pytorch/lib/python3.9/config-3.9-x86_64-linux-gnu -lpython3.9
PYTORCH_LDFLAG=-L/root/anaconda3/envs/pytorch/lib

CUDA_INSTALL_PATH=/usr/local/cuda-11.4
CUDALIB=${CUDA_INSTALL_PATH}/lib64/

LFLAGS=-lSpTRSV -lutils -lcusparse

INCLUDE_DIR=../include
UTILS_DIR=../utils
LIB_DIR=../build

all: main

main: main_model AG_run
	$(CPP) $^ -o $@ -L$(LIB_DIR) $(LFLAGS) $(CFLAG) $(PY_CFLAG) $(PYTORCH_LDFLAG) $(PY_LDFLAG) -lcudart 

main_model: main.cpp
	$(CPP) $^ -c -o $@ -I$(INCLUDE_DIR) -I$(UTILS_DIR) -L$(LIB_DIR) $(LFLAGS) $(CFLAG) $(PY_CFLAG) $(PY_LDFLAG)

AG_run: AG_run.cu
	$(NVCC) $^ -c -o $@ -I$(INCLUDE_DIR) -I$(UTILS_DIR) -L$(LIB_DIR) $(LFLAGS)

.PHONY:

clean:
	rm main main_model AG_run

