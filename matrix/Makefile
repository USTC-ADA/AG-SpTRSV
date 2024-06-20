#!/bin/bash

CC=g++
NVCC=nvcc

INCLUDE_DIR=../include
UTILS_DIR=../utils
LIB_DIR=../build

all: transfer info

transfer: transfer_to_tri.cpp
	$(CC) $^ -o $@ -fpermissive -I${UTILS_DIR} -L${LIB_DIR} -lutils

info: matrix_info.cpp
	$(NVCC) $^ -o $@ -I$(INCLUDE_DIR) -I$(UTILS_DIR) -L$(LIB_DIR) $(LFLAGS) -lSpTRSV -lutils

.PHONY:

clean:
	rm transfer info