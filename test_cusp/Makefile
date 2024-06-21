CC=nvcc

FLAG=-O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_86,code=sm_86

LFLAGS=-lSpTRSV -lutils -lcusparse

INCLUDE_DIR=../include
UTILS_DIR=../utils
LIB_DIR=../build

all: test_cusp

test_cusp: main_cusp.cu
	$(CC) $^ $(FLAG) -o $@ -I$(INCLUDE_DIR) -I$(UTILS_DIR) -L$(LIB_DIR) $(LFLAGS)

.PHONY:

clean:
	rm test_cusp