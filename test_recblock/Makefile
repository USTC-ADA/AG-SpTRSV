#compilers
CC=nvcc

#GLOBAL_PARAMETERS
VALUE_TYPE_DOUBLE = double
VALUE_TYPE_FLOAT = float

#CUDA_PARAMETERS
#-Xcompiler -openmp 
NVCC_FLAGS = -O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_86,code=sm_86

#ENVIRONMENT_PARAMETERS
#CUDA_INSTALL_PATH = /usr/local/cuda-11.1

#includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/include

#libs
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
#-lgomp 
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse
LIBS = $(CUDA_LIBS)

#options
#OPTIONS = -std=c99

make:
	$(CC) $(NVCC_FLAGS)  main.cu -o sptrsv-double $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE_DOUBLE)
#$(CC) $(NVCC_FLAGS) main.cu -o sptrsv-float $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE_FLOAT)

.PHONY:

clean:
	rm sptrsv-double
