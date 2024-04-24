# AG-SpTRSV
An Automatic Framework to Optimize Sparse Triangular Solve on GPUs

## Environmental setup
### Configurations to run AG-SpTRSV
GPU:   NVIDIA Telsa A100, RTX3080 Ti  
CUDA:  11.4  
GCC:   >= 7.5.0  
CMAKE: >= 3.5  
Linux: Ubuntu 18.04
### Extra configurations for the performance model
Python: 3.9  
Necessary Python packages and recommended versions: Pytorch-3.1.12 scikit-learn-1.1.3 scikit-learn numpy-1.23.3 pandas-1.5.3 

## Build up & Execution
### Compile
```
cd scripts
sh build_all.sh
```

### Transform the matrix
1. Download the matrix from SuiteSparse Matrix Collection (<https://sparse.tamu.edu/>). Sample matrices are in ``matrix/matrix_sample``.
2. Transform the general matrix into a triangular matrix. Run ``./matrix/transfer {.mtx file} {output file}``. For example: 
```
./matrix/transfer matrix/matrix_sample/delaunay_n13.mtx matrix/matrix_sample_csr/delaunay_n13.csr
