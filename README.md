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

## Quick start
### Setup and compile
Before compiling AG-SpTRSV, set `SM_NUM` and `BLOCK_NUM` in `include/GPU_setup.h` as the number of multi-processors on the running GPU. Then compile with
```
sh scripts/build_all.sh
```

### Transform the matrix
1. Download the matrix from SuiteSparse Matrix Collection (<https://sparse.tamu.edu/>). Sample matrices are in ``matrix/matrix_sample``.
2. Transform the general matrix into a triangular matrix. Run ``./matrix/transfer {.mtx file} {output file}``. For example: 
```
./matrix/transfer matrix/matrix_sample/delaunay_n13.mtx matrix/matrix_sample_csr/delaunay_n13.csr
```

### Run AG-SpTRSV with manually specified strategies

1. Modify Line 151 in code file ``test/main_ag.cu``. Strategies are provided in ``include/common.h``. Users can change the following parameters:

The size of the thread_block `tb_size` (which influences *nw*).

The number of subwarps per warp `subwarp_size` (*ng*).

Recommended strategies are listed as follows:

Three strategies in Transform stage (*row_s*):

| Transform Strategy  (PREPROCESSING_STRATEGY) | Description | Parameter(row_alpha) | ID |
| ------------------------------------------ | ----------- | --------- | -- |
| ROW_BLOCK | Merge nodes with every $rb$ rows | $rb$ | 0 |
| ROW_BLOCK_THRESH | Merge nodes until one row has more than $rb$ nnz or there are 32 nodes | $rb$ | 1 |
| ROW_BLOCK_AVG | Merge nodes if the average nnz of 32 rows is more that $rb$ | $rb$ | 2 |

Two strategies for the order of scheduling (*level_ps*):

| Scheduling Order (LEVEL_PART_STRATEGY) | Description | ID |
| ------------------------------------- | ----------- | -- |
| LEVEL_WISE | Schedule nodes in the level-set order | 0 |
| ROW_WISE   | Schedule nodes in the order of rows | 1 |

Two heuristic strategies in the first hierarchy of *Schedule* stage (*rg_ss*):
| Scheduling Strategy (ROW_GROUP_SCHED_STRATEGY) | Description | ID |
| ---------------------------------------- | ----------- | -- |
| RG_SIMPLE | Groups nodes sequentially in the order of their starting row indices (*rg_sqeuential*) | 0 |
| RG_BALANCE | sorts the nodes within the same level based on the number of non-zeros and schedule in descreasing order (*rg_balance*) | 1 |

Three heuristic strategies in the second hierarchy of *Schedule* stage (*schedule_ss*):

| Scheduling Strategy(LEVEL_PART_STRATEGY) | Description | ID |
| ---------------------------------------- | ----------- | -- |
| SIMPLE | Schedule nodes in a round-roubin way (*ws_rr*), with hardware scheduler | 0 |
| SIMPLE2 | Schedule nodes in a round-roubin way (*ws_rr*), with software scheduler | 1 |
| WORKLOAD_BALANCE | Schedule nodes in a load-balancing way (*ws_balance*) | 2 |

One strategy in the third hierarchy of *Schedule* stage (*level_ss*):
| Scheduling Strategy(LEVEL_PART_STRATEGY) | Description | ID |
| ---------------------------------------- | ----------- | -- |
| ONE_LEVEL | calculate all the solution in one CUDA kernel | 0 |

Create `anaparas` (scheme information) for AG-SpTRSV, the initiation function is
```
anaparas(int tb_size, int subwarp_size, PREPROCESSING_STRATEGY row_s, int row_alpha, LEVEL_PART_STRATEGY level_ps, LEVEL_SCHED_STRATEGY level_ss, SCHEDULE_STRATEGY schedule_ss, ROW_GROUP_SCHED_STRATEGY rg_ss);
```
For example, for the structured matrix **atmosmodd**, a good initiation is
```
anaparas(1024, 4, ROW_BLOCK, 1, LEVEL_WISE, ONE_LEVEL, SIMPLE, RG_SIMPLE);
```

2. Run ``sh scripts/run_ag.sh {input matrix}`` to evaluate the performance of AG-SpTRSV with a single matrix. For example:
```
sh scripts/run_ag.sh matrix/matrix_sample_csr/delaunay_n13.csr
```

### Run AG-SpTRSV with exaustive search
Run ``sh scripts/run_ag_search.sh {input matrix}`` to evaluate the performance of AG-SpTRSV with a single matrix. This script enables AG-SpTRSV to search among all the available schemes in the optimization space. The search space is defined in ``include/search.h``. Set ``#define PRINT_LOG true`` in ``test/main_search.cu`` to output information of the scheme being evaluated and the best scheme evaluated so far, and set ``#define PRINT_LOG false`` to disable outputing. The process of search may take a while. For example:
```
sh scripts/run_ag_search.sh matrix/matrix_sample_csr/delaunay_n13.csr
```
Run ``sh scripts/run_ag_search.sh {input matrix} {output file}`` to write evaluation statistics to files (perf file in CSV format). The perf file is used for traning of the performance model. The user can write multiple results of execution into the same CSV file. For example:
```
sh scripts/run_ag_search.sh matrix/matrix_sample_csr/delaunay_n13.csr sample.csv
```

### Train performance model with historical data
Run ``python mlp.py {perf file} {matrix info} {output file} {train_prop}`` to train the proposed MLP model. ``{perf file}`` is the output CSV file, ``{matrix info}`` is the CSV file for matrix information, ``{output file}`` is the output model file (in pickle format), and ``{train_prop}`` is the proportion of the training set (the rest of datasets are used as the testing set). We have provided the CSV file of our experimental results in ``data/historical_data/history_XXX.csv``, and information of the matrices used in ``data/matrix_info/info.csv``. For example:
```
python mlp.py ../data/history_data/history_3080Ti.csv ../data/matrix_info/info.csv try_model.out 0.7
```

### Run AG-SpTRSV with performance model
Use the trained performance model for an efficient end-to-end solution. Before compilation, please modify the including and linking directory of *python* (``PY_CFLAG`` and ``PY_LDFLAG``), the linking directory of *pytorch* (``PYTORCH_LDFLAG``) and CUDA installing directory (``CUDA_INSTALL_PATH``). Then, compile and run ``./main -i {matrix file} -m {model file}``. Before execution, set the linking directory of *pytorch* as an environment variable. 

We have provided trained models on two NVIDIA (3080Ti and A100) platforms at ``data/trained_model/model_cpu_XXX.out``. Users can use the model trained with historical data, or use our models for a quick solution. For example,
```
cd test_model
make
export LD_LIBRARY_PATH={pytorch linking library}
./main -i ../matrix/matrix_sample_csr/delaunay_n13.csr -m ../data/trained_model/model_cpu_3080Ti.out
```
