# CAGRA dataset / graph stats

## Build
```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/hootomo/raft-cagra-dataset-eval.git --recursive
cd raft-cagra-dataset-eval

mkdir build
cd build
cmake ..
make -j32 # it will take for a while
cd ..
```

## Set env vars
```bash
# Input
export RAFT_INDEX_PATH=/path/to/raft/index
export DATASET_DTYPE=float

# Output
export INTERNAL_DATASET_PATH=tmp.idataset
export INTERNAL_GRAPH_PATH=tmp.igraph
```

## Run
### Convert raft CAGRA index to internal CAGRA graph / dataset files
```bash
./build/conv $RAFT_INDEX_PATH $DATASET_DTYPE $INTERNAL_GRAPH_PATH $INTERNAL_DATASET_PATH
```

### Graph stats
```
./build/graph_stats $INTERNAL_GRAPH_PATH

# needs `pip install scipy`
python ./connected_components.py $INTERNAL_GRAPH_PATH
```

### Dataset stats
```
./build/dataset_stats $INTERNAL_DATASET_PATH $DATASET_DTYPE
```
