# CAGRA dataset / graph stats

## Build
```bash
mkdir build
cd build
cmake ..
make -j32 # It takes for a while
cd ..
```

## Env var
```bash
# Input
export RAFT_INDEX_PATH=/path/to/raft/index

# Output
export INTERNAL_DATASET_DTYPE=float
export INTERNAL_DATASET_PATH=tmp.idataset
export INTERNAL_GRAPH_PATH=tmp.igraph
```

## Run
### Convert raft CAGRA index to internal CAGRA graph / dataset files
```bash
./build/conv $RAFT_INDEX_PATH $INTERNAL_DATASET_DTYPE $INTERNAL_GRAPH_PATH $INTERNAL_DATASET_PATH
```

### Graph stats
```
./build/graph_stats $INTERNAL_GRAPH_PATH

# needs `pip install scipy`
python ./connected_components.py $INTERNAL_GRAPH_PATH
```

### Dataset stats
```
./build/dataset_stats $INTERNAL_DATASET_PATH $INTERNAL_DATASET_DTYPE
```
