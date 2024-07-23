import sys
import struct
import numpy
import scipy
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

if len(sys.argv) != 2:
    print('usage: No filename set.')
    sys.exit(-1)

filename = sys.argv[1]
print(f"filename: {filename}")

with open(filename, 'rb') as f:
    #num_rows = f.read(8)
    #if not num_rows:
    #    sys.exit(-1)
    #num_rows = struct.unpack('Q', num_rows)[0]    # Q: 64-bit unsigned integer
    #print(f"num_rows: {num_rows}")

    #degree = f.read(8)
    #if not degree:
    #    sys.exit(-1)
    #degree = struct.unpack('Q', degree)[0]    # Q: 64-bit unsigned integer
    #print(f"degree: {degree}")
    num_rows = f.read(4)
    if not num_rows:
        sys.exit(-1)
    num_rows = struct.unpack('I', num_rows)[0]    # Q: 64-bit unsigned integer
    print(f"num_rows: {num_rows}")

    degree = f.read(4)
    if not degree:
        sys.exit(-1)
    degree = struct.unpack('I', degree)[0]    # Q: 64-bit unsigned integer
    print(f"degree: {degree}")
    
    col_list = []
    row_list = []
    dat_list = []
    
    num_rows = 0
    while True:
        col = f.read(4*degree)
        if not col:
            break
        format = '<' + 'i' * degree
        col = struct.unpack(format, col)
        col = numpy.array(col, dtype='i')
        col_list.append(col)

        row = numpy.full(degree, num_rows, dtype='i')
        row_list.append(row)

        dat = numpy.full(degree, 1, dtype='i')
        dat_list.append(dat)
        
        num_rows += 1

    print(f"num_rows: {num_rows}")

    np_row = numpy.concatenate(row_list)
    np_col = numpy.concatenate(col_list)
    np_dat = numpy.concatenate(dat_list)
    print(f"np_row.shape: {np_row.shape}")
    print(f"np_col.shape: {np_col.shape}")
    print(f"np_dat.shape: {np_dat.shape}")
    print(f"np_row: {np_row}")
    print(f"np_col: {np_col}")
    print(f"np_dat: {np_dat}")

    graph = csr_array((np_dat, (np_row, np_col)), shape=(num_rows, num_rows))
    print(f"graph.sahpe: {graph.shape}")
    print(f"graph.nnz: {graph.nnz}")
    
    n_components, labels = connected_components(graph, directed=True, connection='weak', return_labels=True)
    print(f"n_components (weak): {n_components}")
    # print(f"labels: {labels}")

    n_components, labels = connected_components(graph, directed=True, connection='strong', return_labels=True)
    print(f"n_components (strong): {n_components}")
    # print(f"labels: {labels}")
