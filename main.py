import os
import time
import pandas as pd
from tane import *
import pickle

def indices_to_attr_name(column_names, idxs):
    return column_names[list(idxs)].values

def print_func_deps(func_deps, column_names):
    print ('\t=> {} Rules Found'.format(sum(len(v) for v in func_deps.values())))
    for lhs, rhs in func_deps.items():
        print(f"{indices_to_attr_name(column_names, lhs)} -> {indices_to_attr_name(column_names, rhs)}")
    print("\n\n")

def get_col_names(filename):
    return pd.read_csv(filename).columns

def get_tane_rules(csv_filename, cache_filename, min_partition_size, max_lhs_size, error_threshold, ignore_nulls):
    t, table_size = read_db(csv_filename, ignore_nulls)
    tane = TANE(t, table_size=table_size, error_threshold=error_threshold, min_diff_values=min_partition_size, max_lhs_size=max_lhs_size)
    t0 = time.time()
    tane.run()
    print ("\t=> Execution Time: {} seconds".format(time.time()-t0))
    func_deps = {}
    for lhs, rhs in tane.rules:
        if not func_deps.get(lhs):
            func_deps[lhs] = set()
        func_deps[lhs].add(rhs)

    with open(cache_filename, "wb") as f:
        pickle.dump(func_deps, f)
    print ('\t=> {} Rules Found'.format(sum(len(v) for v in func_deps.values())))
    return func_deps

def find_fds(csv_filename, cache_filename, min_partition_size, max_lhs_size, error_threshold=0.0,ignore_nulls=True,
             print_results=True):
    column_names = get_col_names(csv_filename)

    if os.path.exists(cache_filename):
        with open(cache_filename, "rb") as f:
            func_deps = pickle.load(f)
    else:
        func_deps = get_tane_rules(csv_filename, cache_filename, min_partition_size, max_lhs_size, error_threshold,
                                   ignore_nulls)
    if print_results:
        print_func_deps(func_deps, column_names)


if __name__ == '__main__':
    MIN_PARTITION_SIZE = 1
    MAX_LHS_SIZE = 100
    data_filename = "adult-rand-500"
    CSV_FILENAME = f"./data/{data_filename}.csv"
    CACHE_FILENAME = f"./data/cache/{data_filename}.pkl"
    find_fds(CSV_FILENAME, CACHE_FILENAME, MIN_PARTITION_SIZE, MAX_LHS_SIZE, 0.2, True)

