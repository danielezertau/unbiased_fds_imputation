import os
import time
import pandas as pd
from src.tane import *
from src.llm import *
import pickle
from tqdm.auto import tqdm

def indices_to_attr_name(column_names, idxs):
    return column_names[list(idxs)].values

def print_func_deps(func_deps, column_names):
    print ('\t=> {} Rules Found'.format(sum(len(v) for v in func_deps.values())))
    for lhs, rhs in func_deps.items():
        print(f"{indices_to_attr_name(column_names, lhs)} -> {indices_to_attr_name(column_names, rhs)}")
    print("\n\n")

def get_col_names(filename):
    return pd.read_csv(filename).columns

def get_tane_rules(csv_filename, cache_filename, min_num_partitions, max_lhs_size, error_threshold, ignore_nulls):
    t, table_size = read_db(csv_filename, ignore_nulls)
    tane = TANE(t, table_size=table_size, error_threshold=error_threshold, min_diff_values=min_num_partitions, max_lhs_size=max_lhs_size)
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

def find_fds(csv_filename, cache_filename, min_num_partitions, max_lhs_size, error_threshold=0.0, ignore_nulls=True,
             print_results=True):
    column_names = get_col_names(csv_filename)

    if os.path.exists(cache_filename):
        with open(cache_filename, "rb") as f:
            func_deps = pickle.load(f)
    else:
        func_deps = get_tane_rules(csv_filename, cache_filename, min_num_partitions, max_lhs_size, error_threshold,
                                   ignore_nulls)
    if print_results:
        print_func_deps(func_deps, column_names)
    
    return func_deps

def main(csv_filename, cache_filename, min_num_partitions, max_lhs_size, error_threshold):
    col_names = get_col_names(csv_filename)
    func_deps = find_fds(csv_filename, cache_filename, min_num_partitions, max_lhs_size, error_threshold)
    load_env_file()
    biased_fds = {}
    unbiased_fds = {}

    for lhs, rhs_group in tqdm(func_deps.items()):
        for rhs in rhs_group:
            if is_fd_biased(lhs=", ".join(indices_to_attr_name(col_names, lhs)),
                                rhs=", ".join(indices_to_attr_name(col_names, (rhs, )))):
                biased_fds[lhs] = (rhs, )
            else:
                unbiased_fds[lhs] = (rhs, )

    print("BIASED FDS:")
    print_func_deps(biased_fds, col_names)

    print("UNBIASED FDS:")
    print_func_deps(unbiased_fds, col_names)


if __name__ == '__main__':
    MIN_NUM_PARTITIONS = 3
    MAX_LHS_SIZE = 4
    ERROR_THRESHOLD = 0.01
    data_filename = "adult-rand-1000"
    CSV_FILENAME = f"./data/{data_filename}.csv"
    CACHE_FILENAME = f"./data/cache/{data_filename}-{MAX_LHS_SIZE}-{MIN_NUM_PARTITIONS}-{ERROR_THRESHOLD}.pkl"
    main(CSV_FILENAME, CACHE_FILENAME, MIN_NUM_PARTITIONS, MAX_LHS_SIZE, ERROR_THRESHOLD)