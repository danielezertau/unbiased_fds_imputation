import os
import time
from src.tane import *
from src.llm import *
from src.utils import *
from tqdm.auto import tqdm


def get_tane_rules(csv_filename, min_num_partitions, max_lhs_size, error_threshold=0.0, ignore_nulls=True):
    t, table_size = read_db(csv_filename, ignore_nulls)
    tane = TANE(t, table_size=table_size, error_threshold=error_threshold, min_diff_values=min_num_partitions,
                max_lhs_size=max_lhs_size)
    t0 = time.time()
    tane.run()
    print ("\t=> Execution Time: {} seconds".format(time.time()-t0))
    func_deps = {}
    for lhs, rhs in tane.rules:
        if not func_deps.get(lhs):
            func_deps[lhs] = set()
        func_deps[lhs].add(rhs)

    print ('\t=> {} Rules Found'.format(sum(len(v) for v in func_deps.values())))
    return func_deps

def find_unbiased_fds(csv_filename, cache_filename, min_num_partitions, max_lhs_size, error_threshold):
    col_names = get_col_names(csv_filename)

    if os.path.exists(cache_filename):
        unbiased_fds = read_from_cache(cache_filename)
    else:
        func_deps = get_tane_rules(csv_filename, min_num_partitions, max_lhs_size, error_threshold)
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
    write_to_cache(cache_filename, unbiased_fds)

    return unbiased_fds


if __name__ == '__main__':
    MIN_NUM_PARTITIONS = 2
    MAX_LHS_SIZE = 3
    ERROR_THRESHOLD = 0.1
    data_filename = "adult-rand-100"
    CSV_FILENAME = f"./data/{data_filename}.csv"
    CACHE_FILENAME = f"./data/cache/{data_filename}-{MAX_LHS_SIZE}-{MIN_NUM_PARTITIONS}-{ERROR_THRESHOLD}.pkl"
    find_unbiased_fds(CSV_FILENAME, CACHE_FILENAME, MIN_NUM_PARTITIONS, MAX_LHS_SIZE, ERROR_THRESHOLD)