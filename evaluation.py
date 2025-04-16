import argparse
import os

import numpy as np
import pandas as pd
from src.fd_imp import find_fds_and_impute


def rand_null_data(data_dir, input_filename, num_rand_cells, num_experiments):
    # Params and setup
    min_num_partitions, max_lhs_size, err_threshold, use_biased, balance_power, use_simp, simp_strat = (
        2, 3, 0.1, True, 0.5, True, "most_frequent")
    eval_dir = "./eval"
    data_file = f"{eval_dir}/eval-tmp.csv"
    output_file = f"{eval_dir}/eval-out.csv"
    cache_suffix = f"{input_filename}-{max_lhs_size}-{min_num_partitions}-{err_threshold}" 
    cache_file = f"{eval_dir}/eval-cache-{cache_suffix}.pkl"
    print(cache_file)
    os.makedirs(eval_dir, exist_ok=True)

    df = pd.read_csv(f"{data_dir}/{input_filename}.csv")
    sum_imp_ub, sum_imp_b, sum_imp_s = 0, 0, 0
    for i in range(num_experiments):
        print(f"\n\nRunning experiment number {i+1}")
        # Filter out null values
        new_df = df[df.notnull().all(axis=1)]
        n_rows, n_cols = new_df.shape
    
        # Create binary mask
        flat_indices = np.random.choice(n_rows * n_cols, size=num_rand_cells, replace=False)
        row_indices, col_indices = np.unravel_index(flat_indices, (n_rows, n_cols))
        mask = np.zeros((n_rows, n_cols), dtype=bool)
        mask[row_indices, col_indices] = True
    
        # Set NAN by the mask
        print(f"Setting NULL values in {mask.sum()} cells")
        new_df = new_df.mask(mask, np.nan)
    
        # Write eval df to file
        new_df.to_csv(data_file, index=False, na_rep="NULL")
    
        # impute
        imp_ub, imp_b, imp_s = find_fds_and_impute(data_file, cache_file, min_num_partitions,
                                                   max_lhs_size, err_threshold,
                                                   output_file,use_biased,balance_power,
                                                   use_simp,simp_strat)
        sum_imp_ub += imp_ub / num_rand_cells
        sum_imp_b += imp_b / num_rand_cells
        sum_imp_s += imp_s / num_rand_cells
        os.remove(output_file)
        os.remove(data_file)

    print("\n\n RESULTS:")
    print(f"Average fraction of imputed tuples with unbiased FDs: {sum_imp_ub / num_experiments}")
    print(f"Average fraction of imputed tuples with biased FDs: {sum_imp_b / num_experiments}")
    print(f"Average fraction of imputed tuples with SimpleImputer: {sum_imp_s / num_experiments}")

def parse_args():
    parser = argparse.ArgumentParser(description="Find functional dependencies.")

    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Input CSV file")

    parser.add_argument("--input_file", type=str, default="adult-rand-1000",
                        help="Input CSV file")

    parser.add_argument("--num_rand_cells", type=int, default=50,
                        help="Number of random cells to insert.")

    parser.add_argument("--num_iterations", type=int, default=50,
                        help="Number of iterations of the experiment.")
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    rand_null_data(args.data_dir, args.input_file, args.num_rand_cells, args.num_iterations)
