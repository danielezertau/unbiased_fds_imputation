import os
import numpy as np
import pandas as pd
from fd_imp_cli import cli_main


def rand_null_data(input_file_dir, input_filename, args, num_null_cells, num_experiments):
    # Params and setup
    eval_dir = "./eval"
    os.makedirs(eval_dir, exist_ok=True)

    df = pd.read_csv(f"{input_file_dir}/{input_filename}.csv")

    # Filter out null values
    no_null_df = df[df.notnull().all(axis=1)]
    n_rows, n_cols = no_null_df.shape

    sum_imp_ub, sum_imp_b, sum_imp_s = 0, 0, 0
    correct_imp = 0
    for i in range(num_experiments):
        print(f"\n\nRunning experiment number {i+1}")
        # Create binary mask
        flat_indices = np.random.choice(n_rows * n_cols, size=num_null_cells, replace=False)
        row_indices, col_indices = np.unravel_index(flat_indices, (n_rows, n_cols))
        mask = np.zeros((n_rows, n_cols), dtype=bool)
        mask[row_indices, col_indices] = True
    
        # Set NAN by the mask
        print(f"Setting NULL values in {mask.sum()} cells")
        new_df = no_null_df.mask(mask, np.nan)
    
        # Write eval df to file
        new_df.to_csv(f"{eval_dir}/{input_filename}.csv", index=False, na_rep="NULL")
        args += ["--data_filename", input_filename]
    
        # impute
        output_filename, (imp_ub, imp_b, imp_s) = cli_main(args)
        
        # Compare
        imputed_df = pd.read_csv(output_filename)
        correct_imp += 1 - ((imputed_df.values != no_null_df.values).sum().item() / num_null_cells)
        sum_imp_ub += imp_ub / num_null_cells
        sum_imp_b += imp_b / num_null_cells
        sum_imp_s += imp_s / num_null_cells

    print("\n\n RESULTS:")
    print(f"Average fraction of imputed tuples with unbiased FDs: {sum_imp_ub / num_experiments}")
    print(f"Average fraction of imputed tuples with biased FDs: {sum_imp_b / num_experiments}")
    print(f"Average fraction of imputed tuples with SimpleImputer: {sum_imp_s / num_experiments}")
    print(f"Average fraction of correctly imputed cells: {correct_imp / num_experiments}")

if __name__ == '__main__':
    args_dict = [
        '--data_dir', "./eval/",
        '--cache_dir', "./eval/cache",
        '--output_dir', "./eval/out",
        '--min_num_partitions', '2',
        '--max_lhs_size', '3',
        '--error_threshold', '0.05',
        '--use_biased_fds', 'True',
        '--balancing_power', '0.5',
        '--use_simple_imputer', 'True',
        '--simple_imputer_strategy', 'most_frequent'
    ]

    rand_null_data("./data", "adult-rand-500", args_dict, 50, 50)
