import os
import numpy as np
import pandas as pd
from fd_imp_cli import cli_main
from src.imputation import get_imputation_sucess_rate

def set_rand_nulls(no_null_df, num_null_cells):
    n_rows, n_cols = no_null_df.shape
    # Create binary mask
    flat_indices = np.random.choice(n_rows * n_cols, size=num_null_cells, replace=False)
    row_indices, col_indices = np.unravel_index(flat_indices, (n_rows, n_cols))
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[row_indices, col_indices] = True

    # Set NAN by the mask
    print(f"Setting NULL values in {mask.sum()} cells")
    new_df = no_null_df.mask(mask, np.nan)
    return new_df


def rand_null_data(input_file_dir, input_filename, args, num_null_cells, num_experiments):
    # Params and setup
    eval_dir = "./eval"
    os.makedirs(eval_dir, exist_ok=True)

    input_df = pd.read_csv(f"{input_file_dir}/{input_filename}.csv")

    # Filter out null values
    no_null_df = input_df[input_df.notnull().all(axis=1)]

    sum_imp_ub, sum_imp_b, sum_imp_s = 0, 0, 0
    correct_imp_ub, correct_imp_b, correct_imp_s = 0, 0, 0
    for i in range(num_experiments):
        print(f"\n\nRunning experiment number {i+1}")

        # Set NULLs in random cells
        rand_null_df = set_rand_nulls(no_null_df, num_null_cells)

        # Write eval df to file
        rand_null_df.to_csv(f"{eval_dir}/{input_filename}.csv", index=False, na_rep="NULL")
        args += ["--data_filename", input_filename]
    
        # impute
        output_filename, (df_ub, df_b, df_s, imp_ub, imp_b, imp_s) = cli_main(args)
        
        # Compare
        ub_correct = get_imputation_sucess_rate(rand_null_df, df_ub, no_null_df)
        b_correct = get_imputation_sucess_rate(df_ub, df_b, no_null_df)
        s_correct = get_imputation_sucess_rate(df_b, df_s, no_null_df)

        # Update metrics
        correct_imp_ub += ub_correct
        correct_imp_b += b_correct
        correct_imp_s += s_correct
        sum_imp_ub += imp_ub / num_null_cells
        sum_imp_b += imp_b / num_null_cells
        sum_imp_s += imp_s / num_null_cells

    print_avg_results(sum_imp_ub, correct_imp_ub, sum_imp_b, correct_imp_b, sum_imp_s, correct_imp_s, num_experiments)

def rand_null_expr():
    for (input_file, err_thresh, num_null) in [("adult-rand-500", "0.03", 25), ("adult-rand-500", "0.05", 25),
                                               ("adult-rand-1000", "0.06", 50), ("adult-rand-1000", "0.1", 50)]:
        args_dict = [
            '--data_dir', "./eval/",
            '--cache_dir', "./eval/cache",
            '--output_dir', "./eval/out",
            '--min_num_partitions', '2',
            '--max_lhs_size', '3',
            '--error_threshold', err_thresh,
            '--use_biased_fds', 'True',
            '--balancing_power', '0.2',
            '--use_simple_imputer', 'True',
            '--simple_imputer_strategy', 'most_frequent'
        ]
    
        print("#" * 180)
        print(input_file, err_thresh, num_null)
        rand_null_data("./data", input_file, args_dict, num_null, 50)    

def print_avg_results(sum_imp_ub, correct_imp_ub, sum_imp_b, correct_imp_b, sum_imp_s, correct_imp_s, num_experiments):
    print("\n\n RESULTS:")
    print(f"Average fraction of imputed tuples with unbiased FDs: {sum_imp_ub / num_experiments}")
    print(f"Average fraction of correctly imputed cells with unbiased FDs: {correct_imp_ub / num_experiments}")
    print(f"Average fraction of imputed tuples with biased FDs: {sum_imp_b / num_experiments}")
    print(f"Average fraction of correctly imputed cells with biased FDs: {correct_imp_b / num_experiments}")
    print(f"Average fraction of imputed tuples with SimpleImputer: {sum_imp_s / num_experiments}")
    print(f"Average fraction of correctly imputed cells with SimpleImputer: {correct_imp_s / num_experiments}")

if __name__ == '__main__':
    rand_null_expr()
