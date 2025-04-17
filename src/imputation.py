import numpy as np
from src.utils import balance_prob_dist

def find_fds_for_rhs(func_deps, fd_rhs):
    matching_lhs = []
    for lhs, rhs in func_deps.items():
        if set(fd_rhs).issubset(set(rhs)):
            matching_lhs.append(lhs)
    return matching_lhs

def get_possible_completions(func_deps, fd_rhs, row, no_null_df, balancing_power):
    column_names = no_null_df.columns
    fd_rhs_col_name = column_names[list(fd_rhs)].values
    # If we don't have a suitable FD, there are no completions
    if fd_rhs not in func_deps.values():
        return None, None

    matching_lhs = find_fds_for_rhs(func_deps, fd_rhs)

    fd_rhs_values = np.unique(no_null_df[fd_rhs_col_name].values.flatten())
    imputation_prob = np.zeros_like(fd_rhs_values, dtype=np.float64)
    i = 1
    for lhs in matching_lhs:
        lhs_col_names = column_names[list(lhs)].values
        values, probs = get_imputation_distribution(no_null_df, row, lhs_col_names, fd_rhs_col_name)

        # Running average of imputation distribution
        imputation_prob = imputation_prob * ((i - 1) / i)
        imputation_prob[np.isin(fd_rhs_values, values)] += probs / i
        i += 1

    # Fix for when we don't find viable completions
    if imputation_prob.sum() == 0:
        return None, None
    
    return fd_rhs_values, balance_prob_dist(imputation_prob, balancing_power)

def get_imputation_distribution(no_null_df, row, lhs_col_names, fd_rhs_col_name):
    row_values_lhs_cols = row[lhs_col_names].values
    rows_matching_lhs_values = no_null_df[no_null_df[lhs_col_names].values == row_values_lhs_cols]
    matching_rows_rhs_values = rows_matching_lhs_values[fd_rhs_col_name].values.flatten()

    values, counts = np.unique(matching_rows_rhs_values, return_counts=True)
    probs = counts / counts.sum()
    return values, probs

def impute_row(df, no_null_df, row, row_index, func_deps, balancing_power):
    imputed = False
    row_null_cols = row[row.isnull()].keys().values

    imputed_row = row.copy()
    for col in row_null_cols:
        col_index_tup = (df.columns.get_loc(col),)
        values, probs = get_possible_completions(func_deps, col_index_tup, row, no_null_df, balancing_power)
        if values is not None:
            rand_value = np.random.choice(values, p=probs)
            imputed_row[col] = rand_value
            imputed = True

    # Impute
    if imputed:
        df.loc[row_index] = imputed_row

def impute_by_func_deps(df, func_deps, balancing_power):
    rows_with_nulls = df[df.isnull().any(axis=1)]
    no_null_df = df[df.notnull().all(axis=1)]
    for i, row in rows_with_nulls.iterrows():
        impute_row(df, no_null_df, row, i, func_deps, balancing_power)

def get_imputation_sucess_rate(before_df, after_df, ground_truth_df):
    # Where the value was originally missing
    missing_mask = before_df.isna().values

    # Where the imputation actually filled something in
    imputed_mask = missing_mask & after_df.notna().values

    # Where the imputed value is equal to the ground truth
    correct_imputations = (after_df.values == ground_truth_df.values) & imputed_mask

    num_correct = correct_imputations.sum().sum()
    num_imputed = imputed_mask.sum().sum()
    success_rate = num_correct / num_imputed if num_imputed > 0 else 0

    return success_rate