import pandas as pd
import numpy as np

def get_possible_completions(func_deps, fd_rhs, row, no_null_df, balance_probs):
    column_names = no_null_df.columns
    fd_rhs_col_name = column_names[list(fd_rhs)].values
    if fd_rhs not in func_deps.values():
        return tuple()
    matching_lhs = []
    for lhs, rhs in func_deps.items():
        if set(fd_rhs).issubset(set(rhs)):
            matching_lhs.append(lhs)

    fd_rhs_values = np.unique(no_null_df[fd_rhs_col_name].values.flatten())
    imputation_prob = np.zeros_like(fd_rhs_values, dtype=np.float64)
    i = 1
    for lhs in matching_lhs:
        lhs_col_names = column_names[list(lhs)].values
        row_values_lhs_cols = row[lhs_col_names].values
        rows_matching_lhs_values = no_null_df[no_null_df[lhs_col_names].values == row_values_lhs_cols]
        matching_rows_rhs_values = rows_matching_lhs_values[fd_rhs_col_name].values.flatten()

        values, counts = np.unique(matching_rows_rhs_values, return_counts=True)
        # Running average of imputation distribution
        imputation_prob = imputation_prob * ((i - 1) / i)
        imputation_prob[np.isin(fd_rhs_values, values)] += (counts / counts.sum()) / i
        i += 1

    return fd_rhs_values, imputation_prob


def impute_by_func_deps(full_df, func_deps, output_filename):
    rows_to_append = []
    rows_with_nulls = full_df[full_df.isnull().any(axis=1)]
    no_null_table = full_df[full_df.notnull().all(axis=1)]
    for i, row in rows_with_nulls.iterrows():
        full_df.drop(i, inplace=True)
        row_null_cols = row[row.isnull()].keys().values

        completions = {}
        for col in row_null_cols:
            col_index_tup = (full_df.columns.get_loc(col), )
            completions[col] = get_possible_completions(func_deps, col_index_tup, row, no_null_table)

        imputed_row = row.copy()
        for col, imputations in completions.items():
            values, probs = imputations
            rand_value = np.random.choice(values, p=probs)
            imputed_row[col] = rand_value
            rows_to_append.append(imputed_row)

    if len(rows_to_append) > 0:
        full_df = pd.concat([full_df, pd.DataFrame(rows_to_append)], ignore_index=True)

    full_df.to_csv(output_filename, index=False)

