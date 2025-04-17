import os
from sklearn.impute import SimpleImputer
from src.tane import *
from src.llm import *
from src.utils import *
from tqdm.auto import tqdm
from src.imputation import *

def load_fds_from_cache(cache_filename):
    print("Reading functional dependencies from cache")
    all_fds = read_from_cache(cache_filename)
    biased_fds = all_fds["biased"]
    unbiased_fds = all_fds["unbiased"]
    return biased_fds, unbiased_fds

def mine_for_fds(csv_filename, col_names, min_num_partitions, max_lhs_size, error_threshold):
    print("Mining for functional dependencies")
    func_deps = get_tane_rules(csv_filename, min_num_partitions, max_lhs_size, error_threshold)
    load_env_file()
    biased_fds = {}
    unbiased_fds = {}

    print("Checking for biases in functional dependencies")
    for lhs, rhs_group in tqdm(func_deps.items()):
        for rhs in rhs_group:
            if is_fd_biased(lhs=", ".join(indices_to_attr_name(col_names, lhs)),
                            rhs=", ".join(indices_to_attr_name(col_names, (rhs, )))):
                biased_fds[lhs] = (rhs, )
            else:
                unbiased_fds[lhs] = (rhs, )
    return biased_fds, unbiased_fds

def find_fds(csv_filename, cache_filename, min_num_partitions, max_lhs_size, error_threshold):
    col_names = get_col_names(csv_filename)

    if os.path.exists(cache_filename):
        biased_fds, unbiased_fds = load_fds_from_cache(cache_filename)
    else:
        biased_fds, unbiased_fds = mine_for_fds(csv_filename, col_names, min_num_partitions, max_lhs_size,
                                                error_threshold)
    print("BIASED FDS:")
    print_func_deps(biased_fds, col_names)

    print("UNBIASED FDS:")
    print_func_deps(unbiased_fds, col_names)
    write_to_cache(cache_filename, {"biased": biased_fds, "unbiased": unbiased_fds})

    return biased_fds, unbiased_fds

def impute_with_fds_and_report(df, fds, fd_type, balancing_power):
    num_nulls_before = count_nulls(df)
    print(f"Imputing with {fd_type} FDs")
    imputed_df = df.copy()
    impute_by_func_deps(imputed_df, fds, balancing_power)
    num_nulls_after = count_nulls(imputed_df)
    num_imputed = num_nulls_before - num_nulls_after
    print(f"Imputed {num_imputed} cells of missing information using {fd_type} FDs")
    return imputed_df, num_imputed

def impute_with_simp_imp_and_report(df, strategy):
    num_nulls_before = count_nulls(df)
    print(f"Imputing with SimpleImputer strategy {strategy}")
    imp = SimpleImputer(strategy=strategy)
    imputed_df = df.copy()
    imputed_df[:] = imp.fit_transform(df)
    num_nulls_after = count_nulls(imputed_df)
    num_imputed = num_nulls_before - num_nulls_after
    print(f"Imputed {num_imputed} cells of missing information using SimpleImputer strategy {strategy}")
    return imputed_df, num_imputed

def find_fds_and_impute(csv_filename, cache_filename, min_num_partitions, max_lhs_size, error_threshold,
                        output_filename, use_biased_fds, balancing_power, use_simple_imputer, simp_imp_strategy):

    num_imputed_unbiased, num_imputed_biased, num_imputed_simple = 0, 0, 0
    biased_fds, unbiased_fds = find_fds(csv_filename, cache_filename, min_num_partitions, max_lhs_size,
                                        error_threshold)
    full_df = pd.read_csv(csv_filename)
    
    print(f"Total number of NULL cells: {count_nulls(full_df)}")

    # Impute with unbiased FDs
    df_unbiased, num_imputed_unbiased = impute_with_fds_and_report(full_df, unbiased_fds, "unbiased", 1)

    # If we still have missing values, use biased FDs
    if use_biased_fds and df_unbiased.isnull().values.any():
        df_biased, num_imputed_biased = impute_with_fds_and_report(df_unbiased, biased_fds, "biased", balancing_power)
    else:
        df_biased = df_unbiased

    # If we still have missing values, use the most common value
    if use_simple_imputer and df_biased.isnull().values.any():
        df_simple, num_imputed_simple = impute_with_simp_imp_and_report(df_biased, simp_imp_strategy)
    else:
        df_simple = df_biased

    df_simple.to_csv(output_filename, index=False, na_rep="NULL")
    
    return df_unbiased, df_biased, df_simple, num_imputed_unbiased, num_imputed_biased, num_imputed_simple
