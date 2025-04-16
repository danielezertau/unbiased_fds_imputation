import os
from sklearn.impute import SimpleImputer
from src.tane import *
from src.llm import *
from src.utils import *
from tqdm.auto import tqdm
from src.imputation import *

def find_fds(csv_filename, cache_filename, min_num_partitions, max_lhs_size, error_threshold):
    col_names = get_col_names(csv_filename)

    if os.path.exists(cache_filename):
        print("Reading functional dependencies from cache")
        all_fds = read_from_cache(cache_filename)
        biased_fds = all_fds["biased"]
        unbiased_fds = all_fds["unbiased"]
    else:
        print("Mining for functional dependencies")
        func_deps = get_tane_rules(csv_filename, min_num_partitions, max_lhs_size, error_threshold)
        load_env_file()
        biased_fds = {}
        unbiased_fds = {}

        print("ALL FDs")
        print_func_deps(func_deps, col_names)

        print("Checking for biases in functional dependencies")
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
    write_to_cache(cache_filename, {"biased": biased_fds, "unbiased": unbiased_fds})

    return biased_fds, unbiased_fds

def find_fds_and_impute(csv_filename, cache_filename, min_num_partitions, max_lhs_size, error_threshold,
                        output_filename, use_biased_fds, balancing_power, use_simple_imputer, simp_imp_strategy):

    num_imputed_unbiased, num_imputed_biased, num_imputed_simple = 0, 0, 0
    biased_fds, unbiased_fds = find_fds(csv_filename, cache_filename, min_num_partitions, max_lhs_size,
                                        error_threshold)
    full_df = pd.read_csv(csv_filename)
    prev_num_null_cells = count_nulls(full_df)
    # Don't balance the distribution in unbiased FDs
    print(f"Total number of NULL cells: {prev_num_null_cells}")
    print("Imputing with unbiased FDs")
    imputed_df = impute_by_func_deps(full_df, unbiased_fds, balancing_power=1)
    num_imputed_unbiased = prev_num_null_cells - count_nulls(imputed_df)
    print(f"Imputed {num_imputed_unbiased} "
          f"cells of missing information using unbiased FDs")

    # If we still have missing values, use biased FDs
    if use_biased_fds and imputed_df.isnull().values.any():
        print("Imputing with biased FDs")
        prev_num_null_cells = count_nulls(imputed_df)
        imputed_df = impute_by_func_deps(imputed_df, biased_fds, balancing_power)
        num_imputed_biased = prev_num_null_cells - count_nulls(imputed_df)
        print(f"Imputed {num_imputed_biased} "
              f"cells of missing information using biased FDs")

    # If we still have missing values, use the most common value
    if use_simple_imputer and imputed_df.isnull().values.any():
        prev_num_null_cells = count_nulls(imputed_df)
        print(f"Imputing with SimpleImputer strategy {simp_imp_strategy}")
        imp = SimpleImputer(strategy=simp_imp_strategy)
        imputed_df[:] = imp.fit_transform(imputed_df)
        num_imputed_simple = prev_num_null_cells - count_nulls(imputed_df)
        print(f"Imputed {num_imputed_simple} "
              f"cells of missing information using SimpleImputer strategy {simp_imp_strategy}")

    imputed_df.to_csv(output_filename, index=False, na_rep="NULL")
    
    return num_imputed_unbiased, num_imputed_biased, num_imputed_simple
