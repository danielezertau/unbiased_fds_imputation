import time
import os
from sklearn.impute import SimpleImputer
from src.tane import *
from src.llm import *
from src.utils import *
from tqdm.auto import tqdm
from src.imputation import *

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

def find_fds(csv_filename, cache_filename, min_num_partitions, max_lhs_size, error_threshold):
    col_names = get_col_names(csv_filename)

    if os.path.exists(cache_filename):
        all_fds = read_from_cache(cache_filename)
        biased_fds = all_fds["biased"]
        unbiased_fds = all_fds["unbiased"]
    else:
        func_deps = get_tane_rules(csv_filename, min_num_partitions, max_lhs_size, error_threshold)
        load_env_file()
        biased_fds = {}
        unbiased_fds = {}

        print("ALL FDs")
        print_func_deps(func_deps, col_names)

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
    biased_fds, unbiased_fds = find_fds(csv_filename, cache_filename, min_num_partitions, max_lhs_size,
                                        error_threshold)
    full_df = pd.read_csv(csv_filename)
    full_df["Imputed"] = "No"
    # Don't balance the distribution in unbiased FDs
    imputed_df = impute_by_func_deps(full_df, unbiased_fds, balancing_power=1)

    # If we still have missing values, use biased FDs
    if use_biased_fds and imputed_df.isnull().values.any():
        print("Imputing with biased FDs")
        imputed_df = impute_by_func_deps(imputed_df, biased_fds, balancing_power)

    # If we still have missing values, use the most common value
    if use_simple_imputer and imputed_df.isnull().values.any():
        print(f"Imputing with SimpleImputer strategy {simp_imp_strategy}")
        imp = SimpleImputer(strategy=simp_imp_strategy)
        imputed_df[:] = imp.fit_transform(imputed_df)

    imputed_df.to_csv(output_filename, index=False, na_rep="NULL")
