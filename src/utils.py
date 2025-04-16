import csv
import pickle
import pandas as pd

def indices_to_attr_name(column_names, idxs):
    return column_names[list(idxs)].values

def print_func_deps(func_deps, column_names):
    print ('\t=> {} Rules Found'.format(sum(len(v) for v in func_deps.values())))
    for lhs, rhs in func_deps.items():
        print(f"{indices_to_attr_name(column_names, lhs)} -> {indices_to_attr_name(column_names, rhs)}")
    print("\n\n")

def get_col_names(filename):
    return pd.read_csv(filename).columns

def write_to_cache(cache_file, data):
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

def read_from_cache(cache_filename):
    with open(cache_filename, "rb") as f:
        return pickle.load(f)

def balance_prob_dist(probs, power):
    flattened = probs ** power
    return flattened / flattened.sum()

def read_db(path, ignore_nulls):
    hashes = {}
    num_lines = 0
    with open(path, 'r') as fin:
        reader = csv.DictReader(fin)
        for t, line in enumerate(reader):
            # Ignore lines with null values
            if "NULL" in line.values() and ignore_nulls:
                continue
            num_lines += 1
            for i, s in enumerate(line.values()):
                hashes.setdefault(i, {}).setdefault(s, set([])).add(t)# [(i, s)] = len(hashes)
        return [(list(hashes[k].values())) for k in sorted(hashes.keys())], num_lines

def count_nulls(df):
    return df.isnull().values.sum()
