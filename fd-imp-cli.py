import argparse
import os
from main import find_fds_and_impute

def parse_args():
    parser = argparse.ArgumentParser(description="Find functional dependencies.")

    parser.add_argument("--min_num_partitions", type=int, default=3, help="Minimum number of partitions"
                                                                          " allowed for a functional dependency")
    parser.add_argument("--max_lhs_size", type=int, default=3, help="Maximum LHS size of a "
                                                                    "functional dependency")
    parser.add_argument("--error_threshold", type=float, default=0.06, help="Approximate functional "
                                                                           "dependency error threshold")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory containing "
                                                                       "input data")
    parser.add_argument("--data_filename", type=str, default="adult-rand-1000", help="Input CSV file name")

    return parser.parse_args()


def cli_main():
    args = parse_args()

    output_dir = f"{args.data_dir}/out"
    os.makedirs(output_dir, exist_ok=True)

    csv_filename = f"{args.data_dir}/{args.data_filename}.csv"
    output_suffix = f"{args.data_filename}-{args.max_lhs_size}-{args.min_num_partitions}-{args.error_threshold}"
    cache_filename = f"{args.data_dir}/cache/{output_suffix}.pkl"
    output_filename = f"{output_dir}/{output_suffix}.csv"

    find_fds_and_impute(csv_filename, cache_filename, args.min_num_partitions, args.max_lhs_size, args.error_threshold,
                        output_filename)

if __name__ == "__main__":
    cli_main()
