import pandas as pd

# Parameters
subset_size = 5000
input_file = 'data/adult.csv'
output_file = f'data/adult-rand-{subset_size}.csv'


if __name__ == '__main__':
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Sample a random subset of rows
    subset = df.sample(n=min(subset_size, len(df)), random_state=42)

    # Write the subset to a new CSV file, including the header
    subset.to_csv(output_file, index=False, na_rep="NULL")
