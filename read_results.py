import argparse
import pandas as pd
import sys
from pathlib import Path
import subprocess

def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default='grid_search_results.txt',
        help="Output file name to read results of grid search.",
    )
    parser.add_argument(
        "--default_index",
        type=int,
        default=0,
        help="The index for default experiment",
    )

    return parser.parse_args()


def check_default_index_exists(df, default_index):
    return f'default {default_index}' in df['default_index'].values


def get_top_10_abs_difference(df, default_index):
    filtered_df = df[df['default_index'] == f'default {default_index}'].copy()

    # Convert columns to float, coercing errors to NaN
    filtered_df['good_defence'] = pd.to_numeric(filtered_df['good_defence'], errors='coerce')
    filtered_df['wrong_rejection'] = pd.to_numeric(filtered_df['wrong_rejection'], errors='coerce')

    # Remove rows with NaN values
    filtered_df = filtered_df.dropna(subset=['good_defence', 'wrong_rejection'])

    filtered_df['abs_difference'] = filtered_df['good_defence'] - filtered_df['wrong_rejection']
    top_10 = filtered_df.nlargest(10, 'abs_difference')
    return top_10[['std', 'd1', 'd2', 'good_defence', 'wrong_rejection']]


def run_detect_adversarial_examples(std, d1, d2, default_index):
    cmd = f"source ~/NeuralNets/MatrixStatistics/matrix/bin/activate &&" \
          f" python detect_adversarial_examples.py --std {std} --d1 {d1} --d2 {d2} " \
          f"--default_index {default_index}"

    # Run the command and capture the output
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, executable="/bin/bash"
    )

    if result.returncode != 0:
        print(f"Error running script with params --std {std} --d1 {d1} --d2 {d2}: {result.stderr}")
    else:
        print(f"Successfully ran script with params --std {std} --d1 {d1} --d2 {d2}: {result.stdout}")

if __name__ == "__main__":
    args = parse_args()

    path_output = Path(f'experiments/{args.default_index}/grid_search/grid_search_{args.default_index}.txt')

    # Read the results file into a DataFrame
    df = pd.read_csv(path_output)

    # Check if the default_index exists
    if not check_default_index_exists(df, args.default_index):
        print(f"Error: Default index {args.default_index} does not exist in {args.output_file}.")
        sys.exit(1)

    # Get the top 10 values for the highest absolute difference
    top_10_abs_diff = get_top_10_abs_difference(df, args.default_index)
    print("Top 10 values for highest absolute difference:")
    print(top_10_abs_diff)

    for _, row in top_10_abs_diff.iterrows():
        run_detect_adversarial_examples(row['std'], row['d1'], row['d2'], args.default_index)

