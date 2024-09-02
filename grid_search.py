import itertools
import subprocess
import argparse
from multiprocessing import Pool, Manager
import os
import pandas as pd
from pathlib import Path
import json


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--std_values",
        type=float,
        nargs='+',
        default=[0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
        help="The values of std to sweep",
    )
    parser.add_argument(
        "--d1_values",
        type=float,
        nargs='+',
        default=[0.01, 0.1, 0.3, 0.5, 0.75, 0.8, 1, 1.25, 1.5],
        help="The values of d1 to sweep",
    )
    parser.add_argument(
        "--d2_values",
        type=float,
        nargs='+',
        default=[0.01, 0.1, 0.3, 0.5, 0.75, 0.8, 1, 1.25, 1.5],
        help="The values of d2 to sweep",
    )
    parser.add_argument(
        "--default_index",
        default=0,
        type=int,
        help="The index for default experiment",
    )
    parser.add_argument(
        "--nb_workers",
        type=int,
        default=8,
        help="Number of threads for parallel computation",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Temporary directory to save and read data. Useful when using clusters.",
    )
    parser.add_argument(
        "--rej_lev",
        type=int,
        default=1,
        help="Wheter or not to compute rejection level. If 1 then it computes rejection level using only std and d1, "
             "if 0 it runs the detection method on d2 with precomputed rejection level.",
    )
    parser.add_argument(
        "--extensive_search",
        type=int,
        default=0,
        help="If 1, it runs grid search on smaller and more values for std, d1 and d2. "
             "Experiments such as 7 and 11 require this for good results.",
    )

    return parser.parse_args()


# Function to check if the default index exists in the output file
def check_index_exists(output_file, default_index):
    if not os.path.exists(output_file):
        return False

    df = pd.read_csv(output_file)
    return f'default {default_index}' in df['default_index'].values


# Function to check if a parameter combination exists in the output file
def check_param_combination_exists(output_file, std, d1, d2, index):
    if not os.path.exists(output_file):
        return False

    df = pd.read_csv(output_file)
    return not df[(df['std'] == std) & (df['d1'] == d1) & (df['d2'] == d2) & (df['default_index'] == f'default {index}')].empty


# Function to run the generate_adversarial_examples.py script with given parameters
def run_adv_examples_script(params):
    std, d1, d2, index, lock, output_file, temp_dir, rej_lev_flag = params
    print(f'Running parameters: {params}', flush=True)
    reject_path = f'experiments/{index}/rejection_levels/reject_at_{std}_{d1}.json'

    if os.path.exists(reject_path):
        print("Loading rejection level...", flush=True)
        file = open(reject_path)
        reject_at = json.load(file)[0]
        if reject_at < 1:
            print("Rejection level too low", flush=True)
            return

    if rej_lev_flag == 1:
        if temp_dir is not None:
            cmd = f"source ENV/bin/activate &&" \
                  f" python compute_rejection_level.py --std {std} --d1 {d1} " \
                  f"--default_index {index} --temp_dir {temp_dir}"
        else:
            # Assumes the environment is named "matrix"
            cmd = f"source matrix/bin/activate &&" \
                  f" python compute_rejection_level.py --std {std} --d1 {d1} " \
                  f"--default_index {index}"

        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, executable="/bin/bash"
        )

        if result.returncode != 0:
            print(f"Error running compute_rejection_level.py with params {params}: {result.stderr}",flush=True)
            return

        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if "Rejection level" in line:
                rej_lev = float(line.split()[-1].strip(':'))
                print(f"Rejection level: {rej_lev}", flush=True)

        return

    if temp_dir is not None:
        cmd = f"source ENV/bin/activate &&" \
                  f" python detect_adversarial_examples.py --std {std} --d1 {d1} --d2 {d2} " \
                  f"--default_index {index} --temp_dir {temp_dir}"
    else:
        # Assumes the environment is named "matrix"
        cmd = f"source matrix/bin/activate &&" \
              f" python detect_adversarial_examples.py --std {std} --d1 {d1} --d2 {d2} " \
              f"--default_index {index}"

    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, executable="/bin/bash"
    )

    if result.returncode != 0:
        print(f"Error running detect_adversarial_examples.py with params {params}: {result.stderr}", flush=True)
        return

    # Extract the metrics from the output
    output_lines = result.stdout.split('\n')
    good_defences = None
    wrong_rejection = None
    for line in output_lines:
        if "Percentage of good defences" in line:
            good_defences = float(line.split()[-1].strip(':'))
        if "Percentage of wrong rejections" in line:
            wrong_rejection = float(line.split()[-1].strip(':'))

    if good_defences is not None and wrong_rejection is not None:
        result_line = f"{std},{d1},{d2},default {index},{good_defences},{wrong_rejection}\n"

    print(result_line.strip())

    # Write the result to the file
    with lock:
        with open(output_file, 'a') as f:
            f.write(result_line)


def main():
    import numpy as np
    print("Grid search starting...", flush=True)
    args = parse_args()

    experiment_path = Path(f'experiments/{args.default_index}/grid_search/')
    experiment_path.mkdir(parents=True, exist_ok=True)
    output_file = experiment_path / f'grid_search_{args.default_index}.txt'

    # Define the parameter grid
    if args.extensive_search == 1:
        vals = np.geomspace(0.00001, 1, num=10)
        std_values = vals
        d1_values = vals
        d2_values = vals
    else:
        std_values = args.std_values
        d1_values = args.d1_values
        d2_values = args.d2_values

    indexes = [args.default_index]

    # Define the directory path
    dir_path = f'experiments/{args.default_index}/rejection_levels/'

    # Initialize an empty list to store the files with value >= 1
    files_to_keep = []

    if args.rej_lev == 0:
        # Iterate over the files in the directory
        for filename in os.listdir(dir_path):
            # Check if the file starts with 'reject_at_'
            if filename.startswith('reject_at_'):
                filepath = os.path.join(dir_path, filename)
                if os.path.exists(filepath):
                    with open(os.path.join(dir_path, filename), 'r') as f:
                        data = json.load(f)
                    # Check if the value is greater than or equal to 1
                    if data[0] >= 1:
                        # If true, add the file to the list
                        files_to_keep.append(filename)

    param_grid = list(itertools.product(std_values, d1_values, d2_values, indexes, [args.rej_lev]))

    # Use Manager to create a Lock
    with Manager() as manager:
        lock = manager.Lock()

        # Initialize the output file if it doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                f.write("std,d1,d2,default_index,good_defence,wrong_rejection\n")

        # Prepare arguments
        param_grid_with_lock = [(std, d1, d2, index, lock, output_file, args.temp_dir, args.rej_lev) for std, d1, d2, index, _ in param_grid]

        # This case assumes rejection levels were already computed.
        if args.rej_lev == 0:
            param_grid_filtered = [(std, d1, d2, index, lock, output_file, args.temp_dir, args.rej_lev) for std, d1, d2, index, lock, output_file, args.temp_dir, _ in param_grid_with_lock if f'reject_at_{std}_{d1}.json' in files_to_keep]
            with Pool(processes=args.nb_workers) as pool:
                pool.map(run_adv_examples_script, param_grid_filtered)
        # This case computes rejection levels only using std and d1.
        else:
            param_grid_with_lock_rej_lev = [(std, d1, 0, index, lock, output_file, args.temp_dir, args.rej_lev) for std, d1, d2, index, _ in param_grid]
            with Pool(processes=args.nb_workers) as pool:
                pool.map(run_adv_examples_script, param_grid_with_lock_rej_lev)


if __name__ == "__main__":
    main()
