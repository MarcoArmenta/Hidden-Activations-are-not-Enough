import itertools
import subprocess
import argparse
from multiprocessing import Pool, Manager
import os
import pandas as pd
from pathlib import Path


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
        default=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75, 0.8, 1, 1.25, 1.5, 1.75],
        help="The values of d1 to sweep",
    )
    parser.add_argument(
        "--d2_values",
        type=float,
        nargs='+',
        default=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75, 0.8, 1, 1.25, 1.5, 1.75],
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
    parser.add_argument("--temp_dir", type=str, default=None)

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
    std, d1, d2, index, lock, output_file, temp_dir = params
    print(f'Running parameters: {params}', flush=True)

    if check_param_combination_exists(output_file, std, d1, d2, index):
        print(f"Skipping existing params: std={std}, d1={d1}, d2={d2}, default_index={index}")
        return

    if temp_dir is None:
        cmd = f"source ~/NeuralNets/MatrixStatistics/matrix/bin/activate &&" \
              f" python compute_rejection_level.py --std {std} --d1 {d1} " \
              f"--default_index {index}"
    else:
        cmd = f"source ENV/bin/activate &&" \
              f" python compute_rejection_level.py --std {std} --d1 {d1} " \
              f"--default_index {index} --temp_dir {temp_dir}"
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

    if temp_dir is None:
        cmd = f"source ~/NeuralNets/MatrixStatistics/matrix/bin/activate &&" \
              f" python detect_adversarial_examples.py --std {std} --d1 {d1} --d2 {d2} " \
              f"--default_index {index} --temp_dir {temp_dir}"
    else:
        cmd = f"source ENV/bin/activate &&" \
              f" python detect_adversarial_examples.py --std {std} --d1 {d1} --d2 {d2} " \
              f"--default_index {index} --temp_dir {temp_dir}"

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
    else:
        result_line = f"{std},{d1},{d2},default {index},rej_lev={rej_lev}\n"

    print(result_line.strip())

    # Write the result to the file
    with lock:
        with open(output_file, 'a') as f:
            f.write(result_line)


def main():
    print("Grid search starting...",flush=True)
    args = parse_args()

    experiment_path = Path(f'experiments/{args.default_index}/grid_search/')
    experiment_path.mkdir(parents=True, exist_ok=True)
    output_file = experiment_path / f'grid_search_{args.default_index}.txt'

    # Define the parameter grid
    std_values = args.std_values
    d1_values = args.d1_values
    d2_values = args.d2_values
    indexes = [args.default_index]

    param_grid = list(itertools.product(std_values, d1_values, d2_values, indexes))

    # Use Manager to create a Lock
    with Manager() as manager:
        lock = manager.Lock()

        # Initialize the output file if it doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                f.write("std,d1,d2,default_index,good_defence,wrong_rejection\n")

        # Prepare arguments for the workers
        param_grid_with_lock = [(std, d1, d2, index, lock, output_file, args.temp_dir) for std, d1, d2, index in param_grid]

        #for args in param_grid_with_lock:
        #    print(f'Running parameters: {args}')
        #    run_adv_examples_script(args)

        # Use multiprocessing Pool to run the scripts in parallel
        with Pool(processes=args.nb_workers) as pool:
            pool.map(run_adv_examples_script, param_grid_with_lock)


if __name__ == "__main__":
    main()
