import itertools
import subprocess
import argparse
from multiprocessing import Pool, Manager
import os
import pandas as pd


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--std_values",
        type=list,
        default=[0.75, 1, 1.5, 2],
        help="The values of std to sweep",
    )
    parser.add_argument(
        "--d1_values",
        type=list,
        default=[0.01, 0.1, 0.3, 0.5, 0.8, 1],
        help="The values of d1 to sweep",
    )
    parser.add_argument(
        "--d2_values",
        type=list,
        default=[0.01, 0.1, 0.3, 0.5, 0.8, 1],
        help="The values of d2 to sweep",
    )
    parser.add_argument(
        "--default_index",
        type=int,
        default=0,
        help="The index for default experiment",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default='grid_search_results.txt',
        help="Output file name to save results of grid search.",
    )
    parser.add_argument(
        "--nb_workers",
        type=int,
        default=8,
        help="Number of threads for parallel computation",
    )

    return parser.parse_args()


# Function to check if the default index exists in the output file
def check_index_exists(output_file, default_index):
    if not os.path.exists(output_file):
        return False

    df = pd.read_csv(output_file)
    return f'default {default_index}' in df['default_index'].values


# Function to run the adversarial_attacks.py script with given parameters
def run_script(params):
    std, d1, d2, index, lock, output_file = params

    cmd = f"source ~/NeuralNets/MatrixStatistics/matrix/bin/activate &&" \
          f" python adversarial_attacks.py --std {std} --d1 {d1} --d2 {d2} " \
          f"--default_hyper_parameters " \
          f"--default_index {index}"
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, executable="/bin/bash"
    )

    if result.returncode != 0:
        print(f"Error running script with params {params}: {result.stderr}")
        result_line = f"{std},{d1},{d2},{index},ERROR\n"
    else:
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
            result_line = f"{std},{d1},{d2},default {index},None\n"

    print(result_line.strip())

    # Write the result to the file
    with lock:
        with open(output_file, 'a') as f:
            f.write(result_line)


if __name__ == "__main__":
    args = parse_args()

    # Check if the file exists and if the default index already exists
    if check_index_exists(args.output_file, args.default_index):
        print(f"Default index {args.default_index} already exists in {args.output_file}. Exiting without changes.")
        exit()

    # Define the number of processes to use
    num_processes = args.nb_workers  # Adjust this number based on your system's capabilities

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
        if not os.path.exists(args.output_file):
            with open(args.output_file, 'w') as f:
                f.write("std,d1,d2,default_index,good_defence,wrong_rejection\n")

        # Prepare arguments for the workers
        param_grid_with_lock = [(std, d1, d2, index, lock, args.output_file) for std, d1, d2, index in param_grid]

        # Use multiprocessing Pool to run the scripts in parallel
        with Pool(processes=num_processes) as pool:
            pool.map(run_script, param_grid_with_lock)
