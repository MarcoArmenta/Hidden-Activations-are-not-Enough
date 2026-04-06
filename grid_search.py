import os
import json
import itertools
import pandas as pd
from pathlib import Path
from numpy import geomspace
from multiprocessing import Pool
import multiprocessing as mp
from argparse import ArgumentParser, Namespace
from typing import Union

from compute_rejection_level import main as compute_rejection_level
from detect_adversarial_examples import main as detect_adversarial_examples


def parse_args(
        parser:Union[ArgumentParser, None] = None
    ) -> Namespace:
    """
        Args:
            parser: the parser to use.
        Returns:
            The parsed arguments.
    """
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        "--t_epsilon_values",
        type = float,
        nargs = '+',
        default = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
        help = "The values of t_epsilon to sweep",
    )
    parser.add_argument(
        "--epsilon_values",
        type = float,
        nargs = '+',
        default = [0.01, 0.1, 0.3, 0.5, 0.75, 0.8, 1, 1.25, 1.5],
        help = "The values of epsilon to sweep",
    )
    parser.add_argument(
        "--epsilon_p_values",
        type = float,
        nargs = '+',
        default = [0.01, 0.1, 0.3, 0.5, 0.75, 0.8, 1, 1.25, 1.5],
        help = "The values of epsilon_p to sweep",
    )
    parser.add_argument(
        "--experiment_name",
        default = 'alexnet_cifar10',
        type = str,
        help = "The experiment name <model>_<dataset>",
    )
    parser.add_argument(
        "--nb_workers",
        type = int,
        default = 8,
        help = "Number of threads for parallel computation",
    )
    parser.add_argument(
        "--temp_dir",
        type = str,
        default = None,
        help = "Temporary directory to save and read data. Useful when using clusters.",
    )
    parser.add_argument(
        "--rej_lev",
        type = int,
        default = 1,
        help = "Wheter or not to compute rejection level. If 1 then it computes rejection level using only t_epsilon and epsilon, "
             "if 0 it runs the detection method on epsilon_p with precomputed rejection level.",
    )
    parser.add_argument(
        "--extensive_search",
        type = int,
        default = 0,
        help = "If 1, it runs grid search on smaller and more values for t_epsilon, epsilon and epsilon_p. "
               "Experiments such as 7 and 11 require this for good results.",
    )
    parser.add_argument(
        "--baseline_only",
        action="store_true",
        default=False,
        help="Run only baseline methods, skip grid search.",
    )
    parser.add_argument(
        "--grid_chunk_id",
        type=int,
        default=None,
        help="Chunk index for array job parallelization (0-based)",
    )
    parser.add_argument(
        "--total_grid_chunks",
        type=int,
        default=None,
        help="Total number of grid chunks",
    )
    parser.add_argument(
        "--merge_chunks",
        action="store_true",
        default=False,
        help="Merge grid_search_chunk_*.txt files into grid_search.txt",
    )
    return parser.parse_args()


def check_index_exists(
        output_file: str, 
        experiment_name: str
    ) -> bool:
    """
        Args:
            output_file: the path to the output file.
            experiment_name:
        Returns:
            True if the default index exists in the output file, False otherwise.
    """
    if not os.path.exists(output_file): return False
    df = pd.read_csv(output_file)
    return f'{experiment_name}' in df['experiment_name'].values


def check_param_combination_exists(
        output_file: str, 
        t_epsilon: float, 
        epsilon: float, 
        epsilon_p: float, 
        experiment_name: str
    ) -> bool:
    """
        Args:
            output_file: the path to the output file.
            t_epsilon: the value of t_epsilon.
            epsilon: the value of epsilon.
            epsilon_p: the value of epsilon_p.
            experiment_name
        Returns:
            True if the parameter combination exists in the output file, False otherwise.
    """
    if not os.path.exists(output_file):
        return False

    df = pd.read_csv(output_file)
    return not df[(df['t_epsilon'] == t_epsilon) & (df['epsilon'] == epsilon) & (df['epsilon_p'] == epsilon_p) & (df['experiment_name'] == f'{experiment_name}')].empty


def run_adv_examples_script(params: tuple) -> None:
    """
        Run the generate_adversarial_examples.py script with given parameters

        Args:
            params: the parameters to run the script with.
    """
    t_epsilon, epsilon, epsilon_p, experiment_name, lock, output_file, temp_dir, rej_lev_flag, baseline = params
    print(f'Running parameters: {params}', flush=True)
    reject_path = Path(f'experiments/{experiment_name}/rejection_levels/reject_at_{t_epsilon}_{epsilon}.json')

    if os.path.exists(reject_path):
        print("Loading rejection level...", flush=True)
        file = open(reject_path)
        reject_at = json.load(file)[0]
        if reject_at < 1:
            print("Rejection level too low", flush=True)
            return

    if rej_lev_flag == 1:
        compute_rejection_level(
            experiment_name = experiment_name,
            t_epsilon = t_epsilon,
            epsilon = epsilon,
            temp_dir = temp_dir
        )
    elif rej_lev_flag == 0:
        print('DETECTING... \n', flush=True)
        result = detect_adversarial_examples(
            experiment_name = experiment_name,
            t_epsilon = t_epsilon,
            epsilon = epsilon,
            epsilon_p = epsilon_p,
            temp_dir = temp_dir,
            baseline = baseline
        )
        print('DETECTION FINISHED...\n', flush=True)
        # Extract the metrics from the output
        output_lines = result.split('\n')
        good_defences = None
        wrong_rejection = None
        for line in output_lines:
            if "Percentage of good defences" in line:
                good_defences = float(line.split()[-1].strip(':'))
            if "Percentage of wrong rejections" in line:
                wrong_rejection = float(line.split()[-1].strip(':'))

        if good_defences is not None and wrong_rejection is not None:
            result_line = f"{t_epsilon},{epsilon},{epsilon_p},{experiment_name},{good_defences},{wrong_rejection}\n"
            print(result_line.strip())

            # Write the result to the file
            with lock:
                with open(output_file, 'a') as f:
                    f.write(result_line)
        print('Trial finished...\n', flush=True)
    else:
        print("Error: Invalid rej_lev flag", flush=True)
        return


def main() -> None:
    """
        Main function to run the grid search.
    """
    print("Grid search starting...", flush=True)
    args = parse_args()

    if args.baseline_only:
        print("Running baselines only...", flush=True)
        from detect_adversarial_examples import (
            reject_predicted_attacks_baseline,
            reject_predicted_attacks_baseline_matrices,
        )
        from utils.utils import get_input_shape, get_num_classes
        from constants.constants import DEFAULT_EXPERIMENTS

        exp = DEFAULT_EXPERIMENTS[args.experiment_name]
        input_shape = get_input_shape(exp['dataset'])
        num_classes = get_num_classes(exp['dataset'])
        epoch = exp['epochs']
        if args.temp_dir:
            wp = Path(f'{args.temp_dir}/experiments/{args.experiment_name}/weights/epoch_{epoch}.pth')
        else:
            wp = Path(f'experiments/{args.experiment_name}/weights/epoch_{epoch}.pth')

        reject_predicted_attacks_baseline(
            experiment_name=args.experiment_name, weights_path=wp,
            architecture_index=exp['architecture_index'],
            input_shape=input_shape, num_classes=num_classes,
            verbose=True, temp_dir=args.temp_dir)
        print("PENULTIMATE BASELINE FINISHED.", flush=True)

        reject_predicted_attacks_baseline_matrices(
            experiment_name=args.experiment_name,
            num_classes=num_classes, dataset=exp['dataset'],
            verbose=True, temp_dir=args.temp_dir)
        print("MATRIX BASELINE FINISHED.", flush=True)
        return

    experiment_path = Path(f'experiments/{args.experiment_name}/grid_search/')
    experiment_path.mkdir(parents=True, exist_ok=True)

    if args.merge_chunks:
        import glob
        output_file = experiment_path / 'grid_search.txt'
        chunk_files = sorted(glob.glob(str(experiment_path / 'grid_search_chunk_*.txt')))
        with open(output_file, 'w') as out:
            out.write("t_epsilon,epsilon,epsilon_p,experiment_name,good_defence,wrong_rejection\n")
            for cf in chunk_files:
                with open(cf) as inp:
                    inp.readline()  # skip header
                    out.write(inp.read())
        print(f"Merged {len(chunk_files)} chunks into {output_file}")
        return

    if args.grid_chunk_id is not None:
        output_file = experiment_path / f'grid_search_chunk_{args.grid_chunk_id}.txt'
    else:
        output_file = experiment_path / 'grid_search.txt'

    # Define the parameter grid
    if args.extensive_search == 1:
        vals = geomspace(
                    start = 0.00001,
                    stop = 1, 
                    num = 5
                )
        t_epsilon_values = vals
        epsilon_values = vals
        epsilon_p_values = vals
    else:
        t_epsilon_values = args.t_epsilon_values
        epsilon_values = args.epsilon_values
        epsilon_p_values = args.epsilon_p_values

    # Define the directory path
    dir_path = Path(f'experiments/{args.experiment_name}/rejection_levels/')

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

    param_grid = list(
        itertools.product(
            t_epsilon_values, 
            epsilon_values, 
            epsilon_p_values, 
            [f'{args.experiment_name}'],
            [args.rej_lev]
        )
    )

    # Initialize the output file if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write("t_epsilon,epsilon,epsilon_p,experiment_name,good_defence,wrong_rejection\n")
    lock = mp.Manager().Lock()
    # Prepare arguments
    param_grid_with_lock = [(t_epsilon, epsilon, epsilon_p, index, lock, output_file, args.temp_dir, args.rej_lev)
                            for t_epsilon, epsilon, epsilon_p, index, _ in param_grid]

    # This case assumes rejection levels were already computed.
    if args.rej_lev == 0:
        param_grid_filtered = [(t_epsilon,
                                epsilon,
                                epsilon_p,
                                experiment_name,
                                lock,
                                output_file,
                                args.temp_dir,
                                args.rej_lev)
                               for t_epsilon, epsilon, epsilon_p, experiment_name, lock, output_file, args.temp_dir, _ in param_grid_with_lock
                               if f'reject_at_{t_epsilon}_{epsilon}.json' in files_to_keep]
        for i in range(len(param_grid_filtered)):
            param_grid_filtered[i] = param_grid_filtered[i] + (False,)
        # Slice grid for array job parallelization
        if args.grid_chunk_id is not None and args.total_grid_chunks is not None:
            N = len(param_grid_filtered)
            base = N // args.total_grid_chunks
            rem = N % args.total_grid_chunks
            if args.grid_chunk_id < rem:
                start = args.grid_chunk_id * (base + 1)
                end = start + (base + 1)
            else:
                start = args.grid_chunk_id * base + rem
                end = start + base
            param_grid_filtered = param_grid_filtered[start:end]
            print(f"Grid chunk {args.grid_chunk_id}/{args.total_grid_chunks}: combos [{start},{end}) = {len(param_grid_filtered)} items")
        with Pool(processes=args.nb_workers) as pool:
            pool.map(run_adv_examples_script, param_grid_filtered)
    # This case computes rejection levels only using std and d1.
    else:
        param_grid_with_lock_rej_lev = [(t_epsilon,
                                         epsilon,
                                         0,
                                         experiment_name,
                                         lock,
                                         output_file,
                                         args.temp_dir,
                                         args.rej_lev,
                                         False)
                                        for t_epsilon, epsilon, epsilon_p, experiment_name, _ in param_grid]

        with Pool(processes=args.nb_workers) as pool:
            pool.map(run_adv_examples_script, param_grid_with_lock_rej_lev)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
