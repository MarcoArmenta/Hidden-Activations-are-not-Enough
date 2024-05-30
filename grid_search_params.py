import itertools
import subprocess
from multiprocessing import Pool

# Define the parameter grid
std_values = [0.75, 1, 1.5, 2]
d1_values = [0.01, 0.1, 0.3, 0.5, 0.8, 1]
d2_values = [0.01, 0.1, 0.3, 0.5, 0.8, 1]
index = [0, 1, 2, 3]

param_grid = list(itertools.product(std_values, d1_values, d2_values, index))

# File to store the results
results_file = 'grid_search_results.txt'


# Function to run the adversarial_attacks.py script with given parameters
def run_script(params):
    std, d1, d2, index = params

    cmd = f"source ~/NeuralNets/MatrixStatistics/matrix/bin/activate &&" \
          f" python adversarial_attacks.py --std {std} --d1 {d1} --d2 {d2} " \
          f"--default_index {index}"
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, executable="/bin/bash"
    )

    # Extract the metrics from the output
    output_lines = result.stdout.split('\n')
    print(result)
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
    return result_line


if __name__ == "__main__":
    # Define the number of processes to use
    num_processes = 8  # Adjust this number based on your system's capabilities

    # Use multiprocessing Pool to run the scripts in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_script, param_grid)

    # Write all results to the results file
    with open(results_file, 'w') as f:
        f.write(f"std,d1,d2,default_index,good_defence,wrong_rejection\n")
        f.writelines(results)
