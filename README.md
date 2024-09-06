# MatrixStatistics

This repository refers to the paper: Matrix Statistics for Data Representations of Neural Networks

## Repository content

* [constants](constants/constants.py) : contains the MLP architectures used with the 17 different hyperparameters used to train them, as well as the adversarial attack methods used. 
* [matrix_construction](matrix_construction/matrix_construction.py): contains the functions for constructing the matrices induced by a neural network. 
* [representation](matrix_construction/representation.py): contains the class to construct a quiver representation given a neural network and a single data sample and the induced matrix.
* [mlp](model_zoo/mlp.py): contains the mlp model that saves activations and pre-activations of neurons on a single forward pass to construct a quiver representation. 
* [test_mlp_rep](unit_test/test_mlp_rep.py): contains unit testing for construction of induced quiver representation for random inputs on random MLP architectures. 
* [utils](utils/utils.py): contains utils for datasets, models, computation of mean and standard deviation of matrices, etc.
* [test_mlp_rep](unit_test/test_mlp_rep.py): contains unit testing for construction of induced quiver representation for random inputs on random MLP architectures.


## Setting up 

```bash
virtualenv matrix
source matrix/bin/activate
pip install -r requirements.txt
```

## Running experiments

For a fixed index `idx` between 0 and 16 you train the default experiment specified in `constants/constants.py`.
You can add more default experiments by adding them to the DEFAULT_EXPERIMENTS dictionary in `constants/constants.py` and specifying their architecture and training hyperparameters.

### Hierarchy of experiment directory

The following scripts will create a directory `experiments` in which a subdirectory `experiments/{idx}/` will contain all the metadata for default experiment `idx`.

### 1. Train the networks
To train a network run the training script like this:

```bash
python training.py --default_index {idx}
```
This will save the weights and training history in `experiments/{idx}/weights/`.

### 2. Generate matrices
To generate matrices on a trained network run the following:

```bash
python generate_matrices.py --default_index {idx} --num_samples_per_class {num} --nb_workers {workers}
```

By default, `num=1000` and 'workers=8`.

This will create and save all the matrices in the directory `experiments/{idx}/matrices/{k}` where `k` runs over all the classes in the dataset. 
### 3. Compute matrix statistics
To compute matrix statistics, i.e., mean and standard deviation matrices per class run the following:

```bash
python compute_matrix_statistics.py --default_index {idx}
```
This will create a json file in `experiments/{idx}/matrices/matrix_statistics.json`.
### 4. Generate adversarial examples
To generate adversarial examples on the 21 attack methods specified in `constants/constants.py` run the following:

```bash
python generate_adversarial_examples.py --default_index {idx} --test_size {ts} --nb_workers {workers}
```
By default, `ts=-1`, which takes the whole test set to compute adversarial examples.
This will create files `experiments/{idx}/adversarial_examples/{attack_method}/adversarial_examples.pth` which contain all the generated adversarial examples per `attack_method`.
### 5. Generate adversarial matrices
To generate the matrices of each adversarial example generated in step 4 run the following:
```bash
python generate_adversarial_matrices.py --default_index {idx} --nb_workers {workers}
```
This will create files `experiments/{idx}/adversarial_matrices/{attack_method}/{i}/matrix.pth` corresponding to the different `attack_method`s where `i` runs from 0 to the total number of adversarial examples generated per attack method.
### 6. Compute matrices for rejection level
To generate matrices from a random subset of the train data to later compute the rejection level as specified in the paper, run the following:
```bash
python compute_matrices_for_rejection_level.py --default_index {idx} --num_samples_rejection_level {N} --nb_workers {workers}
```
By default, `N=10k`.
This will create files `experiments/{idx}/rejection_levels/matrices/{i}/matrix.pth` together with the prediction of the network on that sample `experiments/{idx}/rejection_levels/matrices/{i}/prediction.pth`.
### 7. Compute rejection levels
To run a grid search to compute several rejection levels run the following:
```bash
python grid_search.py --default_index {idx} --rej_level 1 --nb_workers {workers}
```
This will create files `experiments/{idx}/rejection_levels/reject_at_s_d.json` for different values of `s, d1`.
### 8. Detect adversarial examples
To run the detection algorithm using the pre-computed rejection levels run the following:
```bash
python grid_search.py --default_index {idx} --rej_level 0 --nb_workers {workers}
```
This will create a file `experiments/{idx}/grid_search/grid_search_{idx}.txt` which contains the percentages of good defence and wrong rejections for each combination of the parameters `s, d1, d2`.
### 9. Read results
To filter the results from the grid search run the following:
```bash
python read_results.py --default_index {idx}
```
This will create files `experiments/{idx}/results/{i}_output_{s}_{d1}_{d2}.txt` containing the output of the detection algorithm ran with parameters `s, d1, d2` with the top 3 results given by `i=0,1,2`.

## Order for running scripts
```mermaid
flowchart TD
    A[Training] --> B[Generate matrices]
    A --> C[Adversarial examples]
    A --> D[Matrices for rejection]
    B --> E[Compute matrix statistics]
    C --> F[Adversarial matrices] 
    E --> G[Rejection level]
    F --> G
    D --> G
    G --> H[Grid search]
    H --> I[Read results]
```


If you have access to a SLURM cluster, run the following jobs

**WARNING** The MLP model
