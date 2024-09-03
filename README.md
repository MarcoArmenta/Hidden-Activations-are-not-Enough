# MatrixStatistics

This repository refers to the paper: ...

## Repository content

* [neuralteleportation](neuralteleportation) : contains the main classes for network teleportation. 
* [layers](neuralteleportation/layers): contains the classes necessary for teleporting individual layers. 
* [models](neuralteleportation/models): contains frequently used models such as MLP, VGG, ResNet and DenseNet.
* [experiments](neuralteleportation/experiments): contains experiments using teleportation. 
* [tests](tests): contains black-box tests for network teleportation. 

## Setting up 
First, install dependencies   
```bash
# set-up project's environment
cd neuralteleportation
conda env create -f requirements/neuralteleportation.yml

# activate the environment so that we install the project's package in it
conda activate neuralteleportation
pip install -e .

```
To test that the project was installed successfully, you can try the following command from the Python REPL:
```python

# now you can do:
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel   
```

## Running experiments

If you have access to resources on a cluster through SLURM, the following will
automatically submit a SLURM job for each training job:

**WARNING** The MLP model

### Train the network

```bash
python neuralteleportation/utils/statistics_teleportations.py
```

Run the following scripts in order

`training.py'

`matrices_on_epoch.py'

`adversarial_attacks.py'

## Grid search on s, d1 and d2

Run the script:

`grid_search.py'

### Read results

The script `read_results.py' will read and take all the results written in experiment/{default_index}/grid_search/grid_search_{default_index}.txt and run the `detect_adversarial_examples.py' script on the top 3 results that have the highets difference between percentage of good defences and percentage of wrong rejections, and write the outputs which contain the number of adversarial examples detected and generated per attack method in the directory experiment/{default_index}/results/. The best result will have 0 as suffix and the next 1 and so on.
