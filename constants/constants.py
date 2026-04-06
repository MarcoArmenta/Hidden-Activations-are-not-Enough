"""
This module defines constants used throughout the experiments:

ARCHITECTURES: A list of tuples defining different neural network architectures.
              Each tuple contains layer sizes for MLPs or (conv_layers, fc_layers) for CNNs.
              The architectures vary in depth and width. 
              Note that it's possible to use ResNet, AlexNet, and VGG architectures.

ATTACKS: A list of adversarial attack methods used in the experiments.

DEFAULT_EXPERIMENTS: A dictionary of default experiment configurations.
                     Each key represents an experiment name, and the value is a dictionary
                     containing parameters for the experiment.
"""

ARCHITECTURES = [
    # 0 -> 0, 1, 2, 3
    (500, 500, 500, 500, 500),
    # 1 -> 4, 5, 6
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    # 2 -> 7, 8
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
    1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    # 3 -> 9, 10,
    (10000, 10000),
    # 4 -> 11, 12
    (10000, 10000, 10000, 10000, 10000),
    # 5 -> 13, 14
    (675000, 1500, 1500, 1500, 1500),
    # 6 -> 15, 16
    (2500000, ),
    # 7 -> 8
    (814, 351, 118, 467, 823, 191, 756, 628, 935, 270),
    # CNNs
    ((10,10,10,10), (300,)),
    ((32,64), (128,)),
    ((16,32,64), (128,)),
    ("AlexNet"),
    ("ResNet"),
    ("VGG")
]

# 16 adversarial attacks (see ATTACK_CATEGORIES for grouping)
ATTACKS = [
    # --- Gradient-based attacks ---
    "GN",           # Gaussian Noise (baseline)
    "FGSM",         # Fast Gradient Sign Method (Goodfellow et al., 2015)
    "PGD",          # Projected Gradient Descent (Madry et al., 2018)
    "EOTPGD",       # Expectation Over Transformation PGD
    "MIFGSM",       # Momentum Iterative FGSM (Dong et al., 2018)
    "VMIFGSM",      # Variance-reduced MI-FGSM
    "CW",           # Carlini & Wagner L2 (Carlini & Wagner, 2017)
    "DeepFool",     # DeepFool (Moosavi-Dezfooli et al., 2016)
    # --- AutoAttack ensemble (Croce & Hein, ICML 2020) ---
    # The full AutoAttack suite: APGD-CE + APGD-T + FAB + Square
    "APGD",         # Auto-PGD with CE loss (APGD-CE)
    "APGDT",        # Auto-PGD with targeted DLR loss (APGD-T)
    "FAB",          # Fast Adaptive Boundary attack
    "Square",       # Square Attack (gradient-free, score-based)
    # --- Gradient-free / perturbation-based attacks ---
    "Pixle",        # Pixle attack (perturbation-free, pixel remapping)
    "SPSA",         # Simultaneous Perturbation Stochastic Approximation (gradient-free)
    # --- Elastic-net attacks ---
    "EADL1",        # Elastic-net Attack L1 (Chen et al., 2018)
    "EADEN",        # Elastic-net Attack Decision-based
]

# AutoAttack ensemble components for verification against Croce & Hein (ICML 2020):
# APGD-CE (APGD), APGD-T (APGDT), FAB, Square
# torchattacks uses default AutoAttack parameterization matching the original paper.
AUTOATTACK_COMPONENTS = ["APGD", "APGDT", "FAB", "Square"]

# Subset of attacks for ImageNet experiments (covers gradient, optimization,
# ensemble, and gradient-free categories while keeping compute manageable)
IMAGENET_ATTACKS = ["FGSM", "PGD", "CW", "DeepFool", "APGD", "Square"]

# Attack categories for paper presentation
ATTACK_CATEGORIES = {
    "gradient_based": ["FGSM", "PGD", "EOTPGD", "MIFGSM", "VMIFGSM", "CW", "DeepFool"],
    "autoattack":     ["APGD", "APGDT", "FAB", "Square"],
    "gradient_free":  ["Square", "SPSA", "Pixle"],
    "elastic_net":    ["EADL1", "EADEN"],
    "baseline_noise": ["GN"],
}

DEFAULT_EXPERIMENTS = {
    'resnet_cifar100': { # accuracy: ~0.77 (SOTA-range with cosine schedule)
        'pretrained': False,
        'dataset': 'cifar100',
        'batch_size': 128,
        'lr': 0.1,
        'epochs': 200,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'scheduler': 'cosine',
        'architecture_index': -2,
    },
    'resnet_cifar10': {
        'pretrained': False,
        'dataset': 'cifar10',
        'batch_size': 32,
        'lr': 0.009230996304925737,
        'epochs': 100,
        'optimizer': 'sgd',
        'momentum': 0.8231700796140718,
        'weight_decay': 0.0007273616737214423,
        'scheduler': 'multi',
        'architecture_index': -2,
    },
    'alexnet_cifar10': {
        'epochs': 70,
        'batch_size': 16,
        'lr': 5.9727572025986934e-05,
        'optimizer': 'adam',
        'momentum': 0,
        'weight_decay': 0.0015747834011850491,
        'dataset': 'cifar10',
        'architecture_index': -3,
        'scheduler': 'multi',
    },
    'alexnet_imagenet': {
        'pretrained': True,
        'dataset': 'imagenet',
        'architecture_index': -3,
        'epochs': 0,
        'batch_size': 64,
        'lr': 0.0,
        'optimizer': 'sgd',
        'momentum': 0.0,
        'weight_decay': 0.0,
        'scheduler': 'cosine',
    },
    'resnet_imagenet': {
        'pretrained': True,
        'dataset': 'imagenet',
        'architecture_index': -2,
        'epochs': 0,
        'batch_size': 64,
        'lr': 0.0,
        'optimizer': 'sgd',
        'momentum': 0.0,
        'weight_decay': 0.0,
        'scheduler': 'cosine',
    },
    'vgg_imagenet': {
        'pretrained': True,
        'dataset': 'imagenet',
        'architecture_index': -1,
        'epochs': 0,
        'batch_size': 64,
        'lr': 0.0,
        'optimizer': 'sgd',
        'momentum': 0.0,
        'weight_decay': 0.0,
        'scheduler': 'cosine',
    },
    'mlp_mnist': { # accuracy 0.98
        'pretrained': False,
        'dataset': 'mnist',
        'epochs': 5,
        'layers': (512, 512, 512),
        'batch_size': 32,
        'lr': 0.06941109118141582,
        'optimizer': 'sgd',
        'momentum': 0.6215073814724885,
        'weight_decay': 6.914150600886057e-05,
        'scheduler': None,
    },
    'vgg_cifar10': {
        'epochs': 5,
        'dataset': 'cifar10',
        'batch_size': 32,
        'lr': 0.00012344300494603974,
        'optimizer': 'adam',
        'momentum': 0.0,
        'weight_decay': 0.0001391604103021994,
        'scheduler': 'multi',
        'architecture_index': -1,
    },
    'vgg_cifar100': {
        'epochs': 150,
        'dataset': 'cifar100',
        'batch_size': 32,
        'lr': 0.00012344300494603974,
        'optimizer': 'adam',
        'momentum': 0.0,
        'weight_decay': 0.0001391604103021994,
        'scheduler': 'multi',
        'architecture_index': -1,
    },
    'lenet_cifar10': {
        'epochs': 507,
        'dataset': 'cifar10',
        'architecture_index': -4,
        'batch_size': 256,
        'lr': 0.09997332160161512,
        'optimizer': 'sgd',
        'momentum': 0.356455066927686,
        'weight_decay': 0.0039547952367518496,
        'reduce_lr': 500,
        'scheduler': None,
    },


    'experiment_0': {
        'architecture_index': 0,
        'dataset': 'mnist',
        'optimizer': 'sgd',
        'lr': 0.01,
        'batch_size': 8,
        'epoch': 1,
        'reduce_lr_each': 5,
        'save_every_epochs': 2,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_1': {
        'architecture_index': 0,
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every_epochs': 2,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_2': {
        'architecture_index': 0,
        'optimizer': 'adam',
        'dataset': 'fashion',
        'lr': 1e-06,
        'batch_size': 32,
        'epoch': 81, # 21
        'reduce_lr_each': 20,
        'save_every_epochs': 10,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_3': {
        'architecture_index': 0,
        'optimizer': 'sgd',
        'dataset': 'fashion',
        'lr': 0.1,
        'batch_size': 16,
        'epoch': 51, # 35
        'reduce_lr_each': 20,
        'save_every_epochs': 10,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_4': {
        'architecture_index': 1,
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 7,
        'reduce_lr_each': 5,
        'save_every_epochs': 2,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_5': {
        'architecture_index': 1,
        'optimizer': 'momentum',
        'dataset': 'fashion',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every_epochs': 5,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_6': {
        'architecture_index': 1,
        'optimizer': 'adam',
        'dataset': 'mnist',
        'lr': 0.001,
        'batch_size': 128,
        'epoch': 6,
        'reduce_lr_each': 3,
        'save_every_epochs': 2,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_7': {
        'architecture_index': 2,
        'optimizer': 'adam',
        'dataset': 'fashion',
        'lr': 0.0001,
        'batch_size': 16,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every_epochs': 5,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_8': {
        'architecture_index': 7,
        'optimizer': 'adam',
        'dataset': 'mnist',
        'lr': 0.001,
        'batch_size': 128,
        'epoch': 6,
        'reduce_lr_each': 20,
        'save_every_epochs': 5,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_9': {
        'architecture_index': 3,
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 1,
        'reduce_lr_each': 5,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_10': {
        'architecture_index': 3,
        'optimizer': 'momentum',
        'dataset': 'fashion',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_11': {
        'architecture_index': 4,
        'optimizer': 'sgd',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 64,
        'epoch': 6,
        'reduce_lr_each': 20,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_12': {
        'architecture_index': 4,
        'optimizer': 'sgd',
        'dataset': 'fashion',
        'lr': 0.01,
        'batch_size': 64,
        'epoch': 16,
        'reduce_lr_each': 20,
        'save_every_epochs': 5,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_13': { #this only trains on 40 GBs GPU
        'architecture_index': 5,
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.001,
        'batch_size': 128,
        'epoch': 11,
        'reduce_lr_each': 40,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 1e-5,
        'dropout': 0.5,
    },
    'experiment_14': { #this only trains on 40 GBs GPU
        'architecture_index': 5,
        'optimizer': 'momentum',
        'dataset': 'fashion',
        'lr': 0.001,
        'batch_size': 128,
        'epoch': 11,
        'reduce_lr_each': 40,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 1e-5,
        'dropout': 0.5,
    },
    'experiment_15': { #this only trains on 40 GBs GPU
        'architecture_index': 6,
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.001,
        'batch_size': 256,
        'epoch': 11,
        'reduce_lr_each': 40,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_16': { #this only trains on 40 GBs GPU
        'architecture_index': 6,
        'optimizer': 'momentum',
        'dataset': 'fashion',
        'lr': 0.001,
        'batch_size': 256,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_17': {
        'architecture_index': 8,
        'optimizer': 'sgd',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 11,
        'reduce_lr_each': 40,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_18': {
        'architecture_index': 8,
        'optimizer': 'momentum',
        'dataset': 'cifar10',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 61,
        'reduce_lr_each': 55,
        'save_every_epochs': 5,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_19': {
        'architecture_index': 9,
        'optimizer': 'adam',
        'dataset': 'cifar10',
        'lr': 0.001,
        'batch_size': 64,
        'epoch': 61,
        'reduce_lr_each': 55,
        'save_every_epochs': 5,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_20': {
        'architecture_index': -2,  # ResNet
        'optimizer': 'sgd',
        'dataset': 'cifar10',
        'lr': 0.01,
        'batch_size': 16,
        'epoch': 5,
        'reduce_lr_each': 3,
        'save_every_epochs': 1,
        'residual': True,
        'weight_decay': 0,
        'dropout': 0,
    },
    'experiment_21': {
        'architecture_index': 9,
        'optimizer': 'sgd',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 16,
        'epoch': 5,
        'reduce_lr_each': 50,
        'save_every_epochs': 1,
        'residual': False,
        'weight_decay': 0,
        'dropout': 0,
    },
}
