ARCHITECTURES = [(500, 500, 500, 500, 500),

                 (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),

                 (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                  1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),

                 (10000, 10000),
               ]

ATTACKS = ["GN", "FGSM", "RFGSM", "PGD", "EOTPGD", "FFGSM", "TPGD", "MIFGSM", "UPGD", "DIFGSM", "NIFGSM",
           "PGDRS", "SINIFGSM", "VMIFGSM", "VNIFGSM", "CW", "PGDL2", "PGDRSL2", "DeepFool", "SparseFool",
           "OnePixel", "Pixle", "FAB"]


DEFAULT_EXPERIMENTS = {
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
    },
    'experiment_2': {
        'architecture_index': 0,
        'optimizer': 'adam',
        'dataset': 'fashion',
        'lr': 1e-06,
        'batch_size': 32,
        'epoch': 81,
        'reduce_lr_each': 20,
        'save_every_epochs': 10,
        'residual': False,
    },
    'experiment_3': {
        'architecture_index': 0,
        'optimizer': 'sgd',
        'dataset': 'fashion',
        'lr': 0.1,
        'batch_size': 16,
        'epoch': 51,
        'reduce_lr_each': 20,
        'save_every_epochs': 10,
        'residual': False,
    },
    'experiment_4': {
        'architecture_index': 1,
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 21,
        'reduce_lr_each': 20,
        'save_every_epochs': 5,
        'residual': True,
    },
    'experiment_5': {
        'architecture_index': 2,
        'optimizer': 'adam',
        'dataset': 'fashion',
        'lr': 0.001,
        'batch_size': 16,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every_epochs': 5,
        'residual': False,
    },
    'experiment_6': {
        'architecture_index': 3,
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 1,
        'reduce_lr_each': 5,
        'save_every_epochs': 1,
        'residual': False,
    }
}
