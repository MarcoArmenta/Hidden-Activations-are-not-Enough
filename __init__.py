DEFAULT_TRAININGS = {
    'experiment_0': {
        'dataset': 'mnist',
        'optimizer': 'sgd',
        'lr': 0.01,
        'batch_size': 8,
        'epoch': 21,
        'reduce_lr_each': 5,
        'save_every': 2
    },
    'experiment_1': {
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every': 2
    },
    'experiment_2': {
        'optimizer': 'adam',
        'dataset': 'fashion',
        'lr': 1e-06,
        'batch_size': 16,
        'epoch': 51,
        'reduce_lr_each': 20,
        'save_every': 10
    },
    'experiment_3': {
        'optimizer': 'sgd',
        'dataset': 'fashion',
        'lr': 0.1,
        'batch_size': 16,
        'epoch': 51,
        'reduce_lr_each': 20,
        'save_every': 10
    }
}
