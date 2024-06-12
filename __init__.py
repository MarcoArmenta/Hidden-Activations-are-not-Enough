HIDDEN_SIZE = [(500, 500, 500, 500, 500),

               (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),

               (500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
                500, 500, 500, 500, 500, 500, 500, 500, 500, 500),

               (100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                ]

DEFAULT_TRAININGS = {
    'experiment_0': {
        'dataset': 'mnist',
        'optimizer': 'sgd',
        'layers': HIDDEN_SIZE[0],
        'lr': 0.01,
        'batch_size': 8,
        'epoch': 21,
        'reduce_lr_each': 5,
        'save_every': 2
    },
    'experiment_1': {
        'optimizer': 'momentum',
        'dataset': 'mnist',
        'layers': HIDDEN_SIZE[0],
        'lr': 0.01,
        'batch_size': 32,
        'epoch': 11,
        'reduce_lr_each': 5,
        'save_every': 2
    },
    'experiment_2': {
        'optimizer': 'adam',
        'dataset': 'fashion',
        'layers': HIDDEN_SIZE[0],
        'lr': 1e-06,
        'batch_size': 16,
        'epoch': 51,
        'reduce_lr_each': 20,
        'save_every': 10
    },
    'experiment_3': {
        'optimizer': 'sgd',
        'dataset': 'fashion',
        'layers': HIDDEN_SIZE[0],
        'lr': 0.1,
        'batch_size': 16,
        'epoch': 51,
        'reduce_lr_each': 20,
        'save_every': 10
    }
}
