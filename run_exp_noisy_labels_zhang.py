import glob
import itertools
import copy
import torch

from core.experiment import ExpNoisyLabeledDataZhang
from nb_common import load_results
from pathlib import Path


def fn_wrapper(*args, **kwargs):
    f = ExpNoisyLabeledDataZhang(**kwargs)
    f()


def check_args_equal(first, second):
    
    # assert type(first) == type(second), (first, second)
    if isinstance(first, (int, float, str, type(None))):
        return first == second 
    elif isinstance(first, dict):
        return all((
            check_args_equal(first[k], second[k]) for k in first if k != 'tag'
        ))
    elif isinstance(first, (list, tuple)):
        if len(first) == 2 and isinstance(first[0],str) and isinstance(first[1], dict):
            assert len(second) == 2 and isinstance(second[0],str) and isinstance(second[1], dict)
            return first[0] == second[0] and check_args_equal(first[1], second[1])
        else:        
            return all((
                check_args_equal(first_i, second_i) for first_i, second_i in zip(first, second)
            ))
    else:
        raise ValueError(first, second)


if __name__ == '__main__':
    from pytorch_utils.gpu_fn_scatter import configs_from_grid, scatter_fn_on_devices

    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    output_root_dir = '/home/pma/chofer/repositories/py_supcon_vs_ce/results_xmas_noisy_label_zhang'

    args_template = \
        { 
            'output_root_dir' : output_root_dir, 
            'num_batches' : 100000, 
            'label_noise_fraction': None,
            'tag' : 'ce_fixed_cifar100', 
            'eval_interval' : 500, 
            'num_runs' : 1, 
            'num_samples' : None, 
            'model' : None, 
            'lr_init' : 0.1, 
            'weight_decay' : 0.0001, 
            'ds_train' : None, 
            'ds_test' : None, 
            'momentum' : 0.9, 
            'augment' : 'none', 
            'batch_size' : None, 
            'losses' : None, 
            'losses_track_only' : (), 
            'w_losses' : None, 
            'evaluation_policies': (
                'linear', 'retrained_linear', 'explicit_linear'), 
            'scheduler': 'exponential'
        }

    supcon = {
        'losses': (('SupConLoss', {'temperature': 0.1}),), 
        'model': ("ResNet18", {
                'compactification_cfg': ('sphere_l2', {}), 
                'latent_dim': None, 
                'linear_cfg': ('Linear', {'bias': False}), 
                'batch_norm': True, 
            })
    }

    ce_vanilla = {
        'losses': [('CrossEntropy', {'reduction': 'mean'})], 
        'model': ("ResNet18", {
                'compactification_cfg': ('none', {}), 
                'latent_dim': None, 
                'linear_cfg': ('Linear', {'bias': False}), 
                'batch_norm': True, 
            }),
        'losses_track_only': (
            ('SupConLoss', {'temperature': 0.1, 'project_input_to_sphere': True})
            ,)
    }

    ce_fixed_weights = {
        'losses': [('CrossEntropy', {'reduction': 'mean'})], 
        'model': ("ResNet18", {
                'compactification_cfg': ('none', {}), 
                'latent_dim': None, 
                'linear_cfg': ('FixedSphericalSimplexLinear', {}), 
                'batch_norm': True, 
            }),
        'losses_track_only': (
            ('SupConLoss', {'temperature': 0.1, 'project_input_to_sphere': True})
            ,)
    }

    model_and_loss = [
        # supcon, 
        # ce_vanilla, 
        ce_fixed_weights
    ]

    label_noise_fraction = [
        {'label_noise_fraction': p} 
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ]

    data = [
        # {'ds_train': 'cifar10_train', 'ds_test': 'cifar10_test'}, 
        {'ds_train': 'cifar100_train', 'ds_test': 'cifar100_test'}, 
    ]

    batch_size = [
        {'batch_size': b} 
        for b in 
        [
            # 64, 
            # 128, 
            256
        ]
    ]

    l_args = []
    for m, l, d, b in itertools.product(model_and_loss, label_noise_fraction, data, batch_size):
        args = copy.deepcopy(args_template)
        args.update(m)
        args.update(l)
        args.update(d)
        args.update(b)
        l_args.append(args)

    existing_result_args = [
        r.experiment_args for r in 
        load_results(Path(output_root_dir))
    ]

    for arg in existing_result_args: del arg['experiment_type']
    tmp = []

    for arg in l_args:
        if not any((
            check_args_equal(arg, ex_arg) for ex_arg in existing_result_args
        )):
            tmp.append(arg)

    print("Experiments to compute: {}".format(len(l_args)))
    print("Allready computed results: {}".format(len(existing_result_args)))
    print("Remaining experiments: {}".format(len(tmp)))

    config = [((), a) for a in tmp]

    scatter_fn_on_devices(fn_wrapper, config, [1, 2, 3], 2)
