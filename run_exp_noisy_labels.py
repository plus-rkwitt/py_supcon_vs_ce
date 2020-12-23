import glob
import itertools
import copy
import torch

from core.experiment import ExpNoisyLabeledData


def fn_wrapper(*args, **kwargs):
    f = ExpNoisyLabeledData(**kwargs)
    f()


if __name__ == '__main__':
    from pytorch_utils.gpu_fn_scatter import configs_from_grid, scatter_fn_on_devices

    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    output_root_dir = '/home/pma/chofer/repositories/py_supcon_vs_ce/results_xmas_noisy_label'

    args_template = \
        { 
            'output_root_dir' : output_root_dir, 
            'num_batches' : 100000, 
            'label_noise_fraction': None,
            'tag' : 'noisy_labels', 
            'eval_interval' : 1000, 
            'num_runs' : 1, 
            'num_samples' : None, 
            'model' : None, 
            'lr_init' : 0.1, 
            'weight_decay' : 0.0001, 
            'ds_train' : None, 
            'ds_test' : None, 
            'momentum' : 0.9, 
            'augment' : 'none', 
            'batch_size' : 256, 
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

    model_and_loss = [supcon, ce_vanilla, ce_fixed_weights]

    label_noise_fraction = [
        {'label_noise_fraction': p} 
        for p in 
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ]

    data = [
        {'ds_train': 'cifar10_train', 'ds_test': 'cifar10_test'}, 
        {'ds_train': 'cifar100_train', 'ds_test': 'cifar100_test'}, 
    ]

    l_args = []
    for m, l, d in itertools.product(model_and_loss, label_noise_fraction, data):
        args = copy.deepcopy(args_template)
        args.update(m)
        args.update(l)
        args.update(d)
        l_args.append(args)

    config = [((), a) for a in l_args]

    # for i, (_, a) in enumerate(config):
    #     with torch.cuda.device('cuda:0'):
    #         print(i)
    #         ExpNoisyLabeledData(**a)()

    scatter_fn_on_devices(fn_wrapper, config, [0, 1, 2, 3], 1)
