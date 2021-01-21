import glob
import itertools
import copy
import torch

from core.experiment import Experiment


def fn_wrapper(*args, **kwargs):
    f = Experiment(**kwargs)
    f()


if __name__ == '__main__':
    from pytorch_utils.gpu_fn_scatter import configs_from_grid, scatter_fn_on_devices

    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    output_root_dir = '/home/pma/chofer/repositories/py_supcon_vs_ce/results_xmas_performance'

    args_template = \
        {
            'output_root_dir': output_root_dir,
            'num_batches': 100000,
            'tag': 'performance_rerun_fixed_weights',
            'eval_interval': None,
            'num_runs': 1,
            'num_samples': None,
            'model': None,
            'lr_init': 0.1,
            'weight_decay': 0.0001,
            'ds_train': None,
            'ds_test': None,
            'momentum': 0.9,
            'augment': None,
            'batch_size': None,
            'losses': None,
            'losses_track_only': (),
            'w_losses': None,
            'evaluation_policies': (
                'linear', 'retrained_linear', 'explicit_linear'),
            'scheduler': None
        }

    model_and_loss = []
    supcon = {
        'losses': (('SupConLoss', {'temperature': 0.1}),),
        'model': ("ResNet18", {
            'compactification_cfg': ('sphere_l2', {}),
            'latent_dim': None,
            'linear_cfg': ('Linear', {'bias': False}),
            'batch_norm': True,
        })
    }

    ce_template = {
        'losses': [('CrossEntropy', {'reduction': 'mean'})],
        'model': ("ResNet18", {
            'compactification_cfg': None,
            'linear_cfg': None,
            'batch_norm': True, 'latent_dim': None,
        }),
        'losses_track_only': (
            ('SupConLoss', {'temperature': 0.1, 'project_input_to_sphere': True}),)
    }

    ce_vanilla = copy.deepcopy(ce_template)
    ce_vanilla['model'][1].update(
        {
            'compactification_cfg': ('none', {}),
            'linear_cfg': ('Linear', {'bias': False}),
        }
    )

    ce_fixed_weights = copy.deepcopy(ce_template)
    ce_fixed_weights['model'][1].update(
        {
            'compactification_cfg': ('none', {}),
            'linear_cfg': ('FixedSphericalSimplexLinear', {}),
        }
    )

    ce_samples_on_sphere = copy.deepcopy(ce_template)
    ce_samples_on_sphere['model'][1].update(
        {
            'compactification_cfg': ('sphere_l2', {}),
            'linear_cfg': ('Linear', {'bias': False}),
        }
    )
    model_and_loss = [
        # supcon, 
        # ce_vanilla, 
        ce_fixed_weights, 
        # ce_samples_on_sphere, 
    ]

    data = [
        {'ds_train': 'cifar10_train', 'ds_test': 'cifar10_test'},
        {'ds_train': 'cifar100_train', 'ds_test': 'cifar100_test'},
    ]

    scheduler = [
        {'scheduler': p} for p in ['exponential', 'cosine']
    ]

    augment = [
        {'augment': p} for p in ['none', 'standard']
    ]

    batch_size = [
        {'batch_size': b} for b in [256, 512]
    ]

    l_args = []
    for m, d, s, a, b in itertools.product(model_and_loss, data, scheduler, augment, batch_size):
        args = copy.deepcopy(args_template)
        args.update(m)
        args.update(d)
        args.update(s)
        args.update(a)
        args.update(b)
        l_args.append(args)

    config = [((), a) for a in l_args]
    # print(len(config))

    # for i, (_, a) in enumerate(config):
        
    #     with torch.cuda.device('cuda:0'):
    #         print(i)
    #         Experiment(**a)()

    scatter_fn_on_devices(fn_wrapper, config, [1, 3], 1)
