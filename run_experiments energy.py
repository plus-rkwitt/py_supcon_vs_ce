import glob

from core.experiment_energy import ExpEnergyRandomLabeledData, ExpEnergy


def fn_wrapper(*args, **kwargs):
    f = ExpEnergy(**kwargs)
    f()


if __name__ == '__main__':
    from pytorch_utils.gpu_fn_scatter import configs_from_grid, scatter_fn_on_devices

    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    # output_root_dir = '/scratch2/chofer/toporeg_sandbox'
    output_root_dir = '/tmp/grid_1/'

    grid = \
        {
            "num_runs": [1],
            "output_root_dir": ["/tmp/tmp_results/"],
            "num_batches": [1000],
            "augment": ["standard"],
            "tag": [""],
            "model": [
                ('SimpleCNN13',
                 {'batch_norm': True,
                  'drop_out': False,
                  'final_bn': True,
                  'compactification': 'sphere_l2',
                  'linear_bias': False,
                  'cls_norm': 'none',
                  'latent_dim': None})
            ],
            "batch_cfg": [(4, {'+': 32, '-': 32})],
            "lr_init": [0.1],
            "momentum": [0.9],
            "weight_decay": [0.001],
            "ds_train": ["cifar10_train"],
            "ds_test": ["cifar10_test"],
            "energies": [
                (
                    ('CrossEntropy', {}),
                    ('SphereRepulsion', {
                     'sub_batch_agg': 'soft_min', 'temperature': 0.1, 'negatives_gradient': True}),
                    ('SphereAttraction', {
                     'sub_batch_agg': 'soft_max', 'temperature': 0.1})
                )
            ],
            "w_energies": [None],
            "eval_interval": [None],
            "evaluation_policies": [('optimized_linear', 'explicit_linear')],
        }

    cfgs = configs_from_grid(grid)
    print(len(cfgs))
    for c in cfgs: print(c['energies'])

    # for c in cfgs:
    #     c['output_root_dir'] = output_root_dir
    #     c['eval_interval'] = c['num_batches']

    config = [((), c) for c in cfgs]
    # scatter_fn_on_devices(fn_wrapper, config, [2, 3], 1)
