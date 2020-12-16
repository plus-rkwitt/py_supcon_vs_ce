import glob
import tempfile

import core.experiment
from pytorch_utils.logging import LoggerReader
from pytorch_utils.evaluation import apply_model, argmax_and_accuracy


def load_results(root, tag=None):
    
    results = [LoggerReader(r) for r in glob.glob(str(root / '*'))]
    
    if tag is not None:
        results = [r for r in results if r.experiment_args['tag'] == tag]
        
    results = sorted(results, key=lambda x: x.date)

    return results


def load_experiment_context(path, run_i=0):

    r = LoggerReader(path)
    exp_args = r.experiment_args



    exp_type = getattr(core.experiment, exp_args['experiment_type'].split('.')[-1])

    del exp_args['experiment_type']
    for k, v in exp_args.items():
        if isinstance(v, list):
            exp_args[k] = tuple(v)


    exp_args['losses_track_only'] = ()
    with tempfile.TemporaryDirectory() as pth:
        exp_args['output_root_dir'] = str(pth)
        exp = exp_type(**exp_args)

    it = iter(exp.ds_setup_iter())
    for _ in range(run_i+1):
        next(it)

    ret = {k: v for k, v in exp.__dict__.items() if k[:2] != "__"}
    ret['model'] = r.load_model(run_i, 'model')

    return ret


def compute_latent(path, run_i=0, train=True, device='cpu'):
    exp_context = load_experiment_context(path, run_i=run_i)
    
    if train:
        ds = exp_context['ds_train']
    else:
        ds = exp_context['ds_test']
        
    feat_ext = exp_context['model'].feat_ext
    
    Z, Y = apply_model(dataset=ds, model=feat_ext, device=device)
    
    return Z, Y