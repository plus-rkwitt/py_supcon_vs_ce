import glob
import tempfile
import pandas as pd

import core.experiment
from pytorch_utils.logging import LoggerReader
from pytorch_utils.evaluation import apply_model, argmax_and_accuracy


def load_results(root, tag=None):
    
    results = [LoggerReader(r) for r in glob.glob(str(root / '*'))]
    
    if tag is not None:
        results = [r for r in results if r.experiment_args['tag'] == tag]
        
    results = sorted(results, key=lambda x: x.date)

    return results


def progress(logger_reader):
    n_runs = len(logger_reader)
    
    if len(logger_reader) == 0 or 'batch_i' not in logger_reader[-1]:
        return "not started"    
    
    n_batches = logger_reader[-1]["batch_i"]+1        
        
    args = logger_reader.experiment_args
    
    if n_runs == args['num_runs'] and n_batches == args['num_batches']:
        return True
    else:
        s = '{}/{}'.format(n_runs, str(args['num_runs']))
        s += ' {:.2%} '.format(n_batches/float(args['num_batches']))
        return s
    

def args_df_from_results(results, args_white_list=None, args_simple=None):
    args_white_list = {} if args_white_list is None else args_white_list
    args_simple = dict() if args_simple is None else args_simple
    R = []
    
    for i, r in enumerate(results):        
        df = {}
        df.update({k: f(r.experiment_args) for k, f in args_simple.items()})
        df.update({k: str(v) if isinstance(v, (list, tuple)) else v for  k, v in r.experiment_args.items() if k in args_white_list})
        df['experiment'] = r.experiment_args['experiment_type']
        df['date'] = r.date
        df = pd.DataFrame(df, index=[i]) 
        df['progress'] = progress(r)
        
        R.append(df)   
        
    return pd.concat(R, sort=False)


def load_experiment_context(path, run_i=0):

    r = LoggerReader(path)
    exp_args = r.experiment_args

    for k, v in exp_args.items(): 
        if isinstance(v, list):
            exp_args[k] = tuple(v)

    exp_type = getattr(core.experiment, exp_args['experiment_type'].split('.')[-1])

    del exp_args['experiment_type']

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
    
    Z, Y = apply_model(dataset=ds, model=feat_ext, device=device, shuffle=False)
    
    return Z, Y
