import torch
import numpy as np
import datetime
import uuid
import json
import pickle
import glob


from collections import defaultdict, Counter
from pathlib import Path


def convert_value_to_built_in_type(v):
    """Convert torch tensors to built in types, i.e., non zero-dimensional`tensor` is converted to `list` and zero-dimensional tensor is mapped to its containing value (`float` or `int`).
    For collections (`list`, `tupple`, `dict`) a cascade of this functionality 
    is applied.
    """
    if isinstance(v, torch.Tensor):
        new_v = v.detach().cpu()

        if v.ndimension() == 0:
            new_v = v.item()
        else:
            new_v = v.tolist()
    elif isinstance(v, np.ndarray):
        new_v = v.tolist()
    elif isinstance(v, dict):
        new_v = {k: convert_value_to_built_in_type(vv) for k, vv in v.items()}
    elif isinstance(v, list):
        new_v = [convert_value_to_built_in_type(vv) for vv in v]
    elif isinstance(v, tuple):
        new_v = tuple((convert_value_to_built_in_type(vv) for vv in v))
    else:
        new_v = v

    return new_v


def get_an_id(tag):
    """Returns an unique id which includes the execution time upon seconds. 
    
    Args:
        tag (str): optional tag which is appended to the date. 
    
    Returns:
        str: an string of the form ``"%m-%d-%Y-%H-%M-%S__<tag>__<uiid4>"``. 
    """
    tag = str(tag)

    exp_id = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    if tag != '':
        exp_id = '__'.join([exp_id, tag])

    exp_id = '__'.join([exp_id,  str(uuid.uuid4())])

    return exp_id


class Logger():
    def __init__(self, log_dir, experiment_args: dict):
        self.log_dir = Path(log_dir)
        self.current_cv_run = -1

        with open(self.log_dir / 'args.json', 'w') as fid:
            json.dump(obj=experiment_args, fp=fid)

        self._value_buffer = None

    def new_run(self):
        if self.current_cv_run > -1:
            self.write_logged_values_to_disk()

        self.current_cv_run += 1
        self._current_write_dir = self.log_dir / str(self.current_cv_run)
        self._current_write_dir.mkdir()
        self._value_buffer = defaultdict(list)

    def log_value(self, key: str, value):
        assert isinstance(key, str)
        assert self._value_buffer is not None
        v = convert_value_to_built_in_type(value)
        self._value_buffer[key].append(v)

    def write_value(self, key: str, value):
        assert isinstance(key, str)
        assert self._value_buffer is not None
        v = convert_value_to_built_in_type(value)
        self._value_buffer[key] = v

    def write_logged_values_to_disk(self):
        for k, v in self._value_buffer.items():
            pth = self._current_write_dir / (k + '.pkl')
            with open(pth, 'bw') as fid:
                pickle.dump(obj=v, file=fid)

    def write_model_to_disk(self, key: str, model):
        assert isinstance(key, str)
        assert self.current_cv_run > -1

        torch.save(model, self._current_write_dir / (key + '.pth'))


class _LazyFolderReader:

    def __init__(self, path, pickle_ext='.pkl'):
        self.path = Path(path)
        self.pickle_ext = pickle_ext
        self._dict = {}

        files = glob.glob(str(self.path) + '/*' + self.pickle_ext)
        files = [Path(x) for x in files]

        for fl in files:
            def value(path=str(fl)):
                with open(path, 'br') as fid:
                    v = pickle.load(fid)

                return v

            self._dict[fl.name.split(self.pickle_ext)[0]] = value 

    def __getitem__(self, idx):
        if hasattr(self._dict[idx], '__call__'):
            self._dict[idx] = self._dict[idx]()

        return self._dict[idx]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def items(self):
        for k in self:
            yield k, self[k]

    def keys(self):
        return self._dict.keys()


class LoggerReader:
    _args_file_name = 'args.json'
    _module_ext = '.pth'
    _pickle_ext = '.pkl'

    def __init__(self, folder_path: Path):
        self.path = Path(folder_path)
        self._runs = []

        with open(self.path / self._args_file_name, 'r') as fid:
            self._experiment_args = json.load(fid)

        
        run_folders = [x for x in self.path.iterdir() if x.is_dir()]

        if len(run_folders) > 0:
            
            # check for validity...
            folders_int = [int(x.name) for x in run_folders]
            folders_int = sorted(folders_int)
            assert folders_int == list(range(max(folders_int)+1))

            run_folders = sorted(run_folders, key=lambda x: int(x.name))

            for fd in run_folders:
                self._runs.append(_LazyFolderReader(fd)) 

    def __getitem__(self, idx):
        return self._runs[idx]

    def __iter__(self):
        yield from iter(self._runs) 

    def __len__(self):
        return len(self._runs)

    def load_model(self, run, key):
        p = self.path / (str(run) + '/' + key + self._module_ext)
        return torch.load(p)

    @property
    def experiment_args(self):
        return dict(self._experiment_args)

    @property
    def date(self):
        time_sig = str(self.path.name).split('__')[0]
        time_sig = time_sig.split('-')
        time_sig = [int(x) for x in time_sig]

        return datetime.datetime(
            year=time_sig[2],
            month=time_sig[0],
            day=time_sig[1],
            hour=time_sig[3],
            minute=time_sig[4],
            second=time_sig[5]
        )
