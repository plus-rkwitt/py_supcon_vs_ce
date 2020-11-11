import uuid
import json
import torch
import torch.nn as nn


import pytorch_utils.logging as logging


from collections import defaultdict
from pathlib import Path

from fastprogress import progress_bar


class ExperimentBase(object):

    args = {
        'output_root_dir': str,
        'num_batches': int,
        'tag': str,
        'eval_interval': (int, type(None))
    }

    def check_args(self, args):
        assert 'experiment_type' not in args

        for k, v in args.items():
            assert k in self.args, "Unknown keyword argument {}".format(k)

            check = self.args[k]
            if isinstance(check, (type, tuple)):
                assert isinstance(v, check), "{} (={}) is not {}".format(k, v, check)
            elif hasattr(check, '__call__'):
                check(v, args)        

    def __init__(self, **kwargs):
        self.check_args(kwargs)

        self.args = kwargs

        if self.args['eval_interval'] is None:
            self.args['eval_interval'] = self.args['num_batches']

        self.args['experiment_type'] = \
            '.'.join([self.__module__, type(self).__qualname__])

        self.device = 'cuda'

        output_dir = Path(
            self.args['output_root_dir']) / logging.get_an_id(self.args['tag'])
        output_dir.mkdir()

        self.output_dir = output_dir

        self.logger = logging.Logger(output_dir, self.args)

        self.ds_train = None
        self.ds_test = None
        self.model = None
        self.opt = None
        self.scheduler = None
        self.dl_train = None

        self.batch_x = None
        self.batch_y = None
        self.batch_i = None
        self.batch_loss = None
        self.batch_model_output = None

    def one_run(self):
        self.logger.new_run()
        
        self.pre_run()

        self.setup_model()

        self.setup_opt()
        self.setup_scheduler()
        self.setup_dl_train()

        self.pb = progress_bar(range(self.args['num_batches']))

        self.model.train()

        for (batch_x, batch_y), batch_i in zip(self.dl_train, self.pb):
            self.batch_i = batch_i

            self.batch_x, self.batch_y = batch_x, batch_y

            self.setup_batch()

            self.forward_model()

            self.compute_loss()

            if isinstance(self.opt, list):
                for o in self.opt:
                    o.zero_grad()
            else:
                self.opt.zero_grad()

            self.batch_loss.backward()

            if isinstance(self.opt, list):
                for o in self.opt:
                    o.step()
            else:
                self.opt.step()

            self.post_batch()

            if self.scheduler is not None:
                if isinstance(self.scheduler, list):
                    for s in self.scheduler:
                        s.step()
                else:
                    self.scheduler.step()
            
                if (self.batch_i + 1) % self.args['eval_interval'] == 0 \
                        or \
                        self.batch_i == self.args['num_batches'] - 1:

                    self.evaluate()
                    self.model.train()

            self.logger.write_value('batch_i', self.batch_i)
            self.logger.write_logged_values_to_disk()

        self.logger.write_model_to_disk('model', self.model.cpu())

    def __call__(self):
        try:
            for _ in self.ds_setup_iter():
                self.one_run()

        except Exception as ex:
            self.error = ex
            self.handle_error()

    # necessary hooks

    def forward_model(self):
        self.batch_model_output = self.model(self.batch_x)

    # TODO This should be renamed in init_run_iter or so!
    def ds_setup_iter(self):
        raise NotImplementedError()

    def setup_model(self):
        raise NotImplementedError()

    def setup_opt(self):
        raise NotImplementedError()

    def setup_scheduler(self):
        raise NotImplementedError()

    def setup_dl_train(self):
        raise NotImplementedError()

    def setup_batch(self):
        raise NotImplementedError()

    def compute_loss(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def handle_error(self):
        raise self.error

    # optional hooks

    def post_batch(self):
        pass

    def pre_run(self):
        pass

    # class methods

    @classmethod
    def args_template(cls):
        s = " : , \n    "
        s = s.join([r"'{}'".format(k) for k in cls.args.keys()])
        s = "{ \n    " + s + "\n}"
        return s
