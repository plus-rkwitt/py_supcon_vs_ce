
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


import core
import core.models
import pytorch_utils.data.dataset as ds_utils


from collections import defaultdict
from typing import List


from pytorch_utils.evaluation import apply_model, argmax_and_accuracy
from .experiment_base import ExperimentBase
from .augmentation.auto_aug import AutoAugment
from .augmentation.random_aug import RandomAugment
from .augmentation.supcon_aug import supcon_aug


class RandomBatchSampler:
    def __init__(self, high, num_batches, batch_size, replace=True):
        self.high = high
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.replace = replace
        
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        for _ in range(self.num_batches):
            
            batch_i = np.random.choice(
                self.high,
                self.batch_size,
                replace=self.replace
            )
            
            yield batch_i


class RandomLabels(torch.utils.data.dataset.Dataset):
    def __init__(self, wrappee, num_labels):
        self._labels = torch.randint(num_labels, (len(wrappee),)).tolist()
        self._labels = [int(i) for i in self._labels]
        self._wrappee = wrappee

    def __getitem__(self, idx):
        return self._wrappee[idx][0], self._labels[idx]

    def __len__(self):
        return len(self._wrappee)


# region Losses


class LossBase(nn.Module):
    def __init__(self):
        super().__init__()
        
    def _compute_loss(self, model_output, labels):
        raise NotImplementedError()

    def __call__(self, model_output, labels):
        return self._compute_loss(model_output, labels)


class CrossEntropy(LossBase):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction

    def _compute_loss(self, model_output, labels):
        y_hat, _ = model_output
        l = F.cross_entropy(y_hat, labels, reduction=self.reduction)
               
        return l


class CrossEntropyForTracking(CrossEntropy):
    def _compute_loss(self, model_output, labels):
        _, z = model_output

        M = torch.zeros(
            labels.max()+1, z.size(0), 
            dtype=z.dtype, 
            device=z.device)

        M[labels, torch.arange(z.size(0))] = 1

        M = torch.nn.functional.normalize(M, p=1, dim=1)
        
        weight = torch.mm(M, z)

        y_hat = torch.nn.functional.linear(z, weight)

        return F.cross_entropy(y_hat, labels, reduction=self.reduction)


def _lazy_set_inv_eye(obj, z):
    try:
        return obj._inv_eye

    except AttributeError:
        obj._inv_eye = ~torch.eye(
        z.size(0), # = size of sub-batch
        device=z.device,
        dtype=bool)

        return obj._inv_eye 


class SupConLoss(LossBase):
    def __init__(self,
                 temperature):
        super().__init__()
        self.temperature = temperature

    def _compute_loss(self, model_output, labels):

        # latent representation
        _, z = model_output

        y = labels.view(-1, 1)

        # mask(i,j) = true if label_i == label_j
        mask = torch.eq(y, y.T)

        # weighting factor
        cnt = mask.sum(-1, keepdim=True).float()-1
        w = 1./cnt.float()

        # inner product matrix (scaled by temp)
        ips = torch.div(torch.matmul(z, z.T), self.temperature)

        # get selector for off-diagonal entries
        inv_eye = _lazy_set_inv_eye(self, z)

        # new inner product matrix where diagonal is removed
        ips = ips.masked_select(inv_eye).view(z.size(0), z.size(0) - 1)

        # new mask where diagonal is removed
        mask = mask.masked_select(inv_eye).view(z.size(0), z.size(0) - 1)

        #remove entries where there are no same-class partners in the batch
        I = (cnt.squeeze() > 0)
        ips = ips[I]
        mask = mask[I]
        w = w[I]

        # apply log-softmax along dim=1 to inner product matrix with removed diagonal entries
        ips = nn.functional.log_softmax(ips, dim=1)*w

        # select entries from log-softmax result that correspond to samples that have the same
        # labels - always reduce
        loss = -ips.masked_select(mask).sum() / float(y.size(0))

        return loss


class SupConLossWeighted(LossBase):
    def __init__(self,
                 temperature,
                 weight):
        super().__init__()
        self.temperature = temperature     
        self.log_w = np.log(weight) if weight > 0 else -float('inf')

    def __call__(self, model_output, labels):

        # latent representation
        _, z = model_output

        y = labels.view(-1, 1)

        # mask(i,j) = true if label_i == label_j
        mask = torch.eq(y, y.T)

        # weighting factor
        cnt = mask.sum(-1, keepdim=True).float()-1
        w = 1./cnt.float()

        # get selector for off-diagonal entries
        inv_eye = _lazy_set_inv_eye(self, z)

        # inner product matrix (scaled by temp)
        ips = torch.div(torch.matmul(z, z.T), self.temperature)

        # new inner product matrix where diagonal is removed
        ips = ips.masked_select(inv_eye).view(z.size(0), z.size(0) - 1)

        # new mask where diagonal is removed
        mask = mask.masked_select(inv_eye).view(z.size(0), z.size(0) - 1)

        #remove entries where there are no same-class partners in the batch
        I = (cnt.squeeze() > 0)
        ips = ips[I]
        mask = mask[I]
        w = w[I]

        # 1. Innerclass Contraction
        con = -(ips*w)[mask]        
        con = con.sum()

        #2. Repulsion
        mask_log = mask.float()
        mask_log[mask] = self.log_w
        rep = torch.logsumexp(ips+mask_log, dim=-1)

        rep = rep.sum()

        loss = (con + rep)/float(y.size(0))

        return loss


# endregion


class Experiment(ExperimentBase):

    @ staticmethod
    def arg_check_losses(arg, all_args):
        msg = "Argument losses has to be a tuple of elements of the type tuple[str, dict]! Got {}".format(
            arg)

        assert isinstance(arg, tuple), msg

        for a_1, a_2 in arg:
            assert isinstance(a_1, str), msg
            assert isinstance(a_2, dict), msg

    args = {k: v for k, v in ExperimentBase.args.items()}
    args.update(
        {   
            'num_batches': int, 
            'num_runs': int,
            'num_samples': (int, type(None)), 
            'model': tuple,
            'lr_init': float,
            'weight_decay': float,
            'ds_train': str,
            'ds_test': str,
            'momentum': float,
            'augment': str, 
            'batch_size': int,
            'losses': arg_check_losses,
            'losses_track_only': arg_check_losses,
            'w_losses': (tuple, type(None)),
            'evaluation_policies': tuple
        }
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.args['augment'] in [
            'none', 'standard', 'random', 'auto', 'supcon']

        for e in self.args['evaluation_policies']:
            assert e in ('linear', 'retrained_linear', 'explicit_linear')

        self.losses, self.losses_track_only = [], []

        for name, kwargs in self.args['losses']:

            fn = globals()[name](**kwargs).to(self.device)
            self.losses.append((name, fn))

        for name, kwargs in self.args['losses_track_only']:

            fn = globals()[name](**kwargs).to(self.device)
            self.losses_track_only.append((name, fn))

        w_losses = self.args['w_losses']
        if w_losses is None:
            self.w_losses = [1.] * len(self.losses)

        else:
            assert len(w_losses) == len(self.losses)
            assert all((isinstance(w, float) for w in w_losses))
            self.w_losses = w_losses

    def ds_setup_iter(self):

        if self.args['num_samples'] is None:
            ds_train_splits = [core.data.ds_factory(self.args['ds_train'])]*self.args['num_runs']
        else:
            ds_train_splits = core.data.ds_split_factory(
                self.args['ds_train'], 
                self.args['num_samples']
            )
            ds_train_splits = ds_train_splits[:self.args['num_runs']]

        for ds_split_i_original in ds_train_splits:

            ds_test = core.data.ds_factory(self.args['ds_test'])

            ds_stats = ds_utils.ds_statistics(ds_split_i_original)

            to_tensor_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    ds_stats['channel_mean'],
                    ds_stats['channel_std']),
            ])

            if self.args['augment'] != 'none':

                if self.args['augment'] == 'standard':
                    augmenting_tf = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        to_tensor_tf
                    ])

                elif self.args['augment'] == 'random':
                    augmenting_tf = transforms.Compose([
                        RandomAugment(),
                        to_tensor_tf
                    ])

                elif self.args['augment'] == 'auto':
                    augmenting_tf = transforms.Compose([
                        AutoAugment(),
                        to_tensor_tf
                    ])

                elif self.args['augment'] == 'supcon':
                    augmenting_tf = transforms.Compose([
                        supcon_aug,
                        to_tensor_tf
                    ])

                ds_train = ds_utils.Transformer(
                    ds_split_i_original,
                    augmenting_tf
                )
            else:
                ds_train = ds_utils.Transformer(
                    ds_split_i_original,
                    to_tensor_tf
                )

            self.num_classes = ds_stats['num_classes']

            self.ds_train = ds_train
            self.ds_test = ds_utils.Transformer(ds_test, to_tensor_tf)

            yield None

    def setup_model(self):
        id, kwargs = self.args['model']
        kwargs['num_classes'] = self.num_classes
        self.model = getattr(core.models, id)(**kwargs)
        self.model.to(self.device)

    def setup_opt(self):
        self.opt = \
            torch.optim.SGD(
                self.model.parameters(),
                weight_decay=self.args['weight_decay'],
                lr=self.args['lr_init'],
                momentum=self.args['momentum'],
                nesterov=False)

    def setup_scheduler(self):
        num_batches = float(self.args['num_batches'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt,
            lambda i: ((num_batches - i)/num_batches)**5, last_epoch=-1)

    def setup_dl_train(self):
        self.dl_train = torch.utils.data.DataLoader(
            self.ds_train, 
            batch_sampler=RandomBatchSampler(
                len(self.ds_train), 
                self.args['num_batches'], 
                self.args['batch_size']
            )
        )

    def setup_batch(self):
        self.batch_x = self.batch_x.to(self.device)
        self.batch_y = self.batch_y.to(self.device)

    def compute_loss(self):

        batch_loss = 0.

        for (name, fn), w in zip(self.losses, self.w_losses):
            l_name = fn(self.batch_model_output, self.batch_y)
            self.logger.log_value('batch_{}'.format(name), l_name)

            batch_loss = batch_loss + w*l_name

        batch_loss = batch_loss

        for (name, fn) in self.losses_track_only:
            with torch.no_grad():
                l_name = fn(self.batch_model_output, self.batch_y)
                self.logger.log_value('batch_{}_tracked'.format(name), l_name)

        self.batch_loss = batch_loss

    def evaluate(self):

        feat_ext = self.model.feat_ext
        feat_ext.to(self.device)
        feat_ext.eval()

        input_dim = self.ds_train[0][0].size()
        z = feat_ext(torch.randn(10, *input_dim).to(self.device))
        assert z.ndim == 2
        latent_dim = z.size(1)

        def evaluate_classifier_and_save(classifier, policy):
            classifier.eval()

            datasets = {
                'train': self.ds_train,
                'test': self.ds_test
            }
            for k, ds in datasets.items():
                m = nn.Sequential(feat_ext, classifier)
                X, Y = apply_model(dataset=ds, model=m, device=self.device)
                acc = argmax_and_accuracy(X, Y)

                self.logger.log_value('{}_{}'.format(policy, k), acc)

            self.logger.write_model_to_disk(
                policy, classifier.cpu())

        if 'retrained_linear' in self.args['evaluation_policies']:

            num_batches = 10000

            dl_train = torch.utils.data.DataLoader(
                self.ds_train,
                batch_sampler=RandomBatchSampler(
                    len(self.ds_train), 
                    num_batches, 
                    128, 
                )
            )

            classifier = nn.Linear(
                latent_dim, self.num_classes).to(self.device)
            classifier.train()

            opt = torch.optim.SGD(classifier.parameters(),
                                  lr=0.1, weight_decay=10e-4)

            sch = torch.optim.lr_scheduler.LambdaLR(
                opt,
                lambda i: ((num_batches - i)/num_batches)**5, last_epoch=-1)

            for batch_i, (batch_x, batch_y) in enumerate(dl_train):
                batch_x, batch_y = batch_x.to(
                    self.device), batch_y.to(self.device)
                opt.zero_grad()

                with torch.no_grad():
                    z = feat_ext(batch_x)

                y_hat = classifier(z)

                l = nn.functional.cross_entropy(y_hat, batch_y)
                self.logger.log_value('retrained_linear_loss', l)

                l.backward()
                opt.step()
                sch.step()

            evaluate_classifier_and_save(classifier, 'retrained_linear')

        if 'explicit_linear' in self.args['evaluation_policies']:

            X, Y = apply_model(
                dataset=self.ds_train,
                model=feat_ext,
                device=self.device)

            X, Y = torch.tensor(X), torch.tensor(Y)

            rows = []
            for y in range(self.num_classes):
                X_y = X[Y == y]
                r = X_y.mean(dim=0)
                rows.append(r)

            weight_cls = torch.stack(rows, dim=0)

            classifier = nn.Linear(latent_dim, self.num_classes, bias=False)
            classifier.weight.data = weight_cls

            evaluate_classifier_and_save(classifier, 'explicit_linear')

        if 'linear' in self.args['evaluation_policies']:
            evaluate_classifier_and_save(self.model.cls, 'linear')


class ExpRandomeLabeledData(Experiment):
    def ds_setup_iter(self):
        for _ in super().ds_setup_iter():
            self.ds_train = RandomLabels(
                self.ds_train,
                self.num_classes)

            yield None


