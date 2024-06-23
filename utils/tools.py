import os
import math
import json
import glob

import logging
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from distutils.util import strtobool
from datetime import datetime
from pathlib import Path

my_logger = 'lazy'
logger = logging.getLogger(my_logger)

def gen_version(args):
    dirs = glob.glob(os.path.join('saved/'+ args.data+'/'+args.model, '*'))
    if not dirs:
        return 0

    version = max([int(x.split(os.sep)[-1].split('_')[-1])
                for x in dirs]) + 1
    return version

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** (epoch - 1))}    
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.learning_rate * (0.6 ** epoch)}  
    elif args.lradj == "type3":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
            print('next learning rate is {}'.format(lr))

class EarlyStopping:
    def __init__(self, args, verbose=False, delta=0):
        self.patience = args.patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.use_multi_gpu = args.use_multi_gpu
        if self.use_multi_gpu:
            self.local_rank = args.local_rank
        else:
            self.local_rank = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            if self.use_multi_gpu:
                if self.local_rank == 0:
                    self.save_checkpoint(val_loss, model, path)
                dist.barrier()
            else:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.use_multi_gpu:
                if self.local_rank == 0:
                    self.save_checkpoint(val_loss, model, path)
                dist.barrier()
            else:
                self.save_checkpoint(val_loss, model, path)
            if self.verbose:
                if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0


    def save_checkpoint(self, val_loss, model, path):
        param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
        }
        state_dict = model.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        torch.save(state_dict, path + '/' + f'checkpoint.pth')


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def set_logger(log_dir, model_name, dataset_name, verbose_level=1):
    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger(my_logger)
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False

def serializable_parts_of_dict(d):
    serializable_dict = {k: v for k, v in d.items() if is_json_serializable(v)}
    return serializable_dict

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

