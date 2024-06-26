import os
import logging
import torch
from torch import optim, Tensor
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Union
from functools import partial

from models import *
from utils.data_loader import StandardScaler, get_dataloader
from utils.metrics import masked_mae, masked_mse, huber_loss, masked_huber_loss

my_logger = 'lazy'
logger = logging.getLogger(my_logger)

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = args.gpu
        self.model_name = args.model
        self.pt_dir = args.pt_dir
        self.save_iter = args.save_iter
        self.train_epochs = args.train_epochs
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_amp = args.use_amp
        self.patience = args.patience
        self.max_grad_norm = args.max_grad_norm
        self.need_pt = args.need_pt
        self.lr_adj = args.lradj
        self.horizon = args.pred_len
        
        self.model_dict = {
            # spatial-temporal
            'gwnet': GWNet,
            'mtgnn': MTGnn,
            'pdformer': PDFormer,
            'gpt_st': Gpt_ST,
            # LLM
            'timellm': TimeLLM,
            'autoTimes': AutoTimes,
            'test': Test,
            'gpt4ts': GPT4ts,
            'allm4ts': aLLM4TS,
            # Ours
            'st4llm':ST4llm,
        }
        self.dataloader = self._get_data()
        self.model = self._build_model()
        self.optimizer = self._select_optimizer()
        self.loss_fn = self._select_criterion()

    def _build_model(self): 
        return self.model_dict[self.model_name](self.args).to(self.device)

    def _get_data(self):
        dataloader, scalers = get_dataloader(self.args)
        self.scaler = StandardScaler(scalers[0], scalers[1])
        return dataloader
    
    def _inverse_transform(self, tensors: Union[Tensor, List[Tensor]]):
        n_output_dim = 1
        def inv(tensor):
            tensor = self.scaler.inverse_transform(tensor)
            return tensor

        return [inv(tensor) for tensor in tensors]

    def _save_model(self, save_path, cur_exp):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = f'final_model_{cur_exp}.pt'
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
        return filename

    def _load_model(self, save_path, cur_exp):
        filename = f'final_model_{cur_exp}.pt'
        self.model.load_state_dict(torch.load(os.path.join(save_path, filename)))
        return filename

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return model_optim

    def _select_criterion(self, metric_func='mask_mae'):
        if metric_func == 'mask_mae':
            return partial(masked_mae, null_val=0)
        elif metric_func == 'mask_mse':
            return partial(masked_mse, null_val=0)
        elif metric_func == 'mae':
            return torch.nn.L1Loss()
        elif metric_func == 'mse':
            return torch.nn.MSELoss()
        elif metric_func == 'huber':
            return partial(huber_loss, delta=2)
        elif metric_func == 'mask_huber':
            return partial(masked_huber_loss, delta=2, null_val=0)
        else:
            raise ValueError

    def train(self):
        pass

    def valid(self):
        pass

    def test(self):
        pass
