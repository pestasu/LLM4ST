import os
import time
import warnings
import ipdb
import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
from torch import optim
from torch import optim, Tensor
from typing import Optional, List, Union

from exp.exp_basic import Exp_Basic
from utils import metrics, graph
from utils.tools import adjust_learning_rate, set_logger, serializable_parts_of_dict, gen_version
from utils.data_loader import StandardScaler, get_dataloader
from models.ha import HistoricalAverage

warnings.filterwarnings('ignore')
my_logger = 'lazy'
logger = logging.getLogger(my_logger)

class Exp_HA:
    def __init__(self, args, ii):
        self.cur_exp = ii
        self.args = args
        self.horizon = args.pred_len
        self.data_floder = args.data_floder
        self.pt_dir = args.pt_dir

        self.model = self._build_model()
        self.dataloader = self._get_data()
    
    def _build_model(self): 
        return HistoricalAverage()
    
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
        
    def test(self, setting, is_test=False):
        preds = []
        trues = []

        for i, (batch_x, batch_y) in enumerate(self.dataloader['test']):
            if self.args.model == 'var':
                self.model.fit(batch_x)
            output = self.model.predict(batch_x)
            pred, true = self._inverse_transform([output, batch_y[..., :1]])
            
            preds.append(pred)
            trues.append(true)
                
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)


        mae_day = []
        rmse_day = []
        mae, rmse = metrics.compute_all_metrics(preds, trues)
        logger.info(f'***** Average Horizon, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f} *****')
        mae_day.append(mae)
        rmse_day.append(rmse)

        if self.horizon == 24: 
            for i in range(0, self.horizon, 8):
                pred = preds[:,:, i: i + 8]
                true = trues[:,:, i: i + 8]
                result = metrics.compute_all_metrics(pred, true)
                mae_day.append(result[0])
                rmse_day.append(result[1])

            logger.info(f'***** 0-7 (1-24h) Test MAE: {mae_day[1]:.4f}, Test RMSE: {rmse_day[1]:.4f} *****')
            logger.info(f'***** 8-15 (25-48h) Test MAE: {mae_day[2]:.4f}, Test RMSE: {rmse_day[2]:.4f} *****')
            logger.info(f'***** 16-23 (49-72h) Test MAE: {mae_day[3]:.4f}, Test RMSE: {rmse_day[3]:.4f} *****')

            results = pd.DataFrame(columns=['Time','Test MAE', 'Test RMSE'], index=range(5))
            Time_list=['Average','1-24h','25-48h','49-72h', 'SuddenChange']
            for i in range(4):
                results.iloc[i, 0]= Time_list[i]
                results.iloc[i, 1]= mae_day[i]
                results.iloc[i, 2]= rmse_day[i]
        else:
            print('The output length is not 24!!!')


        mask_sudden_change = metrics.sudden_changes_mask_times(trues, datapath=self.data_floder, null_val=0.0, threshold_start=75, threshold_change=20)
        results.iloc[4, 0] = Time_list[4]
        mae_sc, rmse_sc = metrics.compute_sudden_change(mask_sudden_change, preds, trues, null_value=0.0)
        results.iloc[4, 1:] = [mae_sc, rmse_sc]
        logger.info(f'***** Sudden Changes MAE: {mae_sc:.4f}, Test RMSE: {rmse_sc:.4f} *****')

        # results.to_csv(os.path.join(folder_path, f'metrics_{self.cur_exp}.csv'), index = False)
        results.to_csv(os.path.join(self.pt_dir, f'metrics_{self.cur_exp}.csv'), index = False)

