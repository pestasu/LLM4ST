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
from models.var import VARModel, ARIMAModel

warnings.filterwarnings('ignore')
my_logger = 'lazy'
logger = logging.getLogger(my_logger)

class Exp_VAR:
    def __init__(self, args, ii):
        self.cur_exp = ii
        self.args = args
        self.train_epochs = 100
        self.patience =5
        self.device = args.gpu
        self.horizon = args.pred_len
        self.p = args.p
        self.d = args.d
        self.q = args.q
        self.data_floder = args.data_floder
        self.pt_dir = args.pt_dir
        self.model = self._build_model()
        self.dataloader = self._get_data()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()


    def _build_model(self): 
        return ARIMAModel(self.p, self.d, self.q, ).to(self.device)

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
        
    def early_stop(self, epoch, best_loss):
        logger.info(f'Early stop at epoch {epoch}, loss = {best_loss:.6f}')

    def train_batch(self, x, y):
        '''
        the training process of a batch
        '''   
        self.optimizer.zero_grad()
        x = x[..., :1].to(self.device)
        y = y[:, :, 0, :1].to(self.device)
        outputs = self.model(x)

        pred, true = self._inverse_transform([outputs, y])
        loss = self.loss_fn(pred, true) 

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train(self, setting):
        train_steps = len(self.dataloader['train'])

        self.saved_epoch = -1
        self.val_losses = [np.inf]       
        time_now = time.time()
        for epoch in range(self.train_epochs):
            self.model.train()

            iter_count = 0
            train_losses, pred_losses, rec_losses= [], [], []

            if epoch - self.saved_epoch > self.patience:
                self.early_stop(epoch, min(self.val_losses))
                np.savetxt(os.path.join(self.pt_dir, f'val_loss_{self.cur_exp}.txt'), self.val_losses, fmt='%.4f', delimiter=',')
                break
            logger.info('------start training!------')
            start_time = time.time()
            for i, (batch_x, batch_y) in enumerate(self.dataloader['train']):
                iter_count += 1
                loss = self.train_batch(batch_x, batch_y)
                train_losses.append(loss)

                if (i + 1) % 100 == 0:
                    logger.info(f'\titers: {i+1}, epoch: {epoch+1} | loss: {loss:.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_epochs - epoch) * train_steps - i)
                    logger.info(f'\tspeed: {speed:.4f}s/iter | left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            end_time = time.time()
            logger.info(f'{epoch}-epoch complete')
            logger.info('------evaluating now!------')

            val_loss, val_time = self.valid(epoch)
            logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f}, val_time:{val_time:.1f}s | train_mae:{np.mean(train_losses):.4f}')


    def valid(self, epoch):
        preds, trues = [], []
        self.model.eval()
        total_time = 0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.dataloader['valid']):
                x = batch_x[..., :1].to(self.device)
                y = batch_y[:, :, 0, :1].to(self.device)

                time_now = time.time()

                outputs = self.model(x)

                total_time += time.time() - time_now

                pred, true = self._inverse_transform([outputs, y])

                preds.append(pred.cpu())
                trues.append(true.cpu())

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        val_loss = self.loss_fn(preds, trues)
        
        if val_loss < np.min(self.val_losses):
            saved_model_file = self._save_model(self.pt_dir, self.cur_exp)
            logger.info(f'Valid loss decrease: {np.min(self.val_losses)} -> {val_loss}, saving to {saved_model_file}')
            self.val_losses.append(val_loss)
            self.saved_epoch = epoch

        self.model.train()
        return val_loss, total_time

    def test(self, setting, is_test=False):
        if is_test:
            logger.info(f'------------Test process~load model({self.args.model}-{self.args.version})------------')
            self._load_model(self.pt_dir, self.cur_exp)

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.dataloader['test']):
                x = batch_x[..., :1].to(self.device)
                y = batch_y[..., :1].to(self.device)
                
                predictions = []

                for _ in range(self.args.pred_len):
                    output = self.model(x[:, :, -self.args.seq_len:])
                    predictions.append(output.unsqueeze(2))  
                    x = torch.cat([x, output.unsqueeze(2)], dim=2)

                pred = torch.cat(predictions, dim=2)
                pred, true = self._inverse_transform([pred, y])
                preds.append(pred.cpu())
                trues.append(true.cpu())
                
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


