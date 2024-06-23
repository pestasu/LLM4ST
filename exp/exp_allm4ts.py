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
import matplotlib.pyplot as plt

from exp.exp_basic import Exp_Basic
from utils import metrics
from utils.tools import adjust_learning_rate
from utils.data_loader import StandardScaler, get_dataloader
from models import *

warnings.filterwarnings('ignore')
my_logger = 'lazy'
logger = logging.getLogger(my_logger)

class Exp_allm4ts(Exp_Basic):
    def __init__(self, args, ii):
        super(Exp_allm4ts, self).__init__(args)
        self.cur_exp = ii

    def _get_data(self):
        dataloader, scalers = get_dataloader(self.args, need_location=False)
        self.scaler = StandardScaler(scalers[0], scalers[1])
        return dataloader


    def build_model(self):
        model = self.model_dict[self.model_name].Model(self.args).float()

        if self.need_pt:
            pt_model = torch.load(os.path.join(self.pt_dir, "final_model.pt"), map_location='cpu')

            model_dict =  model.state_dict()
            state_dict = {k:v for k,v in pt_model.items() if k in model_dict.keys()}

            del pt_model
            
        return model
    
    def early_stop(self, epoch, best_loss):
        logger.info(f'Early stop at epoch {epoch}, loss = {best_loss:.6f}')
        # np.savetxt(os.path.join(self.pt_dir, f'val_loss_{self.cur_exp}.txt'), [best_loss], fmt='%.4f', delimiter=',')

    def train_batch(self, x, y):
        '''
        the training process of a batch
        '''   
        self.optimizer.zero_grad()
        x = x.to(self.device)
        y = y[..., :1].to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(y[:, -self.args.pred_len:, :]).float().to(self.device)
        dec_inp = torch.cat([y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        outputs = self.model(x, None, dec_inp, None)
        pred, true = self._inverse_transform([outputs, y])
        loss = self.loss_fn(pred, true) 

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                    max_norm=self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def train(self, setting):
        train_steps = len(self.dataloader['train'])

        if self.use_amp:
            grad_scaler = torch.cuda.amp.GradScaler()

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer = self.optimizer,
                                        steps_per_epoch = train_steps,
                                        pct_start = self.pct_start,
                                        epochs = self.train_epochs,
                                        max_lr = self.learning_rate)
        
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

                if iter_count % self.save_iter == 0:
                    val_loss, _ = self.valid(epoch)
                    logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f} | train_mae:{np.mean(train_losses):.4f}')
            
            end_time = time.time()
            logger.info(f'{epoch}-epoch complete')
            logger.info('------evaluating now!------')

            val_loss, val_time = self.valid(epoch)
            logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f}, val_time:{val_time:.1f}s | train_mae:{np.mean(train_losses):.4f}')

            if self.lr_adj == 'cosine':
                scheduler.step()
                print(f'lr = {self.optimizer.param_groups[0]["lr"]:.10f}')
            else:
                adjust_learning_rate(self.optimizer, epoch+1, self.args)

    def valid(self, epoch):
        preds, trues = [], []
        self.model.eval()
        total_time = 0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.dataloader['valid']):
                x = batch_x.to(self.device)
                y = batch_y[..., :1].to(self.device)

                time_now = time.time()

                # decoder input
                dec_inp = torch.zeros_like(y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(x, None, dec_inp, None)

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
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.dataloader['test']):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y[..., :1].to(self.device)

                outputs = self.model(batch_x)
                pred, true = self._inverse_transform([outputs, batch_y])
                
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
                pred = preds[:,i: i + 8]
                true = trues[:,i: i + 8]
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

        datapath = self.data_floder + './mask_sudden_change_times/'
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        mask_sudden_change = metrics.sudden_changes_mask_times(trues, datapath=datapath, null_val=0.0, threshold_start=75, threshold_change=20)
        results.iloc[4, 0] = Time_list[4]
        mae_sc, rmse_sc = metrics.compute_sudden_change(mask_sudden_change, preds, trues, null_value=0.0)
        results.iloc[4, 1:] = [mae_sc, rmse_sc]
        logger.info(f'***** Sudden Changes MAE: {mae_sc:.4f}, Test RMSE: {rmse_sc:.4f} *****')

        # results.to_csv(os.path.join(folder_path, f'metrics_{self.cur_exp}.csv'), index = False)
        results.to_csv(os.path.join(self.pt_dir, f'metrics_{self.cur_exp}.csv'), index = False)

