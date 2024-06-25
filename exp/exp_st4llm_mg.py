import os
import time
import warnings
import ipdb
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
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

class Exp_st4llm_mg(Exp_Basic):
    def __init__(self, args, ii, accelerator=None):
        super(Exp_st4llm_mg, self).__init__(args)
        self.cur_exp = ii
        self.mode = args.mode
        self.accelerator = accelerator
        self.train_loader, self.valid_loader, self.test_loader = self.dataloader['train'], self.dataloader['valid'], self.dataloader['test']
        train_steps = len(self.dataloader['train'])
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer = self.optimizer,
                                        steps_per_epoch = train_steps,
                                        pct_start = args.pct_start,
                                        epochs = args.train_epochs,
                                        max_lr = args.learning_rate)
        self.train_loader, self.valid_loader ,self.test_loader, self.model, self.optimizer, self.scheduler = accelerator.prepare(
            self.train_loader, self.valid_loader ,self.test_loader, self.model, self.optimizer, scheduler)

    def _build_model(self):
        return self.model_dict[self.model_name](self.args).float()

    def early_stop(self, epoch, best_loss):
        logger.info(f'Early stop at epoch {epoch}, loss = {best_loss:.6f}')
        # np.savetxt(os.path.join(self.pt_dir, f'val_loss_{self.cur_exp}.txt'), [best_loss], fmt='%.4f', delimiter=',')

    def train_batch(self, x, y):
        '''
        the training process of a batch
        '''   
        self.optimizer.zero_grad()
        with self.accelerator.autocast():
            x = x.to(self.accelerator.device)
            y = y[..., :1].to(self.accelerator.device)

            if self.args.mode == 'pretrain':
                rec_t, true_t, rec_s, true_s = self.model(x)
                rec_t, true_t = self._inverse_transform([rec_t, true_t])
                rec_s, true_s = self._inverse_transform([rec_s, true_s])
                loss_rec_t = self.loss_fn(rec_t, true_t)
                loss_rec_s = self.loss_fn(rec_s, true_s)
                loss = loss_rec_t + loss_rec_s
            else:
                output = self.model(x)
                pred, true = self._inverse_transform([outputs, y])
                loss = self.loss_fn(pred, true) 


        self.accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                    max_norm=self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def train(self, setting):
        train_steps = len(self.train_loader)

        if self.use_amp:
            grad_scaler = torch.cuda.amp.GradScaler()

        self.saved_epoch = -1
        self.val_losses = [np.inf]
        time_now = time.time()
        for epoch in range(self.train_epochs):
            self.model.train()

            iter_count = 0
            # train_losses, pred_losses, rec_losses= [], [], []
            train_losses = []
            if epoch - self.saved_epoch > self.patience:
                self.early_stop(epoch, min(self.val_losses))
                np.savetxt(os.path.join(self.pt_dir, f'val_loss_{self.cur_exp}.txt'), self.val_losses, fmt='%.4f', delimiter=',')
                break

            logger.info('------start training!------')
            start_time = time.time()
            for i, (batch_x, batch_y) in enumerate(self.train_loader):
                iter_count += 1
                # loss, pred_loss, rec_loss = self.train_batch(batch_x, batch_y)
                loss = self.train_batch(batch_x, batch_y)
                train_losses.append(loss)
                # pred_losses.append(pred_loss)
                # rec_losses.append(rec_loss)

                if (i + 1) % 100 == 0:
                    logger.info(f'\titers: {i+1}, epoch: {epoch+1} | loss: {loss:.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_epochs - epoch) * train_steps - i)
                    logger.info(f'\tspeed: {speed:.4f}s/iter | left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if iter_count % self.save_iter == 0:
                    val_loss, _ = self.valid(epoch)
                    logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f} | train_loss:{np.mean(train_losses):.4f}')
                    # logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f} | train_loss:{np.mean(train_losses):.4f}, train_mae:{np.mean(pred_losses):.4f}, train_rec:{np.mean(rec_losses):.4f}')
            
            end_time = time.time()
            logger.info(f'{epoch}-epoch complete')
            logger.info('------evaluating now!------')

            val_loss, val_time = self.valid(epoch)
            logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f}, val_time:{val_time:.1f}s | train_mae:{np.mean(train_losses):.4f}')

            if self.lr_adj == 'cosine':
                scheduler.step()
                self.accelerator.print(f'lr = {self.optimizer.param_groups[0]["lr"]:.10f}')
            else:
                adjust_learning_rate(self.optimizer, epoch+1, self.args, self.accelerator)

    def valid(self, epoch):
        preds, trues = [], []
        preds2, trues2 = [], []
        self.model.eval()
        total_time = 0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.valid_loader):
                batch_x = batch_x.to(self.accelerator.device)
                batch_y = batch_y[..., :1].to(self.accelerator.device)

                time_now = time.time()
                if self.args.mode == 'pretrain':
                    rec_t, true_t, rec_s, true_s = self.model(batch_x)
                    rec_t, true_t = self._inverse_transform([rec_t, true_t])
                    rec_s, true_s = self._inverse_transform([rec_s, true_s])
                    preds.append(rec_t.cpu())
                    trues.append(true_t.cpu())
                    preds2.append(rec_s.cpu())
                    trues2.append(true_s.cpu())

                else:
                    output = self.model(batch_x)
                    pred, true = self._inverse_transform([outputs, y])
                    preds.append(pred.cpu())
                    trues.append(true.cpu())

                total_time += time.time() - time_now

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        val_loss = self.loss_fn(preds, trues)

        if self.args.mode == 'pretrain':
            preds2 = torch.cat(preds2, dim=0)
            trues2 = torch.cat(trues2, dim=0)
            val_loss += self.loss_fn(preds2, trues2)
        
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

        preds, trues = [], []
        preds2, trues2 = [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.test_loader):
                batch_x = batch_x.to(self.accelerator.device)
                batch_y = batch_y[..., :1].to(self.accelerator.device)

                if self.args.mode == 'pretrain':
                    rec_t, true_t, rec_s, true_s = self.model(batch_x)
                    rec_t, true_t = self._inverse_transform([rec_t, true_t])
                    rec_s, true_s = self._inverse_transform([rec_s, true_s])
                    preds.append(rec_t.cpu())
                    trues.append(true_t.cpu())
                    preds2.append(rec_s.cpu())
                    trues2.append(true_s.cpu())
                else:
                    output = self.model(batch_x)
                    pred, true = self._inverse_transform([outputs, y])
                    preds.append(pred.cpu())
                    trues.append(true.cpu())
                
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        # dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        # all_preds = self.accelerator.gather(torch.tensor(preds, device=self.accelerator.device))
        # all_labels = self.accelerator.gather(torch.tensor(trues, device=self.accelerator.device))

        # if self.accelerator.is_main_process:
        if self.mode == 'pretrain':
            preds2 = torch.cat(preds2, dim=0)
            trues2 = torch.cat(trues2, dim=0)
            mae1, rmse1 = metrics.compute_all_metrics(preds, trues)
            mae2, rmse2 = metrics.compute_all_metrics(preds2, trues2)
            mae = (mae1+mae2)/2
            rmse = (rmse1+rmse2)/2
            logger.info(f'***** Average Horizon, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f} *****')
            results = pd.DataFrame(columns=['Time','Test MAE', 'Test RMSE'], index=range(1))
            results.iloc[0, 0]= 'Average'
            results.iloc[0, 1]= mae
            results.iloc[0, 2]= rmse

        else:
            maes = []
            rmses = []
            mae, rmse = metrics.compute_all_metrics(preds, trues)
            logger.info(f'***** Average Horizon, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f} *****')
            maes.append(mae)
            rmses.append(rmse)
            if self.horizon == 24: 
                for i in range(0, self.horizon, 8):
                    pred = preds[:,i: i + 8]
                    true = trues[:,i: i + 8]
                    result = metrics.compute_all_metrics(pred, true)
                    maes.append(result[0])
                    rmses.append(result[1])
                if 'AIR' in self.data:
                    logger.info(f'***** (0-7) 1-8h Test MAE: {maes[1]:.4f}, Test RMSE: {rmses[1]:.4f} *****')
                    logger.info(f'***** (8-15) 9-16h Test MAE: {maes[2]:.4f}, Test RMSE: {rmses[2]:.4f} *****')
                    logger.info(f'***** (16-23) 17-24h Test MAE: {maes[3]:.4f}, Test RMSE: {rmses[3]:.4f} *****')

                    results = pd.DataFrame(columns=['Time','Test MAE', 'Test RMSE'], index=range(5))
                    Time_list=['Average','1-8h','9-16h','17-24h', 'SuddenChange']
                else:
                    logger.info(f'***** (0-7) 1-40min Test MAE: {maes[1]:.4f}, Test RMSE: {rmses[1]:.4f} *****')
                    logger.info(f'***** (8-15) 41-80min Test MAE: {maes[2]:.4f}, Test RMSE: {rmses[2]:.4f} *****')
                    logger.info(f'***** (16-23) 81-120min Test MAE: {maes[3]:.4f}, Test RMSE: {rmses[3]:.4f} *****')

                    results = pd.DataFrame(columns=['Time','Test MAE', 'Test RMSE'], index=range(5))
                    Time_list=['Average','1-40min','41-80min','81-120min', 'SuddenChange']

                for i in range(4):
                    results.iloc[i, 0]= Time_list[i]
                    results.iloc[i, 1]= maes[i]
                    results.iloc[i, 2]= rmses[i]
            elif self.horizon == 12: 
                for i in range(0, self.horizon, 4):
                    pred = preds[:,i: i + 4]
                    true = trues[:,i: i + 4]
                    result = metrics.compute_all_metrics(pred, true)
                    maes.append(result[0])
                    rmses.append(result[1])
                if 'AIR' in self.args.data:
                    logger.info(f'***** (0-3) 1-4h Test MAE: {maes[1]:.4f}, Test RMSE: {rmses[1]:.4f} *****')
                    logger.info(f'***** (4-7) 5-8h Test MAE: {maes[2]:.4f}, Test RMSE: {rmses[2]:.4f} *****')
                    logger.info(f'***** (8-11) 9-12h Test MAE: {maes[3]:.4f}, Test RMSE: {rmses[3]:.4f} *****')

                    results = pd.DataFrame(columns=['Time','Test MAE', 'Test RMSE'], index=range(5))
                    Time_list=['Average','1-4h','5-8h','9-12h', 'SuddenChange']
                else:
                    logger.info(f'***** (0-3) 1-20min Test MAE: {maes[1]:.4f}, Test RMSE: {rmses[1]:.4f} *****')
                    logger.info(f'***** (4-7) 21-40min Test MAE: {maes[2]:.4f}, Test RMSE: {rmses[2]:.4f} *****')
                    logger.info(f'***** (8-11) 41-60min Test MAE: {maes[3]:.4f}, Test RMSE: {rmses[3]:.4f} *****')

                    results = pd.DataFrame(columns=['Time','Test MAE', 'Test RMSE'], index=range(5))
                    Time_list=['Average','1-20min','21-40min','41-60min', 'SuddenChange']

                for i in range(4):
                    results.iloc[i, 0]= Time_list[i]
                    results.iloc[i, 1]= maes[i]
                    results.iloc[i, 2]= rmses[i]
            else:
                print('The output length is not 24 or 12!!!')

            mask_sudden_change = metrics.sudden_changes_mask(trues, datapath=self.args.data_root_path, null_val=0.0, threshold_start=75, threshold_change=20, horizon=self.horizon)
            results.iloc[4, 0] = Time_list[4]
            mae_sc, rmse_sc = metrics.compute_sudden_change(mask_sudden_change, preds, trues, null_value=0.0)
            results.iloc[4, 1:] = [mae_sc, rmse_sc]
            logger.info(f'***** Sudden Changes MAE: {mae_sc:.4f}, Test RMSE: {rmse_sc:.4f} *****')
        
        results.to_csv(os.path.join(self.pt_dir, f'metrics_{self.cur_exp}.csv'), index = False)
        # results.to_csv(os.path.join(folder_path, f'metrics_{self.cur_exp}.csv'), index = False)
