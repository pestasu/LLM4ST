import os
import time
import warnings
import ipdb
import json
import pickle
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from tslearn.clustering import TimeSeriesKMeans, KShape
import scipy.sparse as sp
from fastdtw import fastdtw

from exp.exp_basic import Exp_Basic
from utils import metrics, data_loader
from utils.tools import adjust_learning_rate
from utils.serialization import load_pkl
from utils.data_loader import StandardScaler, get_dataloader
from models import *

warnings.filterwarnings('ignore')
my_logger = 'lazy'
logger = logging.getLogger(my_logger)

class Exp_pdformer(Exp_Basic):
    def __init__(self, args, ii):
        super(Exp_pdformer, self).__init__(args)
        self.cur_exp = ii
        self.random_flip = True
        # self.loss_fn = self._select_criterion(metric_func='huber')


    def _build_model(self): 
        self.adj_mx = load_pkl(os.path.join(self.args.data_root_path, 'adj_mx.pkl'))
        self.lap_mx = self._cal_lape(self.adj_mx).to(self.device)
        sd_mx, sh_mx = self._load_rel()
        dtw_matrix = self._get_dtw()
        self.pattern_key_file = os.path.join(self.args.data_root_path, 'pattern_keys_{}_{}_{}_{}_{}.npy'.format(
        self.args.cluster_method, self.args.cand_key_days, self.args.s_attn_size, self.args.n_cluster, self.args.cluster_max_iter))
        self._cal_pattern_keys()
        return self.model_dict[self.model_name](self.args, pattern_keys=self.pattern_keys, adj_mx=self.adj_mx, sd_mx=sd_mx, sh_mx=sh_mx, dtw_matrix=dtw_matrix).to(self.device)
    
    def _get_dtw(self):
        if 'AIR' in self.args.data:
            points_per_hour = 1
        else:
            points_per_hour = 12
        cache_path = f'{self.args.data_root_path}/dtw_{self.args.data}1.npy'
        if not os.path.exists(cache_path):
            data_file_path = os.path.join(self.args.data_root_path, self.args.data_file_path)
            df = load_pkl(data_file_path)["processed_data"]
            df_train = df[:int(0.6*df.shape[0])]
            data_mean = np.mean(
                [df_train[24 * points_per_hour * i: 24 * points_per_hour * (i + 1)]
                 for i in range(df_train.shape[0] // (24 * points_per_hour))], axis=0)
            dtw_distance = np.zeros((self.args.num_nodes, self.args.num_nodes))
            for i in tqdm(range(self.args.num_nodes)):
                for j in range(i, self.args.num_nodes):
                    dtw_distance[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
            for i in range(self.args.num_nodes):
                for j in range(i):
                    dtw_distance[i][j] = dtw_distance[j][i]
            np.save(cache_path, dtw_distance)
        dtw_matrix = np.load(cache_path)
        logger.info('Load DTW matrix from {}'.format(cache_path))
        return dtw_matrix

    def _load_rel(self):
        adj_mx_file = os.path.join(self.args.data_root_path, 'adj_mx1.pkl')
        sh_mx_file = os.path.join(self.args.data_root_path, f'{self.args.data}.npy')
        if not os.path.exists(adj_mx_file):
            relfile = pd.read_csv(f'{self.args.data_root_path}/{self.args.data}.csv')
            set_weight_link_or_dist, init_weight_inf_or_zero = 'dist', 'inf'
            logger.info('set_weight_link_or_dist: {}'.format(set_weight_link_or_dist))
            logger.info('init_weight_inf_or_zero: {}'.format(init_weight_inf_or_zero))
            if len(relfile.columns) != 3:
                raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
            else:
                weight_col = relfile.columns[-1]
                distance_df = relfile[~relfile[weight_col].isna()][['from', 'to', weight_col]]
            self.adj_mx = np.zeros((self.args.num_nodes, self.args.num_nodes), dtype=np.float32)
            if init_weight_inf_or_zero.lower() == 'inf' and set_weight_link_or_dist.lower() != 'link':
                self.adj_mx[:] = np.inf
            for row in distance_df.values:
                if set_weight_link_or_dist.lower() == 'dist':
                    self.adj_mx[int(row[0]), int(row[1])] = row[2]
                else:
                    self.adj_mx[int(row[0]),int(row[1])] = 1
            logger.info('Max adj_mx value = {}'.format(self.adj_mx.max()))

            with open(adj_mx_file, "wb") as f:
                pickle.dump(self.adj_mx, f)

        self.adj_mx = load_pkl(adj_mx_file)
        logger.info('Load new adj matrix from {}'.format(adj_mx_file))
        sd_mx = None
        sh_mx = None
        if not os.path.exists(sh_mx_file):
            sh_mx = self.adj_mx.copy()
            if self.args.type_short_path == 'hop':
                sh_mx[sh_mx > 0] = 1
                sh_mx[sh_mx == 0] = 511
                for i in range(self.args.num_nodes):
                    sh_mx[i, i] = 0
                for k in range(self.args.num_nodes):
                    for i in range(self.args.num_nodes):
                        for j in range(self.args.num_nodes):
                            sh_mx[i, j] = min(sh_mx[i, j], sh_mx[i, k] + sh_mx[k, j], 511)
                np.save(sh_mx_file, sh_mx)
        sh_mx = np.load(sh_mx_file)
        logger.info('Load short hop matrix from {}'.format(sh_mx_file))
        return sd_mx, sh_mx
    
    def _calculate_adjacency_matrix(self):
        self._logger.info("Start Calculate the weight by Gauss kernel!")
        sd_mx = self.adj_mx.copy()
        distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.adj_mx / std))
        self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0
        if self.args.type_short_path == 'dist':
            sd_mx[self.adj_mx == 0] = np.inf
            for k in range(self.args.num_nodes):
                for i in range(self.args.num_nodes):
                    for j in range(self.args.num_nodes):
                        sd_mx[i, j] = min(sd_mx[i, j], sd_mx[i, k] + sd_mx[k, j])
        return sd_mx

    def _cal_pattern_keys(self):
        if not os.path.exists(self.pattern_key_file):
            cand_key_time_steps = self.args.cand_key_days * self.args.points_per_day
            data_index = data_loader.get_dataset(self.args, mode='train').index
            data_all = data_loader.get_dataset(self.args, mode='train').data[..., :self.args.feat_dims[0]]
            data_train = [data_all[start:end].unsqueeze(0) for start, end, _ in data_index]
            data_train = torch.cat(data_train, 0)
            pattern_cand_keys = data_train[:cand_key_time_steps, :, :self.args.s_attn_size].reshape(-1, self.args.s_attn_size, self.args.feat_dims[0])
            logger.info("Clustering...")
            if self.args.cluster_method == "kshape":
                km = KShape(n_clusters=self.args.n_cluster, max_iter=self.args.cluster_max_iter).fit(pattern_cand_keys.numpy())
            else:
                km = TimeSeriesKMeans(n_clusters=self.n_cluster, metric="softdtw", max_iter=self.args.cluster_max_iter).fit(pattern_cand_keys.numpy())
            self.pattern_keys = km.cluster_centers_
            np.save(self.pattern_key_file, self.pattern_keys)
            logger.info("Saved at file " + self.pattern_key_file + ".npy")
        else:
            self.pattern_keys = np.load(self.pattern_key_file)
            logger.info("Loaded file " + self.pattern_key_file + ".npy")

    def _calculate_normalized_laplacian(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        logger.info(f"Number of isolated points: {isolated_point_num}")
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian, isolated_point_num

    def _cal_lape(self, adj_mx):
        L, isolated_point_num = self._calculate_normalized_laplacian(adj_mx)
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

        laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: self.args.lape_dim + isolated_point_num + 1]).float()
        laplacian_pe.require_grad = False
        return laplacian_pe

    def early_stop(self, epoch, best_loss):
        logger.info(f'Early stop at epoch {epoch}, loss = {best_loss:.6f}')

    def train_batch(self, x, y, epoch):
        '''
        the training process of a batch
        '''   
        self.optimizer.zero_grad()
        x = x.to(self.device)
        y = y[..., :1].to(self.device)

        batch_lap_pos_enc = self.lap_mx.to(x.device)
        if self.random_flip:
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(self.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        out = self.model(x, batch_lap_pos_enc)
        pred, true = self._inverse_transform([out, y])
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
            train_losses = []

            if epoch - self.saved_epoch > self.patience:
                self.early_stop(epoch, min(self.val_losses))
                np.savetxt(os.path.join(self.pt_dir, f'val_loss_{self.cur_exp}.txt'), self.val_losses, fmt='%.4f', delimiter=',')
                break

            logger.info('------start training!------')
            start_time = time.time()
            for i, (batch_x, batch_y) in enumerate(self.dataloader['train']):
                iter_count += 1
                loss = self.train_batch(batch_x, batch_y, epoch)
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
                    logger.info(f'Epoch [{epoch}/{self.train_epochs}]({iter_count}) | val_mae:{val_loss:.4f} | train_loss:{np.mean(train_losses):.4f}')
            
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
        preds, trues, masks = [], [], []
        self.model.eval()
        total_time = 0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.dataloader['valid']):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y[..., :1].to(self.device)

                time_now = time.time()


                output = self.model(batch_x, self.lap_mx)
                pred, true = self._inverse_transform([output, batch_y])

                total_time += time.time() - time_now

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
            logger.info(f'------------Test process~load model({self.args.model}-{self.args.data}-{self.args.version})------------')
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

                output = self.model(batch_x, lap_mx=self.lap_mx)
                pred, true = self._inverse_transform([output, batch_y])
                preds.append(pred.cpu())
                trues.append(true.cpu())
                
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

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

