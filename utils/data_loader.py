import os
import datetime
import ipdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.serialization import load_pkl
import warnings

warnings.filterwarnings('ignore')

class StandardScaler():
    """
    Standard scaler for input normalization
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class get_dataset(Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ["train", "valid", "test"], "error mode"
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        data_file_path = os.path.join(args.data_root_path, args.data_file_path)
        index_file_path = os.path.join(args.data_root_path, args.index_file_path)
        self._check_if_file_exists(data_file_path, index_file_path)

        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()
        self.scaler = data["scaler"]

        # read index
        self.index = load_pkl(index_file_path)[mode]

    def _check_if_file_exists(self, data_file_path, index_file_path):
        """Check if data file and index file exist.
        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path
        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("Not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("Not find index file {0}".format(index_file_path))

    def __getitem__(self, index):
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x T x N x C.
        """

        idx = list(self.index[index])
        if isinstance(idx[0], int):
            # continuous index
            history_data = self.data[idx[0]:idx[1]]
            future_data = self.data[idx[1]:idx[2]]
        else:
            # discontinuous index or custom index
            # NOTE: current time $t$ should not included in the index[0]
            history_index = idx[0]    # list
            assert idx[1] not in history_index, "current time t should not included in the idx[0]"
            history_index.append(idx[1])
            history_data = self.data[history_index]
            future_data = self.data[idx[1], idx[2]]
        return torch.Tensor(history_data), torch.Tensor(future_data)

    def __len__(self):
        return len(self.index)
        
class get_dataset_1d(Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ["train", "valid", "test"], "error mode"
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        data_file_path = os.path.join(args.data_root_path, args.data_file_path)
        index_file_path = os.path.join(args.data_root_path, args.index_file_path)
        self._check_if_file_exists(data_file_path, index_file_path)

        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()
        self.scaler = data["scaler"]

        # read index
        self.index = load_pkl(index_file_path)[mode]
        
        self._load_data()

    def _load_data(self):
        history_data = [self.data[start:end].unsqueeze(0) for start, end, _ in self.index]
        future_data = [self.data[start:end].unsqueeze(0) for _, start, end in self.index]
        history_data = torch.cat(history_data, 0)
        B, T, N, C = history_data.shape
        future_data = torch.cat(future_data, 0)
        self.history_data = history_data.permute(0, 2, 1, 3).reshape(B*N, T, C)
        self.future_data = future_data.permute(0, 2, 1, 3).reshape(B*N, -1, C)
    
    def _check_if_file_exists(self, data_file_path, index_file_path):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))
        
    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x T x C.
        """

        return torch.Tensor(self.history_data[index]), torch.Tensor(self.future_data[index])


    def __len__(self):
        return len(self.index)


        

def get_dataloader(args, need_location=True):

    get_dataset_fuc = get_dataset if need_location else get_dataset_1d
    
    datasets = {
        'train': get_dataset_fuc(args, mode='train'),
        'valid': get_dataset_fuc(args, mode='valid'),
        'test': get_dataset_fuc(args, mode='test')
    }

    scalers = datasets['train'].scaler
    dataLoader = {
        ds: DataLoader(datasets[ds],
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False if ds == 'test' else True)
        for ds in datasets.keys()
    }
    return dataLoader, scalers