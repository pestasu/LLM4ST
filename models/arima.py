import torch
import torch.nn as nn
import torch.optim as optim

class ARIMAModel(nn.Module):
    def __init__(self, p, d, q, num_features):
        super(ARIMAModel, self).__init__()
        self.p = p
        self.d = d
        self.q = q
        self.num_features = num_features
        self.ar_params = nn.Parameter(torch.randn(num_features, p))  # AR参数
        self.ma_params = nn.Parameter(torch.randn(num_features, q))  # MA参数

    def forward(self, x):
        batch_size, num_station, seq_len, num_features = x.size()
        
        x = self.differencing(x, self.d)
        
        # 计算AR部分
        ar_part = torch.zeros((batch_size, num_station, seq_len, num_features)).to(x.device)
        for i in range(self.p, seq_len):
            ar_part[:, :, i, :] = torch.sum(self.ar_params.unsqueeze(0).unsqueeze(0) * x[:, :, i-self.p:i, :], dim=2)
        
        # 计算MA部分
        ma_part = torch.zeros((batch_size, num_station, seq_len, num_features)).to(x.device)
        for i in range(self.q, seq_len):
            ma_part[:, :, i, :] = torch.sum(self.ma_params.unsqueeze(0).unsqueeze(0) * x[:, :, i-self.q:i, :], dim=2)

        return ar_part + ma_part

    def differencing(self, x, d):
        for _ in range(d):
            x = x[:, :, 1:, :] - x[:, :, :-1, :]
        return x