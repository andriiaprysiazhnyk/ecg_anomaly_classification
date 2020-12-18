import os
import torch

from ecg_anomaly_classification.data_api.data_api import read_ecg


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, ptb_path, df, transform=None):
        self.ptb_path = ptb_path
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        file_name = self.df.iloc[item]["filename_lr"]
        x = read_ecg(os.path.join(self.ptb_path, file_name))
        y = self.df.iloc[item]["target"]

        if self.transform:
            return self.transform(x), y

        mean = torch.FloatTensor([-0.0020, -0.0015, 0.0006, 0.0018, -0.0012, -0.0004, 0.0002, -0.0009,
                                  -0.0016, -0.0014, -0.0008, -0.0024])
        std = torch.FloatTensor([0.1662, 0.1642, 0.1723, 0.1410, 0.1481, 0.1463, 0.2348, 0.3369, 0.3336,
                                 0.2984, 0.2732, 0.2803])
        return (torch.FloatTensor(x) - mean) / std, y
