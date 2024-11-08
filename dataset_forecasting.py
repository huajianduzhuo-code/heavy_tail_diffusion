import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch


class Forecasting_Dataset(Dataset):
    def __init__(self, mode="train",config=None):
        self.config = config
        self.main_data = np.load(config["data"]["train_data_path"]) # size * L * K = 10000 * 2 * 5
        self.mask_data = np.ones_like(self.main_data)

        self.size, self.L, self.K = self.main_data.shape

        if config["data"]["normalize_method"] == "normalize":
            self.mean_data = self.main_data.mean((0,))
            self.std_data = self.main_data.std((0,))
            
            self.main_data = (self.main_data - self.mean_data) / self.std_data
        elif config["data"]["normalize_method"] == "reflect_normalize":
            self.mean_data = np.zeros(self.main_data.shape[1:])
            self.std_data = np.sqrt((self.main_data**2).mean((0,)))

            self.main_data = (self.main_data - self.mean_data) / self.std_data
        else:
            raise ValueError("not valid normalization")

        total_length = len(self.main_data)
        if mode == 'train': 
            start = 0
            end = int(self.main_data.shape[0]*0.9)
            self.use_index = np.arange(start,end,1)
        if mode == 'valid': #valid
            start = int(self.main_data.shape[0]*0.9)
            end = int(self.main_data.shape[0])
            self.use_index = np.arange(start,end)

    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]
        target_mask = self.mask_data[index].copy()

        condition_L = self.config["data"]["condition_L"]
        target_L = self.config["data"]["target_L"]
        assert condition_L + target_L == target_mask.shape[0], f"{condition_L}+{target_L}!={target_mask.shape[0]}"
        target_mask[condition_L:, :] = 0.0
        s = {
            'observed_data': self.main_data[index],
            'observed_mask': self.mask_data[index],
            'gt_mask': target_mask,
            'timepoints': np.arange(self.L) * 1.0, 
            'feature_id': np.arange(self.K) * 1.0, 
        }

        return s
    def __len__(self):
        return len(self.use_index)

class Forecasting_Dataset_Test(Dataset):
    def __init__(self, config=None):
        self.config = config
        self.main_data = np.load(config["data"]["train_data_path"]) # size * L * K = 10000 * 2 * 5
        self.test_data = np.load(config["data"]["test_data_path"]) # size * L * K = 1000 * 2 * 5
        self.test_data = self.test_data[:config["test"]["batch_size"],:,:]
        self.mask_data = np.ones_like(self.main_data)

        self.size, self.L, self.K = self.main_data.shape

        if config["data"]["normalize_method"] == "normalize":
            self.mean_data = self.main_data.mean((0,))
            self.std_data = self.main_data.std((0,))
            
            self.main_data = (self.main_data - self.mean_data) / self.std_data
            self.test_data = (self.test_data - self.mean_data) / self.std_data
        elif config["data"]["normalize_method"] == "reflect_normalize":
            self.mean_data = np.zeros(self.main_data.shape[1:])
            self.std_data = np.sqrt((self.main_data**2).mean((0,)))

            self.main_data = (self.main_data - self.mean_data) / self.std_data
            self.test_data = (self.test_data - self.mean_data) / self.std_data  
        else:
            raise ValueError("not valid normalization")

        self.mask_data = np.ones_like(self.test_data)
        self.use_index = np.arange(0,config["test"]["batch_size"])
        
    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]
        target_mask = self.mask_data[index].copy()
        
        condition_L = self.config["data"]["condition_L"]
        target_L = self.config["data"]["target_L"]
        assert condition_L + target_L == target_mask.shape[0], f"{condition_L}+{target_L}!={target_mask.shape[0]}"
        target_mask[condition_L:, :] = 0.0
        s = {
            'observed_data': self.test_data[index],
            'observed_mask': self.mask_data[index],
            'gt_mask': target_mask,
            'timepoints': np.arange(self.L) * 1.0, 
            'feature_id': np.arange(self.K) * 1.0, 
        }

        return s
    def __len__(self):
        return len(self.use_index)


def get_dataloader(device,batch_size=8,config=None):
    dataset = Forecasting_Dataset(mode='train',config=config)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Forecasting_Dataset(mode='valid',config=config)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=0)

    scaler = torch.from_numpy(dataset.std_data).to(device).float()
    mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    return train_loader, valid_loader, scaler, mean_scaler



def get_test_dataloader(device,batch_size=8,config=None):
    test_dataset = Forecasting_Dataset_Test(config=config)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=0)

    scaler = torch.from_numpy(test_dataset.std_data).to(device).float()
    mean_scaler = torch.from_numpy(test_dataset.mean_data).to(device).float()

    return test_loader, scaler, mean_scaler