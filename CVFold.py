from EEGDataset import EEGDataset
from torch.utils.data import DataLoader
import torch
import numpy as np

class CVFold:
    def __init__(self, fold_num, 
             train_data, train_lbls,
             val_data, val_lbls,
             test_data, test_lbls
             ):
        
        self.fold = fold_num
        
        # 80% of train data
        self.train_data = train_data
        self.train_lbls = train_lbls
        
        # 20% of train data
        self.val_data = val_data
        self.val_lbls = val_lbls
        
        # For testing both methods
        self.test_data = test_data
        self.test_lbls = test_lbls
            
    def train_loader(self, batch_size):
        return self.make_dataset(self.train_data, self.train_lbls, batch_size)

    def val_loader(self, batch_size):
        return self.make_dataset(self.val_data, self.val_lbls, batch_size)

    def test_loader(self, batch_size):
        return self.make_dataset(self.test_data, self.test_lbls, batch_size)

    def make_dataset(self, data, lbls, batch_size):
        return DataLoader(EEGDataset(data, lbls), batch_size=batch_size, shuffle=True)