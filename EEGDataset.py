import torch
from torch.utils.data import Dataset

# Turns numpy datasets into datasets PyTorch can process, adding a channel dimension for cnn input
class EEGDataset(Dataset):
    def __init__(self, eegs, labels):
        self.eegs = torch.FloatTensor(eegs).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.eegs)
    
    def __getitem__(self, idx):
        return self.eegs[idx], self.labels[idx]
