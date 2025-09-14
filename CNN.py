import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, Dropout
from torchvision import transforms

class CNN(nn.Module):
   def __init__(self):
      super(CNN, self).__init__()
      self.conv1 = Conv2d(1, 16, kernel_size=3, padding=1)
      self.bn1 = BatchNorm2d(16)
      self.conv2 = Conv2d(16, 32, kernel_size=3, padding=1)
      self.bn2 = BatchNorm2d(32)
      self.conv3 = Conv2d(32, 64, kernel_size=3, padding=1)
      self.bn3 = BatchNorm2d(64)
      self.pool = MaxPool2d(2, 2)
      self.fc1 = Linear(64 * 28 * 28, 128)
      self.dropout = Dropout(0.5)
      self.fc2 = Linear(128, 2)

   def forward(self, x):
      x = self.pool(self.bn1(torch.relu(self.conv1(x))))
      x = self.pool(self.bn2(torch.relu(self.conv2(x))))
      x = self.pool(self.bn3(torch.relu(self.conv3(x))))
      x = x.view(x.size(0), -1)
      x = torch.relu(self.fc1(x))
      x = self.dropout(x)
      x = self.fc2(x)
      return x