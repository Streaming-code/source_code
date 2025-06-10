from torch.utils.data import Dataset, DataLoader
import numpy as np

# 自定义数据集类
class ExperienceDataset(Dataset):
    def __init__(self, states, labels):
        self.states = states
        self.labels = labels

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        label = self.labels[idx]
        return state, label