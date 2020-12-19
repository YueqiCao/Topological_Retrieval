import torch
import pickle
import pandas
from torch.utils.data import Dataset, DataLoader

class TreeDataset(Dataset):
    def __init__(self,csv_path,**kwargs):
        super(TreeDataset, self).__init__()
        self.path = csv_path
        self.tree = self.get_hypernimity()

    def get_hypernimity(self):
        hypernyms = dict()
        with open(self.csv_path)