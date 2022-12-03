import pandas as pd
import torch
from utils import USER_COL, ITEM_COL, GROUP_COL

class TagRecDataset(torch.utils.data.Dataset):
  def __init__(self, df: pd.DataFrame):
    self.df = df
    self.data = torch.tensor(df[[USER_COL, ITEM_COL, GROUP_COL]].values, dtype=torch.long)
  def __len__(self):
    return len(self.data)
    
  def __getitem__(self, idx):
    return self.data[idx][0], self.data[idx][1], self.data[idx][2],