import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader import TagRecDataset
from models import STR, SimpleX
from utils import RecHelper, TagRecHelper, load_bili, load_public, printt, set_seed, USER_COL, ITEM_COL
from config import str_config, str_config_gowalla, str_config_yelp, simplex_config, sweep_configuration
from matplotlib import pyplot as plt
import wandb
import random



def run(model: torch.nn.Module, helper: TagRecHelper, config: dict, use_wandb: bool = False):
  train_ds = TagRecDataset(helper.train_set)
  # seed = random.randint(0, 100000)
  seed = 47
  set_seed(seed)
  print('init model...')
  model = model(helper, config=config)
  total_params = sum(p.numel() for p in model.parameters())
  print(f'{"Total parameters:":20} {total_params}')
  print(f'{"Number of users:":20} {model.nuser}')
  print(f'{"Number of items:":20} {model.nitem}')
  if use_wandb: wandb.watch(model)
  loader = DataLoader(train_ds, batch_size=config['batch_size'], num_workers=2, shuffle=True, pin_memory=True)
  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

  printt('Hyperparameters')
  for k, v in config.items():
    print(f'{f"{k}:":20} {v}')

  printt('Start training...')
  for epoch in range(config['n_epoch']):
    # if 'ccl_neg_num' in model.config:
    #   model.generate_neg_sample(model.config['ccl_neg_num'])
    loader = tqdm(loader, ascii=" #", desc=f"Epoch {epoch+1:02d}", total=len(loader), bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt:5s}/{total_fmt:5s} [{elapsed}<{remaining}] {postfix}]")
    model.train()
    train(model, optimizer, loader)
    model.eval()
    res_mean = helper.test(model)
    if use_wandb: 
      wandb.log({
        'Recall': res_mean[0],
        'Precision': res_mean[1],
        'F1': res_mean[2],
        'NDCG': res_mean[3],
        'MAP': res_mean[4],
        'MRR': res_mean[5],
        'HitRate': res_mean[6],
      })
  printt('Training finished.')
  # for u in ul:
  #   res = model.recommend(u)
  #   res_list.append(res)
  # res_list = torch.cat(res_list, dim=0).tolist()
  # f = Recall(20)
  # recall_data = [f(res_list[i], true_tags[i]) for i in range(model.nuser)] 
  # plt.hist(recall_data, bins=10)
  # plt.show()
  torch.save(model.state_dict(), './output/model.pth')


def main():
  wandb.init(project="STR", entity="jannchie", name=f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
  run(STR, load_public('yelp2018'), wandb.config, use_wandb=True)
  
def train(model, optimizer, loader):
  for u, i, g in loader:
    u = u.to(model.device)
    i = i.to(model.device)
    g = g.to(model.device)
    optimizer.zero_grad()
    loss = model(u, i)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    optimizer.step()
    loader.set_postfix(loss=f'{loss.item():08.3f}')
    if use_wandb: wandb.log({'loss': loss.item()})
    
if __name__ == '__main__':
  use_wandb = False
  model = SimpleX
  dataset = 'yelp2018'
  trd = load_public(dataset)
  run(model, trd, simplex_config, use_wandb=use_wandb)
  
  # if use_wandb: 
  #   wandb.login()
  #   sweep_id = wandb.sweep(sweep=sweep_configuration, project='STR')
  #   wandb.agent(sweep_id, function=main)

  #   # wandb.init(project="STR", entity="jannchie", name=f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
  #   # wandb.config = config
  #   # run(STR, load_bili(), config)
  #   # wandb.init(project="SimpleX", entity="jannchie", name=f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
  #   # wandb.config = simplex_config
  

  