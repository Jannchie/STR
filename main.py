import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader import TagRecDataset
from models import STR, SimpleX
from utils import RecHelper, TagRecHelper, load_bili, load_public, printt, set_seed, USER_COL, ITEM_COL
from config import str_config, str_config_gowalla, str_config_yelp, simplex_config, sweep_configuration, mf_bpr_config
from matplotlib import pyplot as plt
import random

try:
  import wandb
except Exception:
  use_wandb = False
use_wandb = True


def run(model: torch.nn.Module, helper: TagRecHelper, config: dict, dataset: str):
  set_seed(47)
  train_ds = TagRecDataset(helper.train_set)
  # seed = random.randint(0, 100000)
  print('init model...')
  model = model(helper, config=config)
  total_params = sum(p.numel() for p in model.parameters())
  print(f'{"Total parameters:":20} {total_params}')
  print(f'{"Number of users:":20} {model.nuser}')
  print(f'{"Number of items:":20} {model.nitem}')
  if use_wandb:
    wandb.watch(model)
  loader = DataLoader(train_ds, batch_size=config.get(
    'batch_size', 512), num_workers=2, shuffle=True, pin_memory=True)
  optimizer = torch.optim.Adam(model.parameters(), lr=config.get(
    'lr', 1e-3), weight_decay=config.get('weight_decay', 1e-5))

  printt('Hyperparameters')
  for k, v in config.items():
    print(f'{f"{k}:":20} {v}')

  if dataset == 'bilibili':
    rec_helper = RecHelper(model, helper)
  patience_init = 2
  patience = patience_init
  current_max = 0
  printt('Start training...')
  best_model  = None
  best_res = None
  for epoch in range(config.get('n_epoch', 10)):
    loader = tqdm(loader, ascii=" #", desc=f"Epoch {epoch+1:02d}", total=len(
      loader), bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt:5s}/{total_fmt:5s} [{elapsed}<{remaining}] {postfix}]")
    model.train()
    train(model, optimizer, loader)
    model.eval()
    if dataset == 'bilibili':
      rec_list = rec_helper.get_rec_tags('1850091')
      print(f'for 1850091: {rec_list}')
    res_mean, result_df = helper.test(model)
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
    # early stopping & save best model
    if res_mean[0] > current_max:
      current_max = res_mean[0]
      best_model = model.state_dict()
      best_res = result_df
      patience = patience_init
    else:
      patience -= 1
      if patience == 0:
        break
  printt('Training finished.')
  print(f'Best Recall: {current_max}')
  print(f'Best result: \n {best_res}')
  path = f'./output/{model.__class__.__name__}-{dataset}_model.pth'
  torch.save(best_model, path)
  print(f'Model saved to {path}')


def record_neptune_data(neptune_run, res_mean):
  neptune_run['Recall'].log(res_mean[0])
  neptune_run['Precision'].log(res_mean[1])
  neptune_run['F1'].log(res_mean[2])
  neptune_run['NDCG'].log(res_mean[3])
  neptune_run['MAP'].log(res_mean[4])
  neptune_run['MRR'].log(res_mean[5])
  neptune_run['HitRate'].log(res_mean[6])


def train(model, optimizer, loader):
  scaler = torch.cuda.amp.GradScaler()
  for u, i, g in loader:
    u = u.to(model.device)
    i = i.to(model.device)
    g = g.to(model.device)
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
      loss = model(u, i)
    scaler.scale(loss).backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    scaler.step(optimizer)
    scaler.update()
    loader.set_postfix(loss=f'{loss.item():08.3f}')
    if use_wandb:
      wandb.log({'loss': loss.item()})


if __name__ == '__main__':
  config = str_config
  model_name = 'STR'
  dataset = 'bilibili'
  
  trd = load_bili() if dataset == 'bilibili' else load_public(dataset)
  if model_name == 'STR':
    model = STR
  elif model_name == 'SimpleX':
    model = SimpleX

  if use_wandb: wandb.init(project=f"{model_name}-{dataset}", entity="jannchie", name=f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
  if use_wandb: wandb.config.update(config)
  run(model, trd, config, dataset)

  # def run_sweep():
  #   wandb.init(name=f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
  #   run(model, trd, wandb.config, dataset)

  # if use_wandb:
  #   wandb.login()
  #   sweep_id = wandb.sweep(sweep=sweep_configuration,
  #                          project=f"{model_name}-{dataset}", entity="jannchie")
  #   wandb.agent(sweep_id, function=run_sweep)
