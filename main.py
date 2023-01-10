import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader import TagRecDataset
from models import MF, STR, SimpleX
from utils import RecHelper, TagRecHelper, load_bili, load_bili_2, load_public, printt, set_seed, USER_COL, ITEM_COL
from config import str_config, str_config_gowalla, str_config_yelp, simplex_config, mf_bpr_config, str_config_amazon
from matplotlib import pyplot as plt
import random

try:
  import wandb
except Exception:
  use_wandb = True
use_wandb = True

now = datetime.datetime.now()

def run(model: torch.nn.Module, helper: TagRecHelper, config: dict, dataset: str):
  # sourcery skip: low-code-quality
  set_seed(47)
  # seed = random.randint(0, 100000)
  print('init model...')
  model = model(helper, config=config)
  path = f'./output/{model.__class__.__name__}-{dataset}_model.pth'
  # model.load_state_dict(torch.load(path))
  total_params = sum(p.numel() for p in model.parameters())
  print(f'{"Total parameters:":20} {total_params}')
  print(f'{"Number of users:":20} {model.nuser}')
  print(f'{"Number of items:":20} {model.nitem}')
  if use_wandb:
    wandb.watch(model)
  bs = config.get('batch_size', 512)
  lr = config.get('lr', 1e-3)
  wd = config.get('weight_decay', 0)
  loader = DataLoader(TagRecDataset(helper.train_set), batch_size=bs, num_workers=2, shuffle=True, pin_memory=True)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
  print(optimizer)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, verbose=True, factor=0.1, min_lr=lr * 0.01)
  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)
  printt('Hyperparameters')
  for k, v in config.items():
    print(f'{f"{k}:":20} {v}')
  if dataset == 'bilibili':
    rec_helper = RecHelper(model, helper)
  patience_init = 5
  patience = patience_init
  current_max = 0
  # helper.test(model, should_mask=dataset != 'bilibili')
  printt('Start training...')
  best_model  = None
  best_res_df = None
  best_res_mean_df = None
  for epoch in range(config.get('n_epoch', 10)):
    loader = tqdm(loader, ascii=" #", desc=f"Epoch {epoch+1:02d}", total=len(
      loader), bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt:5s}/{total_fmt:5s} [{elapsed}<{remaining}] {postfix}]")
    model = model.train()
    train(model, optimizer, loader)
    model = model.eval()
    if dataset == 'bilibili':
      rec_list = [d['name'] for d in rec_helper.get_rec_tags('1850091')]
      print(f'for 1850091: {rec_list}')
    res_df, res_mean_df = helper.test(model, should_mask=dataset != 'bilibili')
    res_mean = res_mean_df.values[0]
    scheduler.step(res_mean[0])
    # scheduler.step()
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
      best_res_mean_df = res_mean_df
      best_res_df = res_df
      patience = patience_init
    elif patience == 0:
      break
    else:
      patience -= 1
      # if patience == patience_init - 2:
      # print(f'Load best model at epoch {epoch}')
      # model.load_state_dict(best_model)
  printt('Training finished.')
  print(f'Best Recall: {current_max}')
  print(f'Best result: \n {best_res_mean_df}')

  best_res_df.to_csv(f'./output/{model.__class__.__name__}-{dataset}_result_{now.strftime("%Y%m%d%H%M%S")}.csv')
  should_save = True
  if should_save:
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
      if model.__class__.__name__ in ['SimpleX', 'MF']:
        loss = model(u, i)
      elif model.__class__.__name__ == 'STR':
        loss = model(u, i, g)
    scaler.scale(loss).backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    scaler.step(optimizer)
    scaler.update()
    # loss.backward()
    # optimizer.step()
    loader.set_postfix(loss=f'{loss.item():08.3f}')
    if use_wandb:
      wandb.log({'loss': loss.item()})


if __name__ == '__main__':
  model_name = 'STR'
  dataset = 'bilibili'
  # dataset = 'yelp2018'
  trd = load_bili_2() if dataset == 'bilibili' else load_public(dataset)
  if model_name == 'STR':
    if dataset == 'bilibili':
      config = str_config
    elif dataset == 'gowalla':
      config = str_config_gowalla
    elif dataset == 'yelp2018':
      config = str_config_yelp
    elif dataset == 'amazon':
      config = str_config_amazon
    model = STR
  elif model_name == 'SimpleX':
    config = simplex_config
    model = SimpleX
  elif model_name == 'MF-BPR':
    config = mf_bpr_config
    model = MF
  sweep = False
  if not sweep:
    if use_wandb: wandb.init(project=f"{model_name}-{dataset}", entity="jannchie", name=f"{now.strftime('%Y%m%d%H%M%S')}", dir='./models')
    if use_wandb: wandb.config.update(config)
    run(model, trd, config, dataset)
  else:
    sweep_configuration = {
      'method': 'grid',
      'name': '230107',
      'metric': {'goal': 'maximize', 'name': 'Recall'}, 
      'parameters': {key: {'value': value} for key, value in config.items()}
    }


    def run_sweep():
      def r():
        wandb.init(name=f"{now.strftime('%Y%m%d%H%M%S')}")
        run(model, trd, wandb.config, dataset)
      if use_wandb:
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"{model_name}-{dataset}", entity="jannchie")
        wandb.agent(sweep_id, function=r)
    # sweep_configuration['parameters']['w_g'] = {'values': [0, 0.1, 0.2, 0.3, 0.4, 0.5]}
    # sweep_configuration['parameters']['aggregate'] = {'values': ['weighted-sum']}
    # sweep_configuration['parameters']['loss_neg_k'] = {'values': [4, 8, 16, 32, 64]}
    # sweep_configuration['parameters']['w_ii'] = {'value': 1}
    # sweep_configuration['parameters']['loss_neg_n'] = {'values': [800, 1000, 1200, 1600, 2000]}
    # sweep_configuration['parameters']['dropout'] = {'value': 0.1}
    # sweep_configuration['parameters']['loss_neg_k'] = {'values': [10, 50, 100, 150, 200, 250, 300, 500]}
    # sweep_configuration['parameters']['loss_neg_w'] = {'values': [130, 150, 200, 300]}
    # sweep_configuration['parameters']['popular_alpha'] = {'values': [0, 0.2, -0.2]}
    # sweep_configuration['parameters']['n_interactive_items'] = {'values': [4, 8, 16]}
    # sweep_configuration['parameters']['loss_neg_m'] = {'values': [0.0]}
    # sweep_configuration['parameters']['aggregate_a'] = {'values': [0.2, 0, -0.2] }
    # sweep_configuration['parameters']['loss_neg_w'] = {'values': [30, 40, 50, 60]}
    # sweep_configuration['parameters']['loss_neg_a'] = {'values': [] }
    run_sweep()
