import csv
import os
import pickle
import random
from typing import Literal

import numpy as np
import pandas as pd
import torch
from recommenders.datasets.python_splitters import python_stratified_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader
from metrics import F1, MAP, MRR, NDCG, HitRate, Precision, Recall

USER_COL = 'user_id'
ITEM_COL = 'item_id'
SCORE_COL = 'score'
SEED = 42
GROUP_COL = 'item_group'

class TagRecHelper:
    def __init__(self, train_set, test_set, uid2idx, iid2idx, gid2idx):
        self.train_set = train_set
        self.test_set = test_set
        self.uid2idx = uid2idx
        self.iid2idx = iid2idx
        self.gid2idx = gid2idx
        self.nuser = len(uid2idx)
        self.nitem = len(iid2idx)
        self.ntag = len(gid2idx)
        self.uidx2id = {i: uid for uid, i in uid2idx.items()}
        self.iidx2id = {i: iid for iid, i in iid2idx.items()}
        self.gidx2id = {i: gid for gid, i in gid2idx.items()}
        truth_dict = self.test_set.groupby(USER_COL)[ITEM_COL].apply(list).to_dict()
        self.truth = [truth_dict.get(uid, []) for uid in range(self.nuser)]
        self.mask = self.train_set.groupby(USER_COL)[ITEM_COL].apply(list).to_list()
        users = torch.arange(0, self.nuser, dtype=torch.long)
        self.ul = DataLoader(users, batch_size=128, pin_memory=True)
        
    def test(self, model):
        res_list = []
        for u in tqdm(self.ul, ascii=" #", desc="Test    ", total=len(self.ul), bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt:5s}/{total_fmt:5s} [{elapsed}<{remaining}] {postfix}"):
            res = model.recommend(u, mask=self.mask)
            res_list.append(res)
        res_list = torch.cat(res_list, dim=0).tolist()
        # print(res_list[3])
        # self.test_target_user(model, 3)
        k = 20
        h = ['Recall', 'Precision', 'F1', 'NDCG', 'MAP', 'MRR', 'HitRate']
        f = [Recall(k), Precision(k), F1(k), NDCG(k), MAP(k), MRR(k), HitRate(k)]
        res_mean = [ np.mean([m(res_list[i], self.truth[i]) for i in range(len(res_list))]) for m in f ]
        result_df = pd.DataFrame([res_mean], columns=h)
        print('Test Result:', result_df, '', sep='\n')
        return res_mean, result_df

    def test_target_user(self, model, u):
        res = model.recommend(torch.tensor([u]), mask=self.mask)
        res_list = res.tolist()
        k = 20
        h = ['Recall', 'Precision', 'F1', 'NDCG', 'MAP', 'MRR', 'HitRate']
        f = [Recall(k), Precision(k), F1(k), NDCG(k), MAP(k), MRR(k), HitRate(k)]
        res_mean = [m(res_list[0], self.truth[u]) for m in f]
        result_df = pd.DataFrame([res_mean], columns=h)
        print(result_df)
    
class RecHelper:
  def __init__(self, model: torch.nn.Module, data:TagRecHelper):
    self.tag_info = { d['tag_id']:d['tag_name'] for d in list(csv.DictReader(open('./db/tag_info.csv', 'r', encoding='utf-8-sig')))}
    self.model = model
    self.data = data
    
  def get_name(self, id):
    return self.tag_info[id]
  
  def get_rec_tags(self, id:str, n=20):
    if id not in self.data.uid2idx:
      return []
    idx = self.data.uid2idx[id]
    tag_idxs = [int(x) for x in self.model.recommend(torch.tensor([idx]))[0]]
    return [self.get_name(self.data.iidx2id[i]) for i in tag_idxs[:n]]

def printt(arg0):
  print('*' * 20)
  print(arg0)
  print('*' * 20)

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"{'Seed:':20} {seed}")

def load_public(name: Literal['ml-1m', 'amazon-book', 'yelp2018', 'gowalla']) -> TagRecHelper:
    print(f"Loading public {name} dataset...")
    train, test = load_df_from_public_txt(name, 'train'), load_df_from_public_txt(name, 'test')
    uid2idx = {uid: i for i, uid in enumerate(train[USER_COL].unique())}
    iid2idx = {iid: i for i, iid in enumerate(train[ITEM_COL].unique())}
    gid2idx = {iid: i for i, iid in enumerate(train[USER_COL].unique())}
    for df in [train, test]:
        df.loc[:, USER_COL] = df[USER_COL].apply(lambda x: uid2idx.get(x, -1))
        df.loc[:, ITEM_COL] = df[ITEM_COL].apply(lambda x: iid2idx.get(x, -1))
        df.loc[:, GROUP_COL] = df[GROUP_COL].apply(lambda x: gid2idx.get(x, -1))
    return TagRecHelper(train, test, uid2idx, iid2idx, gid2idx)

def load_bili() -> TagRecHelper:
    print("Loading bilibili tag dataset...")
    data = []
    if os.path.exists('./data/bilibili/bilib-ds'):
        return pickle.load(open('./data/bilibili/bilib-ds', 'rb'))
    with open('./db/video_data.csv','r', encoding='utf-8-sig') as f:
        for line in tqdm(f):
            row = line.rstrip().split(',')
            mid, aid, tags = row[0], row[1], row[2:]
            data.append([mid, aid, tags])
    df = pd.DataFrame(data, columns=[USER_COL, GROUP_COL, ITEM_COL])
    df = df[df['user_id'].map(df['user_id'].value_counts()) > 1]
    temp = df.explode(ITEM_COL)
    uid2idx = {uid: i for i, uid in enumerate(temp[USER_COL].unique())}
    iid2idx = {iid: i for i, iid in enumerate(temp[ITEM_COL].unique())}
    gid2idx = {iid: i for i, iid in enumerate(temp[GROUP_COL].unique())}

    train_set, test_set, _, _ = train_test_split(df, df['user_id'], test_size=df['user_id'].nunique(), stratify=df['user_id'])
    train_set = train_set.explode(ITEM_COL)
    test_set = test_set.explode(ITEM_COL)
    for df in [train_set, test_set]:
        df.loc[:, USER_COL] = df[USER_COL].apply(lambda x: uid2idx.get(x, -1))
        df.loc[:, ITEM_COL] = df[ITEM_COL].apply(lambda x: iid2idx.get(x, -1))
        df.loc[:, GROUP_COL] = df[GROUP_COL].apply(lambda x: gid2idx.get(x, -1))


    ds = TagRecHelper(train_set, test_set, uid2idx, iid2idx, gid2idx)
    ds.mask = None
    pickle.dump(ds, open('./data/bilibili/bilib-ds', 'wb'))
    return ds
    
def load_ml_1m_df(type: Literal['train', 'test'] = 'train'):
    return load_df_from_public_txt('ml-1m', type)

def load_amazon_book_df(type: Literal['train', 'test'] = 'train'):
    return load_df_from_public_txt('amazon-book', type)

def load_bili_small(type: Literal['train', 'test'] = 'train'):
    if os.path.exists(f'./data/bilibili/mt8-{type}'):
        return pd.read_pickle(f'./data/bilibili/mt8-{type}')
    df = pd.read_csv('./data/bilibili/mt8.csv', names=[USER_COL, ITEM_COL, GROUP_COL])
    df[SCORE_COL] = 1
    train, test = python_stratified_split(df, ratio=[0.8, 0.2], filter_by='user', min_rating=10, col_user=USER_COL, col_item=ITEM_COL)
    uid2idx = {uid: i for i, uid in enumerate(train[USER_COL].unique())}
    iid2idx = {iid: i for i, iid in enumerate(train[ITEM_COL].unique())}
    gid2idx = {iid: i for i, iid in enumerate(train[GROUP_COL].unique())}
    # uidx2id = {i: uid for uid, i in uid2idx.items()}
    # iidx2id = {i: iid for iid, i in iid2idx.items()}
    # gidx2id = {i: gid for gid, i in gid2idx.items()}
    
    for df in [train, test]:
        df.loc[:, USER_COL] = df[USER_COL].apply(lambda x: uid2idx.get(x, -1))
        df.loc[:, ITEM_COL] = df[ITEM_COL].apply(lambda x: iid2idx.get(x, -1))
        df.loc[:, GROUP_COL] = df[GROUP_COL].apply(lambda x: gid2idx.get(x, -1))
        df.drop(df[(df[USER_COL] == -1) | (df[ITEM_COL] == -1) | (df[GROUP_COL] == -1)].index, inplace=True)
    
    train = train.reset_index()
    train.to_pickle('./data/bilibili/mt8-train')
    test = test.reset_index()
    test.to_pickle('./data/bilibili/mt8-test')
    return train if (type == 'train') else test

def load_df_from_public_txt(dataset: Literal['ml-1m', 'amazon-book', 'yelp2018', 'gowalla'],type: Literal['train', 'test'] = 'train'):
    if os.path.exists(f'./data/{dataset}/{type}.csv'):
        df = pd.read_csv(f'./data/{dataset}/{type}.csv')
    else:
        with open(f'./data/{dataset}/{type}.txt', 'r') as f:
            data = []
            for line in tqdm(f):
                arr = line.split(' ')
                if len(arr) == 2 and arr[1] == '\n':
                    continue
                user_id, items = int(arr[0]), [int(item)
                                               for item in arr[1:]]
                data.extend((user_id, item) for item in items)
            df = pd.DataFrame(data, columns=[USER_COL, ITEM_COL])
            df.to_csv(f'./data/{dataset}/{type}.csv', index=False)
    df[GROUP_COL] = df[USER_COL]
    return df
