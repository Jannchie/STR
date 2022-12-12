import torch
import numpy as np
from tqdm import tqdm
from utils import USER_COL, ITEM_COL, SCORE_COL

from typing import Literal


class HParams:
  ratio_neg_per_user = 0
  neg_num = 300
  neg_w = 300
  latent_factors = 128
  method: Literal['cos', 'dot'] = 'cos'
  loss_function: Literal['ccl', 'L1', 'mse', 'bce'] = 'bce'
  epochs = 30
  lr = 0.003
  batch_size = 1024
  weight_decay = 0.0001
  model = 'MF'
  w1 = 0.00001
  w2 = 1
  wii = 2


class BaseModel(torch.nn.Module):
  def __init__(self, train_df, test_df, users: int, items: int, hparams: HParams):
    super().__init__()
    self.name = 'baseMF'
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.test_loader = None
    self.train_df = train_df
    self.u_d = torch.tensor(self.train_df.groupby(
      USER_COL).agg({ITEM_COL: 'count'}).values).view(-1)
    self.i_d = torch.tensor(self.train_df.groupby(
      ITEM_COL).agg({USER_COL: 'count'}).values).view(-1)
    self.test_df = test_df
    self.users = users
    self.items = items
    self.neg_num = hparams.neg_num
    self.w1 = hparams.w1
    self.w2 = hparams.w2
    self.wii = hparams.wii
    self.neg_w = hparams.neg_w

    self.loss_func = hparams.loss_function
    self.latent_factors = hparams.latent_factors
    self.user_embeds = torch.nn.Embedding(
      users, hparams.latent_factors, device=self.device)
    self.item_embeds = torch.nn.Embedding(
      items, hparams.latent_factors, device=self.device)

    self.init_weights()
    self.all_items = torch.arange(self.items, device=self.device)
    self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    self.method = hparams.method
    self.loss_function = hparams.loss_function
    if self.loss_func == 'L1':
      self.criterion = torch.nn.L1Loss()
    elif self.loss_func == 'mse':
      self.criterion = torch.nn.MSELoss()
    elif self.loss_func == 'bce':
      self.criterion = torch.nn.BCEWithLogitsLoss()
    elif self.loss_func == 'ccl':
      def ccl(pos_y: torch.Tensor, neg_y: torch.Tensor):
        return torch.mean(1 - pos_y + 1 / neg_y.shape[0] * torch.sum(torch.max(torch.zeros(neg_y.shape, device='cuda'), neg_y - 0.8)))
      self.criterion = ccl
    else:
      raise ValueError('Loss function not supported')
    print('Model initialized')
    print(f'Users: {self.users}, Items: {self.items}, Latent factors: {self.latent_factors}, Method: {self.method}, loss func: {self.loss_func}')

  def init_weights(self):
    torch.nn.init.normal_(self.user_embeds.weight, std=1e-3)
    torch.nn.init.normal_(self.item_embeds.weight, std=1e-3)

  def forward(self, users, items):
    uv = self.user_vec(users)
    iv = self.item_vec(items)
    if (self.neg_num and self.training):
      neg_iv = self.item_vec(torch.randint(
          self.items, (items.shape[0] * self.neg_num,), device=self.device))
      neg_uv = uv.repeat(self.neg_num, 1)
      return self.affinity(uv, iv), self.affinity(neg_uv, neg_iv)
    return self.affinity(uv, iv)

  def affinity(self, user_vecs, item_vecs):
    if self.method != 'cos':
      return torch.sum(user_vecs * item_vecs, dim=1)
    return self.cosine_similarity(user_vecs, item_vecs)

  def user_vec(self, users):
    return self.user_embeds(users)

  def item_vec(self, items):
    return self.item_embeds(items)

  def get_loss(self, users, items, SCORE_COLs):
    y_pred, y_neg_pred = self.forward(users, items)
    SCORE_COLs_neg = torch.zeros(y_neg_pred.shape[0]).to(self.device)
    if self.loss_function == 'ccl':
      return self.criterion(y_pred, y_neg_pred)
    else:
      return self.criterion(y_pred, SCORE_COLs) + self.criterion(y_neg_pred, SCORE_COLs_neg) * self.neg_num


class UltraGCN(BaseModel):

  def __init__(self, train_df, test_df,
               users: int, items: int,
               hparams: HParams):
    super().__init__(train_df, test_df, users, items, hparams)
    self.name = 'UltraGCN'
    self.use_ii = False
    self.use_uu = True
    u, i, v = \
        torch.tensor(self.train_df[USER_COL].values), \
        torch.tensor(self.train_df[ITEM_COL].values), \
        torch.tensor(self.train_df[SCORE_COL].values if SCORE_COL in self.train_df else np.ones(len(self.train_df)))
    self.sp_mat = torch.sparse.LongTensor(torch.stack(
      (u, i)), v, torch.Size([self.users, self.items]))
    self.ui_constraint_mat = self.init_constraint_mat(self.sp_mat)
    self.ii_mat = torch.sparse.mm(
      self.sp_mat.t().float(), self.sp_mat.float()).to(self.device)
    self.ii_constraint_mat = self.init_constraint_mat(self.ii_mat)
    self.uu_mat = torch.sparse.mm(
      self.sp_mat.float(), self.sp_mat.t().float()).to(self.device)
    self.uu_constraint_mat = self.init_constraint_mat(self.uu_mat)

    # if self.use_ii:
    #     self.ii_mat = torch.sparse.mm(self.sp_mat.t().float(), self.sp_mat.float()).to(self.device)
    #     self.ii_constraint_mat = self.init_constraint_mat(self.ii_mat)
    #     self.init_ii('item')
    # elif self.use_uu:
    #     self.ii_mat = torch.sparse.mm(self.sp_mat.float(), self.sp_mat.t().float()).to(self.device)
    #     self.ii_constraint_mat = self.init_constraint_mat(self.ii_mat)
    #     self.init_ii('user')

    # item_neibers = defaultdict(list)
    # for user_id in tqdm(range(self.users), total=self.users):
    #     items = self.sp_mat[user_id].coalesce()
    #     neibors = items.indices()
    #     for item in neibors:
    #         item_neibers[]
    for user in range(users):
      self.sp_mat[user].coalesce().indices()

    if self.loss_func == 'L1':
      self.criterion = torch.nn.L1Loss(reduction='none')
    elif self.loss_func == 'mse':
      self.criterion = torch.nn.MSELoss(reduction='none')
    elif self.loss_func == 'bce':
      self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

  def init_ii(self, type, num_neighbors=10):
    self.ii_nei_idx = torch.zeros(
      (self.items, num_neighbors), dtype=torch.long, device=self.device)
    self.ii_nei_val = torch.zeros(
      (self.items, num_neighbors), dtype=torch.float, device=self.device)
    idx_length = self.items if type == 'item' else self.users
    for i in tqdm(range(idx_length), total=idx_length):
      di = self.ii_mat[i].to_dense()
      oi = torch.mul(
        self.ii_constraint_mat['beta_id'][i], self.ii_constraint_mat['beta_id'])
      row = oi * di
      row_vals, row_idxs = torch.topk(row, num_neighbors)
      self.ii_nei_idx[i] = row_idxs
      self.ii_nei_val[i] = row_vals

  def init_constraint_mat(self, sp_mat):
    di = torch.sparse.sum(sp_mat, dim=0).values()
    dj = torch.sparse.sum(sp_mat, dim=1).values()
    b_dj = (torch.sqrt(dj + 1) / dj)
    b_di = (1 / torch.sqrt(di + 1))
    return {"beta_ud": b_dj.reshape(-1).to(self.device), "beta_id": b_di.reshape(-1).to(self.device)}

  def get_omegas(self, users: torch.Tensor, items: torch.Tensor):
    if self.w2 > 0:
      weight = torch.mul(
        self.ui_constraint_mat['beta_ud'][users], self.ui_constraint_mat['beta_id'][items])
      w = self.w1 + self.w2 * weight
    else:
      weight = self.w1 * torch.ones(len(items)).to(self.device)
    return w

  def forward(self, users: torch.Tensor, items: torch.Tensor):
    uv = self.user_vec(users)
    iv = self.item_vec(items)
    if (self.neg_num and self.training):
      neg_items = torch.randint(
        self.items, (items.shape[0] * self.neg_num,), device=self.device)
      neg_users = users.repeat_interleave(self.neg_num)
      neg_uv = self.user_vec(neg_users)
      neg_iv = self.item_vec(neg_items)
      return self.affinity(uv, iv), self.affinity(neg_uv, neg_iv), neg_users, neg_items
    return self.affinity(uv, iv)

  def get_uu_loss(self, users):
    return torch.mul(self.uu_constraint_mat['beta_ud'][users], self.uu_constraint_mat['beta_id'][users])

  def get_ii_loss(self, items):
    return torch.mul(self.ii_constraint_mat['beta_ud'][items], self.ii_constraint_mat['beta_id'][items])
  # def a(self, item):
  #     neighbor_embeds = self.item_embeds(self.ii_nei_idx[pos_items])
  #     omegas = self.ii_nei_val[pos_items].to(self.device)
  #     user_embeds = self.user_embeds(users).unsqueeze(1)
  #     loss = -omegas * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log() # Calculate affinity for user and item's neighbors

  def get_loss(self, users, items, scores):
    y_pred, y_neg_pred, neg_users, neg_items = self.forward(users, items)
    neg_omegas = self.get_omegas(neg_users, neg_items)
    pos_omegas = self.get_omegas(users, items)
    scores_neg = torch.zeros(y_neg_pred.shape[0]).to(self.device)
    neg_loss = self.criterion(y_neg_pred, scores_neg).reshape(
      self.neg_num, -1) * neg_omegas.reshape(self.neg_num, -1)
    loss = (self.criterion(y_pred, scores) * pos_omegas +
            neg_loss.mean(dim=0) * self.neg_w).sum()
    loss += self.get_uu_loss(users)
    loss += self.get_ii_loss(items)
    return loss
