import torch
from torch.nn import functional as F
from models.SimpleX import SimpleX
from utils import CNT_COL, ITEM_COL, USER_COL, TagRecHelper
import numpy as np
def get_reg(reg):
    reg_pair = [] # of tuples (p_norm, weight)
    if isinstance(reg, float):
        reg_pair.append((2, reg))
    elif isinstance(reg, str):
        try:
            if reg.startswith("l1(") or reg.startswith("l2("):
                reg_pair.append((int(reg[1]), float(reg.rstrip(")").split("(")[-1])))
            elif reg.startswith("l1_l2"):
                l1_reg, l2_reg = reg.rstrip(")").split("(")[-1].split(",")
                reg_pair.extend(((1, float(l1_reg)), (2, float(l2_reg))))
            else:
                raise NotImplementedError
        except:
            raise NotImplementedError(f"regularizer={reg} is not supported.")
    return reg_pair

class STR(torch.nn.Module):
  
  def load_params(self, path):
    self.load_state_dict(torch.load(path))
    
  def __init__(self, helper: TagRecHelper, config: dict):
    super().__init__()
    self.helper = helper
    self.config = config
    self.nuser = helper.nuser
    self.nitem = helper.nitem
    self.latent_dim = config.get('latent_dim', 64)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.item_emb = torch.nn.Embedding(self.nitem + 1, self.latent_dim, padding_idx=self.nitem, max_norm=self.latent_dim)
    self.user_emb = torch.nn.Embedding(self.nuser + 1, self.latent_dim, padding_idx=self.nuser, max_norm=self.latent_dim)
    self.dim = config.get('latent_dim', 64)
    self.n_interactive_items = config.get('n_interactive_items', 10)
    self.loss_neg_n = config.get('loss_neg_n', 2000)
    self.loss_neg_a = config.get('loss_neg_a', 1)
    self.loss_neg_w = config.get('loss_neg_w', 50)
    self.loss_neg_m = config.get('loss_neg_m', 0.4)
    self.weight_decay = config.get('weight_decay', 1e-6)
    self.attention_head = config.get('attention_head', 1)


    user_dropout = config.get('dropout', 0.5)
    item_dropout = config.get('item_dropout', 0.8)
    
    self.user_dropout = torch.nn.Dropout(user_dropout).to(self.device)
    item_count = self.helper.train_set.groupby('item_id')['user_id'].count().values
    item_count = item_count ** config.get('popular_alpha', 0)
    self.item_dist = torch.tensor(item_count / item_count.sum(), device=self.device)
    self.item_dropout = torch.nn.Dropout(item_dropout)
    self.to(self.device)

    self.group_loss_gamma = config.get('group_loss_gamma', 0)
    if helper.group_dict != None and self.group_loss_gamma > 0:
      self.group_items =  [torch.tensor(helper.group_dict[gid][:10]) for gid in range(len(helper.group_dict))]
      self.group_items = torch.nn.utils.rnn.pad_sequence(self.group_items, batch_first=True, padding_value=self.nitem).to(self.device)
      # self.group_norm = torch.nn.BatchNorm1d(self.group_items.shape[1], device=self.device)
    print('init model parameters')
    # self.batch_norm = torch.nn.BatchNorm1d(self.dim, affine=False).to(self.device)
    torch.nn.init.normal_(self.user_emb.weight, std=1e-4)
    torch.nn.init.normal_(self.item_emb.weight, std=1e-4)
    torch.nn.init.zeros_(self.user_emb.weight[-1, :])
    torch.nn.init.zeros_(self.item_emb.weight[-1, :])
    # if 'aggregate' in config and config.get('cf_w', 1) != 1:
    
    print('calculate item degree')
    u, i, v = \
        torch.tensor(helper.train_set[USER_COL].values), \
        torch.tensor(helper.train_set[ITEM_COL].values), \
        torch.tensor(helper.train_set[CNT_COL].values if CNT_COL in helper.train_set else np.ones(len(helper.train_set)))
    self.sp_mat = torch.sparse.LongTensor(torch.stack((u, i)), v, torch.Size([self.nuser, self.nitem]))
    self.item_deg = torch.sparse.sum(self.sp_mat, dim=0).values().to(self.device)
    self.item_deg = torch.cat((self.item_deg, torch.tensor([0], device=self.device)))
    
    train_deg = self.item_deg[helper.train_set[ITEM_COL].values].cpu()
    helper.train_set['deg'] = train_deg
    helper.train_set = helper.train_set.sort_values(['user_id', 'deg'], ascending=[True, True])
    print('preparing top items')
    user_top_index = helper.train_set.groupby('user_id')['item_id'].apply(lambda x: torch.tensor(x.values[:self.n_interactive_items])).to_list()
    user_top_count = helper.train_set.groupby('user_id')['deg'].apply(lambda x: torch.tensor(x.values[:self.n_interactive_items])).to_list()
    self.user_top_index = torch.nn.utils.rnn.pad_sequence(user_top_index, batch_first=True, padding_value=self.nitem).to(self.device)
    self.user_top_count = torch.nn.utils.rnn.pad_sequence(user_top_count, batch_first=True, padding_value=0).to(self.device)
    self.user_top_len = torch.tensor([len(x) for x in user_top_index], device=self.device)
    # self.user_top_mask = torch.zeros(self.nuser, n_interactive_items, dtype=torch.bool, device=self.device)
    # for idx, x in enumerate(self.user_top_len):
    #   self.user_top_mask[idx, x:] = True
    if config.get('aggregate', 'mean') == 'self-attention':
      self.attention = torch.nn.MultiheadAttention(self.dim, self.attention_head, dropout=item_dropout, batch_first=True, bias=False).to(self.device)
      

    # append padding
    # self.cmat = self.init_constraint_mat(self.sp_mat)

  def init_constraint_mat(self, sp_mat):
    di = torch.sparse.sum(sp_mat, dim=0).values()
    dj = torch.sparse.sum(sp_mat, dim=1).values()
    b_di = (1 / torch.sqrt(di + 1))
    b_dj = (torch.sqrt(dj + 1) / dj)
    return {"beta_ud": b_dj.reshape(-1).to(self.device), "beta_id": b_di.reshape(-1).to(self.device)}
  
  # def loss_fn(self, ue: torch.Tensor, ie: torch.Tensor, neg_ie: torch.Tensor):
  #   """ Return cosine contractive loss.

  #   Args:
  #       u (torch.Tensor): user embedding. [batch_size, latent_dim]
  #       i (torch.Tensor): item embedding. [batch_size, latent_dim]
  #       margin (float, optional): margin. Defaults to 0.8.
  #       neg_w (int, optional): negative item weight. Defaults to 0.

  #   Returns:
  #       torch.Tensor: mean loss.
  #   """
  #   pos_pred = self.affinity(ue, ie)
  #   pos_loss = F.relu(1 - pos_pred)
    
  #   if self.loss_neg_w == 0:
  #     return pos_loss
  #   m = self.config.get('loss_neg_m', 0)
  #   neg_ie_num = neg_ie.shape[0]
  #   neg_ue = ue.repeat(1, neg_ie_num // ue.shape[0]).view(-1, self.dim)
  #   neg_pred = self.affinity(neg_ue, neg_ie).view(-1, neg_ie_num)
  #   # neg_pred = neg_pred.view(ie.shape[0], -1).topk(self.loss_neg_k)[0]
  #   neg_loss = F.relu(neg_pred - m).mean(dim=1)
  #   # return (pos_loss + neg_loss * self.loss_neg_w)
  #   neg_not_zero_count = (neg_loss != 0).sum()
  #   return (pos_loss + neg_loss.sum(dim=-1) / (neg_not_zero_count + 1.e-12) * self.loss_neg_w)
  
  def affinity(self, ue: torch.Tensor, ie: torch.Tensor) -> torch.Tensor:
    """Return affinity between user and item.
    Args:
        u(torch.FloatTensor): user embedding. [batch_size, latent_dim]
        i(torch.FloatTensor): item embedding. [batch_size, n, latent_dim] or [batch_size, latent_dim]
    Returns:
        pred(torch.FloatTensor): affinity between user and item. [batch_size,]
    """
    if len(ie.shape) == 2:
      return torch.mul(ue, ie).sum(dim=1)
    return torch.bmm(ue.unsqueeze(1), ie.transpose(1, 2)).squeeze().sum(dim=1)
  
  def loss_fn(self, ue: torch.Tensor, ie: torch.Tensor):
    """ Return cosine contractive loss.

    Args:
        u (torch.Tensor): user embedding. [batch_size, latent_dim]
        i (torch.Tensor): item embedding. [batch_size, latent_dim]

    Returns:
        torch.Tensor: mean loss.
    """
    pos_pred = self.affinity(ue, ie)
    return F.relu(1 - pos_pred) ** self.loss_neg_a
  
  def get_neg_loss(self, ue: torch.Tensor, neg_ie: torch.Tensor):
    neg_ie_num = neg_ie.shape[0]
    neg_ue = ue.repeat(1, neg_ie_num // ue.shape[0]).view(-1, self.dim)
    neg_pred = self.affinity(neg_ue, neg_ie).view(-1, neg_ie_num)
    neg_loss = (F.relu(neg_pred - self.loss_neg_m) ** self.loss_neg_a).mean(dim=1)
    return neg_loss * self.loss_neg_w
    # neg_not_zero_count = (neg_loss != 0).sum()
    # return neg_loss.sum(dim=-1) / (neg_not_zero_count + 1.e-12) * self.loss_neg_w 
    
  # def loss_fn(self, ue: torch.Tensor, ie: torch.Tensor, neg_ie: torch.Tensor):
  #   """ Return cosine contractive loss.

  #   Args:
  #       u (torch.Tensor): user embedding. [batch_size, latent_dim]
  #       i (torch.Tensor): item embedding. [batch_size, latent_dim]
  #       margin (float, optional): margin. Defaults to 0.8.
  #       neg_w (int, optional): negative item weight. Defaults to 0.

  #   Returns:
  #       torch.Tensor: mean loss.
  #   """

  #   pos_pred = self.affinity(ue, ie)
  #   pos_loss = torch.relu(1 - pos_pred)
  #   neg_ie_num = neg_ie.shape[0]
  #   neg_ue = ue.repeat(1, neg_ie_num // ue.shape[0]).view(-1, self.dim)
  #   neg_pred = self.affinity(neg_ue, neg_ie).view(-1, neg_ie_num)
  #   neg_loss = torch.relu(neg_pred - self.loss_neg_m)
  #   if self.loss_neg_w != 0:
  #     return (pos_loss + neg_loss.mean(dim=-1) * self.loss_neg_w)
  #   return (pos_loss + neg_loss.sum(dim=-1))
  
  def user_embedding(self, u: torch.Tensor) -> torch.Tensor:
    """Return user embedding.
    Args:
        u(torch.LongTensor): tensor stored user indexes. [batch_size,]
    Returns:
        pred(torch.FloatTensor): user embedding. [batch_size, latent_dim]
    """
    eu = self.user_emb(u)
    # eu = self.batch_norm(eu)
    if self.config.get('affinity', 'dot') == 'cos':
      eu = F.normalize(eu)
    return self.user_dropout(eu)
  
  def item_embedding(self, i: torch.Tensor) -> torch.Tensor:
    """Return item embedding.
    Args:
        i(torch.LongTensor): tensor stored item indexes. [batch_size,]
    Returns:
        pred(torch.FloatTensor): item embedding. [batch_size, latent_dim]
    """
    e = self.item_emb(i)
    # e = self.batch_norm(e)
    if self.config.get('affinity', 'dot') == 'cos':
      e = F.normalize(e)
    return e
  def user_item_embedding(self, u: torch.Tensor) -> torch.Tensor:
    """Return user embedding.
    Args:
        u (torch.Tensor): user index. [batch_size]

    Returns:
        torch.Tensor: user embedding. [batch_size, latent_dim]
    """
    ie =  self.item_emb(self.user_top_index[u]) # [batch_size, top_k, latent_dim]
    # degs = self.item_deg[self.user_top_index[u]] # [batch_size, top_k]
    if self.config.get('affinity', 'dot') == 'cos':
      ie = F.normalize(ie)
    # ie = self.interact_norm(ie)
    ie = self.item_dropout(ie)
    aggregate = self.config.get('aggregate', 'mean')
    if aggregate == 'mean':
      # e = ie.sum(dim=1) / self.user_top_len[u].view(-1, 1)
      mask = ie.sum(dim=-1) != 0
      e = ie.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-12) 
    elif aggregate == 'self-attention':
      e = self.attention(ie, ie, ie, need_weights=False)[0]
      e = e.mean(dim=1)
    elif aggregate == 'weighted-sum':
      mask = ie.sum(dim=-1) != 0
      weights = self.user_top_count[u] ** self.config.get('weighted-sum-alpha', 0.75) * mask.float() 
      weights = weights / weights.sum(dim=1, keepdim=True)
      e = (ie * weights[..., None]).sum(dim=1)
    return e
  
  def get_loss(self, ue, ie) -> torch.Tensor:
    """Return loss.
    Args:
        ue(torch.LongTensor): tensor stored user indexes. [batch_size,]
        ie(torch.LongTensor): tensor stored item indexes. [batch_size,]
    Returns:
        pred(torch.FloatTensor): loss. [batch_size,]
    """
    return self.loss_fn(ue, ie) 
  
  def reg_loss(self):
    l = 0
    emb_reg = get_reg(self.weight_decay)
    for params in self.parameters():
      for emb_p, emb_lambda in emb_reg:
          l += (emb_lambda / emb_p) * torch.norm(params, emb_p) ** emb_p
    return l
  
  def forward(self, user: torch.Tensor, item: torch.Tensor, group: torch.Tensor = None):
    """ Return prediction loss.

    Args:
        user (torch.Tensor): user index. [batch_size]
        item (torch.Tensor): item index. [batch_size]

    Returns:
        loss(torch.Tensor): prediction loss. [batch_size] 
    """
    neg_idx = self.item_dist.multinomial(num_samples=(user.shape[0] * self.config.get('loss_neg_n', 0)), replacement=True)
    neg_ie = self.item_embedding(neg_idx).view(-1, self.dim)

    ie = self.item_embedding(item)
    w = self.config.get('cf_w', 1)
    loss_ii = 0
    loss_ui = 0
    if w != 0:
      ue = self.user_embedding(user)
      loss_ui = self.get_loss(ue, ie)
      # loss_ui = self.get_loss(ue, ie, neg_ie)
    if w != 1:
      loss_ii = self.get_loss(ue, self.item_emb(self.user_top_index[user]))
      # uie = self.user_item_embedding(user)
      # loss_ii = self.get_loss(ue, uie, neg_ie)
    if self.config.get('auto_gamma', False):
      gamma = torch.sigmoid(self.user_gamma(user))
      return (gamma * loss_ui + (1 - gamma) * loss_ii).mean()
    
    loss_g = 0
    if self.group_loss_gamma > 0 and group is not None:
      group_items = self.group_items[group]
      # group_items_embeddings = self.group_norm(self.item_embedding(group_items))
      # group_items_embeddings = self.item_dropout(group_items_embeddings)
      group_items_embeddings = self.item_dropout(self.item_embedding(group_items))
      loss_g = self.get_loss(ue, group_items_embeddings.mean(dim=1), neg_ie).mean()
      
    
    # beta = torch.mul(self.cmat['beta_ud'][user], self.cmat['beta_id'][item])
    
    # neg_i_num = neg_ie.shape[0]
    # neg_u = user.repeat_interleave(neg_i_num // user.shape[0])
    # beta_neg = torch.mul(self.cmat['beta_ud'][neg_u], self.cmat['beta_id'][neg_idx])
    return ((w * loss_ui + (1 - w) * loss_ii).mean() * (1 - self.group_loss_gamma)) + loss_g * self.group_loss_gamma + self.get_neg_loss(ue, neg_ie)

  def rec_by_ii(self, u, k=20, mask=None):
    uie = self.user_item_embedding(u)
    return self.rec(uie, mask, u, k)
  
  def rec_by_ui(self, u, k=20, mask=None):
    ue = self.user_embedding(u)
    return self.rec(ue, mask, u, k)

  def rec(self, vec, mask, u, k):
    items = self.item_embedding(torch.arange(self.nitem, device=self.device)).T
    if self.config.get('affinity', 'dot') == 'cos':
      vec = F.normalize(vec, dim=-1)
      items = F.normalize(items, dim=0)
    iix = torch.mm(vec, items)
    if mask:
      for i, m in enumerate(u):
        iix[i][mask[m]] = -10000.0
    return iix.topk(k, dim=-1)
  
  def recommend(self, u: torch.Tensor, k=20, mask=None) -> torch.Tensor:
    """ Return recommend items.

    Args:
        u (torch.Tensor): user index.
        k (int, optional): top K. Defaults to 20.
        mask (torch.Tensor, optional): Mask. Defaults to None.

    Returns:
        torch.Tensor: recommend items.
    """
    return self.rec_by_ui(u.to(self.device), k, mask)
    # x = 0
    # w = self.config.get('cf_w', 1)
    
    # u = u.to(self.device)
    # if w != 0:
    #   ue = self.user_embedding(u)
    #   x += torch.mm(ue, self.item_embedding(torch.arange(self.nitem, device=self.device)).T) * w
    # if w != 1:
    #   uie = self.user_item_embedding(u)
    #   x_uui = torch.mm(uie, self.item_embedding(torch.arange(self.nitem, device=self.device)).T) * (1 - w)
    #   x += x_uui
    # if mask:
    #   for i, m in enumerate(u):
    #     x[i][mask[m]] = -1e4
    # return torch.topk(x, k=k, dim=1)