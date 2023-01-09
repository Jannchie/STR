import torch
from torch.nn import functional as F
from models.SimpleX import SimpleX
from utils import CNT_COL, ITEM_COL, USER_COL, TagRecHelper
import numpy as np

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
    user_dropout = config.get('dropout', 0.8)
    item_dropout = config.get('item_dropout', 0.1)
    group_dropout = config.get('group_dropout', 0.1)
    self.group_dropout = torch.nn.Dropout(group_dropout).to(self.device)
    self.user_dropout = torch.nn.Dropout(user_dropout).to(self.device)
    item_count = self.helper.train_set.groupby('item_id')['user_id'].count().values
    item_count = item_count ** config.get('popular_alpha', 0)
    self.item_dist = torch.tensor(item_count / item_count.sum(), device=self.device)
    self.item_dropout = torch.nn.Dropout(item_dropout)
    self.to(self.device)

    user_top_index = helper.train_set.groupby('user_id')['item_id'].apply(lambda x: torch.tensor(x.value_counts().index[:self.n_interactive_items])).to_list()
    user_top_count = helper.train_set.groupby('user_id')['item_id'].apply(lambda x: torch.tensor(x.value_counts().values[:self.n_interactive_items])).to_list()
    self.user_top_len = torch.tensor([len(x) for x in user_top_index], device=self.device)
    self.user_top_index = torch.nn.utils.rnn.pad_sequence(user_top_index, batch_first=True, padding_value=self.nitem).to(self.device)
    self.user_top_count = torch.nn.utils.rnn.pad_sequence(user_top_count, batch_first=True, padding_value=0).to(self.device)

    self.w_g = config.get('w_g', 0)
    if helper.group_dict != None and self.w_g > 0:
      self.group_items =  [torch.tensor(helper.group_dict[gid][:10]) for gid in range(len(helper.group_dict))]
      self.group_items = torch.nn.utils.rnn.pad_sequence(self.group_items, batch_first=True, padding_value=self.nitem).to(self.device)

    print('init model parameters')
    torch.nn.init.normal_(self.user_emb.weight, std=1e-4)
    torch.nn.init.normal_(self.item_emb.weight, std=1e-4)
    # torch.nn.init.xavier_normal_(self.user_emb.weight)
    # torch.nn.init.xavier_normal_(self.item_emb.weight)
    torch.nn.init.zeros_(self.user_emb.weight[-1, :])
    torch.nn.init.zeros_(self.item_emb.weight[-1, :])
    # if 'aggregate' in config and config.get('w_ii', 1) != 1:
    u, i, v = \
        torch.tensor(helper.train_set[USER_COL].values), \
        torch.tensor(helper.train_set[ITEM_COL].values), \
        torch.tensor(helper.train_set[CNT_COL].values if CNT_COL in helper.train_set else np.ones(len(helper.train_set)))
   
    self.ui_sp_mat = torch.sparse.LongTensor(torch.stack((u, i)), v, torch.Size([self.nuser, self.nitem])).to(self.device)
    
    if config.get('w_ii', 0) > 0:
      ii_mat = torch.sparse.mm(self.ui_sp_mat.t().float(), self.ui_sp_mat.float()).to_dense() 
      ii_mat = ii_mat * (1 - torch.eye(self.nitem))
      self.item_top_nei = ii_mat.topk(self.n_interactive_items, dim=1)
      self.item_top_nei = self.item_top_nei.indices.to(self.device), self.item_top_nei.values.to(self.device)
      del ii_mat
      torch.cuda.empty_cache()
      
    if config.get('w_uu', 0)  > 0:
      uu_mat = torch.sparse.mm(self.ui_sp_mat.float(), self.ui_sp_mat.t().float())
      self.user_top_nei = uu_mat.to_dense().topk(self.n_interactive_items, dim=1)
      self.user_top_nei = self.user_top_nei.indices.to(self.device), self.user_top_nei.values.to(self.device)
      del uu_mat
      torch.cuda.empty_cache()
    
    if config.get('w_h', 0) > 0:
      d = self.ui_sp_mat.to_dense()
      self.user_nei_item = d.topk(self.n_interactive_items, dim=1)
    if config.get('aggregate', 'mean') == 'self-attention':
      self.attention = torch.nn.MultiheadAttention(self.dim, self.attention_head, dropout=item_dropout, batch_first=True, bias=False).to(self.device)
    print('Finish init model')

  
  def get_ccl_loss(self, ue, ie, neg_ie) -> torch.Tensor:
    """Return loss.
    Args:
        ue(torch.LongTensor): tensor stored user indexes. [batch_size,]
        ie(torch.LongTensor): tensor stored item indexes. [batch_size,]
    Returns:
        pred(torch.FloatTensor): loss. [batch_size,]
    """
    return self.ccl_loss(ue, ie, neg_ie, margin=self.config.get('loss_neg_m', 0), neg_w=self.config.get('loss_neg_w', 0))
  
  def ccl_loss(self, ue, ie, neg_ie, margin=0, neg_w=0):
    """ Return cosine contractive loss.

    Args:
        u (torch.Tensor): user embedding. [batch_size, latent_dim]
        i (torch.Tensor): item embedding. [batch_size, latent_dim]
        margin (float, optional): margin. Defaults to 0.8.
        neg_w (int, optional): negative item weight. Defaults to 0.

    Returns:
        torch.Tensor: mean loss.
    """

    pos_pred = self.affinity(ue, ie)
    pos_loss = torch.relu(1 - pos_pred)
    neg_ie_num = neg_ie.shape[0]
    neg_ue = ue.repeat(1, neg_ie_num // ue.shape[0]).view(-1, self.config.get('latent_dim', 64))
    neg_pred = self.affinity(neg_ue, neg_ie).view(-1, neg_ie_num)
    neg_loss = torch.relu(neg_pred - margin)
    if neg_w != 0:
      return (pos_loss + neg_loss.mean(dim=-1) * neg_w)
    return (pos_loss + neg_loss.sum(dim=-1))
    
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
    return torch.bmm(ue.unsqueeze(1), ie.transpose(1, 2)).squeeze().sum()
  
  def get_neg_loss(self, ue: torch.Tensor, neg_ie: torch.Tensor):
    neg_ie_num = neg_ie.shape[0]
    neg_ue = ue.repeat(1, neg_ie_num // ue.shape[0]).view(-1, self.dim)
    neg_pred = self.affinity(neg_ue, neg_ie).view(-1, neg_ie_num)
    neg_loss = (F.relu(neg_pred - self.loss_neg_m)).mean(dim=1)
    return neg_loss * self.loss_neg_w
  
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
  
  def user_nei_embedding(self, u: torch.Tensor) -> torch.Tensor:
    return self.nei_embedding(u, self.user_top_nei)
  
  def item_nei_embedding(self, i: torch.Tensor) -> torch.Tensor:
    return self.nei_embedding(i, self.item_top_nei)
  
  def nei_embedding(self, idx, nei):
    e = self.item_emb(nei[0][idx]) # [batch_size, top_k, latent_dim]
    if self.config.get('affinity', 'dot') == 'cos':
      e = F.normalize(e, dim=2)
    method = self.config.get('aggregate', 'mean')
    if method == 'mean':
      e = e.mean(dim=1)
    elif method == 'self-attention':
      e = self.attention(e, e, e, need_weights=False)[0]
      e = e.mean(dim=1)
    elif method == 'weighted-sum':
      mask = e.sum(dim=-1) != 0
      weights = nei[1][idx] ** self.config.get('aggregate_a', 0.75) * mask.float() 
      weights = weights / weights.sum(dim=1, keepdim=True)
      e = (e * weights[..., None]).sum(dim=1)
    e = self.item_dropout(e)
    return e
  
  def user_history_embedding(self, u: torch.Tensor) -> torch.Tensor:
    ie =  self.item_emb(self.user_nei_item.indices[u]) # [batch_size, top_k, latent_dim]
    # degs = self.item_deg[self.user_top_index[u]] # [batch_size, top_k]
    if self.config.get('affinity', 'dot') == 'cos':
      ie = F.normalize(ie)
    ie = self.item_dropout(ie)
    method = self.config.get('aggregate', 'mean')
    if method == 'mean':
      mask = ie.sum(dim=-1) != 0
      e = ie.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-12) 
    elif method == 'self-attention':
      e = self.attention(ie, ie, ie, need_weights=False)[0]
      e = e.mean(dim=1)
    elif method == 'weighted-sum':
      alpha = self.config.get('aggregate_a', 0.75)
      mask = ie.sum(dim=-1) != 0
      weights = self.user_nei_item.values[u] ** alpha * mask.float() 
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
    pred = self.affinity(ue, ie)
    return F.relu(1 - pred)
  
  def user_item_embedding(self, u: torch.Tensor) -> torch.Tensor:
    """Return user embedding.
    Args:
        u (torch.Tensor): user index. [batch_size]

    Returns:
        torch.Tensor: user embedding. [batch_size, latent_dim]
    """
    ie = self.item_embedding(self.user_top_index[u])
    aggregate = self.config.get('aggregate', 'mean')
    if (aggregate == 'mean'):
      mask = ie.sum(dim=-1) != 0
      e = ie.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-12)
    elif (aggregate == 'self-attention'):
      e = self.attention(ie, ie, ie, need_weights=False)[0].mean(dim=1)
    elif aggregate == 'weighted-sum':
      alpha = self.config.get('aggregate_a', 0.75)
      mask = ie.sum(dim=-1) != 0
      weights = self.user_top_count[u] ** alpha * mask.float() 
      weights = weights / weights.sum(dim=1, keepdim=True)
      e = (ie * weights[..., None]).sum(dim=1)   
    return e

  def forward(self, user: torch.Tensor, item: torch.Tensor, group: torch.Tensor = None):
      """ Return prediction loss.
      Args:
          user (torch.Tensor): user index. [batch_size]
          item (torch.Tensor): item index. [batch_size]
      Returns:
          loss(torch.Tensor): prediction loss. [batch_size] 
      """
      
      neg_idx = self.item_dist.multinomial(num_samples=(user.shape[0] * self.config.get('loss_neg_n', 0)), replacement=True)
      neg_ie = self.item_embedding(neg_idx).view(-1, self.config.get('latent_dim', 64))
      
      ie = self.item_embedding(item)
      w = self.config.get('w_cf', 1)
      loss_uie = 0
      loss_cf = 0
      if w != 0:
        ue = self.user_embedding(user)
        loss_cf = self.get_ccl_loss(ue, ie, neg_ie)
      if w != 1:
        uie = self.user_item_embedding(user)
        loss_uie = self.get_ccl_loss(uie, ie, neg_ie)
      loss_g = 0
      if self.w_g > 0 and group is not None:
        group_items_embeddings = self.item_dropout(self.item_embedding(self.group_items[group]))
        loss_g = self.get_ccl_loss(ue, group_items_embeddings.mean(dim=1), neg_ie).mean()
      return ((w * loss_cf + (1 - w) * loss_uie).mean() * (1 - self.w_g)) + loss_g * self.w_g
  
  # def forward(self, user: torch.Tensor, item: torch.Tensor, group: torch.Tensor = None):
  #   """ Return prediction loss.
  #   Args:
  #       user (torch.Tensor): user index. [batch_size]
  #       item (torch.Tensor): item index. [batch_size]

  #   Returns:
  #       loss(torch.Tensor): prediction loss. [batch_size] 
  #   """
  #   neg_idx = self.item_dist.multinomial(num_samples=(user.shape[0] * self.config.get('loss_neg_n', 0)), replacement=True)
  #   neg_ie = self.item_embedding(neg_idx).view(-1, self.dim)
    
  #   ie = self.item_embedding(item)
  #   w_cf = self.config.get('w_cf', 1)
  #   w_uu = self.config.get('w_uu', 0)
  #   w_ii = self.config.get('w_ii', 0)
  #   w_gi = self.config.get('w_gi', 0)
  #   w_h = self.config.get('w_h', 0)
    
  #   ue = self.user_embedding(user)
  #   loss_cf = self.get_loss(ue, ie)
    
  #   loss_g = 0
  #   if self.w_g > 0 and group is not None:
  #     group_items = self.group_items[group]
  #     group_items_embeddings = self.item_dropout(self.item_embedding(group_items))
  #     loss_g = self.get_loss(ue, group_items_embeddings.mean(dim=1)).mean()
    
  #   loss_h = self.get_loss(self.user_history_embedding(user), ie) if w_h > 0 else torch.zeros_like(loss_cf)
  #   neg_loss = self.get_neg_loss(ue, neg_ie)
    
  #   return ((w_cf * loss_cf  + w_h * loss_h).mean()) * (1 - self.w_g) + self.w_g * loss_g + neg_loss

  def rec_by_user_history(self, u, k=20, mask=None):
    uie = self.user_history_embedding(u)
    return self.rec(uie, mask, u, k)
  
  def rec_by_ui(self, u, k=20, mask=None):
    ue = self.user_embedding(u)
    return self.rec(ue, mask, u, k)

  def rec(self, vec, mask, u, k):
    items = self.item_embedding(torch.arange(self.nitem, device=self.device)).T
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
    
    x = 0
    w = self.config.get('cf_w', 1)
    true_w = w
    u = u.to(self.device)
    if w != 0:
      ue = self.user_embedding(u)
      x += torch.mm(ue, self.item_embedding(torch.arange(self.nitem, device=self.device)).T) * true_w
    if w != 1:
      uie = self.user_history_embedding(u)
      x_uui = torch.mm(uie, self.item_embedding(torch.arange(self.nitem, device=self.device)).T) * (1 - true_w)
      x += x_uui
    if mask:
      for i, m in enumerate(u):
        x[i][mask[m]] = -1e4
    return torch.topk(x, k=k, dim=1)
    # return self.rec_by_ui(u.to(self.device), k, mask)