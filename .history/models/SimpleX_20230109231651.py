import torch
from torch.nn import functional as F

from utils import TagRecHelper
import numpy as np
import datetime
from tqdm import tqdm


class SimpleX(torch.nn.Module):
  def __init__(self, helper: TagRecHelper, config: dict):
    """ Initialize SimpleX model.

    Args:
        helper (TagRecHelper): TagRecHelper object.
        config (dict): configuration.
    """
    super(SimpleX, self).__init__()
    self.helper = helper
    self.config = config
    self.nuser = helper.nuser
    self.nitem = helper.nitem
    self.latent_dim = config.get('latent_dim', 64)
    self.item_emb = torch.nn.Embedding(self.nitem + 1, self.latent_dim, padding_idx=self.nitem, max_norm=self.latent_dim)
    self.user_emb = torch.nn.Embedding(self.nuser + 1, self.latent_dim, padding_idx=self.nuser, max_norm=self.latent_dim)
    torch.nn.init.xavier_normal_(self.user_emb.weight)
    torch.nn.init.xavier_normal_(self.item_emb.weight)
    torch.nn.init.zeros_(self.user_emb.weight[-1, :])
    torch.nn.init.zeros_(self.item_emb.weight[-1, :])
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = config.get('latent_dim', 64)
    attention_head = config.get('attention_head', 1)
    dropout = config.get('dropout', 0)
    n_interactive_items = config.get('n_interactive_items')
    
    # if 'aggregate' in config and config.get('w_ii', 1) != 1:
    user_top_index = helper.train_set.groupby('user_id')['item_id'].apply(lambda x: torch.tensor(x.value_counts().index[:n_interactive_items])).to_list()
    user_top_count = helper.train_set.groupby('user_id')['item_id'].apply(lambda x: torch.tensor(x.value_counts().values[:n_interactive_items])).to_list()
    self.user_top_len = torch.tensor([len(x) for x in user_top_index], device=self.device)
    # self.user_top_mask = torch.zeros(self.nuser, n_interactive_items, dtype=torch.bool, device=self.device)
    # for idx, x in enumerate(self.user_top_len):
    #   self.user_top_mask[idx, x:] = True
    
    # count > 2
    # user_top_index = [x[x > 2] for x in user_top_index]
    # user_top_count = [x[x > 2] for x in user_top_count]
    
    self.user_top_index = torch.nn.utils.rnn.pad_sequence(user_top_index, batch_first=True, padding_value=self.nitem).to(self.device)
    self.user_top_count = torch.nn.utils.rnn.pad_sequence(user_top_count, batch_first=True, padding_value=0).to(self.device)

    if config.get('aggregate', 'mean') == 'self-attention':
      self.attention = torch.nn.MultiheadAttention(latent_dim, attention_head, dropout=dropout, batch_first=True, bias=False).to(self.device)
    
    self.user_dropout = torch.nn.Dropout(dropout).to(self.device)
    item_count = self.helper.train_set.groupby('item_id')['user_id'].count().values
    item_count = item_count ** config.get('popular_alpha', 0)
    self.item_dist = torch.tensor(item_count / item_count.sum(), device=self.device)
    self.to(self.device)

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

  def bpr_loss(self, ue: torch.Tensor, ie: torch.Tensor, je: torch.Tensor) -> torch.Tensor:
    x_ui = torch.mul(ue, ie).sum(dim=1)
    x_uj = torch.mul(ue, je).sum(dim=1)
    x_uij = x_ui - x_uj
    log_prob = F.logsigmoid(x_uij)
    return -log_prob

  def loss_fn(self, ue: torch.Tensor, ie: torch.Tensor, neg_ie: torch.Tensor, margin=0.8, neg_w=0):
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

  def user_embedding(self, u: torch.Tensor) -> torch.Tensor:
    """Return user embedding.
    Args:
        u(torch.LongTensor): tensor stored user indexes. [batch_size,]
    Returns:
        pred(torch.FloatTensor): user embedding. [batch_size, latent_dim]
    """
    w = self.config.get('w_ii', 1)
    eu = self.user_emb(u)
    if 'aggregate' in self.config and w != 1:
      eui = self.user_item_embedding(u)
      eu = w * eu + (1 - w) * eui
    if self.config.get('affinity', 'dot') == 'cos':
      eu = F.normalize(eu)
    return self.user_dropout(eu)

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
    return e

  def item_embedding(self, i: torch.Tensor) -> torch.Tensor:
    """Return item embedding.
    Args:
        i(torch.LongTensor): tensor stored item indexes. [batch_size,]
    Returns:
        pred(torch.FloatTensor): item embedding. [batch_size, latent_dim]
    """
    e = self.item_emb(i)
    if self.config.get('affinity', 'dot') == 'cos':
      e = F.normalize(e)
    return e

  def get_loss(self, ue, ie) -> torch.Tensor:
    """Return loss.
    Args:
        ue(torch.LongTensor): tensor stored user indexes. [batch_size,]
        ie(torch.LongTensor): tensor stored item indexes. [batch_size,]
    Returns:
        pred(torch.FloatTensor): loss. [batch_size,]
    """
    loss = self.config.get('loss', 'ccl')
    if (loss == 'bpr'):
      je = self.item_embedding(torch.randint(0, self.nitem, (len(ie),), device=self.device))
      return self.bpr_loss(ue, ie, je)
    elif (loss == 'ccl'):
      # neg_idx = torch.randint(0, self.nitem, (ue.shape[0] * self.config.get('loss_neg_n', 0),), device=self.device)
      neg_idx = self.item_dist.multinomial(num_samples=(ue.shape[0] * self.config.get('loss_neg_n', 0)), replacement=True)
      neg_ie = self.item_embedding(neg_idx).view(-1, self.config.get('latent_dim', 64))
      return self.loss_fn(ue, ie, neg_ie, margin=self.config.get('loss_neg_k', 0), neg_w=self.config.get('loss_neg_w', 0))
    else:
      raise ValueError('loss function is not supported')

  def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """Return loss value.
    Args:
        u(torch.LongTensor): tensor stored user indexes. [batch_size,]
        i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
    Returns:
        torch.FloatTensor
    """
    ue = self.user_embedding(u)
    ie = self.item_embedding(i)
    return self.get_loss(ue, ie).mean()

  def recommend(self, u: torch.Tensor, k=20, mask=None) -> torch.Tensor:
    """ Return recommend items.

    Args:
        u (torch.Tensor): user index.
        k (int, optional): top K. Defaults to 20.
        mask (torch.Tensor, optional): Mask. Defaults to None.

    Returns:
        torch.Tensor: recommend items.
    """
    u = u.to(self.device)
    ue = self.user_embedding(u)
    x_ui = torch.mm(ue, self.item_embedding(torch.arange(self.nitem).to(self.device)).t())
    if mask:
      for i, m in enumerate(u):
        x_ui[i][mask[m]] = -1e4
    return torch.topk(x_ui, k=k, dim=1)
