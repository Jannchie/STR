import torch
from torch.nn import functional as F
from models.SimpleX import SimpleX
from utils import TagRecHelper


class STR(SimpleX):

  def __init__(self, helper: TagRecHelper, config: dict):
    super().__init__(helper, config)
    self.item_dropout = torch.nn.Dropout(config.get('item_dropout', 0.8))
    if config.get('auto_gamma', False):
      self.user_gamma =  torch.nn.Embedding(self.nuser, 1, device=self.device) 
      torch.nn.init.xavier_normal_(self.user_gamma.weight, gain=100)
    self.group_loss_gamma = config.get('group_loss_gamma', 0)
    if self.group_loss_gamma > 0:
      self.group_items =  [torch.tensor(helper.group_dict[gid][:10]) for gid in range(len(helper.group_dict))]
      self.group_items = torch.nn.utils.rnn.pad_sequence(self.group_items, batch_first=True, padding_value=self.nitem).to(self.device)
      self.group_loss = torch.nn.BCEWithLogitsLoss()
    
  def user_embedding(self, u: torch.Tensor) -> torch.Tensor:
    """Return user embedding.
    Args:
        u(torch.LongTensor): tensor stored user indexes. [batch_size,]
    Returns:
        pred(torch.FloatTensor): user embedding. [batch_size, latent_dim]
    """
    eu = self.user_emb(u)
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
    ie =  self.item_emb(self.user_top_index[u])
    if self.config.get('affinity', 'dot') == 'cos':
      ie = F.normalize(ie)
    ie = self.item_dropout(ie)
    aggregate = self.config.get('aggregate', 'mean')
    if aggregate == 'mean':
      # e = ie.sum(dim=1) / self.user_top_len[u].view(-1, 1)
      e = ie.mean(dim=1)
    elif aggregate == 'self-attention':
      e = self.attention(ie, ie, ie, need_weights=False)[0]
      e = e.mean(dim=1)
    elif aggregate == 'weighted-sum':
      weights = self.user_top_count[u] ** self.config.get('weighted-sum-alpha', 0.75)
      weights = weights / weights.sum(dim=1, keepdim=True) 
      e = (ie * weights[..., None]).sum(dim=1)
    return e
  
  def get_loss(self, ue, ie, neg_ie) -> torch.Tensor:
    """Return loss.
    Args:
        ue(torch.LongTensor): tensor stored user indexes. [batch_size,]
        ie(torch.LongTensor): tensor stored item indexes. [batch_size,]
    Returns:
        pred(torch.FloatTensor): loss. [batch_size,]
    """
    return self.ccl_loss(ue, ie, neg_ie, margin=self.config.get('ccl_neg_margin', 0), neg_w=self.config.get('ccl_neg_weight', 0))

  def forward(self, user: torch.Tensor, item: torch.Tensor, group: torch.Tensor = None):
    """ Return prediction loss.

    Args:
        user (torch.Tensor): user index. [batch_size]
        item (torch.Tensor): item index. [batch_size]

    Returns:
        loss(torch.Tensor): prediction loss. [batch_size] 
    """
    
    neg_idx = self.item_dist.multinomial(num_samples=(user.shape[0] * self.config.get('ccl_neg_num', 0)), replacement=True)
    neg_ie = self.item_embedding(neg_idx).view(-1, self.config.get('latent_dim', 64))
    
    ie = self.item_embedding(item)
    w = self.config.get('aggregate_w', 1)
    loss_ii = 0
    loss_ui = 0
    if w != 0:
      ue = self.user_embedding(user)
      loss_ui = self.get_loss(ue, ie, neg_ie)
    if w != 1:
      uie = self.user_item_embedding(user)
      loss_ii = self.get_loss(uie, ie, neg_ie)
    if self.config.get('auto_gamma', False):
      gamma = torch.sigmoid(self.user_gamma(user))
      return (gamma * loss_ui + (1 - gamma) * loss_ii).mean()
    
    loss_g = 0
    if self.group_loss_gamma > 0 and group is not None:
      group_items = self.group_items[group]
      group_items_embeddings = self.item_dropout(self.item_embedding(group_items))
      # group_sim = torch.bmm(group_items_embeddings, group_items_embeddings.transpose(1, 2))
      loss_g = self.get_loss(ue, group_items_embeddings.mean(dim=1), neg_ie).mean()
      # group_loss = self.group_loss(-group_sim, torch.zeros_like(group_sim)) 

    return ((w * loss_ui + (1 - w) * loss_ii).mean() * (1 - self.group_loss_gamma)) + loss_g * self.group_loss_gamma

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
    w = self.config.get('aggregate_w', 1)
    true_w = w
    u = u.to(self.device)
    if self.config.get('auto_gamma', False):
      true_w = torch.sigmoid(self.user_gamma(u))
    if w != 0:
      ue = self.user_embedding(u)
      x += torch.mm(ue, self.item_embedding(torch.arange(self.nitem, device=self.device)).T) * true_w
    if w != 1:
      uie = self.user_item_embedding(u)
      x_uui = torch.mm(uie, self.item_embedding(torch.arange(self.nitem, device=self.device)).T) * (1 - true_w)
      x += x_uui
    if mask:
      for i, m in enumerate(u):
        x[i][mask[m]] = -1e4
    return torch.topk(x, k=k, dim=1)[1]