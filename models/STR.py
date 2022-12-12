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
  
  def uesr_item_embedding(self, i: torch.Tensor) -> torch.Tensor:
    """Return item embedding.
    Args:
        i(torch.LongTensor): tensor stored item indexes. [batch_size,]
    Returns:
        pred(torch.FloatTensor): item embedding. [batch_size, latent_dim]
    """
    e =  self.item_emb(i)
    if self.config.get('affinity', 'dot') == 'cos':
      e = F.normalize(e)
    return self.item_dropout(e)
  
  def user_item_embedding(self, u: torch.Tensor) -> torch.Tensor:
    """Return user embedding.
    Args:
        u (torch.Tensor): user index. [batch_size]

    Returns:
        torch.Tensor: user embedding. [batch_size, latent_dim]
    """
    ie = self.uesr_item_embedding(self.user_top[u])
    aggregate = self.config.get('aggregate', 'mean')
    if (aggregate == 'mean'):
      # e = ie.sum(dim=1) / self.user_top_len[u].view(-1, 1)
      e = ie.mean(dim=1)
    elif (aggregate == 'self-attention'):
      e = self.attention(ie, ie, ie, need_weights=False)[0]
      e = e.mean(dim=1)
    return e
  
  def forward(self, user: torch.Tensor, item: torch.Tensor):
    """ Return prediction loss.

    Args:
        user (torch.Tensor): user index. [batch_size]
        item (torch.Tensor): item index. [batch_size]

    Returns:
        loss(torch.Tensor): prediction loss. [batch_size] 
    """
    ie = self.item_embedding(item)
    w = self.config.get('aggregate_w', 1)
    loss_ii = 0
    loss_ui = 0
    if w != 0:
      ue = self.user_embedding(user)
      loss_ui = self.get_loss(ue, ie)
    if w != 1:
      uie = self.user_item_embedding(user)
      loss_ii = self.get_loss(uie, ie)
    if self.config.get('auto_gamma', False):
      gamma = torch.sigmoid(self.user_gamma(user))
      return (gamma * loss_ui + (1 - gamma) * loss_ii).mean()
    return (w * loss_ui + (1 - w) * loss_ii).mean()

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