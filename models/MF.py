import torch
from torch import nn
import torch.nn.functional as F 
from utils import TagRecHelper


    

class BPR(nn.Module):
    def __init__(self, user_size, item_size, dim, weight_decay):
        super().__init__()
        self.W = nn.Parameter(torch.empty(user_size, dim))
        self.H = nn.Parameter(torch.empty(item_size, dim))
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)
        self.weight_decay = weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, u, i, j):
        """Return loss value.
        
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]
        
        Returns:
            torch.FloatTensor
        """
        u = self.W[u, :]
        i = self.H[i, :]
        j = self.H[j, :]
        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).sum()
        regularization = self.weight_decay * (u.norm(dim=1).pow(2).sum() + i.norm(dim=1).pow(2).sum() + j.norm(dim=1).pow(2).sum())
        return -log_prob + regularization

    def recommend(self, u, k=20, mask=None):
        """Return recommended item list given users.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        u = self.W[u, :]
        x_ui = torch.mm(u, self.H.t())
        if mask:
          for i, m in enumerate(u):
            x_ui[i][mask[m]] = -1e4
        return torch.topk(x_ui, k=k, dim=1)
      
class MF(BPR):
  def __init__(self, helper: TagRecHelper, config: dict):
    super().__init__(helper.nuser, helper.nitem, config['latent_dim'], config['weight_decay'])
    self.nuser = helper.nuser
    self.nitem = helper.nitem
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)
  def forward(self, u, i):
    j = torch.randint(0, self.nitem, (u.size(0),), device=self.device)
    return super().forward(u, i, j)