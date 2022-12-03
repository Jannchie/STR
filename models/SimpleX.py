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
        
        self.user_emb = torch.nn.Embedding(self.nuser+1, self.config.get('latent_dim', 64), padding_idx=self.nuser)
        self.item_emb = torch.nn.Embedding(self.nitem+1, self.config.get('latent_dim', 64), padding_idx=self.nitem)
        torch.nn.init.xavier_normal_(self.user_emb.weight)
        torch.nn.init.xavier_normal_(self.item_emb.weight)
        torch.nn.init.zeros_(self.user_emb.weight[-1,:])
        torch.nn.init.zeros_(self.item_emb.weight[-1,:])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'aggregate' in config and config.get('aggregate_w', 1) != 1:
            user_top = helper.train_set.groupby('user_id')['item_id'].apply(lambda x: torch.tensor(x.value_counts().index[:config.get('n_interactive_items', None)])).to_list()
            self.user_top = torch.nn.utils.rnn.pad_sequence(user_top, batch_first=True, padding_value=self.nitem).to(self.device)
            if config.get('aggregate', 'mean') == 'self-attention':
                self.attention = torch.nn.MultiheadAttention(config.get('latent_dim', 64), config.get('attention_head', 1), dropout=config.get('dropout', 0), batch_first=True, bias=False).to(self.device)
        self.user_dropout = torch.nn.Dropout(config.get('dropout', 0)).to(self.device)
        # item_count = self.data.train_set.groupby('item_id')['user_id'].count().values
        # self.item_dist = item_count / item_count.sum()
        self.item_dist = torch.ones(self.nitem) / self.nitem
        self.to(self.device)
        
    def affinity(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Return affinity between user and item.
        Args:
            u(torch.FloatTensor): user embedding. [batch_size, latent_dim]
            i(torch.FloatTensor): item embedding. [batch_size, latent_dim]
        Returns:
            pred(torch.FloatTensor): affinity between user and item. [batch_size,]
        """
        return torch.mul(u, i).sum(dim=1)

    def bpr_loss(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(u)
        i = self.item_embedding(i)
        j = self.item_emb[torch.randint(0, self.nitem, (len(i),)), :]
        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).mean()
        return -log_prob

    def ccl_loss(self, u, i, margin=0.8, neg_n = 100, neg_w = 0):
        """ Return cosine contractive loss.

        Args:
            u (torch.Tensor): user index. [batch_size] 
            i (torch.Tensor): item index. [batch_size]
            margin (float, optional): margin. Defaults to 0.8.
            neg_n (int, optional): negative item number. Defaults to 100.
            neg_w (int, optional): negative item weight. Defaults to 0.

        Returns:
            torch.Tensor: mean loss.
        """
        
        # generate negative items
        neg_idx = torch.randint(0, self.nitem, (u.shape[0] * neg_n,), device=self.device)
        neg_ie = self.item_embedding(neg_idx).view(-1, self.config.get('latent_dim', 64))
       
        ue = self.user_embedding(u)
        ie = self.item_embedding(i)
        
        pos_pred = self.affinity(ue, ie)
        pos_loss = torch.relu(1 - pos_pred)
        neg_ue = ue.repeat(1, neg_n).view(-1, self.config.get('latent_dim', 64))
        neg_pred = self.affinity(neg_ue, neg_ie).view(-1, neg_n)
        neg_loss = torch.relu(neg_pred - margin)
        if neg_w != 0:
            return (pos_loss + neg_loss.mean(dim=-1) * neg_w).mean()
        return (pos_loss + neg_loss.sum(dim=-1)).mean()

    def user_embedding(self, u: torch.Tensor) -> torch.Tensor:
        """Return user embedding.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
        Returns:
            pred(torch.FloatTensor): user embedding. [batch_size, latent_dim]
        """
        w = self.config.get('aggregate_w', 1)
        e1 = self.user_emb(u)
        if 'aggregate' in self.config and w != 1:
            items = self.item_embedding(self.user_top[u])
            aggregate = self.config.get('aggregate', 'mean')
            if (aggregate == 'mean'):
                e2 = items.mean(dim=1)
            elif (aggregate == 'self-attention'):
                e2 = self.attention(items, items, items, need_weights=False)[0].mean(dim=1)
            return self.user_dropout(w * e1 + (1 - w) * e2)
        if self.config.get('affinity', 'dot') == 'cos':
            e1 = F.normalize(e1)
        return self.user_dropout(e1)

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

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Return loss value.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]
        Returns:
            torch.FloatTensor
        """
        loss = self.config.get('loss', 'ccl')
        if (loss == 'bpr'):
            return self.bpr_loss(u, i)
        elif (loss == 'ccl'):
            return self.ccl_loss(u, i, margin=self.config.get('ccl_neg_margin', 0), neg_n=self.config.get('ccl_neg_num', 0), neg_w=self.config.get('ccl_neg_weight', 0))
        else:
            raise ValueError('loss function is not supported')

    def recommend(self, u: torch.Tensor, k=20, mask=None) -> torch.Tensor:
        """Return recommended item list given users.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            mask(torch.LongTensor, optional): tensor stored item indexes which is not recommendable. [batch_size,]
        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        u = u.to(self.device)
        ue = self.user_embedding(u)
        x_ui = torch.mm(ue, self.item_embedding(torch.arange(self.nitem).to(self.device)).t())
        if mask:
            for i, m in enumerate(u):
                x_ui[i][mask[m]] = -1e9
        return torch.topk(x_ui, k=k, dim=1)[1]