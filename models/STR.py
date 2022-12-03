import torch
from torch.nn import functional as F
from utils import TagRecHelper

class STR(torch.nn.Module):
    def __init__(self, helper: TagRecHelper, config: dict):
        super().__init__()
        self.helper = helper
        self.config = config
        self.nuser = helper.nuser
        self.nitem = helper.nitem
        self.W = torch.nn.Parameter(torch.empty(self.nuser, config['latent_dim']))
        self.H = torch.nn.Parameter(torch.empty(self.nitem+1, config['latent_dim'])) # + 1 for padding
        torch.nn.init.xavier_normal_(self.W.data)
        torch.nn.init.xavier_normal_(self.H.data)
        torch.nn.init.zeros_(self.H[-1, :].data)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        if 'aggregate' in config:
            # get every user's N used items to np array
            self.user_top = torch.nn.utils.rnn.pad_sequence(helper.train_set.groupby('user_id')['item_id'].apply(lambda x: torch.tensor(x.value_counts().index[:config['n_interactive_items']])).to_list(), padding_value=len(self.H.data)-1).T.to(self.device)
            if (config['aggregate'] == 'self-attention'):
                self.attention = torch.nn.MultiheadAttention(config['latent_dim'], config['attention_head'], dropout=0.2, batch_first=True, bias=False).to(self.device)

    def affinity(self, u, i):
        """Return affinity between user and item.
        Args:
            u(torch.FloatTensor): user embedding. [batch_size, latent_dim]
            i(torch.FloatTensor): item embedding. [batch_size, latent_dim]
        Returns:
            pred(torch.FloatTensor): affinity between user and item. [batch_size,]
        """
        if (self.config['affinity'] == 'dot'):
            return torch.mul(u, i).sum(dim=1)
        elif (self.config['affinity'] == 'cos'):
            return F.cosine_similarity(u, i, dim=1)
        else:
            raise ValueError('affinity function is not supported, please choose from [dot, cos]')

    def bpr_loss(self, u, i):
        j = self.H[torch.randint(0, self.nitem, (len(i),)), :]
        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).mean()
        return -log_prob

    def ccl_loss(self, u, i, margin=0.8, neg_n = 100, neg_w=1):
        # consine similarity
        pos_pred = self.affinity(u, i)
        pos_loss = torch.relu(1 - pos_pred)
        neg_i = self.H[torch.randint(0, self.nitem, (len(i) * neg_n,)), :]
        neg_pred = self.affinity(u.unsqueeze(1).repeat(1, neg_n, 1).view(-1, u.shape[1]), neg_i).view(-1, neg_n)
        neg_loss = torch.relu(neg_pred - margin)
        return (pos_loss + neg_loss.sum(dim=-1) * neg_w).mean()

    def user_embedding(self, u):
        """Return user embedding.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
        Returns:
            pred(torch.FloatTensor): user embedding. [batch_size, latent_dim]
        """
        return self.W[u, :].view(1, -1) if type(u) is int else self.W[u, :]
    
    def item_embedding(self, i):
        """Return item embedding.
        Args:
            i(torch.LongTensor): tensor stored item indexes. [batch_size,]
        Returns:
            pred(torch.FloatTensor): item embedding. [batch_size, latent_dim]
        """
        return self.H[i, :].view(1, -1) if type(i) is int else self.H[i, :]
    
    def item_loss(self, u, i, neg_n = None):
        """Return loss value.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
        Returns:
            loss(torch.FloatTensor): loss value. [batch_size,]
        """
        ei = self.user_item_embedding(u)
        if (self.config['loss'] == 'bpr'):
            return self.bpr_loss(ei, i)
        elif (self.config['loss'] == 'ccl'):
            if neg_n is None:
                neg_n = self.config['ccl_neg_num']
            return self.ccl_loss(ei, i, margin=self.config['ccl_neg_margin'], neg_n=neg_n, neg_w=self.config['ccl_neg_w'])
        else:
            raise ValueError('loss function is not supported, please choose from [bpr, ccl]')

    def user_item_embedding(self, u):
        items = self.H[self.user_top[u].view(-1),:].view(-1, self.config['n_interactive_items'], self.config['latent_dim'])
        if (self.config['aggregate'] == 'mean'):
            ei = items.mean(dim=1)
        elif (self.config['aggregate'] == 'self-attention'):
            ei = self.attention(items, items, items, need_weights=False)[0].mean(dim=1)
        return ei

    def forward(self, u, i):
        """Return loss value.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]
        Returns:
            torch.FloatTensor
        """
        ue = self.user_embedding(u)
        ie = self.item_embedding(i)
        w = self.config['aggregate_w']
        
        if (self.config['loss'] == 'bpr'):
            uiloss =  self.bpr_loss(ue, ie) 
        elif (self.config['loss'] == 'ccl'):
            uiloss =  self.ccl_loss(ue, ie, margin=self.config['ccl_neg_margin'], neg_n=self.config['ccl_neg_num'], neg_w=self.config['ccl_neg_w'])
        else:
            raise ValueError('loss function is not supported')
        return (1 - w) * uiloss + w * self.item_loss(u, ie)

    def recommend(self, u, k=20):
        """Return recommended item list given users.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        w = self.config['aggregate_w']
        x_ui = (1 - w) * torch.mm(self.user_embedding(u), self.H[:-1, :].T) + w * torch.mm(self.user_item_embedding(u), self.H[:-1, :].T)
        return torch.argsort(x_ui, dim=1, descending=True)[:, :k]
