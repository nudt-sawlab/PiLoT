import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

class GraphAttentionLayer(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        dropout, 
        alpha,
        num_head = 4,
        # dim_head = 32,
        depth=2,
        with_query_type=0,
        concat=True,
        # include_self=True,
        # additional=False,
        # with_linear_transform=True
    ):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_head = num_head
        self.depth = depth
        self.with_query_type = with_query_type
        
        if depth > 1:
            self.Ws = nn.ParameterList()
            # self.As = nn.ParameterList()
            for i in range(depth):
                if i == 0:
                    W = nn.Parameter(torch.empty(size=(num_head, in_features // num_head, out_features)))
                    nn.init.xavier_normal_(W.data, gain=1.414)
                else:
                    W = nn.Parameter(torch.empty(size=(num_head, out_features, out_features)))
                    nn.init.xavier_normal_(W.data, gain=1.414)
                # a = nn.Parameter(torch.empty(size=(out_features, 1)))
                # nn.init.xavier_normal_(a.data, gain=1.414)
                self.Ws.append(W)
                # self.As.append(a)
        else:
            self.W = nn.Parameter(torch.empty(size=(num_head, in_features // num_head, out_features)))
            nn.init.xavier_normal_(self.W.data, gain=1.414)
        
        # self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.a = nn.Parameter(torch.empty(size=(num_head, out_features, 1)))
        # self.a = nn.Parameter(torch.empty(size=(num_head, out_features*2, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        if with_query_type == 0:
            self.W_q = nn.Parameter(torch.empty(size=(in_features, in_features))) # with_query0
            nn.init.xavier_normal_(self.W_q.data, gain=1.414)
        elif with_query_type == 1:
            self.W_q = nn.Sequential(nn.Linear(in_features, in_features),         # with_query1
                                    nn.Linear(in_features, in_features),
                                    # nn.Sigmoid()
                                    )
        elif with_query_type == 2:
            self.W_q = nn.Sequential(nn.Linear(in_features, in_features, bias=False), # with_query2
                                    nn.Linear(in_features, in_features, bias=False),
                                    # nn.Sigmoid()
                                    )
        else:
            raise NotImplementedError
        
        # self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # nn.init.xavier_normal_(self.W.data, gain=1.414)
        # # self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        # self.a = nn.Parameter(torch.empty(size=(out_features, 1)))
        # nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # self.include_self = include_self
        # self.with_linear_transform = with_linear_transform
        # self.additional = additional
    
    def forward_q(self, h_3d):
        if self.with_query_type == 0:
            h_prime_3d = torch.matmul(h_3d, self.W_q)
        else:
            h_prime_3d = self.W_q(h_3d)
        return h_prime_3d

    def forward(self, h_2d, mask=None, h_3d=None):
        B, num_leaf, N, dim= h_2d.shape

        if self.depth > 1:
            h_2d = rearrange(h_2d, 'b c n (h d) -> b c n h d', h = self.num_head)
            wh_2d = h_2d.unsqueeze(-2)
            for W in self.Ws:
                wh_2d = torch.matmul(wh_2d, W)
            wh_2d_ = torch.matmul(wh_2d, self.a)
            wh_2d_ = torch.reshape(wh_2d_, (B, num_leaf, N, self.num_head))
        else:
            h_2d = rearrange(h_2d, 'b c n (h d) -> b c n h d', h = self.num_head)
            wh_2d = torch.matmul(h_2d.unsqueeze(-2), self.W)
            wh_2d_ = torch.matmul(wh_2d, self.a)
            wh_2d_ = torch.reshape(wh_2d_, (B, num_leaf, N, self.num_head))
        
        e = self.leakyrelu(wh_2d_)
        if mask is None:
            attention = F.softmax(e, dim=1)
        else:
            # e_exp = e.exp()
            e_exp = (e - e.max(dim=1, keepdim=True)[0]).exp()
            e_exp_mask = e_exp.masked_fill(~mask[..., None], 0.)
            # e_exp_mask = e_exp * mask[..., None]
            attention = e_exp_mask / e_exp_mask.sum(1, keepdim=True).clamp(min=0.0001)
        h_2d = h_2d * attention[..., None]
        h_prime = h_2d.reshape(B, num_leaf, N, dim).sum(1)

        return h_prime

        if h_3d is not None:
            if self.with_query_type == 0:
                h_prime_3d = torch.matmul(h_3d, self.W_q)
            else:
                h_prime_3d = self.W_q(h_3d)
        else:
            h_prime_3d = None

        # h_2d = h_2d.reshape(B, num_leaf*N, dim)
        # wh_2d = torch.matmul(h_2d, self.W)
        # wh_2d_ = torch.matmul(wh_2d, self.a)
        # wh_2d_ = torch.reshape(wh_2d_, (B, num_leaf, N, -1))
        # e = self.leakyrelu(wh_2d_)
        # attention = F.softmax(e, dim=1)
        # h_2d = torch.reshape(h_2d, (B, num_leaf, N, dim))
        # h_2d = h_2d.permute(0, 2, 1, 3)
        # attention = attention.permute(0, 2, 1, 3)
        # h_prime = torch.einsum('bncd,bncq->bnq', attention, h_2d)

        # for W, a in zip(self.Ws, self.As):
        #     h_2d = h_2d.reshape(B, num_leaf*N, dim)
        #     wh_2d = torch.matmul(h_2d, W)
        #     wh_2d_ = torch.matmul(wh_2d, a)
        #     wh_2d_ = torch.reshape(wh_2d_, (B, num_leaf, N, -1))
        #     e = self.leakyrelu(wh_2d_)
        #     attention = F.softmax(e, dim=1)
        #     h_2d = torch.reshape(h_2d, (B, num_leaf, N, dim))
        #     h_2d = h_2d * attention
        # h_prime = h_2d.sum(1)

        # h_2d = h_2d.permute(0, 2, 1, 3)
        # attention = attention.permute(0, 2, 1, 3)
        # h_prime = torch.einsum('bncd,bncq->bnq', attention, h_2d) / 2.

        # b, n1, dim = h_3d.shape
        # b, n2, dim = h_2d.shape
        # num_leaf = int(n2 / n1)

        # wh_2d = torch.matmul(h_2d, self.W)
        # wh_3d = torch.matmul(h_3d, self.W)

        # e = self._prepare_attentional_mechanism_input(wh_2d, wh_3d, num_leaf, self.include_self)
        # attention = F.softmax(e, dim=2)

        # h_2d = torch.reshape(h_2d, (b, n1, num_leaf, dim))
        # wh_2d = torch.reshape(wh_2d, (b, n1, num_leaf, dim))
        # if self.include_self:
        #     wh_2d = torch.cat(
        #         [wh_3d.unsqueeze(-2), wh_2d], dim=-2
        #     ) # [b, N, 1+num_leaf, d_out]
        #     h_2d = torch.cat(
        #         [h_3d.unsqueeze(-2), h_2d], dim=-2
        #     )

        #     if self.with_linear_transform:
        #         h_prime = torch.einsum('bncd,bncq->bnq', attention, wh_2d)
        #     else:
        #         h_prime = torch.einsum('bncd,bncq->bnq', attention, h_2d)

        #     if self.additional:
        #         h_prime = h_prime + h_3d
        # else:
        #     if self.with_linear_transform:
        #         h_prime = torch.einsum('bncd,bncq->bnq', attention, wh_2d) / 2. + wh_3d
        #     else:
        #         h_prime = torch.einsum('bncd,bncq->bnq', attention, h_2d) / 2. + h_3d

        if self.concat:
            return F.elu(h_prime), h_prime_3d
        else:
            return h_prime, h_prime_3d
    
    def _prepare_attentional_mechanism_input(self, wh_2d, wh_3d, num_leaf, include_self=False):
        b, n1, dim = wh_3d.shape
        b, n2, dim = wh_2d.shape

        wh_2d_ = torch.matmul(wh_2d, self.a[:self.out_features, :]) # [b, N2, 1]
        wh_2d_ = torch.reshape(wh_2d_, (b, n1, num_leaf, -1)) # [b, n1, 6, 1]
        wh_3d_ = torch.matmul(wh_3d, self.a[self.out_features:, :]) # [b, N1, 1]

        if include_self:
            wh_2d_ = torch.cat(
                [wh_3d_.unsqueeze(2), wh_2d_], dim=-2
            ) # [b, N1, 1 + num_leaf, 1]

        e = wh_3d_.unsqueeze(2) + wh_2d_
        return self.leakyrelu(e)