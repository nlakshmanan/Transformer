import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        #self.q_linear = nn.Linear(d_model, d_model)
        #self.v_linear = nn.Linear(d_model, d_model)
        #self.k_linear = nn.Linear(d_model, d_model)
        #initilaize the k,q,z weighted matrices
        self.q_linear1 = nn.Parameter(torch.randn(d_model, d_model))
        self.v_linear1 = nn.Parameter(torch.randn(d_model, d_model))
        self.k_linear1 = nn.Parameter(torch.randn(d_model, d_model))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, dropout=None):

        bs = q.size(0)

        if q.size(1)>230:
            self.h=8
            self.d_k = self.d_model // self.h
        elif q.size(1)<=138 and q.size(1)>230:
            self.h=4
            self.d_k = self.d_model // self.h
        elif q.size(1)<=138 and q.size(1)>0:
            self.h=2
            self.d_k = self.d_model // self.h

        # print("bs:",bs)
        # perform q,k,v product operation and split into N heads
        k= torch.matmul(k,self.k_linear1)
        k= k.view(bs, -1, self.h, self.d_k)
        q = torch.matmul(q, self.q_linear1)
        q = q.view(bs, -1, self.h, self.d_k)
        v = torch.matmul(v, self.v_linear1)
        v = v.view(bs, -1, self.h, self.d_k)
        # perform linear operation and split into N heads
        #k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        #q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        #v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        #print("k:",k.size())
        #print("q:",q.size())
        #print("v:",v.size())
        # transpose to get dimensions bs * N * sl * dk
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        # transpose to get dimension of k as bs * N * dk * sl
        k = k.transpose(-2,-1)
        scores=torch.matmul(q,k)
        # scores has the dimension: bs * N * sl * sl
        scores = scores / math.sqrt(self.d_k)
        #print("scores_or:", scores.size())
        if mask is not None:
            mask = mask.unsqueeze(1)
            #print("mask:",mask.size())
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        #print("mask_score:", scores.size())
        if dropout is not None:
            scores = dropout(scores)
        #scores has the dimension: bs * N * sl * dk
        scores = torch.matmul(scores, v)
        #print("scores:", scores.size())
        # concat has the dimension: bs * sl * d_model
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        #print("concat_atten:", concat.size())
        output = self.out(concat)
        #print("output_atten:", output.size())
        return output
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
