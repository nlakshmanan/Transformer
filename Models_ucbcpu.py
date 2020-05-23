import torch
import torch.nn as nn
from Layers_ucb import EncoderLayer, DecoderLayer,Ucb_bandit
from Embed import Embedder, PositionalEncoder
from Sublayersff import Norm,MultiHeadAttention,FeedForward
import copy
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout,c,n_layers):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.dynam = Ucb_bandit(n_layers,c)
        self.layer = EncoderLayer(d_model, heads, dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask,last_loss,k_n,total_n):
        #print("src:", src.size())
        src_long=src.long()
        x = self.embed(src_long)
        x = self.pe(x)
        #At least pass one layer
        x=self.layer(x, mask)
        action, updated_rewards, k_n= self.dynam(last_loss,k_n,total_n)
        if action == 0:
            return self.norm(x),updated_rewards, k_n,action
        else:
            for i in range(action):
                x = self.layers[i](x, mask)
            return self.norm(x),updated_rewards, k_n,action

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        #print("trg:",trg.size())
        x = self.embed(trg)
        #print("x.embed:",x.size())
        x = self.pe(x)
        #print("x.pe:",x.size())
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        #print("x.attention:",x.size())
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout,c,n_layers):
        super().__init__()
        src_vocab=src_vocab
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout,c,n_layers)
        #self.encoder1 = Encoder(trg_vocab, d_model, N, heads, dropout)
        #self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        #self.out = nn.Linear(d_model, trg_vocab)
        #self.out1 = nn.Linear(d_model, src_vocab)
        self.out2 = nn.Linear(d_model, 2)
    def forward(self, src, trg, src_mask,trg_mask,last_loss,k_n,total_n):
        #print("src_size:",src.size())
        #print("trg_size:",trg.size())
        #print("src:",src)
        #print("trg:",trg)
        bat_size = src.size(0)
        src=src.float()
        trg=trg.float()
        src_mask = src_mask.float()
        trg_mask = trg_mask.float()
        #rewards is the obtained rewards in this epoch
        #knn is the matrices for counting each action
        e_outputs,rewardss,knn,action = self.encoder(src, src_mask,last_loss,k_n,total_n)
        mean_pooled = torch.mean(e_outputs, dim=1)
        y = self.out2(mean_pooled)
        y = F.softmax(y,dim=1)
        f_label = trg.transpose(0, 1)
        y1 = torch.tensor(0).float()
        y2 = torch.tensor(1).float()
        replace=torch.tensor(3).float()
        f_label = torch.where(f_label <= replace, y1, y2)
        #print(f_label)
        f_label = f_label.squeeze(0)
        return y.float(),f_label.long(),rewardss,knn,action

def get_model(opt, src_vocab, trg_vocab):

    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab,opt.d_model, opt.n_layers, opt.heads, opt.dropout,opt.c,opt.n_layers)
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model
    
