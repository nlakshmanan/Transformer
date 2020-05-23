import torch
import torch.nn as nn
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayersff import Norm,MultiHeadAttention,FeedForward
import copy
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        src_long=src.long()
        x = self.embed(src_long)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

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
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        src_vocab=src_vocab
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        #self.encoder1 = Encoder(trg_vocab, d_model, N, heads, dropout)
        #self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        #self.out = nn.Linear(d_model, trg_vocab)
        #self.out1 = nn.Linear(d_model, src_vocab)
        self.out2 = nn.Linear(d_model, 2)
    def forward(self, src, trg, src_mask,trg_mask):
        src=src.float()
        trg=trg.float()
        src_mask = src_mask.float()
        trg_mask = trg_mask.float()
        e_outputs = self.encoder(src, src_mask)
        mean_pooled = torch.mean(e_outputs, dim=1)
        y = self.out2(mean_pooled)
        y = F.softmax(y,dim=1)
        f_label = trg.transpose(0, 1)
        y1 = torch.tensor(0).float()
        y2 = torch.tensor(1).float()
        replace = torch.tensor(3).cuda()
        replace = torch.tensor(replace).cuda()
        y1 = torch.tensor(y1).cuda()
        y2 = torch.tensor(y2).cuda()
        f_label = torch.where(f_label <= replace, y1, y2)
        f_label = f_label.squeeze(0)
        return y.float(),f_label.long()

def get_model(opt, src_vocab, trg_vocab):

    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/test_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    model.cuda()

    return model
    
