import torch
import torch.nn as nn
import numpy as np
from Sublayersff import FeedForward, MultiHeadAttention, Norm
import torch.nn.functional as F

class Ucb_bandit(nn.Module):
    def __init__(self, n_layers, c,dd):
        super().__init__()
        #The number of avaiable actions(maximum layers)
        self.k = n_layers
        #The hyper paramater in bandit
        self.c = c
        #DUniform sequence length of each sample
        self.dd = dd

    def forward(self,last_loss,k_n,total_n,Aa,ba):
        #x is the input tensor
        num_sample=x.size(0)
        # for long sentence, randomly choose the action
        if x.size(1)>self.dd:
            inter1 = torch.randperm((self.k - 1))
            action = inter1[0].long()
        else:
            # padding process for shorter length
            if x.size(1)<self.dd:
                padding = torch.zeros(num_sample, (self.dd - x.size(1)))
                x = torch.cat((x, padding), dim=0)
                # enable the sl and embedding to be mergered as one row
            x = x.view(num_sample, 1, -1)
            x_list = torch.chunk(x, num_sample, dim=0)
            # generate one-hot label for each arm
            one_hot = torch.eye(self.k)
            one_hot_list = torch.chunk(one_hot, self.k, dim=0)
            E = x_list[0]
            # concat all samples to become matrix of context
            for i in num_sample:
                E = torch.cat((E, x_list[i + 1]), dim=0)
            # concat all one-hot to become combination matrix of contexts and one-hot
            for t in self.k:
                xta[t] = torch.cat((E, one_hot_list[t]), dim=0).transpose(0, 1)
            #obtain the value of d
            d=xta[0].size(0)
            # Aa product ba
            oa[action] = torch.mm(torch.inverse(Aa[action]), ba[action])
            # interm, the computation under sqrt
            for action in self.k:
                interm[action] = torch.mm(xta[action].transpose(0, 1), Aa[action].transpose(0, 1))
                interm[action] = torch.mm(interm[action], xta[action])
                # Upper Bound Computation
                pta[action] = torch.mm(xta[action].transpose(0, 1), oa[action]) + self.c * torch.sqrt(interm[action])
                # pta is a list and transfer it to tensor
                pta = torch.tensor(pta)
            # make action decision
            action = torch.argmax(pta, dim=0)
        # Update the number of each action has been taken
        k_n[action] += 1
        Aa[action] = Aa[action] + torch.mm(xta[action], xta[action].transpose(0, 1))
        ba[action] = ba[action] + last_loss[action] * xta[action]
        return action,k_n,Aa,ba

        # last_loss is the loss of last epoch
        # k_n is a matrices that count the number of each action has been taken
        # Select action according to UCB Criteria
        #inter = last_loss + self.c * torch.sqrt((torch.log(total_n))/ k_n)
        #if total_n <= 4000:
        #    action= torch.tensor(self.k-1).long()
        #elif 4000<total_n<=4500:
        #    action = ((total_n)%(self.k)).long()
        #else:
        #    ran = np.random.choice([-1, -2], 1, replace=True, p=[0.9, 0.1])
        #    if ran == -1:
        #        action = torch.argmax(inter).long()
        #        if action<self.k-1:
        #            action = action+revise
        #        else:
        #            action=action
        #    else:
        #        inter1 = torch.randperm((self.k - 1))
        #        action = inter1[0].long()
        #updates_rewards is to update current loss upon action taken
        #updated_rewards=last_loss[action]
        #Update the number of each action has been taken
        #k_n[action] += 1
        #return action, updated_rewards,k_n

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
