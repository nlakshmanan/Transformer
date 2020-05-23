import argparse
import time
import torch
from Models211 import get_model
from Process21_cpu import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch21 import create_masks
import dill as pickle
import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
losss=[]
criterion=F.nll_loss
savep=2
name='aaa_'
def train_model(model, opt,SRC,TRG):
    
    print("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()

    for epoch in range(opt.epochs):
        total_loss = 0
        loss_value = 0

        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')
        for i, batch in enumerate(opt.train):
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            src_mask,trg_mask = create_masks(src, trg_input, opt)
            preds, ys = model(src, trg_input, src_mask,trg_mask)
            preds = torch.log(preds)
            opt.optimizer.zero_grad()
            loss = criterion(preds, ys)
            loss.backward()
            opt.optimizer.step()
            loss1 = loss.detach().numpy()
            print(loss1)
            if opt.SGDR == True:
                opt.sched.step()
            loss_value = loss_value + loss1
            print(loss_value)
        if (epoch // savep)*savep-epoch == 0:
            dst = name + str(epoch)
            os.mkdir(dst)
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
            pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
        losss.append(loss_value)
        print(losss)
        plt.figure()
        plt.plot(losss)
        plt.savefig(name+".png")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=True)
    parser.add_argument('-trg_data', required=True)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-heads', type=int, default=1)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=15000)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=1e-7)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)

    opt = parser.parse_args()
    
    #opt.device = 0 if opt.no_cuda is False else -1
    #if opt.device == 0:
    #    assert torch.cuda.is_available()
    
    read_data(opt)
    SRC, TRG= create_fields(opt)
    #exit()
    opt.train = create_dataset(opt, SRC, TRG)
    #print(len(SRC.vocab))
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
    train_model(model, opt,SRC,TRG)
if __name__ == "__main__":
    main()
