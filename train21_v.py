import argparse
import time
import torch
import os
import shutil
from Models_v import get_model
from Process21 import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch21 import create_masks
import dill as pickle
import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
losss=[]
f1=[]
criterion=F.nll_loss

global src_mask,src,trg_input,trg_mask
def train_model(model, opt, SRC, TRG):
    name = '128_L' + str(opt.n_layers)+str(opt.aaa)
    savep = 5
    print("training model...")
    model.train()
    #Initialize the test epoch
    doc_epoch = 50
    test_epoch= doc_epoch
    dst=name
    dst1=dst+"1"
    losstext = "losstext.csv"
    #dst1=name + str(test_epoch)+'_c_'+str(epoch+opt.restart)
    os.mkdir(dst)
    os.mkdir(dst1)
    best_val=opt.bestval
    best_epoch = 0+opt.restart
    for epoch in range(opt.epochs):
        total_loss = 0
        loss_value = 0
        for i, batch in enumerate(opt.train):
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            src_mask,trg_mask = create_masks(src, trg_input, opt)
            src_mask=torch.tensor(src_mask).cuda()
            src=torch.tensor(src).cuda()
            trg_input=torch.tensor(trg_input).cuda()
            trg_mask=torch.tensor(trg_mask).cuda()
            preds, ys= model(src, trg_input, src_mask, trg_mask)
            preds = torch.log(preds)
            opt.optimizer.zero_grad()
            loss = criterion(preds, ys)
            loss.backward()
            opt.optimizer.step()
            loss1 = loss
            loss1 = loss.detach().cpu().numpy()
            #print("current_loss:",loss1)
            if opt.SGDR == True:
                opt.sched.step()
            loss_value = loss_value + loss1
        if (epoch // savep) * savep - epoch == 0:
            shutil.rmtree(dst)
            dst = name + str(epoch+opt.restart)
            os.mkdir(dst)
            print("saving trained weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/test_weights')
            pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
            pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
        loss_value = loss_value / (i + 1)
        losss.append(loss_value)
        print("loss list:",losss)
        #plt.figure()
        #plt.plot(losss)
        #losstext=name+"val_epo"
        #np.savetxt(losstext, losss)
        #plt.savefig(name + ".png")
        best_epoch,best_val=eval(model, opt, SRC, TRG,epoch,name,best_epoch,best_val)
        print("best_epoch:",best_epoch)
        print("best_val:", best_val)
        print("current_epoch",epoch)
        if best_epoch>= doc_epoch:
            if best_epoch > test_epoch:
                shutil.rmtree(dst1)
                test_epoch = best_epoch
                xianshi=int(best_val*100)
                xianshi=xianshi/100
                dst1 = str(xianshi)+'_'+str(epoch+opt.restart)+'_'+str(opt.aaa)+'_'+str(opt.batchsize)
                os.mkdir(dst1)
                print("saving tested weights to " + dst1 + "/...")
                torch.save(model.state_dict(), f'{dst1}/test_weights')
                pickle.dump(SRC, open(f'{dst1}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{dst1}/TRG.pkl', 'wb'))

def eval(model,opt,SRC,TRG,epoch,name,best_epoch,best_val):
    validation = torch.zeros((1, 2)).float()
    labellll = torch.tensor(0).long()
    labellll = labellll.view(-1, 1)
    validation = torch.tensor(validation).cuda()
    labellll = torch.tensor(labellll).cuda()
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(opt.train1):
            src1 = batch.src.transpose(0, 1)
            trg1 = batch.trg.transpose(0, 1)
            trg_input1 = trg1[:, :-1]
            src_mask1, trg_mask1 = create_masks(src1, trg_input1, opt)
            src_mask1 = torch.tensor(src_mask1).cuda()
            src1 = torch.tensor(src1).cuda()
            trg_input1 = torch.tensor(trg_input1).cuda()
            trg_mask1 = torch.tensor(trg_mask1).cuda()
            val, valabel = model(src1, trg_input1, src_mask1, trg_mask1)
            valabel = valabel.view(-1, 1)
            validation = torch.cat((validation, val), dim=0)
            labellll = torch.cat((labellll, valabel), dim=0)
        validation = validation[1:, :]
        labellll = labellll[1:, :]
        print("vali_size:",labellll.size(0))
        validation = validation.detach().cpu().numpy()
        labellll = labellll.detach().cpu().numpy()
        validation = np.argmax(validation, axis=1)
        validation = validation.reshape(-1, 1)
        F_1 = f1_score(labellll, validation, average='binary')
        if F_1> best_val:
            best_val = F_1
            best_epoch = epoch+opt.restart
        else:
            best_epoch = best_epoch
            best_val = best_val
        print("val:", F_1)
        f1.append(F_1)
        plt.figure()
        plt.plot(f1)
        plt.savefig(name + "validation.png")
        valtext=name+"validation.txt"
        np.savetxt(valtext,f1)
        return best_epoch,best_val

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=True)
    parser.add_argument('-trg_data', required=True)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, required=True)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, required=True)
    parser.add_argument('-restart', type=int, required=True)
    parser.add_argument('-bestval', type=float, required=True)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, required=True)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-aaa',type=float, required=True)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-src_datav', required=True)
    parser.add_argument('-trg_datav', required=True)
    opt = parser.parse_args()

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())

    read_data(opt)
    SRC, TRG = create_fields(opt)
    opt.train = create_dataset(opt, SRC, TRG)
    opt.train1 = create_dataset1(opt, SRC, TRG)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    aaa=opt.aaa
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=aaa*1e-8, betas=(0.9, 0.999), eps=1e-8)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
    train_model(model, opt, SRC, TRG)



if __name__ == "__main__":
    main()
