import argparse
import time
import torch
import os
from Models_ucbcpu import get_model
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
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
losss=[]
f1=[]
criterion=F.nll_loss
name = '4h4l-7dyna_'
global src_mask,src,trg_input,trg_mask
def train_model(model, opt, SRC, TRG):
    savep = 50
    print("training model...")
    model.train()
    #The reward of each epoch
    rewards=torch.zeros(opt.epochs).float()
    #The counting marice for action
    k_n=torch.ones(opt.n_layers).float()
    #Total mean reward
    mean_reward=torch.tensor(0).float()
    #The reward matrix(loss matrix for each layer)
    last_loss=abs(torch.ones(opt.n_layers))
    #The number of total action
    total_n=torch.tensor(1).float()
    #Compute the layer distribution of each epoch
    knn_epoch=torch.zeros(opt.epochs,opt.n_layers)
    for epoch in range(opt.epochs):
        total_loss = 0
        loss_value = 0
        for i, batch in enumerate(opt.train):
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            src_mask,trg_mask = create_masks(src, trg_input, opt)
            preds, ys,rewardss,k_n,action = model(src, trg_input,src_mask,trg_mask,last_loss,k_n,total_n)
            preds = torch.log(preds)
            opt.optimizer.zero_grad()
            loss = criterion(preds, ys)
            loss.backward()
            opt.optimizer.step()
            loss1 = loss
            print(loss1)
            mean_reward, total_n, last_loss, knn_epoch=dyna(action,rewards,k_n,mean_reward,loss,\
                                                            last_loss,total_n,rewardss,knn_epoch,epoch)
            if opt.SGDR == True:
                opt.sched.step()
            loss_value = loss_value + loss1
            print(loss_value)
        #if (epoch // savep) * savep - epoch == 0:
        #    dst = name + str(epoch)
        #    os.mkdir(dst)
        #    print("saving weights to " + dst + "/...")
        #    torch.save(model.state_dict(), f'{dst}/model_weights')
        #    pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
        #    pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
        loss_value=loss_value/(i+1)
        losss.append(loss_value)
        print(losss)
        pl_greedy(k_n, opt)
        pl_epoch(knn_epoch, opt,last_loss,k_n,total_n)
        plt.figure()
        plt.plot(losss)
        plt.savefig(name+".png")
        #eval(model, opt, SRC, TRG)

def dyna(action,rewards,k_n,mean_reward,loss,last_loss,total_n,rewardss,knn_epoch,epoch):
    # Update total rewards
    mean_reward = mean_reward + (rewardss - mean_reward) / total_n
    # The total number of actions which has been taken in whole training
    total_n = total_n + 1
    # Update the rewards after each sample training
    reward_value = loss / (action + 1)
    # Update results for loss matrix (reward matrix)
    last_loss[action] = reward_value + (rewards[action] - reward_value) / k_n[action]
    ##### Need more confirmation
    knn_epoch[epoch][action] +=1
    return mean_reward,total_n,last_loss,knn_epoch

def pl_greedy(k_n,opt):
    plt.figure()
    figure_name = "Each Action Distribution"
    plt.title(figure_name)
    #xx1= np.linspace(1,opt.n_layers, opt.n_layers)
    k_n=k_n.detach().cpu().numpy()
    yy1= k_n
    xx1=np.linspace(1,opt.n_layers,opt.n_layers)
    plt.bar(xx1,yy1,width = 0.35,align='center',color ='blue',alpha=0.8)
    plt.xlabel('Action')
    plt.ylabel('Total_Number')
    plt.savefig(name + figure_name+".png")

def pl_epoch(knn_epoch,opt):
    plt.figure()
    figure_name="Action Distribution of Each Epoch"
    plt.title(figure_name)
    knn_epoch=knn_epoch.detach().cpu().numpy()
    xxx=np.linspace(1,opt.epochs,opt.epochs)
    plt.plot(xxx, knn_epoch[:,0], color='green',linewidth=3, label='One Layer')
    plt.plot(xxx, knn_epoch[:,1], color='red', linewidth=3,label='Two Layer')
    plt.plot(xxx, knn_epoch[:,2], color='blue', linewidth=3,label='Three Layer')
    plt.plot(xxx, knn_epoch[:,3], color='black', linewidth=3,label='Four Layer')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Action Number')
    plt.savefig(name + figure_name+".png")


def eval(model,opt,SRC,TRG,last_loss,k_n,total_n):
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
            val, valabel = model(src1, trg_input1, src_mask1, trg_mask1,last_loss,k_n,total_n)
            valabel = valabel.view(-1, 1)
            validation = torch.cat((validation, val), dim=0)
            labellll = torch.cat((labellll, valabel), dim=0)
        validation = validation[1:, :]
        labellll = labellll[1:, :]
        validation = validation.detach().cpu().numpy()
        labellll = labellll.detach().cpu().numpy()
        validation = np.argmax(validation, axis=1)
        validation = validation.reshape(-1, 1)
        F_1 = f1_score(labellll, validation, average='binary')
        print("val:", F_1)
        f1.append(F_1)
        plt.figure()
        plt.plot(f1)
        plt.savefig(name + "val.png")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=True)
    parser.add_argument('-trg_data', required=True)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-c',  type=float, default=0.9)
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, required=True)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, required=True)
    parser.add_argument('-heads', type=int, default=4)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, required=True)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=1e-7)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    opt = parser.parse_args()

    read_data(opt)
    SRC, TRG = create_fields(opt)
    opt.train = create_dataset(opt, SRC, TRG)
    #opt.train1 = create_dataset1(opt, SRC, TRG)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
    train_model(model, opt, SRC, TRG)



if __name__ == "__main__":
    main()
