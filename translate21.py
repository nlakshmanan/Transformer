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
import torch.optim as optimmatplotlib
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train_model(model, opt):
    prediction = torch.zeros((1, 2)).float()
    label = torch.tensor(5).long()
    label = label.view(-1, 1)
    print("training model...")
    model.train()
    for i, batch in enumerate(opt.train):
        src = batch.src.transpose(0,1)
        trg = batch.trg.transpose(0,1)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input, opt)
        preds,ys = model(src, trg_input, src_mask, trg_mask)
        ys=ys.view(-1,1)
        prediction=torch.cat((prediction,preds),dim=0)
        label=torch.cat((label,ys),dim=0)
        print(i)
    prediction=prediction[1:,:]
    label=label[1:,:]
    print(prediction.size(),label.size())
    prediction = prediction.detach().numpy()
    label = label.detach().numpy()
    prediction=np.argmax(prediction, axis=1)
    prediction=prediction.reshape(-1,1)
    F_measure1 = f1_score(label, prediction, average='weighted')
    print("f1_score:",F_measure1)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=True)
    parser.add_argument('-trg_data', required=True)
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-heads', type=int, default=4)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-batchsize', type=int, default=8000)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-SGDR', action='store_true')
    opt = parser.parse_args()

    #opt.device = 0 if opt.no_cuda is False else -1
    #if opt.device == 0:
    #    assert torch.cuda.is_available()

    read_data(opt)
    SRC, TRG= create_fields(opt)
    opt.train = create_dataset(opt, SRC, TRG)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    #if opt.load_weights is not None and opt.floyd is not None:
        #os.mkdir('weights')
        #pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        #pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    
    train_model(model, opt)
    if opt.floyd is False:
        promptNextAction(model, opt, SRC, TRG)


def promptNextAction(model, opt, SRC, TRG):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'


if __name__ == "__main__":
    main()
