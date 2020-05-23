import argparse
import time
import torch
from Models_v import get_model
from Process21_trans import *
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

def train_model(model, opt, SRC, TRG):
    with torch.no_grad():
        print("testing model...")
        model.train()
        a = 0
        prediction = torch.zeros((1, 2)).float()
        label = torch.tensor(0).long()
        label = label.view(-1, 1)
        prediction = torch.tensor(prediction).cuda()
        label = torch.tensor(label).cuda()
        for i, batch in enumerate(opt.train):
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            src_mask = torch.tensor(src_mask).cuda()
            src = torch.tensor(src).cuda()
            trg_input = torch.tensor(trg_input).cuda()
            trg_mask = torch.tensor(trg_mask).cuda()
            if opt.min_strlen <= src.size(1) < opt.max_strlen:
                preds, ys = model(src, trg_input, src_mask, trg_mask)
                ys = ys.view(-1, 1)
                prediction = torch.cat((prediction, preds), dim=0)
                label = torch.cat((label, ys), dim=0)
                #a1 = opt.test_iteration
                #while a > a1:
                #    prediction = prediction[1:, :]
                #    label = label[1:, :]
                #    prediction = prediction.detach().cpu().numpy()
                #    label = label.detach().cpu().numpy()
                #    prediction = np.argmax(prediction, axis=1)
                #    prediction = prediction.reshape(-1, 1)
                #    F_measure1 = f1_score(label, prediction, average='weighted')
                #    print("f1_score:", F_measure1)
        prediction = prediction[1:, :]
        label = label[1:, :]
        print(prediction.size(), label.size())
        if prediction.size(0) != label.size(0):
            if prediction.size(0) >= label.size(0):
                makeup = torch.tensor((prediction.size(0) - label.size(0))).long()
                one = torch.tensor(1).long()
                label = torch.cat((label.long(), torch.zeros(makeup, one).long().cuda()), dim=0)
            else:
                makeup = torch.tensor((label.size(0) - prediction.size(0))).float()
                two = torch.tensor(2).float()
                prediction = torch.cat((prediction, torch.zeros(makeup, two).cuda()), dim=0)
        prediction1 = prediction.detach().cpu().numpy()
        label1 = label.detach().cpu().numpy()
        prediction1 = np.argmax(prediction1, axis=1)
        prediction1 = prediction1.reshape(-1, 1)
        F_measure1 = f1_score(label1, prediction1, average='binary')
        print(np.shape(prediction1)[0])
        print("f1_score:", F_measure1)
        torch.cuda.empty_cache()
        return F_measure1, prediction, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=True)
    parser.add_argument('-trg_data', required=True)
    parser.add_argument('-load_weights', required=True)
    #parser.add_argument('-index_cuda', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, required=True)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-sta', type=int,required=True)
    parser.add_argument('-end', type=int,required=True)
    parser.add_argument('-i', type=int, required=True)
    #parser.add_argument('-times', type=int, required=True)
    #parser.add_argument('-test_iteration', type=int, required=True)
    parser.add_argument('-heads', type=int, default=4)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-max_strlen',type=int,required=True)
    parser.add_argument('-min_strlen', type=int, required=True)
    #parser.add_argument('-in_cuda', type=int, required=True)
    parser.add_argument('-batchsize', type=int,required=True)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-SGDR', action='store_true')
    opt = parser.parse_args()

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())
    sta = int(opt.sta)
    end = int(opt.end)
    gap=int(end-sta)
    times=10000//gap
    print(times)
    predictions = torch.zeros((1, 2)).float().cuda()
    labels = torch.tensor(0).long().cuda()
    labels = labels.view(-1, 1).cuda()
    f1s=0
    read_data(opt)
    SRC, TRG = create_fields(opt)
    i=opt.i
    while i <=times:
        print("sta:",sta)
        print("end:",end)
        opt.train = create_dataset(opt, SRC, TRG, sta,end)
        model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
        if opt.floyd is False:
            promptNextAction(model, opt)
        final,predic,lab=train_model(model, opt, SRC, TRG)
        predic=predic.cuda()
        lab=lab.cuda()
        f1s=final+f1s
        predictions=torch.cat((predictions,predic),dim=0)
        labels=torch.cat((labels,lab), dim=0)
        p1=predictions
        l1=labels
        p1=  p1.detach().cpu().numpy()
        l1 = l1.detach().cpu().numpy()
        p1 = np.argmax(p1, axis=1)
        F_measure11 = f1_score(p1, l1, average='binary')
        print("tempf1:",F_measure11)
        sta = sta + gap
        end = end + gap
        i=i+1
        print("i:",i)
    predictionss = predictions[1:, :]
    labss = labels[1:, :]
    predictionss = predictionss.detach().cpu().numpy()
    labss = labss.detach().cpu().numpy()
    predictionss = np.argmax(predictionss, axis=1)
    F_measure = f1_score(labss, predictionss, average='binary')
    print("final:",f1s/(i+1), F_measure)

def promptNextAction(model, opt):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'


if __name__ == "__main__":
    main()