import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
from argparse import ArgumentParser

#from train_rep import train_rep,valid_rep
from train_no_rep import train,valid,test

class Option():
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("-hidden",type = int,default=64)
        parser.add_argument("-channel",type = int,default=1)
        parser.add_argument("-batch",type = int,default=100)
        parser.add_argument("-gauss",type = float,default=0.1)
        parser.add_argument("-epoch",type = int,default=8000)
        parser.add_argument("-learn_rate",type = float,default=0.01)
        #parser.add_argument("-alpha",type=int,default=2)
        #parser.add_argument("-no_rep",action="store_true")
        self.parser = parser

    def get_param(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = Option().get_param()
    m= args.hidden
    ch = args.channel
    batch = args.batch
    sigma = args.gauss
    ep = args.epoch
    lr = args.learn_rate
    #alpha = args.alpha
    #if args.no_rep != True :

    #    enc, rep ,dec = train_rep(m,hidden,n,batch,sigma,ep,lr)
    #    valid_rep(enc,rep,dec,m,batch,sigma)
    #else :
    enc, dec = train(batch,sigma,ep,lr,m,ch)
    test(enc,dec,batch,sigma)
