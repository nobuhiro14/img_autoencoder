import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
from functools import reduce

def compose(*func):
  def sub_compose(f, g):
    return lambda x: f(g(x))

  return reduce(sub_compose, func, lambda x : x)


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(3,256,kernel_size=5,stride=2) # 符号化器用レイヤー
        self.norm1 = nn.BatchNorm2d(256)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv2d(256,256,kernel_size=3,stride=2) # 符号化器用レイヤ-
        self.norm2 = nn.BatchNorm2d(256)
        self.act2 = nn.ELU()
        self.conv3 = nn.Conv2d(256,256,kernel_size=3,stride=1)
        self.norm3 = nn.BatchNorm2d(256)
        self.act3 = nn.ELU()
        self.reshape = torch.reshape
        self.conv4 = nn.Conv1d(256,32,kernel_size=1,stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv5 = nn.Conv1d(32,2,kernel_size=1,stride=1)



    def normalize(self, x): # 送信信号の正規化
        # 等電力制約
        #norm = torch.norm(x,dim=1).view(mbs, 1).expand(-1, 2) # Normalization layer
        #x = x/norm
        # 平均エネルギー制約
        mbs ,_,_ = x.shape
        norm = torch.sqrt((x.norm(dim=1)**2).sum()/mbs)
        x = x/norm
        return x


    def forward(self, m):
        s = compose(self.act3,self.norm3,self.conv3,self.act2,self.norm2,self.conv2,self.act1,self.norm1,self.conv1)(m)
        mbs, ch ,_,_ = s.shape
        s = self.reshape(s,(mbs,ch,-1))
        s = compose(self.conv5,self.softmax,self.conv4)(s)
        y = self.normalize(s) # normalization
        return y

class repeater(nn.Module):
    def __init__(self,num_hidden_units,n):
        super(repeater, self).__init__()
        self.mid1 = nn.Linear(1*n, num_hidden_units) # 符号化器用レイヤー
        self.act1 = torch.relu
        self.mid2 = nn.Linear(num_hidden_units,num_hidden_units*2)
        self.act2 = torch.relu
        self.mid3 = nn.Linear(num_hidden_units*2, 2*n) # 符号化器用レイヤー
    def normalize(self, x): # 送信信号の正規化
        # 等電力制約
        #norm = torch.norm(x,dim=1).view(mbs, 1).expand(-1, 2) # Normalization layer
        #x = x/norm
        # 平均エネルギー制約
        mbs ,_,_ = x.shape
        norm = torch.sqrt((x.norm(dim=1)**2).sum()/mbs)
        x = x/norm
        return x
    def detection(self,x):
        y = x[:,0,:]**2 + x[:,1,:]**2
        return y

    def forward(self, m):
        s = self.detection(m)
        s = self.mid1(s)
        s = self.act1(s)
        s = self.mid2(s)
        s = self.act2(s)
        s = self.mid3(s)
        mbs, n_len = s.shape
        n = int(n_len/2)
        s = s.reshape(mbs,n,2)
        y = self.normalize(s)
        return y


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.lin1 = nn.Linear(16,32)
        self.reshape = torch.reshape
        transes = []
        acts = []
        norms = []
        for i in range(3):
            if i ==0:
                transes.append(nn.ConvTranspose2d(2,256,kernel_size=3,stride=1))
            else :
                transes.append(nn.ConvTranspose2d(256,256,kernel_size=3,stride=1))
            acts.append(nn.ELU())
            norms.append(nn.BatchNorm2d(256))
        self.transes = transes
        self.acts = acts
        self.norms = norms
        self.trans1 = nn.ConvTranspose2d(256,256,kernel_size=7,stride=1)
        self.act1 = nn.ELU()
        self.norm1 = nn.BatchNorm2d(256)
        self.trans2  = nn.ConvTranspose2d(256,3,kernel_size=2,stride=2)
        self.tan = torch.tanh
        self.reshape2 = torch.reshape

    def detection(self,x):
        y = x[:,0,:]**2 + x[:,1,:]**2
        return y

    def forward(self, m):
        s = self.detection(m)
        s = self.lin1(s)
        mbs,le = s.shape
        s = self.reshape(s,(mbs,2,4,-1))
        for i in range(3):
            s = self.transes[i](s)
            s = self.norms[i](s)
            s = self.acts[i](s)
        s = compose(self.act1,self.norm1,self.trans1)(s)
        s = compose(self.tan,self.trans2)(s)
        #s = compose(self.tan,self.trans2,self.norm1,self.act1,self.trans1)(s)
        return s
