import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model import encoder, decoder, repeater
from dataset import load_cifar10


def train(batch,sigma,epoch,learn_rate):

    loader = load_cifar10(batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = encoder().to(device)
    dec = decoder().to(device)

    loss_func = nn.MSELoss().to(device)
    enc_opt= optim.Adam(enc.parameters(), lr=learn_rate)
    dec_opt = optim.Adam(dec.parameters(),lr = learn_rate)

    for i in range(epoch):
        enc.train()
        dec.train()
        for img,_ in loader["train"]:
            img = img.to(device)
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            enc_sig = enc(img)
            shape = enc_sig.shape
            gauss = torch.normal(torch.zeros(shape),std=sigma)
            gauss = gauss.to(device)
            noisy = enc_sig + gauss
            m_hat = dec(noisy)
            loss = loss_func(m_hat, img)
            loss.backward()
            enc_opt.step()
            dec_opt.step()
        if i % 10 == 0:
            print(i, loss.item())



    return enc, dec

def valid(enc,dec,batch,sigma):
    loss_func = nn.MSELoss()
    loader = load_cifar10(batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc.to(device)
    dec.to(device)
    enc.eval()
    dec.eval()
    with torch.no_grad():
        for img,_ in loader["test"]:
            img = img.to(device)
            enc.zero_grad()
            dec.zero_grad()
            enc_sig = enc(img)
            shape = enc_sig.shape
            gauss = torch.normal(torch.zeros(shape),std=sigma)
            gauss = gauss.to(device)
            noisy1 = enc_sig + gauss
            m_hat = dec(noisy1)

    score = 0
    before = img[0,:,:,:].to("cpu").detach().numpy()
    after = m_hat[0,:,:,:].to("cpu").detach().numpy()
    before = before.transpose((1,2,0))
    after = after.transpose((1,2,0))

    before = Image.fromarray(before)
    after  =Image.fromarray(after)
    before.save("before.png")
    after.save("after.png")
    m_np = m.detach().numpy()
    m_hat_np = m_hat.detach().numpy()
    for ans , res in zip(m_np,m_hat_np):
      if np.where(ans ==1) == np.where(res == np.max(res)):
        score +=1
    mbs ,_ = m_hat.shape
    print(score/mbs)
    print(loss_func(m,m_hat))
    x_re = enc_sig[:, 0,:].detach().numpy()
    x_im = enc_sig[:, 1,:].detach().numpy()
    n_re = noisy1[:, 0,:].detach().numpy()
    n_im = noisy1[:, 1,:].detach().numpy()
    y_re = noisy2[:, 0,:].detach().numpy()
    y_im = noisy2[:, 1,:].detach().numpy()
    x_amp = np.unique(x_re**2 + x_im**2)
    m_rep = np.unique(m_re**2+m_im**2)
    print(f"encoder sig points: {x_amp.shape}")
    print(f"repeater sig points: {m_rep.shape}")
    print(f"enc points :{x_amp}")
    fig = plt.figure(figsize=(5,5))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(x_re, x_im)
    fig.savefig("encoder.png")
    fig = plt.figure(figsize=(5,5))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(n_re, n_im)
    fig.savefig("noisy1.png")
    fig = plt.figure(figsize=(5,5))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(m_re, m_im)
    fig.savefig("repeater.png")
    fig = plt.figure(figsize=(5,5))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(y_re, y_im)
    fig.savefig("noisy2.png")
