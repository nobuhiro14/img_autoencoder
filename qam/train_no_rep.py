import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model import encoder, decoder, repeater
from dataset import load_cifar10

def get_psnr(est,corr):
    est = est * 0.5 + 0.5
    est = est * 255
    corr = corr * 0.5 + 0.5
    corr = corr * 255
    mse = torch.sum(torch.sum((est-corr)**2 ))
    peak = 1
    psnr = 20 * torch.log(1/mse)
    return psnr

def train(batch,sigma,epoch,learn_rate,m):

    loader = load_cifar10(batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = encoder(m).to(device)
    dec = decoder(m).to(device)

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
            valid(enc,dec,batch,sigma)



    return enc, dec


def valid(enc,dec,batch,sigma):

        loss_func = nn.MSELoss()
        loader = load_cifar10(batch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc.to(device)
        dec.to(device)
        enc.eval()
        dec.eval()
        psnr = 0
        count = 0
        with torch.inference_mode():
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
                psnr += get_psnr(img,m_hat)
                count +=1
                loss  = loss_func(m_hat,img)


        ave_psnr = psnr/count
        print(f"PSNR : {ave_psnr}")
        losses = loss.item()
        print(f"loss : {losses}")

def test(enc,dec,batch,sigma):
    loss_func = nn.MSELoss()
    loader = load_cifar10(batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc.to(device)
    dec.to(device)
    enc.eval()
    dec.eval()
    psnr = 0
    count = 0
    with torch.inference_mode():
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
            psnr += get_psnr(img,m_hat)
            count +=1

    model_path = 'model_enc.pth'
    torch.save(enc.state_dict(), model_path)
    model_path = 'model_dec.pth'
    torch.save(dec.state_dict(), model_path)
    ave_psnr = psnr/count
    print(f"PSNR : {ave_psnr}")
    score = 0
    before = img[0,:,:,:].to("cpu").detach().numpy() * 0.5 + 0.5
    before = before * 255
    after = m_hat[0,:,:,:].to("cpu").detach().numpy() * 0.5 + 0.5
    after = after * 255
    before = before.transpose(2,1,0).astype(np.uint8)
    after = after.transpose(2,1,0).astype(np.uint8)
    before = Image.fromarray(before)
    after  =Image.fromarray(after)
    before.save("before.png")
    after.save("after.png")
    x_re = enc_sig[:, 0,:].detach().to("cpu").numpy()
    x_im = enc_sig[:, 1,:].detach().to("cpu").numpy()
    n_re = noisy1[:, 0,:].detach().to("cpu").numpy()
    n_im = noisy1[:, 1,:].detach().to("cpu").numpy()
    x_amp = np.unique(x_re**2 + x_im**2)
    print(f"encoder sig points: {x_amp.shape}")
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
