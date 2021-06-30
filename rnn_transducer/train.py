#coding: utf-8
import os
import time
import random
import argparse
import logging

import numpy as np
import torch
from torch import nn 

from model import Transducer
from warprnnt_pytorch import RNNTLoss
from data_helper import DataHelper
from data_helper import DataHelperWithLessMem

parser = argparse.ArgumentParser(description='Train RNN Tranducer Model Implemented by PyTorch')
parser.add_argument('--train_file', type=str, default="",
                    help='train file path')
parser.add_argument('--dev_file', type=str, default="",
                    help='dev file path')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--input_size', type=int, default=38, metavar='N',
                    help='input size')
parser.add_argument('--output_size', type=int, default=13000, metavar='N',
                    help='output size')
parser.add_argument('--output_embed_size', type=int, default=13000, metavar='N',
                    help='output embedding size')
parser.add_argument('--hidden_size', type=int, default=250, 
                    help='hidden size for rnn')
parser.add_argument('--proj_size', type=int, default=0, 
                    help='porjection size for rnn')
parser.add_argument('--ctc_layer_nums', type=int, default=3, 
                    help='layer nums of ctc network')
parser.add_argument('--pred_layer_nums', type=int, default=2, 
                    help='layer nums of predict network')
parser.add_argument('--bi', default=False, action='store_true', 
                    help='whether use bidirectional lstm')
parser.add_argument('--noise', default=False, action='store_true',
                    help='add Gaussian weigth noise')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--log-evaluate', type=int, default=10000, metavar='N',
                    help='evaluate interval')
parser.add_argument('--stdout', default=False, action='store_true', 
                    help='log in terminal')
parser.add_argument('--output_dir', type=str, default='exp/rnnt_lr1e-3',
                    help='path to save the final model')
parser.add_argument('--cuda', default=True, action='store_false')
parser.add_argument('--init', type=str, default='',
                    help='Initial am & pm parameters')
parser.add_argument('--initam', type=str, default='',
                    help='Initial am parameters')
parser.add_argument('--grad_clip', default=False, action='store_true')
parser.add_argument('--schedule', default=False, action='store_true')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, 'args'), 'w') as f:
    f.write(str(args))

if args.stdout: logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
else: logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', filename=os.path.join(args.output_dir, 'train.log'), level=logging.INFO)

random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)

# load dataset
trainset = DataHelperWithLessMem(args.train_file, args.batch_size, True)
devset = DataHelperWithLessMem(args.dev_file, args.batch_size, True)

# initialize model
model = Transducer(args.input_size, args.output_size, args.output_embed_size, 
                   args.hidden_size, args.ctc_layer_nums, 
                   args.pred_layer_nums, args.dropout, bidirectional=args.bi,
                   proj_size=args.proj_size)
criterion = RNNTLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

for param in model.parameters():
    torch.nn.init.uniform_(param, -0.1, 0.1)
if args.init: model.load_state_dict(torch.load(args.init))
if args.initam: model.encoder.load_state_dict(torch.load(args.initam))
if args.cuda: model.cuda()


def eval():
    model.eval()
    losses = []
    with torch.no_grad():
        for xs, ys, xlen, ylen in devset:
            xs = torch.IntTensor(xs).cuda()
            zero = torch.zeros((ys.shape[0], 1), dtype=torch.int).cuda()
            ys = torch.IntTensor(ys).cuda()
            yp = torch.cat((zero, ys), dim=1).cuda()
            xlen = torch.IntTensor(xlen).cuda()
            ylen = torch.IntTensor(ylen).cuda()
            out = model(xs, yp)
            loss = criterion(out, ys, xlen, ylen)
            loss = float(loss.data) * len(xlen)
            losses.append(loss)
    return sum(losses) / len(devset)


def train():
    def adjust_learning_rate(optimizer, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    def add_noise(x):
        dim = x.shape[-1]
        noise = torch.normal(torch.zeros(dim), 0.075)
        if x.is_cuda: noise = noise.cuda()
        x.data += noise

    prev_loss = 1000
    best_model = None
    lr = args.lr
    early_stop_cnt = 0
    for epoch in range(1, args.epochs):
        totloss = 0; losses = []
        start_time = time.time()
        for i, (xs, ys, xlen, ylen) in enumerate(trainset):
            xs = torch.IntTensor(xs).cuda()
            if args.noise: add_noise(xs)
            zero = torch.zeros((ys.shape[0], 1), dtype=torch.int).cuda()
            ys = torch.IntTensor(ys).cuda()
            yp = torch.cat((zero, ys), dim=1).cuda()
            xlen = torch.IntTensor(xlen).cuda()
            ylen = torch.IntTensor(ylen).cuda()
            model.train()
            optimizer.zero_grad()
            out = model(xs, yp)
            loss = criterion(out, ys, xlen, ylen)
            loss.backward()
            loss = float(loss.data) * len(xlen)
            totloss += loss; losses.append(loss)
            if args.grad_clip: grad_norm = nn.utils.clip_grad_norm(model.parameters(), 7)
            optimizer.step()


            if i and i % args.log_interval == 0:
                loss = totloss / args.batch_size / args.log_interval
                logging.info('[Epoch %d Batch %d] loss %.2f'%(epoch, i, loss))
                totloss = 0

            if i and i % args.log_evaluate == 0:
                trn_l = sum(losses) / args.batch_size / (i+1)
                val_l = eval()
                logging.info('[Epoch %d Batch %d] time cost %.2fs, train loss %.2f; cv loss %.2f; lr %.3e'%(
                    epoch, i, time.time()-start_time, trn_l, val_l, lr
                ))
                if val_l < prev_loss:
                    early_stop_cnt = 0
                    prev_loss = val_l
                    best_model = '{}/params_epoch{:02d}_bz{:02d}_tr{:.2f}_cv{:.2f}'.format(args.output_dir, epoch, i, trn_l, val_l)
                    torch.save(model.state_dict(), best_model)
                else:
                    early_stop_cnt += 1 
                    torch.save(model.state_dict(), '{}/params_epoch{:02d}_bz{:02d}_tr{:.2f}_cv{:.2f}_rejected'.format(args.output_dir, epoch, i, trn_l, val_l))
                    if early_stop_cnt >= 1000:
                        break

        if early_stop_cnt >= 1000:
            break

        losses = sum(losses) / len(trainset)
        val_l = eval()
        logging.info('[Epoch %d] time cost %.2fs, train loss %.2f; cv loss %.2f; lr %.3e'%(
            epoch, time.time()-start_time, losses, val_l, lr
        ))

        if val_l < prev_loss:
            early_stop_cnt = 0
            prev_loss = val_l
            if epoch and epoch % 10 == 0:
                best_model = '{}/params_epoch{:02d}_tr{:.2f}_cv{:.2f}'.format(args.output_dir, epoch, losses, val_l)
                torch.save(model.state_dict(), best_model)
        else:
            early_stop_cnt += 1 
            if epoch and epoch % 10 == 0:
                torch.save(model.state_dict(), '{}/params_epoch{:02d}_tr{:.2f}_cv{:.2f}_rejected'.format(args.output_dir, epoch, losses, val_l))
            #if early_stop_cnt >= 10:
            #    break
            #model.load_state_dict(torch.load(best_model))
            #if args.cuda: model.cuda()
            #if args.schedule:
            #    lr /= 2
            #    adjust_learning_rate(optimizer, lr)

if __name__ == '__main__':
    train()

