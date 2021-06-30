#coding: utf-8
import math

import torch
from torch import nn
import torch.nn.functional as F


def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a-b)))


class Sequence():
    def __init__(self, seq=None, blank=0):
        if seq is None:
            self.g = [] # predictions of char language model
            self.k = [blank] # prediction char label
            self.h = None
            self.logp = 0 # probability of this sequence, in log scale
        else:
            self.g = seq.g[:] # save for prefixsum
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp

    def __str__(self):
        return 'Prediction: {}\nlog-likelihood {:.2f}\n'.format(' '.join([str(i) for i in self.k]), -self.logp)


class EncoderLayer(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_size, 
                 num_layers, 
                 dropout=.2, 
                 blank=0, 
                 bidirectional=False,
                 proj_size=0):
        super(EncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.blank = blank

        # lstm hidden vector: (h_0, c_0) num_layers * num_directions, batch, hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=(dropout if num_layers > 1 else 0), bidirectional=bidirectional, proj_size=proj_size)
        if proj_size: hidden_size = proj_size
        if bidirectional: hidden_size *= 2
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, xs, hid=None):
        h, hid = self.lstm(xs, hid)
        return self.linear(h), hid


class JointLayer(nn.Module):
    def __init__(self,
                 input_size, 
                 hidden_size,
                 output_size,
                 dropout=0.5):
        super(JointLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, f, g):
        '''
        `f`: encoder lstm output (B,T,U,H)
        `g`: predictor lstm output (B,T,U,H)
        NOTE f and g must have the same size except the last dim
        '''
        out = torch.cat((f, g), dim=-1)
        out = self.dropout(out)
        out = torch.tanh(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


class Transducer(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 output_embed_size, 
                 hidden_size, 
                 ctc_num_layers, 
                 pred_num_layers, 
                 dropout=.5, 
                 blank=0, 
                 bidirectional=False,
                 proj_size=0):
        super(Transducer, self).__init__()
        self.blank = blank
        self.output_size = output_size
        self.hidden_size = hidden_size

        # NOTE encoder & predict only use lstm
        self.input_embed = nn.Embedding(input_size, input_size)
        self.input_embed.weight.data = torch.eye(input_size, input_size)
        self.input_embed.weight.requires_grad = False

        self.encoder = EncoderLayer(input_size, proj_size if proj_size else hidden_size, hidden_size, ctc_num_layers, dropout, bidirectional=bidirectional, proj_size=proj_size)

        #self.output_embed = nn.Embedding(output_size, output_embed_size)
        self.output_embed = nn.Embedding(output_size, output_embed_size)
        self.output_embed.weight.data[1:] = torch.eye(output_size-1, output_embed_size)
        self.output_embed.weight.requires_grad = False
        self.predict = nn.LSTM(output_embed_size, hidden_size, pred_num_layers, batch_first=True, dropout=(dropout if pred_num_layers > 1 else 0), proj_size=proj_size)

        if proj_size: hidden_size = proj_size
        self.joint = JointLayer(2*hidden_size, hidden_size, output_size, dropout)

    def forward(self, xs, ys):
        # encode input sequece
        xe = self.input_embed(xs)
        xe, _ = self.encoder(xe)

        # predict output sequence
        yp = self.output_embed(ys)
        yp, _ = self.predict(yp)

        # prepare input for joint
        xe = xe.unsqueeze(dim=2) # (B, T, H) -> (B, T, 1, H) 
        yp = yp.unsqueeze(dim=1) # (B, U, H) -> (B, 1, U, H)
        xe = xe.expand([-1, -1, yp.shape[2], -1]) # (B, T, 1, H) -> (B, T, U, H)
        yp = yp.expand([-1, xe.shape[1], -1, -1]) # (B, 1, U, H) -> (B, T, U, H)

        # joint encode and predict represation
        out = self.joint(xe, yp)
        return out

    def greedy_decode(self, x):
        x = self.input_embed(x)
        x = self.encoder(x)[0][0]
        vy = torch.IntTensor([0]).view(1,1) # vector preserve for embedding
        if x.is_cuda: vy = vy.cuda()
        y, h = self.predict(self.output_embed(vy)) # decode first zero 
        y_seq = []; logp = 0
        for i in x:
            ytu = self.joint(i, y[0][0])
            out = F.log_softmax(ytu, dim=0)
            p, pred = torch.max(out, dim=0) # suppose blank = -1
            pred = int(pred)
            logp += float(p)
            if pred != self.blank:
                y_seq.append(pred)
                vy.data[0][0] = pred # change pm state
                y, h = self.predict(self.output_embed(vy), h)
        return [(y_seq, -logp)]


    def beam_search(self, xs, W=10, prefix=False):
        '''''
        `xs`: acoustic model outputs
        NOTE only support one sequence (batch size = 1)
        '''''
        use_gpu = xs.is_cuda
        def forward_step(label, hidden):
            ''' `label`: int '''
            label = torch.IntTensor([label]).view(1,1)
            if use_gpu: label = label.cuda()
            label = self.output_embed(label)
            pred, hidden = self.predict(label, hidden)
            return pred[0][0], hidden

        def isprefix(a, b):
            # a is the prefix of b
            if a == b or len(a) >= len(b): return False
            for i in range(len(a)):
                if a[i] != b[i]: return False
            return True

        xs = self.input_embed(xs)
        xs = self.encoder(xs)[0][0]
        B = [Sequence(blank=self.blank)]
        for i, x in enumerate(xs):
            B = sorted(B, key=lambda a: len(a.k), reverse=True) # larger sequence first add
            A = B
            B = []
            if prefix:
                for j in range(len(A)-1):
                    for i in range(j+1, len(A)):
                        if not isprefix(A[i].k, A[j].k): continue
                        # A[i] -> A[j]
                        pred, _ = forward_step(A[i].k[-1], A[i].h)
                        idx = len(A[i].k)
                        ytu = self.joint(x, pred)
                        logp = F.log_softmax(ytu, dim=0)
                        curlogp = A[i].logp + float(logp[A[j].k[idx]])
                        for k in range(idx, len(A[j].k)-1):
                            ytu = self.joint(x, A[j].g[k])
                            logp = F.log_softmax(ytu, dim=0)
                            curlogp += float(logp[A[j].k[k+1]])
                        A[j].logp = log_aplusb(A[j].logp, curlogp)

            while True:
                y_hat = max(A, key=lambda a: a.logp)
                # y* = most probable in A
                A.remove(y_hat)
                # calculate P(k|y_hat, t)
                # get last label and hidden state
                pred, hidden = forward_step(y_hat.k[-1], y_hat.h)
                ytu = self.joint(x, pred)
                logp = F.log_softmax(ytu, dim=0) # log probability for each k
                # TODO only use topk vocab
                for k in range(self.output_size):
                    yk = Sequence(y_hat)
                    yk.logp += float(logp[k])
                    if k == self.blank:
                        B.append(yk) # next move
                        continue
                    # store prediction distribution and last hidden state
                    yk.h = hidden; yk.k.append(k); 
                    if prefix: yk.g.append(pred)
                    A.append(yk)
                # sort A
                # A = sorted(A, key=lambda a: a.logp, reverse=True) # just need to calculate maximum seq
                
                # sort B
                # B = sorted(B, key=lambda a: a.logp, reverse=True)
                y_hat = max(A, key=lambda a: a.logp)
                yb = max(B, key=lambda a: a.logp)
                if len(B) >= W and yb.logp >= y_hat.logp: break

            # beam width
            B = sorted(B, key=lambda a: a.logp, reverse=True)
            B = B[:W]

        # return top W highest probability sequence
        return [(B[i].k, -B[i].logp) for i in range(len(B))]


