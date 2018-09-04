import os
import sys
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import model
import argparse
import time
import math
from tensorboardX import SummaryWriter

import util

parser = argparse.ArgumentParser(description='lm.py')

parser.add_argument('-train_epoches', type=int, default=10,
                    help='Number of epoch to train')
parser.add_argument('-steps_every_print', type=int, default=10,
                    help='Number of step to print training info')
parser.add_argument('-steps_every_evaluate', type=int, default=100,
                    help='Number of step to evaluate model')
parser.add_argument('-steps_every_save', type=int, default=10000,
                    help='Number of step to save model')
parser.add_argument('-summary_dir', default='lm',
                    help="""Model directory to save""")
parser.add_argument('-save_model_dir', default='model',
                    help="""Model directory to save""")
parser.add_argument('-load_model', default='',
                    help="""Model filename to load""")
parser.add_argument('-vocab', default='../data/addr_name_20160101-20180601_large_w2v.voc',
                    help="""Vocabulary filename for 
                    training and validation""")
parser.add_argument('-pretrain_em', default='',
                    help="""Pretained word embedding""")
parser.add_argument('-train', default='../data/train.txt',
                    help="""Text filename for training""")
parser.add_argument('-train_pre_batchs_num', type=int, default=100000,
                    help="""Number of batches to read inte memory,
                    avoid reading full corpus and cause OOM""")
parser.add_argument('-valid', default='../data/valid.txt',
                    help="""Text filename for validation""")                    
parser.add_argument('-valid_pre_batchs_num', type=int, default=10000,
                    help="""Number of batches to read inte memory,
                    avoid reading full corpus and cause OOM""")
parser.add_argument('-rnn_type', default='mlstm',
                    help='mlstm, lstm or gru')
parser.add_argument('-layers', type=int, default=1,
                    help='Number of layers in the rnn')
parser.add_argument('-rnn_size', type=int, default=4096,
                    help='Size of hidden states')
parser.add_argument('-embed_size', type=int, default=300,
                    help='Size of embeddings')
parser.add_argument('-seq_length', type=int, default=20,
                    help="Maximum sequence length")
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-learning_rate', type=float, default=0.001,
                    help="""Starting learning rate.""")
parser.add_argument('-dropout', type=float, default=0.1,
                    help='Dropout probability.')
parser.add_argument('-param_init', type=float, default=0.05,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-clip', type=float, default=5,
                    help="""Clip gradients at this value.""")
parser.add_argument('-seed', type=int, default=1234,
                    help='random seed')   
# GPU
parser.add_argument('-cuda', action='store_true',
                    help="Use CUDA")

opt = parser.parse_args()    

print('the hyper parameters are below:')
param_dict = vars(opt)
for ele in param_dict.keys():
    print("\t%s=%s" % (ele, param_dict[ele]))

learning_rate = opt.learning_rate
batch_size = opt.batch_size
hidden_size =opt.rnn_size
input_size = opt.embed_size
TIMESTEPS = opt.seq_length

vocab_dict = util.Dictionary(opt.vocab)
vocab_size = len(vocab_dict)
print('\tvocab_size=%d' % vocab_size)
print('*' * 20)

writer = SummaryWriter(opt.summary_dir)

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

def create_model():
    if opt.rnn_type == 'gru':
        rnn = model.StackedRNN(nn.GRUCell, opt.layers, input_size, hidden_size, vocab_size, opt.dropout)
    elif opt.rnn_type == 'mlstm':
        rnn = model.StackedLSTM(model.mLSTM, opt.layers, input_size, hidden_size, vocab_size, opt.dropout)
    else:#default to lstm
        rnn = model.StackedLSTM(nn.LSTMCell, opt.layers, input_size, hidden_size, vocab_size, opt.dropout)
    return rnn

if len(opt.load_model) > 0:
    print('load saved model')
    checkpoint = torch.load(opt.load_model)
    embed = checkpoint['embed']
    rnn = checkpoint['rnn']
elif len(opt.pretrain_em) > 0:
    print('load pretrained embedding')
    embed = nn.Embedding.from_pretrained(torch.from_numpy(np.load(opt.pretrain_em)).float(), False)
    print('the shpae of pretrained embedding: (%d, %d)' % (embed.weight.size(0), embed.weight.size(1)))
    rnn = create_model()
else:
    print('using random embedding')
    embed = nn.Embedding(vocab_size, input_size)
    rnn = create_model()

loss_fn = nn.CrossEntropyLoss() 

nParams = sum([p.nelement() for p in rnn.parameters()])
print('the number of parameters: %d' % nParams)

embed_optimizer = optim.SGD(embed.parameters(), lr=learning_rate)
rnn_optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
   
trains = util.Corpus(opt.train, vocab_dict, TIMESTEPS, batch_size, opt.train_pre_batchs_num, opt.train_epoches).gen_batch()
valids = util.Corpus(opt.valid, vocab_dict, TIMESTEPS, batch_size, opt.valid_pre_batchs_num, opt.train_epoches*1000).gen_batch()
print("load data finish")
sys.stdout.flush()

def update_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
    return
    
def clip_gradient_coeff(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))

def calc_grad_norm(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    return math.sqrt(totalnorm)
    
def calc_grad_norms(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    norms = []
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        norms += [modulenorm]
    return norms
    
def clip_gradient(model, clip):
    """Clip the gradient."""
    totalnorm = 0
    for p in model.parameters():
        p.grad.data = p.grad.data.clamp(-clip,clip)

        
def make_cuda(state):
    if isinstance(state, tuple):
        return (state[0].cuda(), state[1].cuda())
    else:
        return state.cuda()
        
def copy_state(state):
    if isinstance(state, tuple):
        return (Variable(state[0].data), Variable(state[1].data))
    else:
        return Variable(state.data)     

def init_hidden():
    hidden_init = rnn.state0(opt.batch_size)            
    if opt.cuda:
        embed.cuda()
        rnn.cuda()
        hidden_init = make_cuda(hidden_init)
    return hidden_init

def evaluate(epoch, step):
    hidden_init = init_hidden()

    total_batch = 0
    avg_loss = 0.0
    start = time.time()
    for i in range(10):
        batch, _ = next(valids)

        hidden = hidden_init
        if opt.cuda:
            batch = batch.cuda()

        loss = 0
        for t in range(TIMESTEPS):                  
            emb = embed(batch[t])
            hidden, output = rnn(emb, hidden)
            loss += loss_fn(output, batch[t+1])

        hidden_init = copy_state(hidden)
        total_batch = total_batch + 1
        cur_loss = loss.item()/TIMESTEPS
        avg_loss = avg_loss + cur_loss
    print('test: epoch=%s global_step=%s cur_loss=%.4f avg_loss=%.4f time_cost=%.4f' % (epoch, step, cur_loss, avg_loss/total_batch, time.time()-start))
    writer.add_scalar('v_cur_loss', cur_loss, step)
    writer.add_scalar('v_avg_loss', avg_loss/total_batch, step)

def save_model(epoch, step, avg_loss):
    if not os.path.exists(opt.save_model_dir): 
        os.makedirs(opt.save_model_dir)

    checkpoint = {
            'rnn': rnn,
            'embed': embed,
            'opt': opt,
            'epoch': epoch
    }
    save_file = ('%s/lm_e%s_s%s_%.2f.pt' % (opt.save_model_dir, epoch, step, avg_loss))
    torch.save(checkpoint, save_file)
    print('Saving to '+ save_file)

def train():
    hidden_init = init_hidden()
    avg_loss = 0.0
    cur_epoch = 1
    global_step = 0
    for batch, epoch in trains:
        if cur_epoch != epoch:
            hidden_init = init_hidden()
            avg_loss = 0

            learning_rate *= 0.7
            update_lr(rnn_optimizer, learning_rate)
            update_lr(embed_optimizer, learning_rate)

            cur_epoch = epoch

        embed_optimizer.zero_grad()
        rnn_optimizer.zero_grad()
        start = time.time()
        hidden = hidden_init
        if opt.cuda:
            batch = batch.cuda()
        loss = 0
        for t in range(TIMESTEPS):                  
            emb = embed(batch[t])
            hidden, output = rnn(emb, hidden)
            loss += loss_fn(output, batch[t+1])
        
        loss.backward()
        
        hidden_init = copy_state(hidden)
        gn =calc_grad_norm(rnn)
        clip_gradient(rnn, opt.clip)
        clip_gradient(embed, opt.clip)
        embed_optimizer.step()
        rnn_optimizer.step()
        cur_loss = loss.item()/TIMESTEPS
        if global_step == 0:
            avg_loss = cur_loss
        else:
            avg_loss = .99*avg_loss + .01*cur_loss
        global_step = global_step + 1

        if global_step % opt.steps_every_print== 0:
            print('train: epoch=%s global_step=%s cur_loss=%.4f avg_loss=%.4f time_cost=%.4f grad_norm=%.4f' % (epoch, global_step, cur_loss, avg_loss, time.time()-start, gn))
            writer.add_scalar('cur_loss', cur_loss, global_step)
            writer.add_scalar('avg_loss', avg_loss, global_step)
        if global_step % opt.steps_every_evaluate == 0:
            evaluate(epoch, global_step)
        if global_step % opt.steps_every_save == 0:
            save_model(epoch, global_step, avg_loss)

try:
    train()
except KeyboardInterrupt:
    print('Exiting from training early')
except StopIteration:
    print('finish all epoch')
