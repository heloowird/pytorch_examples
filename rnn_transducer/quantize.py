#coding: utf-8
import argparse
import logging
import math
import os
import time

import editdistance
import numpy as np
import torch
from torch.quantization import quantize_dynamic

from model import Transducer
from data_helper import DataHelper

parser = argparse.ArgumentParser(description='RNN Transducer Model.')
parser.add_argument('model', help='trained model filename')
parser.add_argument('--beam', type=int, default=0, help='apply beam search, beam width')
parser.add_argument('--input_size', type=int, default=38, metavar='N',
                    help='input size')
parser.add_argument('--output_size', type=int, default=13000, metavar='N',
                    help='output size')
parser.add_argument('--output_embed_size', type=int, default=300, metavar='N',
                    help='output embedding size')
parser.add_argument('--hidden_size', type=int, default=250, 
                    help='hidden size for rnn')
parser.add_argument('--proj_size', type=int, default=0, 
                    help='porjection size for rnn')
parser.add_argument('--ctc_layer_nums', type=int, default=3, 
                    help='layer nums of ctc network')
parser.add_argument('--pred_layer_nums', type=int, default=2, 
                    help='layer nums of predict network')
parser.add_argument('--bi', default=False, action='store_true', help='bidirectional LSTM')
parser.add_argument('--dataset', default='test', help='decoding data set')
parser.add_argument('--input_file', type=str, default='test_data/dev', help='input prefix')
parser.add_argument('--output_dir', type=str, default='', help='eval result output dir')
parser.add_argument('--quantized_model', type=str, default='', help='quantized model file')
args = parser.parse_args()

log_filename = args.output_dir if args.output_dir else os.path.dirname(args.model) + '/eval_' + os.path.basename(args.model) + '.log'
if args.output_dir: os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt="%H:%M:%S", filename=log_filename, level=logging.INFO)

test_set = DataHelper(args.input_file, 1)

# Load model
raw_model = Transducer(args.input_size, args.output_size, args.output_embed_size, 
                   args.hidden_size, args.ctc_layer_nums, 
                   args.pred_layer_nums, bidirectional=args.bi, 
                   proj_size=args.proj_size)
raw_model.load_state_dict(torch.load(args.model, map_location='cpu'))

# Do dynamic quantization for RNN-T model
quantized_model = quantize_dynamic(raw_model)

# Save quantized model
torch.save(quantized_model.state_dict())

torch.set_num_threads(1)

def distance(y, t, blank='<eps>', need_remap=False):
    def remap(y, blank):
        prev = blank
        seq = []
        for i in y:
            if i != blank and i != prev: seq.append(i)
            prev = i
        return seq
    y = remap(y, blank) if need_remap else y
    t = remap(t, blank) if need_remap else t
    return y, t, editdistance.eval(y, t)


def evaluate(model):
    model.eval()

    logging.info('start eval rnn-t model:')
    tot_sent_cnt = tot_sent_err_cnt = 0
    tot_char_cnt = tot_char_err_cnt = 0
    tot_timecost = 0.0
    for xs, ys, xlen, ylen in test_set:
        start = time.time()
        w = [str(i) for i in xs[0]]
        x = torch.LongTensor(xs)
        if use_gpu:
            x = x.cuda()
        if args.beam > 0:
            y, nll = model.beam_search(x, args.beam)[0]
        else:
            y, nll = model.greedy_decode(x)[0]
        tot_timecost += (time.time() - start)
        y = [str(i) for i in y if i != 0]
        t = [str(i) for i in ys[0]]
        y, t, e = distance(y, t)
        tot_char_err_cnt += e
        tot_char_cnt += len(t)
        if ' '.join(y) != ' '.join(t): tot_sen_err_cnt += 1
        tot_sent_cnt += 1
    logging.info('CER {:.2f}%, Top1-Acc {:.2f}%, Avg-Timecost {:.2f}\n'.format(err/cnt*100, (1-tot_sent_err_cnt/tot_sent_cnt)*100, tot_timecost/tot_sent_cnt))


evaluate(raw_model)
evaluate(quantized_model)


