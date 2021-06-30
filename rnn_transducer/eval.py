#coding: utf-8
import argparse
import logging
import math
import os
import time

import editdistance
import numpy as np
import torch

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
parser.add_argument('--output_dir', type=str, default='', help='decoded result output dir')
args = parser.parse_args()

log_filename = args.output_dir if args.output_dir else os.path.dirname(args.model) + '/eval_' + os.path.basename(args.model) + '.log'
if args.output_dir: os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt="%H:%M:%S", filename=log_filename, level=logging.INFO)

test_set = DataHelper(args.input_file, 1)

# Load model
model = Transducer(args.input_size, args.output_size, args.output_embed_size, 
                   args.hidden_size, args.ctc_layer_nums, 
                   args.pred_layer_nums, bidirectional=args.bi, 
                   proj_size=args.proj_size)
model.load_state_dict(torch.load(args.model, map_location='cpu'))

use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()
model.eval()

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


def evaluate():
    logging.info('start testing rnn-t model:')
    err = cnt = 0
    k = 0
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
        y = [str(i) for i in y if i != 0]
        t = [str(i) for i in ys[0]]
        y, t, e = distance(y, t)
        err += e
        cnt += len(t)
        timecost = time.time() - start
        logging.info('[{} input]: {}'.format(k, ' '.join(w)))
        logging.info('[{} timecost]: {}'.format(k, timecost))
        logging.info('[{} groudth]: {}'.format(k, ' '.join(t)))
        logging.info('[{} predict]: {}\nlog-likelihood: {:.2f}\n'.format(k, ' '.join(y), nll))
        print('{}\t{}'.format(' '.join(w), ' '.join(y)))
        k += 1
    logging.info('{} set {} CER {:.2f}%\n'.format(
        args.dataset.capitalize(), 'Transducer', 100*err/cnt))


if __name__ == '__main__':
    evaluate()


