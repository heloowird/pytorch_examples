#!/bin/bash

train_epoches=10
steps_every_print=100
steps_every_evaluate=1000
steps_every_save=100000
summary_dir='./text_classification/log/pretrained_lm'
save_model_dir='./text_classification/model/pretrained_lm'
load_model=''
vocab='../data/addr_name_20160101-20180601_large_w2v.voc'
pretrain_em='../data/addr_name_20160101-20180601_large_w2v.npy'
train='../data/train.txt'
train_pre_batchs_num=100000
valid='../data/valid.txt'
valid_pre_batchs_num=10000

python lm.py -cuda -train_epoches=$train_epoches -steps_every_print=$steps_every_print -steps_every_evaluate=$steps_every_evaluate -steps_every_save=$steps_every_save -summary_dir=$summary_dir -save_model_dir=$save_model_dir -load_model=$load_model -vocab=$vocab -pretrain_em=$pretrain_em -train=$train -train_pre_batchs_num=$train_pre_batchs_num -valid=$valid -valid_pre_batchs_num=$valid_pre_batchs_num 
