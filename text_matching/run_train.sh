#!/bin/bash

set -exu

echo "step0: train word2vec model"
python word2vec.py train ../user_data/model/word2vec/word2vec.sg.600.bin ../tcdata/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv ../tcdata/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv
