#!/usr/bin/env python
#coding:utf-8
#author:zhujianqi

from __future__ import print_function
import os
import sys
import argparse
import json

import torch
from torch.utils.data import DataLoader

from corpus import TextPairDataset
from util import load_word_embedding
from linear_model import LinearModel
from esim.model import ESIM


def predict(model, dataloader):
    model.eval()
    device = model.device

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            text1 = batch['premise'].to(device)
            text1_len = batch['premise_length'].to(device)
            text2 = batch['hypothesis'].to(device)
            text2_len = batch['hypothesis_length'].to(device)
            label = batch['label'].to(device)

            _, prob = model(text1,
                            text1_len,
                            text2,
                            text2_len)

            prob = prob[:,1]
            predictions.extend(torch.flatten(prob).tolist())

    return predictions


def main(test_file,
         output_dir,
         model_file, 
         embedding_file,
         hidden_size,
         num_classes,
         batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(10 * '-', ' Preparing for testing ', 10 * '-')

    output_dir = os.path.normpath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\t* Loading test data...")
    word2id, embeddings = load_word_embedding(embedding_file)
    vocab_size = embeddings.shape[0]
    embed_dim = embeddings.shape[1]
    test_dataset = TextPairDataset(test_file, word2id, max_premise_length=22, max_hypothesis_length=22)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    print("\t* Building model...")
    model = ESIM(vocab_size,
                        embed_dim,
                        hidden_size,
                        num_classes=num_classes,
                        device=device).to(device)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model'])

    print("\t* Predicting prob...")
    predictions = predict(model, test_dataloader)

    print("\t* Saving result...")
    with open(os.path.join(output_dir, "result.csv"), 'w') as output_f:
        for ele in predictions:
            output_f.write('{}\n'.format(ele))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        help='Path to a checkpoint with a trained model')
    parser.add_argument('--config', default='config/esim_model.json',
                        help='Path to a configuration file')
    parser.add_argument('--test_file', default='../tcdata/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv',
                        help='Path to the test data')
    parser.add_argument('--output_dir', default='../prediction_result/esim_result',
                        help='Path to the predicted result')
    args = parser.parse_args()

    with open(os.path.normpath(args.config), 'r') as config_file:
        config = json.load(config_file)

    main(args.test_file,
         args.output_dir,
         args.checkpoint,
         config['embeddings'],
         config['hidden_size'],
         config['num_classes'])


