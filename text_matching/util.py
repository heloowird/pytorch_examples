#!/usr/bin/env python #coding:utf-8
#author:zhujianqi

from __future__ import print_function
import sys
import time

import torch
import numpy as np

from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score


def log_error(info):
    sys.stderr.write('{}\n'.format(info))


def load_word_embedding(model_filename, offset=2):
    model = Word2Vec.load(model_filename)

    word2id = dict([(w, model.wv.vocab[w].index + offset) for w in model.wv.vocab.keys()])

    embedding_dim = model.wv[list(word2id.keys())[0]].shape[0]
    num_words = len(word2id) + offset

    embeddings = np.zeros((num_words, embedding_dim))
    for word, index in word2id.items(): 
        embeddings[index] = np.array(model.wv[word], dtype=float)

    return word2id, embeddings


def train(model,
          dataloader,
          criterion,
          optimizer,
          epoch_num,
          max_gradient_norm=5.0):
    model.train()
    device = model.device

    total_elapsed = 0.0
    total_acc, total_count = 0.0, 0
    total_loss = 0.0
    y_true, y_score = [], []

    log_interval = 100
    epoch_start = time.time()
    for idx, batch in enumerate(dataloader):
        batch_start = time.time()

        text1 = batch['premise'].to(device)
        text1_len = batch['premise_length'].to(device)
        text2 = batch['hypothesis'].to(device)
        text2_len = batch['hypothesis_length'].to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()

        logit, prob = model(text1, text1_len, text2, text2_len)
        loss = criterion(logit, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        total_elapsed = time.time() - batch_start

        total_loss += loss.item()
        total_count += label.size(0)

        prob = prob[:,1]
        total_acc += ((prob > 0.5) == label).sum().item()

        y_true.extend(torch.flatten(label).tolist())
        y_score.extend(torch.flatten(prob).tolist())

        if idx % log_interval == 0 and idx > 0:
            score = roc_auc_score(y_true, y_score)
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                    '| loss {:6.3f} | auc {:6.3f} | acc {:6.3f}'.format(epoch_num, idx, len(dataloader), total_loss/idx+1, score, total_acc/total_count))

    batch_time = time.time() - epoch_start
    batch_loss = total_loss / len(dataloader)
    batch_auc = roc_auc_score(y_true, y_score)
    batch_acc = total_acc / total_count

    return batch_time, batch_loss, batch_auc, batch_acc


def evaluate(model,
             dataloader,
             criterion):
    model.eval()
    device = model.device

    total_acc, total_count = 0.0, 0
    total_loss = 0.0
    y_true, y_score = [], []

    epoch_start = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            text1 = batch['premise'].to(device)
            text1_len = batch['premise_length'].to(device)
            text2 = batch['hypothesis'].to(device)
            text2_len = batch['hypothesis_length'].to(device)
            label = batch['label'].to(device)

            logit, prob = model(text1, text1_len, text2, text2_len)
            loss = criterion(logit, label)

            total_loss += loss.item()

            prob = prob[:,1]
            total_acc += ((prob > 0.5) == label).sum().item()
            total_count += label.size(0)

            y_true.extend(torch.flatten(label).tolist())
            y_score.extend(torch.flatten(prob).tolist())

    batch_time = time.time() - epoch_start
    batch_loss = total_loss / len(dataloader)
    batch_auc = roc_auc_score(y_true, y_score)
    batch_acc = total_acc / total_count

    return batch_time, batch_loss, batch_auc, batch_acc


if __name__ == "__main__":
    pretrained_model_filename = '../user_data/model/word2vec.sg.bin'
    _, word2id = load_word_embedding(pretrained_model_filename)
    print(word2id)


