#!/usr/bin/env python
#coding:utf-8
#author:zhujianqi

from __future__ import print_function
import sys
from enum import Enum

import torch
from torch.utils.data import Dataset

from util import log_error


class CorpusType(Enum):
    WORD2VEC = 1
    TEXT_PAIR = 2


class TextPair():
    def __init__(self, query1, query2, label):
        self.query1 = query1
        self.query2 = query2
        self.label  = label

    def get_query(self, which_query=1):
        return self.query1 if which_query == 1 else self.query2

    def get_label(self):
        return self.label

    def __str__(self):
        return 'query1=[{}], query2=[{}], label={}'.format(
                ' '.join(self.query1), ' '.join(self.query2), self.label)


def parse_data(filename, corpus_type):
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip('\r\n')
            info = line.split('\t')
            if len(info) < 2:
                log_error('bad line: ' + line)
                continue

            text1, text2 = info[0:2]
            query1, query2 = text1.split(' '), text2.split(' ')
            if corpus_type == CorpusType.WORD2VEC:
                yield query1
                yield query2
            elif corpus_type == CorpusType.TEXT_PAIR:
                label = int(info[2]) if len(info) == 3 else -1
                #yield TextPair(query1, query2, label)
                yield (query1, query2, label)
            else:
                raise ValueError('bad corpus type')


class Corpus():
    def __init__(self, filenames, corpus_type, parse_func):
        if type(filenames) is list:
            self.filenames = filenames
        else:
            self.filenames = [filenames]

        self.corpus_type = corpus_type
        self.parse_data = parse_func

    def __iter__(self):
        for filename in self.filenames:
            for ele in self.parse_data(filename, self.corpus_type):
                yield ele


class Word2VecCorpus(Corpus):
    def __init__(self, filenames, parse_func=parse_data):
        super(Word2VecCorpus, self).__init__(filenames, CorpusType.WORD2VEC, parse_func)


class TextPairCorpus(Corpus):
    def __init__(self, filenames, parse_func=parse_data):
        super(TextPairCorpus, self).__init__(filenames, CorpusType.TEXT_PAIR, parse_func)


class TextPairDataset(Dataset): 
    def __init__(self, 
                 data_filename, 
                 word2id=None,
                 oov_idx=0,
                 padding_idx=1,
                 max_premise_length=None,
                 max_hypothesis_length=None):

        self.max_premise_length = max_premise_length
        self.max_hypothesis_length = max_hypothesis_length

        premises_data = []
        hypotheses_data = []
        label_data = []

        text_pair_corpus = TextPairCorpus(data_filename)
        for ele in text_pair_corpus:
            premises_data.append(self.text_to_id(ele[0], word2id, oov_idx))
            hypotheses_data.append(self.text_to_id(ele[1], word2id, oov_idx))
            label_data.append(int(ele[2]))

        self.premises_lengths = [len(seq) for seq in premises_data]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in hypotheses_data]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(premises_data)

        self.data = {"ids": [],
                     "premises": torch.ones((self.num_sequences, self.max_premise_length), dtype=torch.long) * padding_idx,
                     "hypotheses": torch.ones((self.num_sequences, self.max_hypothesis_length), dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(label_data, dtype=torch.long)}
        
        for i, premise in enumerate(premises_data):
            self.data["ids"].append(i)
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])
            hypothesis = hypotheses_data[i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])

    def text_to_id(self, text, word2id, oov_id):
        return [word2id[w] if w in word2id else oov_id for w in text]

    def __len__(self):
        return self.num_sequences
        
    def __getitem__(self, index):
        return {"id": self.data["ids"][index],
                "premise": self.data["premises"][index],
                "premise_length": min(self.premises_lengths[index], self.max_premise_length),
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.hypotheses_lengths[index], self.max_hypothesis_length),
                "label": self.data["labels"][index]}


if __name__ == '__main__':
    corpus = Word2VecCorpus(sys.argv[1])
    #corpus = TextPairCorpus(sys.argv[1])
    for ele in corpus:
        print(ele)


