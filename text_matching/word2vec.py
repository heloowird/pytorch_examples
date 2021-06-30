#!/usr/bin/env python
#coding:utf-8
#author:zhujianqi

from __future__ import print_function
import sys

from gensim.models import Word2Vec

from util import log_error
from corpus import Word2VecCorpus


def train(corpus_filenames, model_filename):
    sentences = Word2VecCorpus(filenames)
    model = Word2Vec(sentences=sentences, size=600, window=5, min_count=4, workers=4, sg=1, seed=17)
    model.save(model_filename)
    model.wv.save_word2vec_format(model_filename + '.wv')


def test(model_filename):
    model = Word2Vec.load(model_filename)
    while True:
        word = input('input word: ')

        if word.lower() in ['exit', 'quit', 'q']:
            break

        if word not in model.wv:
            log_error('{} out of model'.format(word))
            continue

        print(model.wv[word])
        print(model.wv.most_similar(word))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        log_error('Bad args, missing runtime type or model path')
        sys.exit(-1)

    model_filename = sys.argv[2]
    if sys.argv[1] == 'train' and len(sys.argv) > 3:
        filenames = sys.argv[3:]
        train(filenames, model_filename)
    elif sys.argv[1] == 'test':
        test(model_filename)
    else:
        log_error('bad runtime type')
        sys.exit(-1)


