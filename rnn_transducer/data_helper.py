#coding=utf-8
import os
import sys

import numpy as np


np.random.seed(1024)


class DataHelper:
    def __init__(self, corpus_filename, batch_size=1, shuffle=False):
        self.inputs = []
        self.labels = []
        self.batch_size = batch_size
        self.shuffle = shuffle

        # load input and label
        with open(corpus_filename, 'r') as f:
            for line in f:
                info = line.strip('\r\n').split('\t')
                if len(info) != 2:
                    sys.stderr.write('bad sample format\n')
                    continue
                self.inputs.append([int(e) for e in info[0].split(' ')])
                self.labels.append([int(e) for e in info[1].split(' ')])

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        inputs = np.array(self.inputs)
        labels = np.array(self.labels)
        sample_size = len(self.labels)
        if self.shuffle:
            shuffle_indices = np.random.permutation(np.arange(sample_size))
            inputs = inputs[shuffle_indices] 
            labels = labels[shuffle_indices]

        start = 0
        while start + self.batch_size <= sample_size:
            end = start + self.batch_size
            yield self._format_batch(inputs[start:end], labels[start:end])
            start = end

    def _format_batch(self, input_batch, label_batch):
        xs, xlens = self._padding(input_batch, 0) 
        ys, ylens = self._padding(label_batch, 1)
        return xs, ys, xlens, ylens

    def _padding(self, input_batch, padding_idx):
        lens = [len(i) for i in input_batch]
        max_len = max(lens)
        outs = np.full((len(input_batch), max_len), fill_value=padding_idx, dtype=np.int32)
        for idx, inp in enumerate(input_batch):
            outs[idx, :len(inp)] = inp

        return outs, lens


class DataHelperWithLessMem:
    def __init__(self, corpus_filename, batch_size=1, shuffle=False):
        self.corpus_filename = corpus_filename
        self.shuf_corpus_filename = corpus_filename + '.shuf.' + str(os.getpid())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuf_command = 'shuf {} -o {}'.format(self.corpus_filename, self.shuf_corpus_filename)

    def __len__(self):
        lines = os.popen('wc -l {}'.format(self.corpus_filename)).read().split(' ')[0]
        return int(lines)

    def __iter__(self):
        corpus_filename = self.corpus_filename
        if self.shuffle:
            corpus_filename = self.shuf_corpus_filename
            os.system(self.shuf_command)

        inputs = []
        labels = []
        with open(corpus_filename, 'r') as f:
            for line in f:
                info = line.strip('\r\n').split('\t')
                if len(info) != 2:
                    sys.stderr.write('bad sample format\n')
                    continue
                inputs.append([int(e) for e in info[0].split(' ')])
                labels.append([int(e) for e in info[1].split(' ')])

                if len(inputs) == self.batch_size:
                    yield self._format_batch(inputs, labels)
                    inputs = []
                    labels = []

    def _format_batch(self, input_batch, label_batch):
        xs, xlens = self._padding(input_batch, 0) 
        ys, ylens = self._padding(label_batch, 1)
        return xs, ys, xlens, ylens

    def _padding(self, input_batch, padding_idx):
        lens = [len(i) for i in input_batch]
        max_len = max(lens)
        outs = np.full((len(input_batch), max_len), fill_value=padding_idx, dtype=np.int32)
        for idx, inp in enumerate(input_batch):
            outs[idx, :len(inp)] = inp

        return outs, lens

if __name__ == '__main__':
    data_set = DataHelperWithLessMem(sys.argv[1], 3, True)
    print(len(data_set))
    i = 0
    for ele in data_set: 
        print(ele)
        if i == 10: break
        i += 1


