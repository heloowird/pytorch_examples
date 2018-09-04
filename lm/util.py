import os
import torch
from torch.autograd import Variable

class Dictionary(object):
    def __init__(self, filename):
        self.word2idx = {}
        self.idx2word = []
        self._build_dict(filename)

    def _build_dict(self, filename):
        with open(filename, encoding='utf-8') as f:
            for line in f:
                word = line.split(' ')[0]
                if word not in self.word2idx:
                    self.idx2word.append(word)
                    self.word2idx[word] = len(self.idx2word) - 1
                else:
                    raise Exception('duplicate word(%s) in dict'.format(word))

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, dict_, ssz, bsz, pbn, epoch=None):
        self.dictionary = dict_
        self.seq_size = ssz
        self.batch_size = bsz
        self.pre_batchs_num = pbn
        self.epoch = epoch
        self.pre_words_num = self.batch_size * self.seq_size * self.pre_batchs_num
        self.data = self._tokenize_and_batch(path)

    def gen_batch(self):
        try:
            for mini_batch, epoch in self.data:
                n_batch = mini_batch.size(0)//self.seq_size
                for s in range(n_batch-1):
                    yield Variable(mini_batch.narrow(0,s*self.seq_size,self.seq_size+1).long()), epoch
        except StopIteration:
            print("finish all batch")

    def _tokenize_and_batch(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        ids = torch.IntTensor(self.pre_words_num)
        token = 0
        cur_epoch = self.epoch
        while cur_epoch == None or cur_epoch > 0:
            if cur_epoch is not None:
                cur_epoch = cur_epoch - 1

            with open(path, encoding='utf-8') as f:
                for line in f:
                    for word in line.strip('\r'):
                        if word in self.dictionary.word2idx:
                            ids[token] = self.dictionary.word2idx[word]
                        else:
                            ids[token] = 0
                        token += 1
                        if token == self.pre_words_num:
                            yield self._batchify(ids, self.batch_size), self.epoch - cur_epoch
                            ids = torch.IntTensor(self.pre_words_num)
                            token = 0

    def _batchify(self, data, bsz):
        nbatch = data.size(0) // bsz
        data = data.narrow(0, 0, nbatch * bsz)
        data = data.view(bsz, -1).t().contiguous()
        return data        
