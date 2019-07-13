#coding=utf-8
# Helper functions to make the code more readable.
import os
import pickle
import numpy as np
import torch

np.random.seed(1023)

class DataHelper():
    def __init__(self, mode, filename, batch_size, max_length, start_tag, end_tag, unknown_word, pad=True, shuffle=True):
        self.mode = mode
        self.batch_size = batch_size
        self.max_length = max_length 
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.unknown_word = unknown_word
        self.pad = pad
        self.shuffle = shuffle

        self.sentences, \
        self.word2id, \
        self.id2word, \
        self.tag2id, \
        self.id2tag, \
        max_length_ = self._load_data(filename, start_tag, end_tag, unknown_word)

    def get_sentences(self):
        return self.sentences

    def get_word2id(self):
        return self.word2id

    def get_tag2id(self):
        return self.tag2id

    def _load_data(self, filename, start_tag, end_tag, unknown_word):
        sentences = []
        word2id = {}
        id2word = []
        tag2id = {}
        id2tag = []
    
        word2id[unknown_word] = len(word2id)
        id2word.append(unknown_word)
        max_length = 0
        with open(filename, 'rt', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip('\r\n')
                sentences.append(sentence)
                phrases = sentence.split('  ')
                length = 0
                for phrase in phrases:
                    tag_suffix = phrase[-1:]
                    if tag_suffix in ['a', 'b', 'c']:
                        for tag in ['B-'+tag_suffix, 'I-'+tag_suffix, 'E-'+tag_suffix]:
                            if tag not in tag2id:
                                tag2id[tag] = len(tag2id)
                                id2tag.append(tag)
                    else:
                        tag = tag_suffix
                        if tag not in tag2id:
                            tag2id[tag] = len(tag2id)
                            id2tag.append(tag)

                    _phrase = phrase
                    if self.mode in ['train', 'test']:
                        _phrase = phrase[:-2]

                    words = _phrase.split('_')
                    for word in words:
                        if word not in word2id:
                            word2id[word] = len(word2id)
                            id2word.append(word)
                    length += len(words)
                if length > max_length:
                    max_length = length
    
        tag2id[start_tag] = len(tag2id)
        id2tag.append(start_tag)
        tag2id[end_tag] = len(tag2id)
        id2tag.append(end_tag)

        return sentences, word2id, id2word, tag2id, id2tag, max_length

    def gen_batch(self):
        """
        Generates a batch iterator for a dataset.
        """
        batch_size = self.batch_size
        shuffle = self.shuffle
        data = np.array(self.sentences)

        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        while True:
            # shuffle the data at starting of each epoch
            shuffled_data = data
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
    
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield self._format_samples(shuffled_data[start_index:end_index], self.max_length)

            if self.mode in ['train', "pred"]:
                break

    def _format_samples(self, raw_samples, max_length):
        wordss, tagss, lengths = [], [], []
        for raw_sample in raw_samples:
            words = []
            tags = []
            phrases = raw_sample.split('  ')
            for phrase in phrases:
                sentence = phrase
                if self.mode in ['train', 'test']:
                    sentence = phrase[:-2]
                _words = [word for word in sentence.split('_')]
                words.extend(_words)

                tag = phrase[-1:]
                if tag in ['a', 'b', 'c']:
                    tags.append('B-'+tag)
                    if len(_words) > 2:
                        tags.extend(['I-'+tag]*(len(_words) - 2))
                    if len(_words) > 1:
                        tags.append('E-'+tag)
                else:
                    tags.extend([tag]*len(_words))

            words_size = len(words)
            lengths.append(words_size)
            # padding
            if self.pad and words_size < max_length:
                words.extend([self.unknown_word]*(max_length-words_size))
                tags.extend([self.end_tag]*(max_length-words_size))
            if max_length:
                words = words[:max_length]
                tags = tags[:max_length]
            wordss.append(words)
            tagss.append(tags)
        return (wordss, tagss, lengths)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seqs, to_ix):
    idxs = [[to_ix[w] if w in to_ix else 0 for w in seq] for seq in seqs]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def get_tags(path, tag, tag_map):
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    end_tag = tag_map.get("E-" + tag)
    single_tag = tag_map.get("o")
    o_tag = tag_map.get("o")
    begin = -1
    end = 0
    tags = []
    last_tag = 0
    for index, tag in enumerate(path):
        if tag == begin_tag and index == 0:
            begin = 0
        elif tag == begin_tag:
            begin = index
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
        elif tag == o_tag or tag == single_tag:
            begin = -1
        last_tag = tag
    return tags

def f1_score(tar_path, pre_path, tag, tag_map, path_len):
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_path, pre_path, path_len):
        tar, pre, _len = fetch
        tar_tags = get_tags(tar[:_len], tag, tag_map)
        pre_tags = get_tags(pre[:_len], tag, tag_map)

        origin += len(tar_tags)
        found += len(pre_tags)

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1

    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall+precision == 0 else (2*precision*recall)/(precision + recall)
    print("\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(tag, recall, precision, f1))
    return recall, precision, f1

def save_dict(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        
def load_dict(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data
