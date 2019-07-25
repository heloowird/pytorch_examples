#coding=utf-8
import sys
import os
import time

import torch
import torch.autograd as autograd
import torch.optim as optim

from model import BiLSTM_CRF
from util import DataHelper
from util import prepare_sequence
from util import f1_score
from util import save_dict
from util import load_dict

# Use cuda or not
use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
device = torch.device("cuda" if use_cuda else "cpu")

# Train super parameters
start_tag = "<START>"
stop_tag = "<STOP>"
unknown_word = "<UNK>"
max_length = 256
batch_size = 32
epoch_num = 50
pad = True

# Model super parameters
embedding_dim = 300
hidden_dim = 200
dropout = 0

# Input and output parameters
mode = ""
train_file = "../../data/datagrand/train.txt.train"
test_file = "../../data/datagrand/train.txt.test"
pred_file = "../../data/datagrand/test.txt"
model_path = "output/"
word_dict_file = "output/word_dict"
tag_dict_file = "output/tag_dict"

# Load training and testing data
train_helper = DataHelper('train', train_file, batch_size, max_length, start_tag, stop_tag, unknown_word, pad)

#word_to_ix = train_helper.get_word2id()
#tag_to_ix = train_helper.get_tag2id()
#save_dict(word_to_ix, word_dict_file)
#save_dict(tag_to_ix, tag_dict_file)

word_to_ix = load_dict(word_dict_file)
tag_to_ix = load_dict(tag_dict_file)

test_helper = DataHelper('test', test_file, batch_size, max_length, start_tag, stop_tag, unknown_word, pad)

pred_helper = DataHelper('pred', pred_file, batch_size, 0, start_tag, stop_tag, unknown_word, False, False)

# Build model
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim, dropout, batch_size).to(device)

def train():
    # Restore model parameters
    model_file = model_path + 'params.pkl'
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        print("resotre model from {}".format(model_file))

    # Select optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(epoch_num):
        index = 0
        for batch_samples in train_helper.gen_batch():
            starttime = time.time()
            # Step 0. Genetate batch samples.
            wordss, tagss, lengths = batch_samples
    
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
    
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(wordss, word_to_ix)
            if use_cuda:
                sentence_in = sentence_in.cuda()
    
            label_out = prepare_sequence(tagss, tag_to_ix)
            if use_cuda:
                label_out = label_out.cuda()
    
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, label_out, lengths)
    
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
            # print info
            print("epoch:{}, batch:{}, loss:{}, timecost:{}".format(epoch, index, loss.cpu().tolist()[0], (time.time()-starttime)))
    
            # Step 5. Save model
            evaluate()
            if index and index % 10 == 0:
                torch.save(model.state_dict(), model_file)
            index += 1
    torch.save(model.state_dict(), model_file)

def evaluate():
    wordss, tagss, lengths = test_helper.gen_batch().__next__()
    sentence_in = prepare_sequence(wordss, word_to_ix)
    target_tag_seqs = prepare_sequence(tagss, tag_to_ix)
    predict_scores, predict_tag_seqs = model(sentence_in, lengths)
    for tag in ['a', 'b', 'c']:
        f1_score(target_tag_seqs, predict_tag_seqs, tag, tag_to_ix, lengths)

# Set batch_size with 1 
#   or set max_lenght large 
def predict():
    word_to_ix = load_dict(word_dict_file)
    tag_to_ix = load_dict(tag_dict_file)
    ix_to_tag = {v:k for k,v in tag_to_ix.items()}
    model_file = model_path + 'params.pkl'
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))

    for wordss, tagss, lengths in pred_helper.gen_batch():
        sentence_in = prepare_sequence(wordss, word_to_ix)
        predict_scores, predict_ix_seqs = model(sentence_in, lengths)
        for word, ix in zip(wordss[0], predict_ix_seqs[0]):
            print(word, ix_to_tag[ix])
        print()

def main():
    train()
    #predict()

if __name__ == '__main__':
    main()
