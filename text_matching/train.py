#!/usr/bin/env python
#coding:utf-8
#author:zhujianqi


from __future__ import print_function
import sys
import argparse
import json


import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from corpus import TextPairDataset
from util import load_word_embedding
from util import train
from util import evaluate
from util import log_error
from linear_model import LinearModel
from esim.model import ESIM

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main(model_name,
         train_file,
         valid_file,
         embedding_file,
         target_dir,
         hidden_size=300,
         dropout=0.5,
         num_classes=3,
         epochs=64,
         batch_size=32,
         lr=0.0004,
         patience=5,
         max_grad_norm=10.0,
         checkpoint=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # step0: load data
    word2id, embeddings = load_word_embedding(embedding_file)
    embeddings = torch.tensor(embeddings, dtype=torch.float).to(device)

    train_dataset = TextPairDataset(train_file, word2id, max_premise_length=22, max_hypothesis_length=22)
    valid_dataset = TextPairDataset(valid_file, word2id, max_premise_length=22, max_hypothesis_length=22)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        
    # step1: make model 
    model = ESIM(embeddings.shape[0],
                        embeddings.shape[1],
                        hidden_size,
                        embeddings,
                        dropout,
                        num_classes,
                        device=device).to(device)

    # step2: do training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.5,
                                                           patience=0)

    best_score = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        
        print('\t* Training will continue on existing model from epoch {}...'.format(start_epoch))

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epochs_count = checkpoint['epochs_count']
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']

        valid_time, valid_loss, valid_auc, valid_acc = evaluate(model, valid_dataloader, criterion)

        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:6.3f} | valid auc {:6.3f} | '
              'valid acc {:6.3f} '.format(start_epoch,
                                          valid_time,
                                          valid_loss,
                                          valid_auc,
                                          valid_acc))

    print('\n',
          20 * '-',
          'Training model on device: {}'.format(device),
          20 * '-')

    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        epochs_count.append(epoch)

        print('* Training epoch {}:'.format(epoch))
        epoch_time, epoch_loss, epoch_auc, epoch_acc = train(model,
                                                             train_dataloader,
                                                             criterion,
                                                             optimizer,
                                                             epoch,
                                                             max_grad_norm)

        train_losses.append(epoch_loss)

        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:6.3f} | train auc {:6.3f} | '
              'train acc {:6.3f} '.format(epoch,
                                          epoch_time,
                                          epoch_loss,
                                          epoch_auc,
                                          epoch_acc))

        epoch_time, epoch_loss, epoch_auc, valid_acc = evaluate(model,
                                                                valid_dataloader,
                                                                criterion)
        valid_losses.append(epoch_loss)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:6.3f} | valid auc {:6.3f} | '
              'valid acc {:6.3f} '.format(epoch,
                                          epoch_time,
                                          epoch_loss,
                                          epoch_auc,
                                          epoch_acc))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_auc)

        # Early stopping on validation accuracy.
        if epoch_auc < best_score:
            patience_counter += 1
        else:
            best_score = epoch_auc
            patience_counter = 0
            # Save the best model. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best model, use the 'esim_*.pth.tar'
            # checkpoints instead.
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'best_score': best_score,
                        'optimizer': optimizer.state_dict(),
                        'epochs_count': epochs_count,
                        'train_losses': train_losses,
                        'valid_losses': valid_losses},
                       os.path.join(target_dir, '{}_best.pth.tar'.format(model_name)))

        # Save the model at each epoch.
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'best_score': best_score,
                    'optimizer': optimizer.state_dict(),
                    'epochs_count': epochs_count,
                    'train_losses': train_losses,
                    'valid_losses': valid_losses},
                   os.path.join(target_dir, "{}_{}.pth.tar".format(model_name, epoch)))

        if patience_counter >= patience:
            print('-> Early stopping: patience limit reached, stopping...')
            break

    # Plotting of the loss curves for the train and validation sets.
    plt.figure()
    plt.plot(epochs_count, train_losses, '-r')
    plt.plot(epochs_count, valid_losses, '-b')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training loss', 'Validation loss'])
    plt.title('Cross entropy loss')
    plt.show()


if __name__ == "__main__":
    default_config = "config/linear_model.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=default_config,
                        help='Path to a json configuration file')
    parser.add_argument('--checkpoint',
                        default=None,
                        help='Path to a checkpoint file to resume training')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), "r") as config_file:
        config = json.load(config_file)

    main(config['model_name'],
         os.path.normpath(os.path.join(script_dir, config['train_data'])),
         os.path.normpath(os.path.join(script_dir, config['valid_data'])),
         os.path.normpath(os.path.join(script_dir, config['embeddings'])),
         os.path.normpath(os.path.join(script_dir, config['target_dir'])),
         config['hidden_size'],
         config['dropout'],
         config['num_classes'],
         config['epochs'],
         config['batch_size'],
         config['lr'],
         config['patience'],
         config['max_gradient_norm'],
         args.checkpoint)


