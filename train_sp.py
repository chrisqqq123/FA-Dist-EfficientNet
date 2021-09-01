#!/usr/bin/env python3
"""Script to train a model through refined labels on ImageNet's train set."""

import argparse
import logging
import pprint
import os
import sys
import time

import torch
from torch import nn

from models import model_factory
import opts
import test
import utils

import mul_cifar100
import datetime
import time
now = datetime.datetime.now()
name_time = now.strftime("%Y-%m-%d-%H-%M")

def parse_args(argv):
    """Parse arguments @argv and return the flags needed for training."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)

    group = parser.add_argument_group('General Options')
    opts.add_general_flags(group)

    group = parser.add_argument_group('Dataset Options')
    opts.add_dataset_flags(group)

    group = parser.add_argument_group('Model Options')
    opts.add_model_flags(group)

    group = parser.add_argument_group('Label Refinery Options')
    opts.add_label_refinery_flags(group)

    group = parser.add_argument_group('Training Options')
    opts.add_training_flags(group)

    args = parser.parse_args(argv)

    if args.label_refinery_model is not None and args.label_refinery_state_file is None:
        parser.error("You should set --label-refinery-state-file if "
                     "--label-refinery-model is set.")

    return args


class LearningRateRegime:
    """Encapsulates the learning rate regime for training a model.

    Args:
        @intervals (list): A list of triples (start, end, lr). The intervals
            are inclusive (for start <= epoch <= end, lr will be used). The
            start of each interval must be right after the end of its previous
            interval.
    """

    def __init__(self, regime):
        if len(regime) % 3 != 0:
            raise ValueError("Regime length should be devisible by 3.")
        intervals = list(zip(regime[0::3], regime[1::3], regime[2::3]))
        self._validate_intervals(intervals)
        self.intervals = intervals
        self.num_epochs = intervals[-1][1]

    @classmethod
    def _validate_intervals(cls, intervals):
        if type(intervals) is not list:
            raise TypeError("Intervals must be a list of triples.")
        elif len(intervals) == 0:
            raise ValueError("Intervals must be a non empty list.")
        elif intervals[0][0] != 1:
            raise ValueError("Intervals must start from 1: {}".format(intervals))
        elif any(end < start for (start, end, lr) in intervals):
            raise ValueError("End of intervals must be greater or equal than their"
                             " start: {}".format(intervals))
        elif any(intervals[i][1] + 1 != intervals[i + 1][0]
                 for i in range(len(intervals) - 1)):
            raise ValueError("Start of each each interval must be the end of its "
                             "previous interval plus one: {}".format(intervals))

    def get_lr(self, epoch):
        for (start, end, lr) in self.intervals:
            if start <= epoch <= end:
                return lr
        raise ValueError("Invalid epoch {} for regime {!r}".format(
            epoch, self.intervals))


def _set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _get_learning_rate(optimizer):
    return max(param_group['lr'] for param_group in optimizer.param_groups)


############## Pruning #####################
# def threshing_weights(weights, thresh):
# # zero out some weights
#     for wt in weights:
#         norm_ch = wt.pow(2).sum(dim=[0,2,3]).pow(1/2.)
#         for i in range(len(norm_ch)):
#             if norm_ch[i]<thresh:
#                 wt[:,i,:,:] *= 0
#     return weights


def gpls(weights, lamb=1e-2, decay=0.99):
# compute the norm in channels
    rs = 0
    for wt in weights:
        rs *= decay 
        rs += wt.pow(2).sum(dim=[0,2,3]).add(1e-8).pow(1/2.).sum()   
    rs *= lamb
    return rs
########################################


def train_for_one_epoch(model, loss, train_loader, optimizer, epoch_number):
    model.train()
    loss.train()

    data_time_meter = utils.AverageMeter()
    batch_time_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter(recent=100)
    top1_meter = utils.AverageMeter(recent=100)
    top5_meter = utils.AverageMeter(recent=100)

    timestamp = time.time()
    for i, (images, labels) in enumerate(train_loader):
        batch_size = images.size(0)

        if utils.is_model_cuda(model):
            images = images.cuda()
            labels = labels.cuda()

        # Record data time
        data_time_meter.update(time.time() - timestamp)

        # Forward pass, backward pass, and update parameters.
        outputs = model(images)
        loss_output = loss(outputs, labels)


        ############# Pruning ###############
        # weights = [ p for n,p in model.named_parameters() if 'weight' in n and 'se' not in n and 'conv' in n and len(p.size())==4]
        # lamb = 0.001
        # reg_loss = gpls(weights, lamb)
        ####################################


        # Sometimes loss function returns a modified version of the output,
        # which must be used to compute the model accuracy.
        if isinstance(loss_output, tuple):
            loss_value, outputs = loss_output
        else:
            loss_value = loss_output

        ########### Prning #################
        # loss_value += reg_loss
        ####################################

        loss_value.backward()

        # Update parameters and reset gradients.
        optimizer.step()
        optimizer.zero_grad()

        # Record loss and model accuracy.
        loss_meter.update(loss_value.item(), batch_size)
        top1, top5 = utils.topk_accuracy(outputs, labels, recalls=(1, 5))
        top1_meter.update(top1, batch_size)
        top5_meter.update(top5, batch_size)

        # Record batch time
        batch_time_meter.update(time.time() - timestamp)
        timestamp = time.time()
        if i%100==0:
            logging.info(
                'Epoch: [{epoch}][{batch}/{epoch_size}]\t'
                'Time {batch_time.value:.2f} ({batch_time.average:.2f})   '
                'Data {data_time.value:.2f} ({data_time.average:.2f})   '
                'Loss {loss.value:.3f} {{{loss.average:.3f}, {loss.average_recent:.3f}}}    '
                'Top-1 {top1.value:.2f} {{{top1.average:.2f}, {top1.average_recent:.2f}}}    '
                'Top-5 {top5.value:.2f} {{{top5.average:.2f}, {top5.average_recent:.2f}}}    '
                'LR {lr:.5f}'.format(
                    epoch=epoch_number, batch=i + 1, epoch_size=len(train_loader),
                    batch_time=batch_time_meter, data_time=data_time_meter,
                    loss=loss_meter, top1=top1_meter, top5=top5_meter,
                    lr=_get_learning_rate(optimizer)))
    
    ############# Pruning ##############
    # weights = [ p for n,p in model.named_parameters() if 'weight' in n and 'se' not in n and 'conv' in n and len(p.size())==4]
    # for wt in weights:
    #     norm_ch = wt.pow(2).sum(dim=[0,2,3]).pow(1/2.)
    #     for i in range(len(norm_ch)):
    #         if norm_ch[i]<1e-8:
    #             wt[:,i,:,:].data *= 0
    ####################################

    # Log the overall train stats
    logging.info(
        'Epoch: [{epoch}] -- TRAINING SUMMARY\t'
        'Time {batch_time.sum:.2f}   '
        'Data {data_time.sum:.2f}   '
        'Loss {loss.average:.3f}     '
        'Top-1 {top1.average:.2f}    '
        'Top-5 {top5.average:.2f}    '.format(
            epoch=epoch_number, batch_time=batch_time_meter, data_time=data_time_meter,
            loss=loss_meter, top1=top1_meter, top5=top5_meter))


def save_checkpoint(checkpoints_dir, model, optimizer, epoch):
    model_state_file = os.path.join(checkpoints_dir, 'model_state_{:02}.pytar'.format(epoch))
    optim_state_file = os.path.join(checkpoints_dir, 'optim_state_{:02}.pytar'.format(epoch))
    torch.save(model.state_dict(), model_state_file)
    torch.save(optimizer.state_dict(), optim_state_file)


def create_optimizer(model, momentum=0.9, weight_decay=0.0001):
    # Get model parameters that require a gradient.
    model_trainable_parameters = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(model_trainable_parameters, lr=0,
                                momentum=momentum, weight_decay=weight_decay)
    return optimizer


def main(argv):
    """Run the training script with command line arguments @argv."""
    args = parse_args(argv)
    if args.label_refinery_model is None:
        if args.coslinear:
            save_dir = args.save+'/'+args.model+'_cos_'+name_time
        else:
            save_dir = args.save+'/'+args.model+'_'+name_time
    else:
        if args.coslinear:
            save_dir = args.save+'/'+args.model+'_cos_rfn_'+name_time
        else:
            save_dir = args.save+'/'+args.model+'_rfn_'+name_time
    if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
    utils.general_setup(save_dir, args.gpus)

    logging.info("Arguments parsed.\n{}".format(pprint.pformat(vars(args))))

    # Create the train and the validation data loaders.
    train_loader = mul_cifar100.mul_CIFAR100DataLoader(root=args.data_dir, 
        image_size=32, train=True, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = mul_cifar100.mul_CIFAR100DataLoader(root=args.data_dir, 
        image_size=32, train=False, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Create model with optional label refinery.
    model, loss = model_factory.create_model(
        args.model, args.model_state_file, args.gpus, args.label_refinery_model,
        args.label_refinery_state_file, args.coslinear, args.s)
    # logging.info("Model:\n{}".format(model))

    if args.lr_regime is None:
        lr_regime = model.LR_REGIME
    else:
        lr_regime = args.lr_regime
    regime = LearningRateRegime(lr_regime)
    # Train and test for needed number of epochs.
    optimizer = create_optimizer(model, args.momentum, args.weight_decay)
    
    for epoch in range(1, int(regime.num_epochs) + 1):
        lr = regime.get_lr(epoch)
        _set_learning_rate(optimizer, lr)
        train_for_one_epoch(model, loss, train_loader, optimizer, epoch)
        ############# Print results ########
        # weights = [ p for n,p in model.named_parameters() if 'weight' in n and 'se' not in n and 'conv' in n and len(p.size())==4]
        # name = [ n for n,p in model.named_parameters() if 'weight' in n and 'se' not in n and 'conv' in n and len(p.size())==4]
        # j = 0
        # for wt in weights:
        #     zr_ch = 0
        #     rs = wt.pow(2).sum(dim=[0,2,3]).pow(1/2.)
        #     for i in range(len(rs)):
        #         if rs[i]<=1e-15:
        #             zr_ch += 1
        #     csize = list(wt.size())
        #     num_ch = csize[1]
        #     print('Number of zero channels: '+str(zr_ch)+'/'+str(num_ch)+'  in :'+name[j])
        #     j += 1
        ####################################
        test.test_for_one_epoch(model, loss, val_loader, epoch)
        save_checkpoint(save_dir, model, optimizer, epoch)
        

if __name__ == '__main__':
    main(sys.argv[1:])
