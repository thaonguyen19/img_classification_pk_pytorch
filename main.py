#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from colorama import Fore
from importlib import import_module

import config
from dataloader import getDataloaders
from utils import save_checkpoint, get_optimizer, create_save_folder
from args import arg_parser, arch_resume_names

try:
    from tensorboard_logger import configure, log_value
except BaseException:
    configure = None


def getModel(arch, **kargs):
    m = import_module('models.' + arch)
    model = m.createModel(**kargs)
    if arch.startswith('alexnet') or arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model


def main():
    # parse arg and start experiment
    global args
    best_err1 = 100.
    best_epoch = 0

    args = arg_parser.parse_args()
    args.config_of_data = config.datasets[args.data]
    args.num_classes = config.datasets[args.data]['num_classes']
    if configure is None:
        args.tensorboard = False
        print(Fore.RED +
              'WARNING: you don\'t have tensorboard_logger installed' +
              Fore.RESET)

    # optionally resume from a checkpoint
    if args.resume:
        if args.resume and os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            global checkpoint
            checkpoint = torch.load(args.resume)
            old_args = checkpoint['args']
            # set args based on checkpoint
            if args.start_epoch <= 0:
                args.start_epoch = checkpoint['epoch'] + 1
            best_epoch = args.start_epoch - 1
            best_err1 = checkpoint['best_err1']
            for name in arch_resume_names: 
            #['arch', 'depth', 'death_mode', 'death_rate', 'growth_rate', 'bn_size', 'compression']
                if name in vars(args) and name in vars(old_args):
                    setattr(args, name, getattr(old_args, name))
            model = getModel(**vars(args))
            print ("ending resuming")

            # nonparallel_dict = {k[7:]: checkpoint['state_dict'][k] for k in checkpoint['state_dict'].keys()}
            # model.load_state_dict(nonparallel_dict)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print(
                "=> no checkpoint found at '{}'".format(
                    Fore.RED +
                    args.resume +
                    Fore.RESET),
                file=sys.stderr)
            return
    else:
        # create model
        print("=> creating model '{}'".format(args.arch))
        model = getModel(**vars(args))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = get_optimizer(model, args)

    Trainer = import_module(args.trainer).Trainer
    trainer = Trainer(model, criterion, optimizer, args)

    # create dataloader
    if args.evaluate == 'train':
        train_loader, _, _ = getDataloaders(
            splits=('train'), **vars(args))
        trainer.test(train_loader, best_epoch)
        return
    elif args.evaluate == 'val':
        _, val_loader, _ = getDataloaders(
            splits=('val'), **vars(args))
        trainer.test(val_loader, best_epoch)
        return
    elif args.evaluate == 'test':
        _, _, test_loader = getDataloaders(
            splits=('test'), **vars(args))
        if args.test_death_mode == 'none':
            trainer.test(test_loader, best_epoch)
        else:
            print ("Stochastic depth testing...")
            nblocks = (args.depth - 2) // 2
            n = (args.depth - 2) // 6 
            section_reps=[n]*3
            all_top1 = []

            for n in range(0, nblocks): #drop 0, 1, 2, ..., nblocks-1 blocks
                print ("Dropping " + str(n)+ " blocks")
                death_rates_list = [0]*(nblocks-n) + [1]*n
                test_death_rate = []
                count = 0
                for i in range(len(section_reps)):
                    test_death_rate.append(death_rates_list[count:(count+section_reps[i])])
                    count += section_reps[i]
                model = getModel(test_death_rate=test_death_rate, **vars(args))
                model.load_state_dict(checkpoint['state_dict'])
                optimizer = get_optimizer(model, args)
                trainer = Trainer(model, criterion, optimizer, args)
                _, top1, _ = trainer.test(test_loader, best_epoch)
                all_top1.append(top1)

            with open(args.resume.split('/')[1]+'.csv','w') as f:
                writer = csv.writer(f)
                rows = zip(range(0, nblocks), all_top1)
                for row in rows:
                    writer.writerow(row)
        return
        
    else:
        train_loader, val_loader, _ = getDataloaders(
            splits=('train', 'val'), **vars(args))

    # check if the folder exists
    create_save_folder(args.save, args.force)

    # set up logging
    global log_print, f_log
    f_log = open(os.path.join(args.save, 'log.txt'), 'w')

    def log_print(*args):
        print(*args)
        print(*args, file=f_log)
    log_print('args:')
    log_print(args)
    print('model:', file=f_log)
    print(model, file=f_log)
    log_print('# of params:',
              str(sum([p.numel() for p in model.parameters()])))
    f_log.flush()
    torch.save(args, os.path.join(args.save, 'args.pth'))
    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_err1'
              '\tval_err1\ttrain_err5\tval_err']
    if args.tensorboard:
        configure(args.save, flush_secs=5)

    for epoch in range(args.start_epoch, args.epochs + 1):

        # train for one epoch
        train_loss, train_err1, train_err5, lr = trainer.train(
            train_loader, epoch)

        if args.tensorboard:
            log_value('lr', lr, epoch)
            log_value('train_loss', train_loss, epoch)
            log_value('train_err1', train_err1, epoch)
            log_value('train_err5', train_err5, epoch)

        # evaluate on validation set
        val_loss, val_err1, val_err5 = trainer.test(val_loader, epoch)

        if args.tensorboard:
            log_value('val_loss', val_loss, epoch)
            log_value('val_err1', val_err1, epoch)
            log_value('val_err5', val_err5, epoch)

        # save scores to a tsv file, rewrite the whole file to prevent
        # accidental deletion
        scores.append(('{}\t{}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_err1, val_err1, train_err5, val_err5))
        with open(os.path.join(args.save, 'scores.tsv'), 'w') as f:
            print('\n'.join(scores), file=f)

        # remember best err@1 and save checkpoint
        is_best = val_err1 < best_err1
        if is_best:
            best_err1 = val_err1
            best_epoch = epoch
            print(Fore.GREEN + 'Best var_err1 {}'.format(best_err1) +
                  Fore.RESET)
            # test_loss, test_err1, test_err1 = validate(
            #     test_loader, model, criterion, epoch, True)
            # save test
        save_checkpoint({
            'args': args,
            'epoch': epoch,
            'best_epoch': best_epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
        }, is_best, args.save)
        if not is_best and epoch - best_epoch >= args.patience > 0:
            break
    print('Best val_err1: {:.4f} at epoch {}'.format(best_err1, best_epoch))


if __name__ == '__main__':
    main()
