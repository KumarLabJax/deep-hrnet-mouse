# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.focalloss import bce_focal_loss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    # writer_dict = {
    #     'writer': SummaryWriter(log_dir=tb_log_dir),
    #     'train_global_steps': 0,
    #     'valid_global_steps': 0,
    # }

    # THIS FUNCTIONALITY IS BROKEN UNTIL
    # https://github.com/pytorch/pytorch/issues/19374 IS FIXED
    # dump_input = torch.rand(
    #     (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    # )
    # writer_dict['writer'].add_graph(model, (dump_input, ))
    #
    # logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    if cfg.LOSS.USE_FOCAL_LOSS:
        print('USING FOCAL LOSS')
        def fl_crit(input, target, target_weight):
            return bce_focal_loss(input, target)
        criterion = fl_crit
    else:
        print('USING MSE LOSS')
        criterion = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        ).cuda()

    # Data loading code
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    normalize = transforms.Normalize(
        mean=[0.45], std=[0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    print('len(train_dataset):', len(train_dataset))
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    print('len(valid_dataset):', len(valid_dataset))

    print('init train loader')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    print('init valid loader')
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    best_perf = None
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    train_table_fname = os.path.join(final_output_dir, 'training.tsv')
    val_table_fname = os.path.join(final_output_dir, 'validation.tsv')
    with    open(train_table_fname, 'w', newline='') as train_table_f, \
            open(val_table_fname, 'w', newline='') as val_table_f:

        train_header = ['Epoch', 'Batch', 'Loss', 'Accuracy', 'Batch Time', 'Batch Size']
        train_table_writer = csv.DictWriter(train_table_f, fieldnames=train_header, delimiter='\t')
        train_table_writer.writeheader()

        val_header = ['Epoch', 'Loss', 'Accuracy', 'Performance Indicator']
        val_table_writer = csv.DictWriter(val_table_f, fieldnames=val_header, delimiter='\t')
        val_table_writer.writeheader()

        print('entering epoch loop from:', begin_epoch, 'to', cfg.TRAIN.END_EPOCH)
        for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
            lr_scheduler.step()

            # train for one epoch
            train(
                cfg, train_loader, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict=None,
                dict_writer=train_table_writer)


            # evaluate on validation set
            perf_indicator = validate(
                cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict=None,
                dict_writer=val_table_writer, epoch=epoch,
            )

            if best_perf is None or perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
                print('*** NEW BEST ***', perf_indicator)
            else:
                best_model = False

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

        final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
        torch.save(model.module.state_dict(), final_model_state_file)
        # writer_dict['writer'].close()


if __name__ == '__main__':
    main()
