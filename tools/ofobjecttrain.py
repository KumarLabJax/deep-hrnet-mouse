# base this off of mousetrain.py... but we need to use BCEWithLogitsLoss
# https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss

import argparse
import csv
import os
import pprint
import random
import shutil

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.segfunction import train, validate
from dataset.OpenFieldObjDataset import OpenFieldObjDataset, parse_obj_labels
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

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

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=True).cuda()

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)

    criterion = torch.nn.BCEWithLogitsLoss().cuda()

    # Data loading code
    obj_labels = list(parse_obj_labels(cfg.DATASET.CVAT_XML))
    validation_set_filename = cfg.DATASET.TEST_SET
    val_img_names = set()
    if os.path.exists(validation_set_filename):
        with open(validation_set_filename) as val_file:
            for curr_line in val_file:
                img_name = curr_line.strip()
                val_img_names.add(img_name)

    else:
        img_names = {lbl['image_name'] for lbl in obj_labels}
        val_count = round(len(img_names) * cfg.DATASET.TEST_SET_PROPORTION)
        val_img_names = set(random.sample(img_names, val_count))

        logger.info("=> saving validation image names to '{}'".format(validation_set_filename))
        with open(validation_set_filename, 'w') as val_file:
            for img_name in val_img_names:
                val_file.write(img_name)
                val_file.write('\n')

    train_labels = [lbl for lbl in obj_labels if lbl['image_name'] not in val_img_names]
    train_ofods = OpenFieldObjDataset(
        cfg,
        train_labels,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ofods,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    val_labels = [lbl for lbl in obj_labels if lbl['image_name'] in val_img_names]
    val_ofods = OpenFieldObjDataset(
        cfg,
        val_labels,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]),
    )
    valid_loader = torch.utils.data.DataLoader(
        val_ofods,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    logger.info("=> full data set size: {}; training/validation: {} [{}]/{} [{}]".format(
        len(obj_labels), len(train_labels), len(train_ofods), len(val_labels), len(val_ofods)))

    best_perf = None
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    train_table_fname = os.path.join(final_output_dir, 'training.tsv')
    val_table_fname = os.path.join(final_output_dir, 'validation.tsv')
    with    open(train_table_fname, 'w', newline='') as train_table_f, \
            open(val_table_fname, 'w', newline='') as val_table_f:

        train_header = ['Epoch', 'Batch', 'Loss', 'Batch Time', 'Batch Size']
        train_table_writer = csv.DictWriter(train_table_f, fieldnames=train_header, delimiter='\t')
        train_table_writer.writeheader()

        val_header = ['Epoch', 'Loss', 'Accuracy']
        val_table_writer = csv.DictWriter(val_table_f, fieldnames=val_header, delimiter='\t')
        val_table_writer.writeheader()

        logger.info('entering epoch loop from: {} to {}'.format(begin_epoch, cfg.TRAIN.END_EPOCH))
        for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
            lr_scheduler.step()

            # train for one epoch
            train(cfg, train_loader, model, criterion, optimizer, epoch, train_table_writer)

            # evaluate on validation set
            perf_indicator = validate(
                cfg, valid_loader, model,
                criterion, val_table_writer, epoch)

            if best_perf is None or perf_indicator >= best_perf:
                best_perf = perf_indicator
                logger.info('*** NEW BEST *** {}'.format(perf_indicator))
                best_model_state_file = os.path.join(final_output_dir, 'best_state.pth')
                logger.info('=> saving best model state to {}'.format(best_model_state_file))
                torch.save(model.state_dict(), best_model_state_file)

        final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info('=> saving final model state to {}'.format(final_model_state_file))
        torch.save(model.state_dict(), final_model_state_file)


if __name__ == '__main__':
    main()
