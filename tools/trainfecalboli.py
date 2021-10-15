import argparse
import csv
import functools
import itertools
import os
import pprint
import random
import shutil

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.fecalbolifunc import train, validate
from core.assocembedloss import weighted_bcelogit_loss
from dataset.fecalbolidata import FecalBoliDataset, parse_fecal_boli_labels
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import models


def parse_args():
    parser = argparse.ArgumentParser(description='train fecal boli detection network')

    parser.add_argument('--cvat-files',
                        help='list of CVAT XML files to use',
                        nargs='+',
                        required=True,
                        type=str)
    parser.add_argument('--image-dir',
                        help='directory containing images',
                        required=True,
                        type=str)
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

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

    args = parser.parse_args()

    return args

#   python tools/trainfecalboli.py \
#       --cfg experiments/fecalboli/fecalboli_2020-05-0-08.yaml \
#       --cvat-files data/fecal-boli/*.xml \
#       --image-dir data/fecal-boli/images
def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, _ = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    swriter = SummaryWriter(os.path.join(final_output_dir, 'tb'))

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

    if cfg.LOSS.POSE_LOSS_FUNC == 'MSE':
        criterion = torch.nn.MSELoss()
    elif cfg.LOSS.POSE_LOSS_FUNC == 'WEIGHTED_BCE':
        criterion = functools.partial(
                weighted_bcelogit_loss,
                pos_weight = cfg.LOSS.POSITIVE_LABEL_WEIGHT)
    else:
        raise Exception('Unknown pose loss function: {}'.format(cfg.LOSS.POSE_LOSS_FUNC))

    # Data loading code
    fecal_boli_labels = list(itertools.chain.from_iterable(
        parse_fecal_boli_labels(f) for f in args.cvat_files))
    validation_set_filename = cfg.DATASET.TEST_SET
    val_img_names = set()
    if os.path.exists(validation_set_filename):
        with open(validation_set_filename) as val_file:
            for curr_line in val_file:
                img_name = curr_line.strip()
                val_img_names.add(img_name)

    else:
        img_names = {lbl['image_name'] for lbl in fecal_boli_labels}
        val_count = round(len(img_names) * cfg.DATASET.TEST_SET_PROPORTION)
        val_img_names = set(random.sample(img_names, val_count))

        logger.info("=> saving validation image names to '{}'".format(validation_set_filename))
        with open(validation_set_filename, 'w') as val_file:
            for img_name in val_img_names:
                val_file.write(img_name)
                val_file.write('\n')

    transform = transforms.Normalize(mean=[0.485], std=[0.229])

    train_labels = [lbl for lbl in fecal_boli_labels if lbl['image_name'] not in val_img_names]
    train_ds = FecalBoliDataset(
        cfg,
        args.image_dir,
        train_labels,
        True,
        transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
    )

    val_labels = [lbl for lbl in fecal_boli_labels if lbl['image_name'] in val_img_names]
    val_ds = FecalBoliDataset(
        cfg,
        args.image_dir,
        val_labels,
        False,
        transform,
    )
    valid_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    logger.info("=> full data set size: {}; training/validation: {} [{}]/{} [{}]".format(
        len(fecal_boli_labels), len(train_labels), len(train_ds), len(val_labels), len(val_ds)))

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

        val_header = ['Epoch', 'Loss', 'Performance Indicator']
        val_table_writer = csv.DictWriter(val_table_f, fieldnames=val_header, delimiter='\t')
        val_table_writer.writeheader()

        logger.info('entering epoch loop from: {} to {}'.format(begin_epoch, cfg.TRAIN.END_EPOCH))
        for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

            # train for one epoch
            train(
                cfg,
                train_loader,
                model,
                criterion,
                optimizer,
                train_table_writer,
                swriter,
                epoch)

            # evaluate on validation set
            perf_indicator = validate(
                cfg,
                valid_loader,
                model,
                criterion,
                val_table_writer,
                swriter,
                epoch)

            if best_perf is None or perf_indicator >= best_perf:
                best_perf = perf_indicator
                logger.info('*** NEW BEST *** {}'.format(perf_indicator))
                best_model_state_file = os.path.join(final_output_dir, 'best_state.pth')
                logger.info('=> saving best model state to {}'.format(best_model_state_file))
                torch.save(model.state_dict(), best_model_state_file)

            lr_scheduler.step()

        final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info('=> saving final model state to {}'.format(final_model_state_file))
        torch.save(model.state_dict(), final_model_state_file)


if __name__ == '__main__':
    main()
