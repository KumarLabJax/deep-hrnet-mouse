# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# NOTE:
#   This code is based on function.py code. The name was kept
#   function.py by Massimo. I (KSS) renamed it so that the original
#   function.py code will work as is but that means that this code will
#   not work until it is reintegrated back into the codebase

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import matplotlib.pyplot as plt

import numpy as np
import torch

import cv2

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

import imageio

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, summary_writer, writer_dict=None, dict_writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        # if i == 1:
        #     plt.imshow(input[0, 0, :, :])
        #     plt.show()
        #     plt.imshow(outputs[0, 0, :, :].cpu().detach().numpy())
        #     plt.show()
        #     plt.imshow(target[0, :, :])
        #     plt.show()

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        output_array = outputs.cpu().detach().numpy()
        target_array = target.cpu().detach().numpy()
        target_weight_array = target_weight.cpu().detach().numpy()

        # print(output_array.shape)
        # print(target.shape)
        # print(target_weight.shape)
        # print(np.all(target_array==0))

        target_array_im = np.array((target_array[2], target_array[3]), dtype=np.uint8)

        target_array_im = target_array_im = target_array_im.min()
        target_array_im = 255 * (target_array_im / target_array_im.max())
        target_array_im = target_array_im.astype(np.uint8)

        target_array = target_array.astype('uint8')

        imageio.imwrite(('INF/inf%s.png' % meta['image'][0]), target_array[0, :, :])     # target_array_im[:, :, 0])
        # ^ USED TO BE [0, 0, :. :]

        # print(meta['center'])

        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(target_array[0, :, :])
        # print(meta['filename'])
        # print(maxLoc)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            # print(output.shape, target.shape, target_weight.shape)
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        elapsed_time = time.time() - end
        batch_time.update(elapsed_time)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.7f} ({loss.avg:.7f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            # save_debug_images(config, input, meta, target, pred*4, output,
            #                   prefix)

        if dict_writer:
            dict_writer.writerow({
                'Epoch': epoch,
                'Batch': i,
                'Loss': loss.item(),
                'Accuracy': avg_acc,
                'Batch Time': elapsed_time,
                'Batch Size': input.size(0),
            })


        summary_writer.add_scalar(
            'loss/train',
            loss.item(),
            epoch * (len(train_loader)) + i)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, summary_writer, writer_dict=None, dict_writer=None, epoch=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            print(input.shape)
            outputs = model(input)

            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            # s = meta['scale'].numpy()
            s = 1
            # score = meta['score'].numpy()
            score = 1
            print(output.shape)
            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            # ERROR


            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c
            all_boxes[idx:idx + num_images, 2:4] = s
            # all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.7f} ({loss.avg:.7f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                # save_debug_images(config, input, meta, target, pred*4, output,
                #                  prefix)

        try:
            name_values, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path,
                filenames, imgnums
            )
        except NotImplementedError:
            name_values = []
            perf_indicator = -losses.avg

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

        if dict_writer:
            dict_writer.writerow({
                'Epoch': epoch,
                'Loss': losses.avg,
                'Accuracy': acc.avg,
                'Performance Indicator': perf_indicator,
            })

        summary_writer.add_scalar(
            'validate/train',
            loss.item(),
            epoch * (len(val_loader)) + i)

        return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
