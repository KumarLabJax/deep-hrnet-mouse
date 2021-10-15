# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.function import AverageMeter


logger = logging.getLogger(__name__)


def train(
        config,
        train_loader,
        model,
        criterion,
        optimizer,
        dict_writer,
        summary_writer,
        epoch,
        device=None):

    if device is None:
        device = next(model.parameters()).device

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    batch_count = len(train_loader)
    for i, label_batch in enumerate(train_loader):

        batch_size = label_batch['image'].size(0)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        img_batch = label_batch['image'].to(device=device, non_blocking=True)
        if img_batch.size(1) == 1:
            img_batch = torch.cat([img_batch] * 3, dim=1)
        output = model(img_batch)

        loss = criterion(output, label_batch)
        summary_writer.add_scalars(
            'Loss/train',
            criterion.loss_components.copy(),
            epoch * batch_count + i)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), batch_size)

        # measure elapsed time
        elapsed_time = time.time() - end
        batch_time.update(elapsed_time)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.7f} ({loss.avg:.7f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=batch_size/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

        dict_writer.writerow({
            'Epoch': epoch,
            'Batch': i,
            'Loss': loss.item(),
            'Batch Time': elapsed_time,
            'Batch Size': batch_size,
        })


def validate(
        config,
        val_loader,
        model,
        criterion,
        dict_writer,
        summary_writer,
        epoch,
        device=None):

    if device is None:
        device = next(model.parameters()).device

    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, label_batch in enumerate(val_loader):

            batch_size = label_batch['image'].size(0)
            img_batch = label_batch['image'].to(device=device, non_blocking=True)
            if img_batch.size(1) == 1:
                img_batch = torch.cat([img_batch] * 3, dim=1)
            output = model(img_batch)

            loss = criterion(output, label_batch)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.7f} ({loss.avg:.7f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses)
                logger.info(msg)

        perf_indicator = -losses.avg

        dict_writer.writerow({
            'Epoch': epoch,
            'Loss': losses.avg,
            'Performance Indicator': perf_indicator,
        })
        summary_writer.add_scalar(
            'Loss/validation',
            losses.avg,
            epoch)

    return perf_indicator
