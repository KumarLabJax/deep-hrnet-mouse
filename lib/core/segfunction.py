# similar to function.py but modified for segmentation

import time
import logging

import torch
import torch.nn.functional as torchf

from core.function import AverageMeter

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch, dict_writer=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img_batch, target_batch) in enumerate(train_loader):

        # turn grayscale into 3 channels for model
        img_batch = img_batch.cuda()
        img_batch = torch.cat([img_batch] * 3, dim=1)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output_batch = model(img_batch)

        target_batch = target_batch.cuda(non_blocking=True)
        loss = criterion(output_batch, target_batch)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        losses.update(loss.item(), img_batch.size(0))

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
                      speed=img_batch.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

        if dict_writer:
            dict_writer.writerow({
                'Epoch': epoch,
                'Batch': i,
                'Loss': loss.item(),
                'Batch Time': elapsed_time,
                'Batch Size': img_batch.size(0),
            })


def validate(config, val_loader, model, criterion, dict_writer=None, epoch=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (img_batch, target_batch) in enumerate(val_loader):

            # turn grayscale into 3 channels for model
            img_batch = img_batch.cuda()
            img_batch = torch.cat([img_batch] * 3, dim=1)

            num_images = img_batch.size(0)

            # compute output
            output_batch = model(img_batch)

            target_batch = target_batch.cuda(non_blocking=True)
            loss = criterion(output_batch, target_batch)

            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            output_mask = output_batch >= 0.0
            avg_acc = 1.0 - torch.abs(target_batch - output_mask.float()).mean()
            acc.update(avg_acc.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.7f} ({loss.avg:.7f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

        if dict_writer:
            dict_writer.writerow({
                'Epoch': epoch,
                'Loss': losses.avg,
                'Accuracy': acc.avg,
            })

    return acc.avg
