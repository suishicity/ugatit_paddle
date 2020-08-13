import os
import sys
import itertools
import collections
import time
import datetime
import yaml
import argparse
import os.path as osp
import paddle
import paddle.fluid as fluid
import numpy as np
from easydict import EasyDict
from visualdl import LogWriter

from dataset import ImageFolder
from ugatit import UGATIT
from utils import image_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/ugatit.yaml')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()
    cfg = EasyDict(yaml.full_load(open(args.config, 'r')))
    cfg.resume = args.resume
    return cfg


def screen_print(loss_dict, iteraion, num_itrations, time_left):
    msg = "\r[Iteration %d/%d] " % (
        iteraion, num_itrations
    )
    for k, v in loss_dict.items():
        msg += "%s: %.3e " % (k, v)
    time_left = datetime.timedelta(seconds=round(time_left))
    msg += "ETA: %s" % time_left
    sys.stdout.write(msg)

        
def main():
    fluid.enable_dygraph()

    cfg = parse_args()

    trainA = ImageFolder(osp.join(cfg.data_dir, 'trainA'), transform=image_transform(mode='train'))
    trainB = ImageFolder(osp.join(cfg.data_dir, 'trainB'), transform=image_transform(mode='train'))
    testA = ImageFolder(osp.join(cfg.data_dir, 'testA'), transform=image_transform(mode='test'))
    testB = ImageFolder(osp.join(cfg.data_dir, 'testB'), transform=image_transform(mode='test'))

    trainA_loader = fluid.io.batch(fluid.io.shuffle(trainA.reader, 500), batch_size=cfg.batch_size, drop_last=True)
    trainB_loader = fluid.io.batch(fluid.io.shuffle(trainB.reader, 500), batch_size=cfg.batch_size, drop_last=True)
    testA_loader = fluid.io.batch(testA.reader, batch_size=1)
    testB_loader = fluid.io.batch(testB.reader,  batch_size=1)

    trainA_iter = iter(trainA_loader())
    trainB_iter = iter(trainB_loader())
    testA_iter = itertools.cycle(testA_loader())
    testB_iter = itertools.cycle(testB_loader())

    trainer = UGATIT(cfg)
    start = 1
    if cfg.resume:
        start = trainer.resume()
    hist = collections.defaultdict(lambda : 0)

    trainer.sample_images(start, {'A': next(testA_iter), 'B': next(testB_iter)})
    start_time = time.time()
    
    for iteration in range(0, cfg.iterations+1):
        try: 
            imgA, imgB = next(trainA_iter), next(trainB_iter)
        except:
            trainA_iter = iter(trainA_loader)
            trainB_iter = iter(trainB_loader)
            imgA, imgB = next(trainA_iter), next(trainB_iter)
        
        losses = trainer.step({'A': imgA, 'B': imgB})
        time_left = (time.time() - start_time()) / (iteration - start + 1) * (cfg.iterations - iteration + 1)
        screen_print(losses, iteration, cfg.iteraionts, time_left)

        for k, v in losses.items():
            hist[k] += v
        if iteration % cfg.log_freq == 0:
           pass 


if __name__ == "__main__":

    main()