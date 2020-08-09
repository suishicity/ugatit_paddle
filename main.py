import os
import sys
import yaml
import argparse
import paddle
import paddle.fluid as fluid
import numpy as np
from easydict import EasyDict

from dataset import ImageFolder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/ugatit.yaml')
    parser.add_argument('--resume', type=bool, action='store_true')

    args = parser.parse_args()
    cfg = EasyDict(yaml.full_load(open(args.config, 'r')))
    cfg.resume = args.resume
    return cfg


        
def main():
    cfg = parse_args()
    