""" Training augmented model """
import os
import sys
import argparse
import time
import glob
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from core.loss import JointsMSELoss
from core.config import config
from core.config import update_config
from core.function import *

from models.model_augment import Network
from models import genotypes as gt
import dataset

from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import count_parameters_in_MB


device = torch.device("cuda")


        
def parse_args():

    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # searching
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)

    parser.add_argument('--batch_size',
                        help='bs',
                        type=int)

    parser.add_argument('--test_weight',
                        help='weights',
                        type=str)
    
    args = parser.parse_args()

    return args

def reset_config(config, args):

    if args.gpus:
        config.GPUS = args.gpus

    if args.batch_size:
        config.TEST.BATCH_SIZE = args.batch_size

    if args.test_weight:
        config.TEST.MODEL_FILE = args.test_weight


def main():


    args = parse_args()
    reset_config(config, args)

    # tensorboard
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'test', 'valid')
    
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = True
    
    model = Network(config, gt.DARTS)
    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
   
    
    gpus = [int(i) for i in config.GPUS.split(',')]
    criterion = JointsMSELoss(use_target_weight = config.LOSS.USE_TARGET_WEIGHT).to(device)
    model = nn.DataParallel(model, device_ids=gpus).to(device)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
                            config,
                            config.DATASET.ROOT,
                            config.TRAIN.TEST_SET,
                            False,
                            transforms.Compose([
                               transforms.ToTensor(),
                               normalize,
                            ]))
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=config.TEST.BATCH_SIZE*len(gpus),
                                               shuffle=False,
                                               num_workers=config.WORKERS,
                                               pin_memory=True)

  
    validate(config, valid_loader, valid_dataset, model, criterion,
                 final_output_dir, tb_log_dir)
    
    
    
if __name__ == "__main__":
    main()
    
    
    
