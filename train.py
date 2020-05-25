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
    
    args = parser.parse_args()

    return args

def reset_config(config, args):

    if args.gpus:
        config.GPUS = args.gpus


def main():


    args = parse_args()
    reset_config(config, args)

    # tensorboard
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train', 'train')
    
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = True
    
    model = Network(config, gt.DARTS)
    model.init_weights()
    
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    
    logger.info("param size = %fMB", count_parameters_in_MB(model))
    
    
    gpus = [int(i) for i in config.GPUS.split(',')]
    criterion = JointsMSELoss(use_target_weight = config.LOSS.USE_TARGET_WEIGHT).to(device)
    model = nn.DataParallel(model, device_ids=gpus).to(device)
    
    logger.info("Logger is set - training start")


    # weights optimizer
    optimizer = torch.optim.Adam(model.parameters(), config.TRAIN.LR)
                                
                                
    # prepare dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.'+config.DATASET.DATASET)(
                            config,
                            config.DATASET.ROOT,
                            config.TRAIN.TRAIN_SET,
                            True,
                            transforms.Compose([
                               transforms.ToTensor(),
                               normalize,
                            ]))
                            
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
                            config,
                            config.DATASET.ROOT,
                            config.TRAIN.TEST_SET,
                            False,
                            transforms.Compose([
                               transforms.ToTensor(),
                               normalize,
                            ]))
                           

  
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
                                               shuffle=True,
                                               num_workers=config.WORKERS,
                                               pin_memory=True)
                                               
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
                                               shuffle=False,
                                               num_workers=config.WORKERS,
                                               pin_memory=True)
                                               
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)

    # training loop
    best_top1 = 0.
    best_model = False
    for epoch in range(config.TRAIN.EPOCHS):


        # training
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        # validation
        top1 = validate(
            config, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_model = True
        else:
            best_model = False
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': best_top1,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)
        
        lr_scheduler.step()

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    logger.info('=> best accuracy is {}'.format(best_top1))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()



if __name__ == "__main__":
    main()
    
    
    
