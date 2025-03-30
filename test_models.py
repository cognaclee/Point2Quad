# -*- coding: utf-8 -*-
# @Time        : 24/04/2024 23:10 PM
# @Author      : li zezeng
# @Email       : zezeng.lee@gmail.com

import signal
import os
import numpy as np
import sys
import torch
import time
# Dataset
from datasets.point2Quadrilateral import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import QuadNet


if __name__ == '__main__':

    chosen_log = 'results/Log_2024-07-01_13-12-08/'

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = -1

    # Choose to test on validation or test split
    on_val = True

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = 'current_chkp.tar'
    print('chosen_chkp=',chosen_chkp)
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)
    config.phase = 'test'
    config.data_dir = '/your/data/path/'

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    #config.batch_num = 3
    #config.in_radius = 4
    config.validation_size = 200
    config.input_threads = 10
    config.class_w = [1.0,1.5]
    config.in_face_features_dim = 12+17

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'
    
    #test_dataset = point2QuadDataset(config, set='validation', use_potentials=True)
    test_dataset = object2QuadDataset(config, set='testing')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=point2QuadCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)


    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = QuadNet(config)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')
    
    out_dir = './test_results/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    baseNames = [] 
    for i in range(test_dataset.num_mesh):
    #for i in range(100):
        baseNames.append(test_dataset.mesh_names[i][test_dataset.start:-2])
    tester.cloud_reconstruction_test(net, test_loader, config, baseNames, out_dir)
