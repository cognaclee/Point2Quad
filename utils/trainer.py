# -*- coding: utf-8 -*-
# @Time        : 23/04/2022 17:10 PM
# @Description :The code is modified from kpconv
# @Author      : li zezeng
# @Email       : zezeng.lee@gmail.com

# Basic libs
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys


# Metrics
from utils.config import Config
from sklearn.neighbors import KDTree
from metrics.utils import not_quadrilateral_coplane
from metrics.losses import dist_surface_to_quadrilateral_probs,surface_loss


import warnings
warnings.filterwarnings("ignore")

import gc
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#
def degree_loss(out,lower_thresh,upper_thresh):
    nums = out.shape[0]
    up = torch.ge(out, upper_thresh)
    lw = torch.le(out, lower_thresh)
    #loss = torch.true_divide(torch.sum(torch.pow(out[up]-upper_thresh,2))+torch.sum(torch.pow(lower_thresh-out[lw],2)),nums)
    loss = torch.true_divide(torch.sum(up*torch.pow(out-upper_thresh,2))+torch.sum(lw*torch.pow(lower_thresh-out,2)),nums)
    return loss 


def degree_loss(out,lower_thresh,upper_thresh):
    nums = out.shape[0]
    up = torch.ge(out, upper_thresh)
    lw = torch.le(out, lower_thresh)
    loss = torch.true_divide(torch.sum(up)+torch.sum(lw),nums)
    return loss 
 

class ModelTrainer:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, finetune=False, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        ############
        # Parameters
        ############

        # Epoch index
        self.epoch = 0
        self.step = 0
        self.epoch_face = 50

        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in net.named_parameters() if 'offset' in k]
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k]
        deform_lr = config.learning_rate * config.deform_lr_factor
        self.optimizer = torch.optim.SGD([{'params': other_params},
                                          {'params': deform_params, 'lr': deform_lr}],
                                         lr=config.learning_rate,
                                         momentum=config.momentum,
                                         weight_decay=config.weight_decay)

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        if (chkp_path is not None):
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                net.train()
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                config.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------
    def train(self, net, training_loader, config):
        """
        Train the model on a particular dataset.
        """

        ################
        # Initialization
        ################

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps out_loss offset_loss train_accuracy time\n')

            # Killing file (simply delete this file when you want to stop the training)
            PID_file = join(config.saving_path, 'running_PID.txt')
            if not exists(PID_file):
                with open(PID_file, "w") as file:
                    file.write('Launched with PyCharm')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None
            PID_file = None

        # Loop variables
        mean_dt = np.zeros(1)
        best_acc = 0

        # Start training loop
        for epoch in range(config.max_epoch):

            # Remove File for kill signal
            if epoch == config.max_epoch - 1 and exists(PID_file):
                remove(PID_file)

            self.step = 0
            #print("1:{}".format(torch.cuda.memory_allocated(0)))
            for batch in training_loader:

                # Check kill signal (running_PID.txt deleted)
                if config.saving and not exists(PID_file):
                    continue

                ##################
                # Processing batch
                ##################

                # New time
                start_time = time.time()

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = net(batch, config)
                loss = net.loss(outputs, batch.labels.detach())
                
                mid_time = time.time()

                # Backward + optimize
                if self.epoch > self.epoch_face:
                    loss_face = surface_loss(outputs, batch.labels,self.device)
                    loss_all = loss+loss_face
                    loss_all.backward()
                else:
                    #loss_face = 0
                    loss.backward()

                if config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                #print("4:{}".format(torch.cuda.memory_allocated(0)))

                #torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)
                #t += [time.time()]
                end_time = time.time()

                # Average timing
                acc = net.accuracy(outputs.detach(), batch.labels)
                if self.epoch > self.epoch_face:
                    message = 'e{:03d}-i{:04d} => L_BCE={:.3f}  L_Face={:.3f}  acc={:3.0f}% / t(ms): {:5.1f} {:5.1f}'
                    print(message.format(self.epoch, self.step,
                                         loss.item(),
                                         loss_face.item(),
                                         100*acc,
                                         1000 * (mid_time-start_time),
                                         1000 * (end_time-start_time)))
                else:
                    message = 'e{:03d}-i{:04d} => L_BCE={:.3f}  acc={:3.0f}% / t(ms): {:5.1f} {:5.1f}'
                    print(message.format(self.epoch, self.step,
                                         loss.item(),
                                         100*acc,
                                         1000 * (mid_time-start_time),
                                         1000 * (end_time-start_time)))

                torch.cuda.empty_cache()
                self.step += 1
                del loss,  outputs, batch
                gc.collect()
                #print("3:{}".format(torch.cuda.memory_allocated(0)))

            ##############
            # End of epoch
            ##############

            # Check kill signal (running_PID.txt deleted)
            if config.saving and not exists(PID_file):
                break

            # Update learning rate
            if self.epoch in config.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= config.lr_decays[self.epoch]

            # Update epoch
            self.epoch += 1

            # Saving
            if config.saving:
                # Get current best state dict
                '''
                if acc>=best_acc:
                    save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                    # Save current best state of the network (for restoring purposes)
                    checkpoint_path = join(checkpoint_directory, 'best_chkp.tar')
                    torch.save(save_dict, checkpoint_path)
                '''
                save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current best state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)
                

                # Save checkpoints occasionally
                if (self.epoch + 1) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                    torch.save(save_dict, checkpoint_path)

            # Validation
            #net.eval()
            #self.validation(net, val_loader, config)
            #net.train()

        print('Finished Training')
        return

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def validation(self, net, val_loader, config: Config):
        t0 = time.time()
        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        mean_acc = 0

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            predicted = torch.argmax(outputs.data, dim=1).short().cpu().detach().numpy()

            labels = batch.labels.short().cpu().numpy()
            total = labels.size
            correct = np.sum(predicted == labels)
            acc = correct / total
            mean_acc += acc

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Accuracy: {:.1f}%  ; (timings : {:4.2f} {:4.2f})'
                print(message.format(100 *acc,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Print instance mean
        print('{:s} mean accuracy = {:.1f}%'.format(config.dataset, 100*mean_acc/len(val_loader)))

        # Saving (optionnal)
        if config.saving:

            # Save potentials
            if val_loader.dataset.use_potentials:
                pot_path = join(config.saving_path, 'potentials')
                if not exists(pot_path):
                    makedirs(pot_path)
                files = val_loader.dataset.files
                for i, file_path in enumerate(files):
                    pot_points = np.array(val_loader.dataset.input_points[i].data, copy=False)
                    cloud_name = file_path[0].split('/')[-1][:-4]+'.ply'
                    pot_name = join(pot_path, cloud_name)
                    pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
                    
                    write_ply(pot_name,
                            [pot_points.astype(np.float32), pots],
                            ['x', 'y', 'z', 'pots'])

        return

    def cloud_reconstruction(self, net, test_loader, config, baseNames, save_dir = './train_results/'):
        """
        Test method for cloud segmentation models
        """
        mean_acc = 0
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        mean_acc = 0

        # Start validation loop
        for i, batch in enumerate(test_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = net(batch, config)
  
                # Get probs and labels
                predicted = torch.argmax(outputs.data, dim=1).short().cpu().detach().numpy()

                labels = batch.labels.short().cpu().numpy()
                total = labels.size
                correct = np.sum(predicted == labels)
                acc = correct / total
                mean_acc += acc
            
                probs =  outputs[:,1].cpu().detach().numpy()
                faces = batch.faces.cpu().numpy()
                input_points =   batch.points[0].cpu().detach().numpy()
            
                #sub_ply_file = join(save_dir, '{:04d}.npz'.format(i))
                sub_ply_file = join(save_dir, baseNames[i]+'npz')
                np.savez(sub_ply_file, predicted=predicted, labels=labels,cand_faces=faces,input_points=input_points,probability=probs)
            

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Accuracy: {:.1f}%  ; (timings : {:4.2f} {:4.2f})'
                print(message.format(100 *acc,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Print instance mean
        print('{:s} mean accuracy = {:.1f}%'.format(config.dataset, 100*mean_acc/len(test_loader)))

