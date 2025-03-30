# -*- coding: utf-8 -*-
# @Time        : 24/04/2024 23:48 PM
# @Author      : li zezeng
# @Email       : zezeng.lee@gmail.com

import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

# Metrics
from sklearn.metrics import confusion_matrix


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True):
        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        checkpoint = torch.load(chkp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return


    def cloud_reconstruction_test(self, net, test_loader, config, baseName, save_dir = './test_results/'):
        """
        Test method for cloud segmentation models
        """
        
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        mean_acc = 0.0
        mean_pre = 0.0
        mean_rec = 0.0
        
        obj_nums = len(test_loader)

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
                correct_vec = (predicted == labels)
                correct = np.sum(correct_vec)
                acc = correct / total
                mean_acc += acc
                
                TP = np.sum(correct_vec[labels==1])
                precision = TP/(np.sum(predicted))
                recall = TP/(np.sum(labels))
                mean_pre +=precision
                mean_rec +=recall
                
                #print('outputs.shape=',outputs.shape)
                probs =  outputs[:,1].cpu().detach().numpy()
                faces = batch.faces.cpu().numpy()
                input_points =   batch.points[0].cpu().detach().numpy()
                #sub_ply_file = join(save_dir, '{:04d}.npz'.format(i))
                sub_ply_file = join(save_dir, baseName[i]+'npz')
                np.savez(sub_ply_file, predicted=predicted, labels=labels,cand_faces=faces,input_points=input_points,probability=probs)
            

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Accuracy: {:.1f}%  ;  (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * acc,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Print instance mean
        #print('{:s} mean accuracy = {:.1f}%'.format(config.dataset, 100*mean_acc/len(test_loader)))
        print('{:s} mean precision = {:.5f}%, mean recall = {:.5f}%, mean accuracy = {:.5f}%'.format(config.dataset, 100*mean_pre/obj_nums, 100*mean_rec/obj_nums, 100*mean_acc/obj_nums))


