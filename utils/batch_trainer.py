# Basic libs
import torch
import numpy as np
from os import makedirs, remove
from os.path import exists, join
import time

# PLY reader
from utils.ply import write_ply

# Metrics
from utils.config import Config

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


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
        self.edge_threshold = config.edge_threshold

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

    def train(self, net, training_loader, val_loader, config):
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
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start training loop
        for epoch in range(config.max_epoch):

            # Remove File for kill signal
            if epoch == config.max_epoch - 1 and exists(PID_file):
                remove(PID_file)

            self.step = 0
            for points,labels,neighbors in training_loader:

                # Check kill signal (running_PID.txt deleted)
                if config.saving and not exists(PID_file):
                    continue

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    points=points.to(self.device)
                    labels=labels.to(self.device)
                    neighbors=neighbors.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = net(points,neighbors)
                loss = net.loss(outputs, labels)
                
                acc = net.accuracy(outputs, labels)

                t += [time.time()]

                # Backward + optimize
                loss.backward()
                #bias_grd = (outputs-batch.labels).detach()
                #midp.backward(bias_grd)

                if config.grad_clip_norm > 0:
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()

                
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Average timing
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => L={:.3f} acc={:3.0f}% / t(ms): {:5.1f} {:5.1f} {:5.1f})'
                    print(message.format(self.epoch, self.step,
                                         loss.item(),
                                         100*acc,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         1000 * mean_dt[2]))

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        message = '{:d} {:d} {:.3f} {:.3f} {:.3f}\n'
                        file.write(message.format(self.epoch,
                                                  self.step,
                                                  net.output_loss,
                                                  acc,
                                                  t[-1] - t0))


                self.step += 1

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
                # Get current state dict
                save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)

                # Save checkpoints occasionally
                if (self.epoch + 1) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                    torch.save(save_dict, checkpoint_path)

            # Validation
            net.eval()
            self.validation(net, val_loader, config)
            net.train()

        print('Finished Training')
        return

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def validation(self, net, val_loader, config: Config):
        t0 = time.time()
        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        sigmoid = torch.nn.Sigmoid()


        # Number of classes predicted by the model
        nc_model = config.num_neibors


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
        for i, (points,labels,neighbors) in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                points=points.to(self.device)
                labels=labels.to(self.device)
                neighbors=neighbors.to(self.device)

            # Forward pass
            outputs = net(points,neighbors).squeeze()

            # Get probs and labels
            stacked_probs = sigmoid(outputs).cpu().detach().numpy()
            labels = labels.short()
            
            predicted = torch.zeros_like(outputs,dtype=torch.short)
            predicted[stacked_probs>self.edge_threshold]=1
        
            total = labels.size(0)*labels.size(1)
            #print('labels.shape=',labels.shape)
            correct = (predicted == labels).sum().item()
            acc = correct / total
            mean_acc += acc

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Accuracy: {:.1f}%  ; Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(acc,
                                     100 * i / config.validation_size,
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

    def cloud_reconstruction(self, net, test_loader, config, save_dir = './train_results/'):
        """
        Test method for cloud segmentation models
        """
        sigmoid = torch.nn.Sigmoid()

        # Number of classes predicted by the model
        nc_model = config.num_neibors
        mean_acc = 0
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        mean_acc = 0

        # Start validation loop
        for i, (points,labels,neighbors) in enumerate(test_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                points=points.to(self.device)
                labels=labels.to(self.device)
                neighbors=neighbors.to(self.device)
            # Forward pass
            outputs = net(points,neighbors)

            # Get probs and labels
            stacked_probs = sigmoid(outputs).cpu().detach().numpy()
            labels = labels.short()
            neighbor_index = neighbor_index.short()
            
            predicted = torch.zeros_like(outputs,dtype=torch.short)
            predicted[stacked_probs>config.edge_threshold]=1

            total = labels.size(0)*labels.size(1)
            #print('labels.shape=',labels.shape)
            correct = (predicted == labels).sum().item()
            acc = correct / total
            mean_acc += acc
            
            predicted = predicted.cpu().detach().numpy()
            labels = labels.cpu().numpy()
            neighbor_index = neighbor_index.cpu().numpy()
            sub_ply_file = join(save_dir, '{:04d}.npz'.format(i))
            np.savez(sub_ply_file, predicted=predicted, labels=labels,neighbor_index=neighbor_index)
            

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Accuracy: {:.1f}%  ; Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(acc,
                                     100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Print instance mean
        print('{:s} mean accuracy = {:.1f}%'.format(config.dataset, 100*mean_acc/len(test_loader)))

