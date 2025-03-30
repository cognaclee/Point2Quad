# -*- coding: utf-8 -*-
# @Time        : 22/04/2024 17:10 PM
# @Description :The code is modified from kpconv
# @Author      : li zezeng
# @Email       : zezeng.lee@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import *
import numpy as np
from torch.cuda.amp import autocast
from torchvision.ops import sigmoid_focal_loss
import torch.nn.init as init
import gc


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class PointAttention(nn.Module):
    def __init__(self, in_dim):
        super(PointAttention, self).__init__()
        layer = in_dim//4

        self.w_qs = nn.Conv1d(in_dim, layer, 1)
        self.w_ks = nn.Conv1d(in_dim, layer, 1)
        self.w_vs = nn.Conv1d(in_dim, in_dim, 1)
        self.gamma=nn.Parameter(torch.tensor(torch.zeros([1]))).cuda()

    def forward(self, inputs):
        q = self.w_qs(inputs)
        k = self.w_ks(inputs)
        v = self.w_vs(inputs)
        
        s = torch.matmul(q.transpose(2, 1), k)

        beta = F.softmax(s, dim=-1)  # attention map
        o = torch.matmul(v, beta)   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        x = self.gamma * o + inputs
        print('PointAttention.shape=',x.shape)
        return x

class PointResNet(nn.Module):
    def __init__(self, in_dim,layers):
        super(PointResNet, self).__init__()
        
        chs = in_dim
        fea_dims = []
        for i in range(layers):
            fea_dims.append(chs)
            chs *= 2
        fea_dims.append(chs)
        for i in range(layers):
            fea_dims.append(chs)
            chs = chs //2
        #fea_dims.append(in_dim)   
        
        sequence = []
        for i in range(len(fea_dims)-1):
            sequence += [nn.Conv1d(fea_dims[i],fea_dims[i+1], 1),
            nn.GroupNorm(1,fea_dims[i+1]),
            nn.ReLU()]
        
        sequence += [nn.Conv1d(fea_dims[-1],in_dim, 1)]
        self.net = nn.Sequential(*sequence)

    def forward(self, x):
        res = self.net(x)
        out = x + res
        return out

class QuadNet(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config):
        super(QuadNet, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.fea_dim = config.output_features_dim
        self.head_mlp = UnaryBlock(out_dim, self.fea_dim, False, 0) 
        #self.head_softmax = UnaryBlock(self.fea_dim, self.C, False, 0, no_relu=True)
        
        
        sequence = [nn.Conv1d(config.in_face_features_dim, 128, 1),nn.InstanceNorm1d(128, affine=True),nn.LeakyReLU()]
        #sequence += [PointAttention(64)]
        sequence += [nn.Conv1d(128, 128, 1),nn.InstanceNorm1d(128, affine=True),nn.LeakyReLU()]
        sequence += [PointResNet(128,2)]
        sequence += [nn.Conv1d(128, 256, 1),nn.InstanceNorm1d(256, affine=True),nn.LeakyReLU()]
        sequence += [nn.Conv1d(256, 512, 1),nn.InstanceNorm1d(512, affine=True),nn.LeakyReLU()]
        sequence += [nn.Conv1d(512, config.out_face_features_dim, 1)]
        
        self.face_encoder = nn.Sequential(*sequence)
        #initialize_weights(self.face_encoder)
        

        
        self.part_num = 2
        #sequence = [torch.nn.Conv1d(self.fea_dim * 4+config.out_face_features_dim, 512, 1),nn.InstanceNorm1d(512, affine=True),nn.Dropout(p=0.1),nn.ReLU()]
        sequence = [torch.nn.Conv1d(self.fea_dim * 4, 512, 1),nn.InstanceNorm1d(512, affine=True),nn.Dropout(p=0.1),nn.ReLU()]
        sequence += [nn.Conv1d(512, 256, 1),nn.InstanceNorm1d(256, affine=True),nn.ReLU()]
        sequence += [nn.Conv1d(256, 128, 1),nn.InstanceNorm1d(128, affine=True),nn.ReLU()]
        sequence += [nn.Conv1d(128, 64, 1),nn.InstanceNorm1d(64, affine=True),nn.ReLU()]
        sequence += [nn.Conv1d(64, self.part_num, 1)]

        self.tail = nn.Sequential(*sequence)
        initialize_weights(self.tail)
        
        ################
        # Network Losses
        ################

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            #self.criterion = torch.nn.CrossEntropyLoss(weight=config.class_w, ignore_index=-1,reduction='none')
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
            print('class weigted crossEntropy loss')
        else:
            #self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1,reduction='none')
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
            print('vanilla crossEntropy loss')
        return


    @autocast()
    def forward(self, batch, config):
        # Get input features
        x = batch.features.clone().detach()
        faces = batch.faces.clone().detach()
        N, _ = x.size()
        M, _ = faces.size()
        
        #print('x.shape=',x.shape)
        
        f0 = faces.unsqueeze(1).repeat(1, x.shape[1], 1)
        x_f = x.unsqueeze(-1).repeat(1, 1, 4).gather(0, f0).permute(0, 2, 1).contiguous().view(1,x.shape[1] * 4, M)
        f_info = batch.f_infos.clone().detach().permute(1, 0).unsqueeze(0).contiguous()
        #print('x_f.shape=',x_f.shape)
        #print('f_info.shape=',f_info.shape)
        x_f  = torch.cat((x_f,f_info), dim=1)
        x_face1 = self.face_encoder(x_f)
        
        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            torch.cuda.empty_cache()
            x = block_op(x, batch)
            torch.cuda.empty_cache()

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                one = skip_x.pop()
                x = torch.cat([x, one], dim=1)
                del one
            torch.cuda.empty_cache()
            x = block_op(x, batch)
            torch.cuda.empty_cache()
        

        # Head of network
        
        x1 = self.head_mlp(x, batch).view(N, self.fea_dim).transpose(1, 0).unsqueeze(-1).repeat(1, 1, 4)
        faces = faces.unsqueeze(0).repeat(self.fea_dim, 1, 1)
        '''
        x_face = x1.gather(1, faces).permute(0, 2, 1).contiguous().view(1,self.fea_dim * 4, M)
        '''
        x_face2 = x1.gather(1, faces).permute(0, 2, 1).contiguous().view(1,self.fea_dim * 4, M)
        x_face  = torch.cat((x_face1,x_face2), dim=1)
        
        
        x2 = self.tail(x_face).transpose(2, 1).contiguous().view(-1, self.part_num)
        
        #print('x_face2=',torch.isnan(x_face2).any())
        #print('x2=',torch.isnan(x2).any())

        out = F.log_softmax(x2, dim=-1)
        del skip_x
        gc.collect()
        return out

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)
        #self.output_loss = sigmoid_focal_loss(outputs, labels,reduction='mean')
        return self.output_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """
        with torch.no_grad():
            predicted = torch.argmax(outputs.data, dim=1).short().cpu().detach().numpy()
            total = labels.numel()
            labels = labels.short().cpu().detach().numpy()
            correct = np.sum(predicted == labels)

            return correct / total
