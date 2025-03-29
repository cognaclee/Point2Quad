# -*- coding: utf-8 -*-
# @Time        : 22/05/2024 17:10 PM
# @Description :The code is modified from PointCloudDataset
# @Author      : li zezeng
# @Email       : zezeng.lee@gmail.com

import h5py
import numpy as np
import torch
from multiprocessing import Lock
# Dataset parent class
from datasets.common import PointCloudDataset
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')
from glob import glob
import gc
# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class point2QuadDataset(PointCloudDataset):
    """Class to handle mesh dataset."""

    def __init__(self, config, set='training', use_potentials=True):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        PointCloudDataset.__init__(self)

        # Parameters from config
        self.config = config

        # Training or test set
        self.set = set

        # Using potential or random epoch generation
        self.use_potentials = use_potentials

        # Number of models used per epoch
        if self.set == 'training':
            self.epoch_n = config.epoch_steps * config.batch_num
        elif self.set in ['validation', 'test']:
            self.epoch_n = config.validation_size * config.batch_num
        else:
            raise ValueError('Unknown set for mesh data: ', self.set)


        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')


        # Start loading
        self.input_points,self.input_faces,self.input_labels = self.load_points_candidates(self.config.training_data_name)
        self.num_mesh = self.input_points.shape[0]

        ############################
        # Batch selection parameters
        ############################


        self.worker_lock = Lock()

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return self.num_mesh

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """
        if self.use_potentials:
            return self.potential_item(batch_i)
        else:
            return self.random_item(batch_i)

    def potential_item(self, batch_i):
        # Initiate concatanation lists
        p_list = []
        s_list = []

        # Collect labels and colors
        faces = self.input_faces[batch_i,:,:]
        if self.set =='test':
            labels = np.zeros(points.shape[0])       
        else:
            labels = self.input_labels[batch_i,:]

        # Data augmentation
        if self.set == 'training':
            points, scale, R = self.augmentation_transform(points)
        else:
            scale = np.ones(points.shape[1])
            R = np.eye(points.shape[1])

        # Get original height as additional feature
        input_features = points[:, 2:]
        

        # Stack batch
        p_list += [points]
        s_list += [scale]


        ###################
        # Concatenate batch
        ###################

        stacked_points = points
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = R

        # Input features
        if self.config.in_features_dim == 1:
            stacked_features = input_features
        elif self.config.in_features_dim == 2:
            stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
            stacked_features = np.hstack((stacked_features, input_features))
        elif self.config.in_features_dim == 3:
            stacked_features = points
        else:
            raise ValueError('Only accepted input dimensions are 1, 2 and 3 (without and with XYZ)')
        

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)
        
        

        # Add scale and rotation for testing
        input_list += [scales, rots, faces]

        return input_list

    def random_item(self, batch_i):

        # Initiate concatanation lists
        p_list = []
        l_list = []
        s_list = []
        
        # Get points from tree structure
        points = self.input_points[batch_i,:,:]


        # Collect labels and colors

        faces = self.input_faces[batch_i,:,:]
        if self.set =='test':
            labels = np.zeros(points.shape[0])       
        else:
            labels = self.input_labels[batch_i,:]

        # Data augmentation
        if self.set == 'training':
            input_points, scale, R = self.augmentation_transform(input_points)
        else:
            scale = np.ones(input_points.shape[1])
            R = np.eye(input_points.shape[1])

        # Color augmentation
        input_features = points[:, 2:]

        # Stack batch
        p_list += [input_points]
        l_list += [labels]
        s_list += [scale]


        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = R

        # Input features
        if self.config.in_features_dim == 1:
            stacked_features = input_features
        elif self.config.in_features_dim == 2:
            stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
            stacked_features = np.hstack((stacked_features, input_features))
        elif self.config.in_features_dim == 3:
            stacked_features = input_points
        else:
            raise ValueError('Only accepted input dimensions are 1, 2 and 3 (without and with XYZ)')
            

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)

        # Add scale and rotation for testing
        input_list += [scales, rots, faces]

        return input_list


    def load_points_candidates(self,file_name):
        # Fill data containers
        f = h5py.File(file_name, 'r')
        print(f.keys())
        points = f['points'][:]
        faces = f['faces'][:]
        labels = f['labels'][:]
        f.close()
        assert faces.shape[0] == labels.shape[0], 'The number of face and label are not equal'
        print(points.shape)
        print(faces.shape)
        print(labels.shape)
        print(type(points))
        
        return points,faces,labels



class point2QuadCustomBatch:
    """Custom batch definition with memory pinning for Quadrilateral mesh"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 5) // 5

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind]).long()
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.faces = torch.from_numpy(input_list[ind]).long()
        #ind += 1
        #self.samples = torch.from_numpy(input_list[ind])
        ind += 1
        self.f_infos = torch.from_numpy(input_list[ind])
        gc.collect()

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.faces = self.faces.pin_memory()
        self.f_infos = self.f_infos.pin_memory()
        #self.samples = self.samples.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.faces = self.faces.to(device)
        self.f_infos = self.f_infos.to(device)
        #self.samples = self.samples.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def point2QuadCollate(batch_data):
    return point2QuadCustomBatch(batch_data)
    
    
    
class object2QuadDataset(PointCloudDataset):
    """Class to handle mesh dataset."""

    def __init__(self, config, set='training', max_face_nums=25000):
        PointCloudDataset.__init__(self)

        # Parameters from config
        self.config = config
        # Training or test set
        self.set = set

        # Start loading
        self.mesh_names = sorted(glob(config.data_dir+'/*.'+'h5'))
        self.mesh_names = self.mesh_names#[11:66]
        self.start = len(config.data_dir)
        self.num_mesh = len(self.mesh_names)
        self.max_face_nums = max_face_nums
        
        '''print('self.mesh_names[1970]',self.mesh_names[1970])
        print('self.mesh_names[1971]',self.mesh_names[1971])
        print('self.mesh_names[1972]',self.mesh_names[1972])'''
        print('config.data_dir=',config.data_dir)

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return self.num_mesh

    def __getitem__(self, batch_i):
        # Initiate concatanation lists
        p_list = []
        s_list = []

        # Get potential points from tree structure
        #points,faces,labels,samples = self.load_points_candidates(self.mesh_names[batch_i])
        #print('self.mesh_names',self.mesh_names[batch_i])
        points,faces,labels,f_infos = self.load_points_candidates(self.mesh_names[batch_i])
        

        # Collect labels and colors
        

        # Data augmentation
        if self.set == 'training':
            points, scale, R = self.augmentation_transform(points)
            
            #samples = samples - center_point
            #samples = np.sum(np.expand_dims(samples, 2) * R, axis=1) * scale
        else:
            scale = np.ones(points.shape[1])
            R = np.eye(points.shape[1])

        # Get original height as additional feature
        input_features = points[:, 2:]
        

        # Stack batch
        p_list += [points]
        s_list += [scale]


        ###################
        # Concatenate batch
        ###################

        stacked_points = points
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = R

        # Input features
        if self.config.in_features_dim == 1:
            stacked_features = input_features
        elif self.config.in_features_dim == 2:
            stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
            stacked_features = np.hstack((stacked_features, input_features))
        elif self.config.in_features_dim == 3:
            stacked_features = points.copy()
        else:
            raise ValueError('Only accepted input dimensions are 1, 2 and 3 (without and with XYZ)')
        
        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)
        
        

        # Add scale and rotation for testing
        #input_list += [scales, rots, point_inds,faces,samples]
        input_list += [scales, rots, faces,f_infos]
        gc.collect()

        return input_list


    def load_points_candidates(self,file_name):
        # Fill data containers
        #print('file_name: ',file_name)
        f = h5py.File(file_name, 'r')
        points = f['points'][:]
        faces = f['faces'][:]
        labels = f['labels'][:]
        #labels = f['f_labels'][:]
        #f_infos = f['face_infos'][:]
        f_infos_all = f['face_infos'][:]
        f_infos = f_infos_all[:,1:]
        f_infos[:,0] = f_infos_all[:,0]
        #f_infos[:,6:9] = 0.25*(f_infos_all[:,6:9]+f_infos_all[:,9:12]+f_infos_all[:,12:15]+f_infos_all[:,15:18])
        
        f.close()

        face_nums = faces.shape[0]
        assert face_nums == labels.shape[0], 'The number of face and label are not equal'
        '''if face_nums>self.max_face_nums and self.set=='training':
            chs = np.random.randint(low=0,high=face_nums,size=(self.max_face_nums,),dtype='int')
            faces = faces[chs,:]
            labels = labels[chs]
            #samples = samples[chs,:]'''
        gc.collect()
        
        return points,faces,labels,f_infos
