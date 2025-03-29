import signal
import os
import numpy as np
import sys
import time

from torch.utils.data import DataLoader
from utils.config import Config
from models.architectures import QuadNet

from datasets.point2Quadrilateral import *
from utils.trainer import ModelTrainer



class point2QuadrilateralConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    # Dataset name
    dataset = 'point2Quadrilateral'
    phase = 'train'
    data_dir = '/user38/data/mesh/quadrilateral/20240630/'
    
    # Number of CPU threads for the input pipeline
    input_threads = 12

    #########################
    # Architecture definition
    #########################

    # Define layers
    
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']
    '''
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']
    '''

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Radius of the input sphere (decrease value to reduce memory cost)
    in_radius = 1.2

    # Size of the first subsampling grid in meter (increase value to reduce memory cost)
    first_subsampling_dl = 0.03

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 5.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1#.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 128
    #in_features_dim = 5
    in_face_features_dim = 12+17

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02
    '''
    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points
    '''
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 500
    class_w = [1.0,4.0] #[1.0,4]

    # Learning rate management
    learning_rate = 1e-3
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}#150
    grad_clip_norm = 100.0

    # Number of batch (decrease to reduce memory cost, but it should remain > 3 for stability)
    batch_num = 1

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 5

    # Number of epoch between each checkpoint
    checkpoint_gap = 50
  
    # Do we need to save convergence
    saving = True
    saving_path = None


if __name__ == '__main__':

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


    ###############
    # Previous chkp
    ###############

    # Choose here if you want to start training from a previous snapshot (None for new training)
    #previous_training_path = 'Log_2020-03-19_19-53-27'
    previous_training_path = ''

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None


    print('Data and Model Preparation')
    print('****************')
    t1 = time.time()

    # Initialize configuration class
    config = point2QuadrilateralConfig()
    if previous_training_path:
        config.load(os.path.join('results', previous_training_path))
        config.saving_path = None

    # Get path from argument if given
    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]
                       
    # Initialize datasets   
    #training_dataset = point2QuadDataset(config, set='training', use_potentials=True)
    #test_dataset = point2QuadDataset(config, set='validation', use_potentials=True)
    training_dataset = object2QuadDataset(config, set='training')
    
    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,#4
                                 shuffle=True,
                                 collate_fn=point2QuadCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=False)#False/True
    
    net = QuadNet(config)



    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    try:
        #trainer.train(net, training_loader, test_loader, config)
        trainer.train(net, training_loader, config)
    except:
        print('Caught an error')
        os.kill(os.getpid(), signal.SIGINT)
    
    out_dir = './test_results/fake_point/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    config.phase = 'test'
    #config.data_dir = '/user36/data/mesh/quadrilateral/third/test_no_dupl/'
    test_dataset = object2QuadDataset(config, set='test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=point2QuadCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)
    
    baseNames = [] 
    for i in range(test_dataset.num_mesh):
        baseNames.append(test_dataset.mesh_names[i][test_dataset.start:-2])
    trainer.cloud_reconstruction(net, test_loader, config, baseNames, out_dir)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)