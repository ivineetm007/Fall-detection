
import sys
import numpy as np
import os
from data_management import load_videos,load_optical_flow_dataset
import config

# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
from models import ROI_C3D_AE_no_pool,Fusion_C3D_no_pool
from trainer.fusionroigan import Params,Fusion_ROI_3DCAE_GAN3D
from trainer.util import create_diff_mask
import argparse
#Select data_type
parser = argparse.ArgumentParser(description='Fusion and Region based adversarial model training')
parser.add_argument('--epochstrained', default='0',
                    help='Epoch number of the saved model')
parser.add_argument('--lambda_T', default='0.1',
                    help='MSE loss hyperparameter for thermal frames')
parser.add_argument('--lambda_F', default='0.1',
                    help='MSE loss hyperparameter for flow frames')
args = parser.parse_args()

dset = config.track_root_folder
d_type='ROI_Fusion'
#parameters
epochs=300
epochs_trained=int(args.epochstrained)
LOAD_DATA_SHAPE=config.LOAD_DATA_SHAPE
width, height = LOAD_DATA_SHAPE[0],LOAD_DATA_SHAPE[1]
thermal_channels=1
flow_channels=3
win_length=config.WIN_LENGTH

regularizer_list = ['BN']
break_win=config.SPLIT_GAP
stride=config.STRIDE
lambdas=[float(args.lambda_T),float(args.lambda_F)]
#aggreagte all parameters in Params class
param=Params(width=width, height=height,win_length=win_length,thermal_channels=thermal_channels,flow_channels=flow_channels,dset=dset,d_type=d_type,regularizer_list=regularizer_list,break_win=break_win)
param.thermal_lambda=lambdas[0]
param.flow_lambda=lambdas[1]
#-----------------
#Load train data
#-----------------
#load thermal frames
Fall_videos=load_videos(dset='Thermal_track',vid_class='Fall',input_type='ROI_FRAME')
#load flow frames
load_optical_flow_dataset(vid_class='Fall',videos=Fall_videos)

#-----------------
#MODEL Initialization
#-----------------
TR, TR_name, _ = ROI_C3D_AE_no_pool(img_width=param.width, img_height=param.height, win_length=param.win_length, regularizer_list=param.regularizer_list,channels=param.thermal_channels,d_type='thermal')
FR, FR_name, _ = ROI_C3D_AE_no_pool(img_width=param.width, img_height=param.height, win_length=param.win_length-1, regularizer_list=param.regularizer_list,channels=param.flow_channels,d_type='flow')
D, D_name, _ = Fusion_C3D_no_pool(img_width=param.width, img_height=param.height, win_length=param.win_length, regularizer_list=param.regularizer_list,thermal_channels=param.thermal_channels,flow_channels=param.flow_channels)
param.TR_name=TR_name
param.FR_name=FR_name
param.D_name=D_name
D_path = param.get_D_path(epochs_trained)
TR_path = param.get_TR_path(epochs_trained)
FR_path = param.get_FR_path(epochs_trained)

if os.path.isfile(TR_path) and os.path.isfile(D_path) and os.path.isfile(FR_path):
    TR.load_weights(TR_path)
    FR.load_weights(FR_path)
    D.load_weights(D_path)
    print("Model weights loaded successfully........")
else:
    print("Saved model weights not found......")
    sys.exit(0)
#-----------------
#model training
#-----------------
#load Gan trainer
GAN3D=Fusion_ROI_3DCAE_GAN3D(train_par=param,stride=stride)
GAN3D.initialize_model(T_Reconstructor=TR, F_Reconstructor=FR, Discriminator=D)
GAN3D.test(test_videos=Fall_videos, score_type = 'T_R_S', epochs = epochs_trained,plot=True,tolerance_limit=8)
GAN3D.test(test_videos=Fall_videos, score_type = 'F_R', epochs = epochs_trained,plot=False,tolerance_limit=8)
GAN3D.test(test_videos=Fall_videos, score_type = 'R', epochs = epochs_trained,plot=False,tolerance_limit=8)
