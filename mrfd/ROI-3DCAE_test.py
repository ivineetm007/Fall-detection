
import sys
import numpy as np
import os
from data_management import load_videos,load_optical_flow_dataset
import config
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
from models import ROI_C3D_AE_no_pool,C3D_no_pool
from trainer.roigan import Params,ROI_3DCAE_GAN3D
from trainer.util import create_diff_mask
import argparse
#Select data_type
parser = argparse.ArgumentParser(description='Region based adversarial model training')
parser.add_argument('--datatype', default='thermal',
                    help='thermal or flow roi adversarial model')
parser.add_argument('--epochstrained', default='0',
                    help='Epoch number of the saved model')
parser.add_argument('--lambda_', default='0.1',
                    help='Reconstructor loss hyperparameter')
args = parser.parse_args()

dset = config.track_root_folder
d_type=None
model_data_type=args.datatype
if model_data_type=='flow':
    d_type='ROIopticalFLow'
elif model_data_type=='thermal':
    d_type='ROIframe'
else:
     print('incorrect argument')
     sys.exit(0)
#parameters
epochs_trained=int(args.epochstrained)
LOAD_DATA_SHAPE=config.LOAD_DATA_SHAPE
width, height = LOAD_DATA_SHAPE[0],LOAD_DATA_SHAPE[1]
channels=LOAD_DATA_SHAPE[2]
win_length=config.WIN_LENGTH
if model_data_type=='flow':
    channels=3

regularizer_list = ['BN']
break_win=config.SPLIT_GAP
stride=config.STRIDE
lambda_= float(args.lambda_)
#aggreagte all parameters in Params class
param=Params(width=width, height=height,win_length=win_length,channels=channels,dset=dset,d_type=d_type,regularizer_list=regularizer_list,break_win=break_win,lambda_=lambda_)
#-----------------
#Load train data
#-----------------
Fall_videos=load_videos(dset='Thermal_track',vid_class='Fall',input_type='ROI_FRAME')
if model_data_type=='flow':
    load_optical_flow_dataset(vid_class='Fall',videos=Fall_videos)

#-----------------
#MODEL Initialization
#-----------------
if model_data_type=='thermal':
    ##reconstructor model
    R, R_name, _ = ROI_C3D_AE_no_pool(img_width=param.width, img_height=param.height, win_length=param.win_length, regularizer_list=param.regularizer_list,channels=param.channels)
    ##Dicriminator model
    D, D_name, _ = C3D_no_pool(img_width=param.width, img_height=param.height,  win_length=param.win_length, regularizer_list=param.regularizer_list,channels=param.channels)
elif model_data_type=='flow':
    ##reconstructor model
    R, R_name, _ = ROI_C3D_AE_no_pool(img_width=param.width, img_height=param.height, win_length=param.win_length-1, regularizer_list=param.regularizer_list,channels=param.channels)
    ##Dicriminator model
    D, D_name, _ = C3D_no_pool(img_width=param.width, img_height=param.height,  win_length=param.win_length-1, regularizer_list=param.regularizer_list,channels=param.channels)
param.R_name=R_name
param.D_name=D_name

D_path = param.get_D_path(epochs_trained)
R_path = param.get_R_path(epochs_trained)

if os.path.isfile(R_path) and os.path.isfile(D_path):
    R.load_weights(R_path)
    D.load_weights(D_path)
    print("Model weights loaded successfully........")
else:
    print("Saved model weights not found......")
    sys.exit(0)
#-----------------
#model testing
#-----------------
#load Gan trainer
GAN3D=ROI_3DCAE_GAN3D(train_par=param,stride=stride)
GAN3D.initialize_model(Reconstructor=R , Discriminator=D )
GAN3D.test(test_videos=Fall_videos, score_type = 'R', epochs = epochs_trained,plot=True,tolerance_limit=8)
