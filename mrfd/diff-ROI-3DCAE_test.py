
import sys
import numpy as np
import os
from data_management import load_videos
import config
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
from models import diff_ROI_C3D_AE_no_pool,C3D_no_pool
from trainer.diffroigan import Params,Diff_ROI_3DCAE_GAN3D
from trainer.util import create_diff_mask
import argparse

parser = argparse.ArgumentParser(description='Region and difeerence constraint adversarial model training')
parser.add_argument('--epochstrained', default='0',
                    help='Epoch number of the saved model')
parser.add_argument('--lambda_S', default='1',
                    help='ROI MSE loss hyperparameter')
parser.add_argument('--lambda_T', default='1',
                    help='Diff MSE loss hyperparameter')
args = parser.parse_args()

dset = config.track_root_folder
d_type='ROIframe'

#parameters
epochs=300
epochs_trained=int(args.epochstrained)
LOAD_DATA_SHAPE=config.LOAD_DATA_SHAPE
width, height = LOAD_DATA_SHAPE[0],LOAD_DATA_SHAPE[1]
channels=LOAD_DATA_SHAPE[2]
win_length=config.WIN_LENGTH
regularizer_list = ['BN']
break_win=config.SPLIT_GAP
stride=config.STRIDE
lambdas=[float(args.lambda_S),float(args.lambda_T)]#lambda_s ,lambda_t
#aggreagte all parameters in Params class
param=Params(width=width, height=height,win_length=win_length,channels=channels,dset=dset,d_type=d_type,regularizer_list=regularizer_list,break_win=break_win)
param.lambda_S=lambdas[0]
param.lambda_T=lambdas[1]
#-----------------
#Load train data
#-----------------
Fall_videos=load_videos(dset='Thermal_track',vid_class='Fall',input_type='ROI_FRAME')


##reconstructor model
R, R_name, _ = diff_ROI_C3D_AE_no_pool(img_width=param.width, img_height=param.height, win_length=param.win_length, regularizer_list=param.regularizer_list,channels=param.channels,lambda_S=param.lambda_S,lambda_T=param.lambda_T)
##Dicriminator model
D, D_name, _ = C3D_no_pool(img_width=param.width, img_height=param.height,  win_length=param.win_length, regularizer_list=param.regularizer_list,channels=param.channels)

param.R_name=R_name
param.D_name=D_name

D_path = param.get_D_path(epochs_trained)
R_path = param.get_R_path(epochs_trained)

if os.path.isfile(R_path) and os.path.isfile(D_path):
    R.load_weights(R_path)
    D.load_weights(D_path)
    print("Model weights loaded successfully........")
else:
    print(D_path)
    print(R_path)
    print("Saved model weights not found......")
    sys.exit(0)
#-----------------
#model training
#-----------------
GAN3D=Diff_ROI_3DCAE_GAN3D(train_par=param,stride=stride)
GAN3D.initialize_model(Reconstructor=R , Discriminator=D )
GAN3D.test(test_videos=Fall_videos, score_type = 'R_S', epochs = epochs_trained,plot=True,tolerance_limit=8)
GAN3D.test(test_videos=Fall_videos, score_type = 'R_T', epochs = epochs_trained,plot=True,tolerance_limit=8)
