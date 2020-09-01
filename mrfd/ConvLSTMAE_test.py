
import sys
import numpy as np
import os
from data_management import load_videos
import config

# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
from models import CLSTM_AE
from trainer.convlstmae import Params,ConvLSTMAE_trainer
import argparse

parser = argparse.ArgumentParser(description='Deep spatio temporal autoencoder training')
parser.add_argument('--epochstrained', default='0',
                    help='Epoch number of the saved model')
args = parser.parse_args()

dset = config.track_root_folder
d_type='frame'

#parameters
epochs=500
epochs_trained=int(args.epochstrained)
LOAD_DATA_SHAPE=config.LOAD_DATA_SHAPE
width, height = LOAD_DATA_SHAPE[0],LOAD_DATA_SHAPE[1]
channels=LOAD_DATA_SHAPE[2]
win_length=config.WIN_LENGTH
break_win=config.SPLIT_GAP
stride=config.STRIDE
expname='exp1'
batch_size=16#as mentioned in convlstm paper
#aggreagte all parameters in Params class
param=Params(width=width, height=height,win_length=win_length,channels=channels,dset=dset,d_type=d_type,break_win=break_win,hp_name=expname,batch_size=batch_size)


#-----------------
#Load test data
#-----------------
Fall_videos=load_videos(dset='Thermal_track',vid_class='Fall',input_type='FRAME')



#-----------------
#MODEL Initialization
#-----------------

##reconstructor model
R, R_name, _ = CLSTM_AE(img_width=param.width, img_height=param.height, win_len=param.win_length)
param.R_name=R_name
#

R_path = param.get_R_path(epochs_trained)

if os.path.isfile(R_path):
    R.load_weights(R_path)
    print("Model weights loaded successfully........")
else:
    print("Saved model weights not found......")
    sys.exit(0)
#-----------------
#model training
#-----------------
convlstm=ConvLSTMAE_trainer(train_par=param,stride=stride)
convlstm.initialize_model(Reconstructor=R)
convlstm.test(test_videos=Fall_videos, score_type = 'R', epochs = epochs_trained,plot=True,tolerance_limit=8)
