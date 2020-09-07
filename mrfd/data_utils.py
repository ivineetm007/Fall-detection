import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import data, img_as_float
from skimage import exposure
# importing PIL
from PIL import Image
import pandas as pd
from skimage.filters import threshold_otsu
import os
import csv
import glob
import tqdm
import shutil
import sys

"""
box- [y_up, x_left, y_down, x_right]
string- y_up_x_left_y_down_x_right
"""
def box_to_str(box):
    bb = list(map(str, box));
    return '_'.join(bb)

def str_to_box(st):
    return list(map(float,st.split('_')))


def improve_box_cord(box,width,height,offset=10):
    '''
        Change the coordinates that are out of frame and apply offset around the bounding box
    '''
    left, top, right, bottom=box[1]-offset,box[0]-offset,box[3]+offset,box[2]+offset
    if left<0:
        left=0
    if top<0:
        top=0
    if right>width-1:
        right=width-1
    if bottom>height-1:
        bottom=height-1
    return [top,left, bottom,  right]


def sort_frames(frames, dset):
        #Sorting, trying for differnt dataset string formats
        numbers=[]
        if dset == 'SDU' or dset == 'SDU-Filled': #TODO remove try except, failing to sort shoudl stop!
            print('sorting SDU frames...')

            #try:
            frames = sorted(frames, key = lambda x: int(os.path.basename(x).split('.')[0])) #SDU
            # except ValueError:
            #     print('failed to sort SDU vid frames')
            #     pass
        elif dset == 'UR' or dset == 'UR-Filled' or dset == 'Thermal' or 'Thermal_track':
            print('sorting Thermal frames...')
            try:
                frames = sorted(frames, key = lambda x: int(x.split('-')[-1].split('.')[0]))
                numbers=[int(x.split('-')[-1].split('.')[0]) for x in frames]
            except ValueError:
                print('failed to sort UR vid frames')
                return

        elif dset == 'TST':
            try:
                frames = sorted(frames, key = lambda x: int(x.split('_')[-1].split('.')[0]))
            except ValueError:
                print('failed to sort vid frames, trying again....')
                pass
        return frames,numbers

def create_dictionary_csv(csv_path):
    """
    Load csv file and returns dictonary using pandas
    """
    data=pd.read_csv(csv_path, index_col ='Frame number')
    new_data = data.rename(columns = {'Frame number':'number',"Track box (y_up x_left y_down x_right)": "box"})
    return new_data.box.T.to_dict()


def load_ROI_box_csv(csv_path):
    ''''
        Read csv file and returns the bounding boxes
    '''
    num_box=create_dictionary_csv(csv_path)
    keys_sorted=sorted(num_box.keys())
    from config import WIDTH,HEIGHT
    box=[improve_box_cord(str_to_box(num_box[key]),WIDTH,HEIGHT,offset=10) for num,key in enumerate(keys_sorted)]
    return np.array(box),keys_sorted


def load_fall_labels(csvpath):
    ''' #
    Read csv file and create dictionary for start and end label of all Videos
    Ex Fall_labels['Fall1'] is a list [232,254]
    '''
    Fall_labels={}
    if not os.path.isfile(csvpath):
        print("csv file for labels not found at ".csvpath)
        print("Check the csv path in the config file or place the csv file at the right place")
    with open(csvpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        count=0
        for row in csv_reader:
            if count==0:
                count=1
                continue
            Fall_labels[row[0]]=[int(row[1]),int(row[2])]
    return Fall_labels

def split_data_tracks(data,frame_numbers,win_length,gap=24):
    '''
        Break the given data into chunks based on frame numbers. If there is no bounding box more than $gap frames then split it.
    '''
    frame_list=[]
    data_list=[]
    start_ind=0
    end_ind=0

    num=len(frame_numbers)
    while(end_ind<num):
        if(end_ind+1==num):
            if(end_ind+1-start_ind>=win_length):
                data_list.append(data[start_ind:end_ind+1])
                frame_list.append(frame_numbers[start_ind:end_ind+1])
            else:
                print("Frames less than window length")
        elif(frame_numbers[end_ind+1]-frame_numbers[end_ind]>gap):
            if(end_ind+1-start_ind>=win_length):
                data_list.append(data[start_ind:end_ind+1])
                frame_list.append(frame_numbers[start_ind:end_ind+1])
            else:
                print("Frames less than window length")
            start_ind=end_ind+1
        end_ind+=1
    return data_list,frame_list



def preprocess_frames(frames,numbers,process_list,ht,wd,channels,ROI_array=None):
    data=[]
    for x,i in zip(frames, range(0,frames.__len__())):
        #print(x,i)
        img=None
        #FIrst loading image
        if channels==1:
            img=cv2.imread(x,0) #Use this for RGB to GS #TODO change this based on dset
            img=img.reshape(img.shape[0],img.shape[1],1)

            # print(img.shape
        elif channels==3:
            # print("Thermal pose IUV")
            img=cv2.imread(x,1)#1 for color images

        #resize
        # img=img.reshape(ht,wd,3)
        if 'ROI_frame' in process_list:
            box=ROI_array[i,:]
            left, top, right, bottom=int(box[1]),int(box[0]),int(box[3]),int(box[2])
            #Pixel outside the ROI are assigned -1, GAN inpu range []-1,1]
            out=-np.ones(shape=img.shape,dtype='float32')
            patch=img[top:bottom,left:right,:]
            #Normalizing only inside ROI
            if 'Processed' in process_list:
                mean=np.mean(patch,axis=(0,1))
                patch=patch-mean
                patch=patch/ 255.0

            out[top:bottom,left:right,:]=patch
            img=cv2.resize(out,(ht,wd))
            img=img.reshape(ht,wd,channels)#resize on may remove the channels dis

        elif 'Processed' in process_list:
            img=cv2.resize(img,(ht,wd))

            img=img.reshape(ht,wd,channels)

            img=img-np.mean(img,axis=(0,1))#Mean centering
            img=img.astype('float32') / 255. #normalize
        else:
            img=cv2.resize(img,(ht,wd))
            img=img.reshape(ht,wd,channels)

        data.append(img)


    #data = data.reshape((len(data), np.prod(data.shape[1:]))) #Flatten the images
    data=np.array(data)

    print('data.shape', data.shape)

    # print(numbers)
    return data,numbers,frames
def create_img_data_set(fpath, data_shape,dset, process_list=None,ROI_array=None,sort = True):
    '''
    Loading and preprocessing of all images located at fpath as given input in process_list.

    fpath
    data_shape-expected data shape after resizing
    dset
    process_List- transformations list
    ROI_array=None- Bounding box array
    sort = True

    '''
    #Final imagfe dimensions
    ht,wd,channels=data_shape[0],data_shape[1],data_shape[2]
    print('gathering data at', fpath)
    fpath = fpath.replace('\\', '/')
    frames = glob.glob(fpath+'/*.jpg') + glob.glob(fpath+'/*.png')
    numbers=[]
    #sort frames
    if sort == True:
        frames,numbers = sort_frames(frames, dset)

    if 'ROI_frame' in process_list:
        if len(ROI_array)!=len(frames):
            print("Number of ROI and tracked frames are not equal")
            sys.exit(0)
    return preprocess_frames(frames,numbers,process_list,ht,wd,channels,ROI_array)

def create_ROI_mask(ROI_boxes,ROI_numbers,img_shape=(480,640,1),load_shape=(64,64,1),win_length=8,split_gap=10):
    '''
        Create mask images, resize it and create subvideos based on frame_numbers/ROI_numbers.
    '''
    num_frames=len(ROI_numbers)
    ROI_mask=np.zeros(shape=(num_frames,load_shape[0],load_shape[1],load_shape[2]))
    for i in range(num_frames):
        box=ROI_boxes[i,:]
        left, top, right, bottom=int(box[1]),int(box[0]),int(box[3]),int(box[2])
        mask=np.zeros(shape=img_shape)
        mask[top:bottom,left:right,:]=1
        mask=cv2.resize(mask,(load_shape[0],load_shape[1]))
        ROI_mask[i,:,:,:]=mask.reshape(load_shape)
    print("ROI mask data shape",ROI_mask.shape)
    ROI_mask_list,_=split_data_tracks(ROI_mask,ROI_numbers,gap=split_gap,win_length=win_length)
    return ROI_mask_list


def computeOpticalFlow(frames,load_height,load_width):
    '''
        Compute optical flow from franeback method
    '''
    index=1
    num=len(frames)
    prvs = cv2.imread(frames[0],0)
    flow_array=[]

    while(index<num):
        # print(index)
        next = cv2.imread(frames[index],0)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #check for nan and inf
        # if np.isnan(flow).any()==True:
        #     print('nan in optical flow extracted')
        # if np.isinf(flow).any()==True:
        #     print('inf in optical flow extracted')
        #FInd magnitude from x and y flow
        # mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag=np.sqrt(flow[...,0]**2+flow[...,1]**2)
        if np.isnan(mag).any()==True:
            print('nan in optical mag')
        if np.isinf(mag).any()==True:
            print('inf in optical flow mag')
        #Normalizing the flow in range 0 to 255 for visualization
        x_flow= cv2.normalize(flow[...,0],None,-1,1,cv2.NORM_MINMAX)
        # if np.isnan(x_flow).any()==True:
        #     print('nan in optical x flow')
        y_flow= cv2.normalize(flow[...,1],None,-1,1,cv2.NORM_MINMAX)
        # if np.isnan(y_flow).any()==True:
        #     print('nan in optical y flow')
        mag_norm= cv2.normalize(mag,None,-1,1,cv2.NORM_MINMAX)
        # if np.isnan(mag_norm).any()==True:
        #     print('nan in optical mag norm')
        #     print(np.min(mag))
        #     print(np.max(mag))
        #Stack across channels
        flow_con=np.stack((x_flow,y_flow,mag_norm),axis=2)
        flow_con=cv2.resize(flow_con,(load_width,load_height))
        # if np.isnan(flow_con).any()==True:
        #     print('nan in optical flow')
        flow_con=flow_con.reshape(load_height,load_width,3)
        flow_array.append(flow_con)

        prvs = next
        index+=1

    flow_np=np.array(flow_array)

#     print('FLOW_ARRAY.shape', flow_np.shape)
    return flow_np
