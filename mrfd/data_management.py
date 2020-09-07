import os
import glob
# import h5py
import numpy as np
import tqdm
import cv2
# from util import *
import sys
import csv
import config
import shutil
from data_utils import load_ROI_box_csv,create_img_data_set,split_data_tracks,create_ROI_mask,load_fall_labels,sort_frames,computeOpticalFlow
root_drive = config.root_drive



def get_dir_lists(dset, d_type):
    '''
    Returns the path of all video/frames folders inside dset/dtype folder Ex. Thermal_track/csv or Thermal_track/video
    '''
    path_ADL=None
    path_Fall=None

    if dset == 'Thermal' or  dset=='Thermal_track':
        path_Fall = root_drive + '/'+dset+'/'+d_type +'/Fall/Fall*'
        path_ADL = root_drive + '/'+dset+'/'+d_type +'/NonFall/ADL*'
    else:
        raise Exception('Dataset not matched')


    vid_dir_list_0 = glob.glob(path_ADL)
    vid_dir_list_1 = glob.glob(path_Fall)

    return vid_dir_list_0, vid_dir_list_1

def get_mapped_dir_lists(dset, d_type='frame',output_dir=None,output_name=None):
    '''
    recursively map all folders of ADL and FALL to a new directory
    d_type- data type of the dataset whose path is mapped
    '''
    path_ADL=None
    path_Fall=None
    output_path=output_dir
    if output_name!=None:
        output_path=root_drive + '/'+output_name+'/'+d_type

    if dset == config.root_folder or dset=='Thermal_pose' or dset==config.track_root_folder:

        path_Fall = root_drive + '/'+dset+'/'+d_type +'/Fall/Fall*'
        path_ADL = root_drive + '/'+dset+'/'+d_type +'/NonFall/ADL*'
    else:
        print("Dataset not match.... get_mapped_dir_lists function")
    vid_dir_list_0 = glob.glob(path_ADL)
    vid_dir_list_1 = glob.glob(path_Fall)
    ADL_list=[]
    Fall_list=[]
    for path in vid_dir_list_0:
        splits=path.split('/')
        ADL_list.append([path,output_path+'/'+splits[-2]+'/'+splits[-1]])
    for path in vid_dir_list_1:
        splits=path.split('/')
        Fall_list.append([path,output_path+'/'+splits[-2]+'/'+splits[-1]])
    return ADL_list,Fall_list


def get_ROI_boxes(dset=config.track_root_folder,class_='ADL'):
    '''
        Returns a dictionary
            key:video name ex 'ADL1'
            value:dictionary
                key:'BOX'-
                value: numpy array of car_boxes
                key:'NUMBER'
                value: list of frame number of corresponding detected bbox
    '''
    class_ROI={}
    ADL_csv_dir_list,Fall_csv_dir_list=get_dir_lists(dset=dset, d_type='csv')
    if class_=='ADL':
        csv_list=list(map(lambda csv_path: os.path.join(csv_path,os.path.basename(csv_path)+'.csv'),ADL_csv_dir_list))
        if len(csv_list)<config.adl_num:
            print(len(csv_list),"csv folders found instead of",config.adl_num," in track data folder")
            sys.exit(0)
    elif class_=='Fall':
        csv_list=list(map(lambda csv_path: os.path.join(csv_path,os.path.basename(csv_path)+'.csv'),Fall_csv_dir_list))
        if len(csv_list)<config.fall_num:
            print(len(csv_list),"csv folders found instead of",config.fall_num," in track data folder")



    for csv_path in csv_list:
        ROI={}
        ROI['BOX'],ROI['NUMBER']=load_ROI_box_csv(csv_path)
        class_ROI[os.path.basename(csv_path).split('.')[0]]=ROI
    return class_ROI



def copy_file(inp_path,out_path):
    try:
        dest = shutil.copyfile(inp_path,out_path)
        # print("Copied ",dest)
    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")

    # If destination is a directory.
    except IsADirectoryError:
        print("Destination is a directory.")

    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")

    # For other errors
    except :
        print("Error occurred while copying file.",inp_path)
def copy_track_frames_video(input_path,output_path,numbers):
    '''
        Copy frames given frame numbers from input dir to output dir
    '''
    frames = glob.glob(input_path+'/*.jpg') + glob.glob(input_path+'/*.png')
    frames,in_numbers = sort_frames(frames, dset='Thermal')
    os.makedirs(output_path, exist_ok=True)
    for image_path,in_number in zip(frames,in_numbers):
        if in_number in numbers:
            out_image_path=os.path.join(output_path,os.path.basename(image_path))
            copy_file(image_path,out_image_path)


def copy_track_frames_dataset():
    '''
        copy tracked frames from dataset to track folder
    '''
    ADL_ROI=get_ROI_boxes(class_='ADL')
    Fall_ROI=get_ROI_boxes(class_='Fall')
    ADL_list,Fall_list=get_mapped_dir_lists(dset=config.root_folder, d_type='frame',output_name=config.track_root_folder)
    print("Copy tracked frames for ADL videos")
    for input_path,output_path in tqdm.tqdm(ADL_list):
        video_name=os.path.basename(input_path)
        print('Input:',input_path)
        print('Copy Destination folder:',output_path)
        if os.path.exists(output_path) and os.path.isdir(output_path):
            shutil.rmtree(output_path)
        copy_track_frames_video(input_path,output_path,numbers=ADL_ROI[video_name]['NUMBER'])
    print("Copy tracked frames for Fall videos")
    for input_path,output_path in tqdm.tqdm(Fall_list):
        video_name=os.path.basename(input_path)
        print('Input:',input_path)
        print('Copy Destination folder:',output_path)
        if os.path.exists(output_path) and os.path.isdir(output_path):
            shutil.rmtree(output_path)
        copy_track_frames_video(input_path,output_path,numbers=Fall_ROI[video_name]['NUMBER'])

def init_vid(vid_dir = None, vid_class = None,dset='Thermal_track',process_list=['Processed'],win_length=8,split_gap=10,data_shape=(64,64,1),ROI_array=None,fall_labels=None):# TOMOVE
    '''
    vid_class= 1 or 0
    Loads and preprocess a video. First load and preprocess the frames. Sceond create subvideos and labels for them.
    returns list of data, list of numbers
    '''
    print('Loading vid at ', vid_dir)

    #Data as numpy array and list of sorted frame numbers
    data,frame_numbers,frames_path=create_img_data_set(fpath=vid_dir, data_shape=data_shape,dset=dset, process_list=process_list,ROI_array=ROI_array,sort = True)
    #Data as list of numpy array and list of sorted frame numbers
    data_list,frame_numbers_list=split_data_tracks(data,frame_numbers,gap=split_gap,win_length=win_length)
    #Split frames path
    frames_path_list,_=split_data_tracks(frames_path,frame_numbers,gap=split_gap,win_length=win_length)

    vid_dir_name = os.path.basename(vid_dir)
    print('vid_dir_name', vid_dir_name)
    labels = np.array([0] * len(data))

    if vid_class==1:
        start,end=fall_labels
        for i in range(len(frame_numbers)):
            if frame_numbers[i]>=start and frame_numbers[i]<=end:
                labels[i]=1
        print("Start frame:",start)
        print("End frame:",end)

    labels_list,_=split_data_tracks(labels,frame_numbers,gap=split_gap,win_length=win_length)

    total_label=0
    for labels in labels_list:
        total_label+=len(labels)
    print("Total number of labels,",total_label)

    total_frame=0
    for frames in frame_numbers_list:
        total_frame+=len(frames)
    print("Total number of frames,",total_frame)
    assert total_label==total_frame

    return  data_list,frame_numbers_list,labels_list,frames_path_list

def load_videos(dset='Thermal_track',vid_class='ADL',input_type='FRAME'):
    '''
    Load the dataset and precprocess.
    vid_class:- ['ADL','Fall']
    input_type=['FRAME','ROI_FRAME']
    Steps:
        Copy tracked frames into track folder
        Load ROI boxes from csv files
        Load tracked frames, preprocess, split into subvideos, load and create for subvideos
    Returns dictionary containing video frame paths, actual frame numbers, preprocessed subvidoes, their masks and labels
    '''
    ADL_frame_dir_list,Fall_frame_dir_list=get_dir_lists(dset=dset, d_type='frame')
    if (len(ADL_frame_dir_list)<config.adl_num or len(Fall_frame_dir_list)<config.fall_num)   and dset=='Thermal_track':
        print("All videos folder not found in Thermal_track folder. Copying tracked frames from Thermal folder............")
        copy_track_frames_dataset()
        ADL_frame_dir_list,Fall_frame_dir_list=get_dir_lists(dset=dset, d_type='frame')

    if vid_class=='ADL':
        ADL_videos={}
        if input_type=='ROI_FRAME':
            #het ROI boxes and frame numbers
            ADL_ROI=get_ROI_boxes(dset=config.track_root_folder,class_='ADL')#Returns dictionary
        for ADL_frame_dir in ADL_frame_dir_list:
            vid_name=os.path.basename(ADL_frame_dir)
            print("\nLoading Video...........",vid_name)
            ADL={}
            if input_type=='FRAME':
                print("\nLoading frame data...........\n")
                data_list,frame_numbers_list,labels_list,frames_path=init_vid(vid_dir = ADL_frame_dir, vid_class = 0, dset=dset,process_list=['Processed'],win_length=config.WIN_LENGTH,split_gap=config.SPLIT_GAP,data_shape=config.LOAD_DATA_SHAPE,fall_labels=None)
                ADL['FRAME']=data_list
                ADL['NUMBER']=frame_numbers_list
                ADL['LABELS']=labels_list
                ADL['PATH']=frames_path
            elif input_type=='ROI_FRAME':
                print("\nLoading data and masking...........\n")
                data_list,frame_numbers_list,labels_list,frames_path=init_vid(vid_dir = ADL_frame_dir, vid_class = 0, dset=dset,process_list=['Processed','ROI_frame'],win_length=config.WIN_LENGTH,split_gap=config.SPLIT_GAP,data_shape=config.LOAD_DATA_SHAPE,fall_labels=None,ROI_array=ADL_ROI[vid_name]['BOX'])
                ADL['ROI_FRAME']=data_list
                ADL['NUMBER']=frame_numbers_list
                ADL['LABELS']=labels_list
                ADL['PATH']=frames_path
                print("\nCreating MASK data...........\n")
                ADL['MASK']=create_ROI_mask(ROI_boxes=ADL_ROI[vid_name]['BOX'],ROI_numbers=ADL_ROI[vid_name]['NUMBER'],img_shape=(config.HEIGHT,config.WIDTH,1),load_shape=config.LOAD_DATA_SHAPE,win_length=config.WIN_LENGTH,split_gap=config.SPLIT_GAP)
            else:
                print('\Invalid input_type. Skipping this video.........\n')

            ADL_videos[vid_name]=ADL
        return ADL_videos

    elif vid_class=='Fall':
        Fall_videos={}
        if input_type=='ROI_FRAME':
            #het ROI boxes and frame numbers
            Fall_ROI=get_ROI_boxes(dset=config.track_root_folder,class_='Fall')
        #Load labels from csv files
        fall_labels=load_fall_labels(config.label_csv_path)
        #Processing each video
        for Fall_frame_dir in Fall_frame_dir_list:
            vid_name=os.path.basename(Fall_frame_dir)
            print("\nLoading Video...........",vid_name)
            Fall={}
            if input_type=='FRAME':
                print("\nLoading frame data...........\n")
                data_list,frame_numbers_list,labels_list,frames_path=init_vid(vid_dir = Fall_frame_dir, vid_class = 1, dset=dset,process_list=['Processed'],win_length=config.WIN_LENGTH,split_gap=config.SPLIT_GAP,data_shape=config.LOAD_DATA_SHAPE,fall_labels=fall_labels[vid_name])
                Fall['FRAME']=data_list
                Fall['NUMBER']=frame_numbers_list
                Fall['LABELS']=labels_list
                Fall['PATH']=frames_path
            elif input_type=='ROI_FRAME':
                print("\nLoading data and masking...........\n")
                data_list,frame_numbers_list,labels_list,frames_path=init_vid(vid_dir = Fall_frame_dir, vid_class = 1, dset=dset,process_list=['Processed','ROI_frame'],win_length=config.WIN_LENGTH,split_gap=config.SPLIT_GAP,data_shape=config.LOAD_DATA_SHAPE,fall_labels=fall_labels[vid_name],ROI_array=Fall_ROI[vid_name]['BOX'])
                Fall['ROI_FRAME']=data_list
                Fall['NUMBER']=frame_numbers_list
                Fall['LABELS']=labels_list
                Fall['PATH']=frames_path
                print("\nLoading MASK data...........\n")
                mask_list=create_ROI_mask(ROI_boxes=Fall_ROI[vid_name]['BOX'],ROI_numbers=Fall_ROI[vid_name]['NUMBER'],img_shape=(config.HEIGHT,config.WIDTH,1),load_shape=config.LOAD_DATA_SHAPE,win_length=config.WIN_LENGTH,split_gap=config.SPLIT_GAP)
                Fall['MASK']=mask_list
            Fall['START_END']=fall_labels[vid_name]
            Fall_videos[vid_name]=Fall
        return Fall_videos
    else:
        print("Invalid vid_clas type")
        sys.exit(0)


def flow_from_path_list(sub_vid_path_list):
    '''
        Compute optical flow for a
    '''
    vid_flow_list=[]
    for sub_vid_path in tqdm.tqdm(sub_vid_path_list):
        flow=computeOpticalFlow(sub_vid_path,config.LOAD_DATA_SHAPE[0],config.LOAD_DATA_SHAPE[1])
        print('Number of frames',len(sub_vid_path))
        print('Flow shape',flow.shape)
        # print(np.isnan(flow).any())
        vid_flow_list.append(flow)
    return vid_flow_list

def load_flow_from_folder(vid_flow_dir):
    ''''
        load all the stored optical flow computed for all the subvideos of a video
    '''
    files=glob.glob(vid_flow_dir+'/*.npy')
#     print(files)
    sorted_files = sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
#     print(sorted_files)
    num=len(sorted_files)
    vid_flow_list=[]
    for i in range(num):
        flow=np.load(sorted_files[i])
        vid_flow_list.append(flow)
        print(str(i)+' subvid flow shape')
        print(flow.shape)
    return vid_flow_list

def save_flow_folder(vid_flow_list,vid_flow_dir):
    '''
        save optical flow computed for subvideos
    '''
    os.makedirs(vid_flow_dir,exist_ok=True)
    num=len(vid_flow_list)
    for i in range(num):
        print(str(i)+' subvid flow shape')
        print(vid_flow_list[i].shape)
        save_path=os.path.join(vid_flow_dir,str(i))
        np.save(save_path,vid_flow_list[i])


def load_optical_flow_dataset(videos,vid_class='ADL'):
    os.makedirs(config.flow_dir,exist_ok=True)
    if vid_class=='ADL':
        ADL_videos=videos
        for vid_name in ADL_videos.keys():
        # for vid_name in ['ADL3']:
            print('------------')
            print('Video Name', vid_name)
            print('------------')
            vid_flow_dir=os.path.join(config.flow_dir,vid_name)
            if os.path.isdir(vid_flow_dir):
                print("Loading flow from:",vid_flow_dir)
                ADL_videos[vid_name]['FLOW']=load_flow_from_folder(vid_flow_dir)
            else:
                print("Saving flow at:",vid_flow_dir)
                vid_flow_list=flow_from_path_list(ADL_videos[vid_name]['PATH'])
                ADL_videos[vid_name]['FLOW']=vid_flow_list
                save_flow_folder(vid_flow_list,vid_flow_dir)
    elif vid_class=='Fall':
        Fall_videos=videos
        for vid_name in Fall_videos.keys():
            print('------------')
            print('Video Name', vid_name)
            print('------------')
            vid_flow_dir=os.path.join(config.flow_dir,vid_name)
            if os.path.isdir(vid_flow_dir):
                print("Loading flow from:",vid_flow_dir)
                Fall_videos[vid_name]['FLOW']=load_flow_from_folder(vid_flow_dir)
            else:
                print("saving flow at:",vid_flow_dir)
                vid_flow_list=flow_from_path_list(Fall_videos[vid_name]['PATH'])
                Fall_videos[vid_name]['FLOW']=vid_flow_list
                save_flow_folder(vid_flow_list,vid_flow_dir)
