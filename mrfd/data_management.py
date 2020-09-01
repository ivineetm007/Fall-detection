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
        save optical flow computed for scubvideos
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




# #
# #
# #
# # def init_test_only(data_option, img_width, img_height, test_vid):
# #     path = './SplitByVideoData' + '/' + data_option + '/' + 'Videos'+ '_data{}_imgdim{}x{}_proc.h5'.format(data_option, img_width, img_height)
# #     init_videos(img_width = img_width, img_height = img_height, data_option = data_option, path = path)
# #
# #     with h5py.File(path, 'r') as hf:
# #         test_vid = test_vid.replace('\\','/')
# #         test_data = hf[test_vid]['Data'][:]
# #         test_labels = hf[test_vid]['Labels'][:]
# #     test_vid_name = test_vid.split('/')[-1]
# #     return test_data, test_labels, test_vid_name
# #
# # def find_start_index_disc(start_frame_index, NF_frames_indeces):
# #     '''
# #     gets real index(not frame index) where Fall starts in NFFall array
# #     '''
# #
# #     for i in range(len(NF_frames_indeces)):
# #         index = NF_frames_indeces[i]
# #         if index>start_frame_index:
# #             return i
# #         else:
# #             return start_frame_index
# #
# # def get_fall_indeces(Fall_vid_dir = None, use_cropped = False, dset = 'Thermal'):
# #     """
# #     input Fall not NFFall (ie. recreate fall opt1, with opt3 labels)
# #
# #     Gets start/stop indices accoutnign for potentiol discont's from cropping
# #     """
# #     if dset == 'Thermal' or dset == 'UR' or dset == 'UR-Filled':
# #         split_char = '-'
# #     else:
# #         split_char = '_'
# #
# #     print('Fall_vid_dir', Fall_vid_dir)
# #     Fall_vid_dir = Fall_vid_dir.replace('\\','/')
# #     base_Fall = Fall_vid_dir
# #
# #     basename = os.path.basename(Fall_vid_dir)
# #     print(basename)
# #     root = os.path.dirname(os.path.dirname(Fall_vid_dir))
# #     print(root)
# #     base_NFFall = root + '/NonFall/NF' + basename
# #
# #     frames_opt3_Fall = glob.glob( base_Fall + '/*.jpg') + \
# #         glob.glob( base_Fall + '/*.png')
# #
# #     frames_opt3_NFFall = glob.glob( base_NFFall + '/*.jpg') + \
# #         glob.glob( base_NFFall + '/*.png')
# #
# #
# #     #print("\n".join(frames_opt3_Fall))
# #     #print('{} opt3 fall frames found'.format(len(frames_opt3)))
# #
# #     frames_opt3_Fall = sort_frames(frames_opt3_Fall, dset)
# #     frames_opt3_NFFall = sort_frames(frames_opt3_NFFall, dset)
# #
# #     # if dset == 'TST': #Sortign frames, glob returns unsorted
# #     #     frames_opt3_Fall = sorted(frames_opt3_Fall, key = lambda x: int(x.split(split_char)[-1].split('.')[0]))
# #
# #
# #     frames_opt3_Fall[0] = frames_opt3_Fall[0].replace('\\', '/')
# #     print('frames_opt3_Fall[0]', frames_opt3_Fall[0])
# #
# #     start_frame_ind = int(os.path.basename(frames_opt3_Fall[0]).split('.')[0].split(split_char)[-1])
# #     #start_frame_ind = int(frames_opt3_Fall[0].split(split_char)[-1].split('.')[0]) #Thermal
# #     print('start_frame_ind', start_frame_ind)
# #     end_frame_index = start_frame_ind + len(frames_opt3_Fall) #-1? TODO Not used
# #
# #     #print(frames_opt3_NFFall)
# #     NF_frame_indices = [int(os.path.basename(frames_opt3_NFFall[i]).split('.')[0].split(split_char)[-1]) \
# #                             for i in range(len(frames_opt3_NFFall))]
# #
# #     if len(frames_opt3_NFFall) > 0:
# #         new_fall_start_ind = find_start_index_disc(start_frame_ind,\
# #                                                    NF_frame_indices)
# #     else:
# #         print('no NFF frames found')
# #         new_fall_start_ind = start_frame_ind
# #
# #     print('new_fall_start_ind, len(frames_opt3_Fall)', new_fall_start_ind, len(frames_opt3_Fall))
# #     new_end_frame_index = new_fall_start_ind + len(frames_opt3_Fall)
# #
# #     #New means accounts for discont of cropping
# #
# #     return new_fall_start_ind, new_end_frame_index
# #
# #
#
# #
#
# #
# # def create_windowed_labels(labels, stride, tolerance, window_length):
# #     '''
# #     Create labels on seq level
# #
# #     int tolerance: number of fall frames (1's) in a window for it to be labeled as a fall (1). must not exceed window length
# #
# #     '''
# #     output_length = int(np.floor((len(labels) - window_length) / stride))+1
# #     #output_shape = (output_length, window_length, 1)
# #     output_shape = (output_length, 1)
# #
# #     total = np.zeros(output_shape)
# #
# #     i=0
# #     while i < output_length:
# #         next_chunk = np.array([labels[i+j] for j in range(window_length)])
# #
# #         num_falls = sum(next_chunk) #number of falls in the window
# #
# #         if num_falls >= tolerance:
# #             total[i] = 1
# #         else:
# #             total[i] = 0
# #
# #
# #         i = i+stride
# #
# #     labels_windowed = total
# #
# #     return labels_windowed
# #
#
# #
# def init_data_by_class(train_par,data_shape=(64,64,1),vid_class = 'NonFall', use_cropped = False, fill_holes = True):
#
#     '''
#     Loads data seperated by vid class. To load data seperated by video, use init_videos
#     Most usefull for 2D models(which do not require windowing by vid)
#
#     Attributes:
#         bool fill_holes: if True creates UR data set from denoised depth images
#     '''
#
#     ht,wd = train_par.height,train_par.width
#
#     if train_par.dset == 'Thermal' and train_par.d_type == 'frame':
#         if vid_class == 'NonFall':
#             fpath= root_drive + '/Thermal/NonFall/ADL*'
#         else:
#             fpath= root_drive + '/Thermal/Fall/Fall*'
#     elif train_par.dset =='Thermal_track':
#         if vid_class == 'NonFall':
#             fpath= root_drive + '/Thermal_track/'+train_par.d_type+'/NonFall/ADL*'
#         else:
#             fpath= root_drive + '/Thermal_track/'+train_par.d_type+'/Fall/Fall*'
#
#     data = create_img_data_set(fpath=fpath,data_shape=data_shape,dset=train_par.dset, raw=train_par.raw,sort=False) #Don't need to sort
#
#     #path = './H5Data/Data_set_imgdim{}x{}.h5'.format(img_width, img_height) #Old
#     #path = 'N:/FallDetection/Fall-Data/H5Data/Data_set_imgdim{}x{}.h5'.format(img_width, img_height) #Old
#     path = train_par.get_h5py_path()
#
#     root_path = train_par.get_split_path('Split_by_class',vid_class)
#
#     if vid_class == 'NonFall':
#         labels = np.array([0] * len(data))
#     else:
#         labels = np.array([1] * len(data))
#
#
#     with h5py.File(path, 'a') as hf:
#         #root_sub = root.create_group('Split_by_video')
#         print('creating data at ', root_path)
#         if root_path in hf:
#             print('root_path {} found, clearing'.format(root_path))
#             del hf[root_path]
#         root = hf.create_group(root_path)
#
#         root['Data'] = data
#
#         root['Labels'] = labels
#
#
# def load_data(split_by_vid_or_class = 'Split_by_video', train_par=None, vid_class = 'NonFall',return_data=False):
#
#     path = train_par.get_h5py_path()
#
#     #init_h5py(path)
#     #Check for existemce of file
#     if not os.path.isfile(path):
#         print('no h5py path found, initializing...')
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         if split_by_vid_or_class == 'Split_by_class':
#             init_data_by_class(vid_class = vid_class,train_par=train_par)
#         else:
#             init_videos(train_par)
#
#     else:
#         print('h5py path found, loading data_dict..')
#
#     root_path=train_par.get_split_path(split_by_vid_or_class,vid_class)
#     print(root_path)
#
#     #Check for existemce of group
#     if split_by_vid_or_class == 'Split_by_class':
#         #check the split exist or not
#         try:
#             with h5py.File(path, 'r') as hf:
#                 data_dict = hf[root_path]['Data'][:]
#         except:
#             print('component not found, initializing...')
#             init_data_by_class(vid_class = vid_class,train_par=train_par)
#
#         if return_data==True:
#             with h5py.File(path, 'r') as hf:
#                 data_dict = hf[root_path]['Data'][:]
#
#             return data_dict
#
#     elif split_by_vid_or_class == 'Split_by_video':
#         try:
#             with h5py.File(path, 'r') as hf:
#                 data_dict =hf[root_path]
#         except:
#             print('component not found, initializing...')
#             init_videos(train_par)
#
#         if return_data==True:
#             with h5py.File(path, 'r') as hf:
#                 data_dict = hf[root_path]
#             return data_dict
#
# #-------------------------------------------------------------------------------------------------
# #Code for temporal data_management
# #-------------------------------------------------------------------------------------------------
# def create_windowed_arr(arr, stride, window_length):
#     """
#     arr: array of imgs
#     """
#
#     img_width, img_height,channels = arr.shape[1], arr.shape[2], arr.shape[3]
#
#     output_length = int(np.floor((len(arr) - window_length) / stride))+1
#     output_shape = (output_length, window_length, img_width, img_height, channels)
#
#     total = np.zeros(output_shape)
#
#     i=0
#     while i < output_length:
#         next_chunk = np.array([arr[i+j] for j in range(window_length)]) #Can use np.arange if want to use time step \
#         # ie. np.arrange(0,window_length,dt)
#
#         total[i] = next_chunk
#
#         i = i+stride
#
#     # arr_windowed = total
#
#     return total
#
# # def vid_group_to_lists(grp):
#
#
# def sort_str_dict(diction):
#     values=[]
#     keys=[int(key) for key in diction.keys()]
#     keys=sorted(keys)
#     for key in keys:
#         values.append(diction[str(key)])
#     return values
#
# def load_vid_grp(grp):
#     start_index=None
#     end_index=None
#     data_grp=grp['Data']
#     label_grp=grp['Labels']
#     frame_number_grp=grp['frame_numbers']
#     data_dic=dict(data_grp)
#     label_dic=dict(label_grp)
#     frame_number_dic=dict(frame_number_grp)
#     if 'Fall start index' in grp['Data'].attrs:
#         start_index=grp['Data'].attrs['Fall start index']
#     if 'Fall end index' in grp['Data'].attrs:
#         end_index=grp['Data'].attrs['Fall end index']
#
#     data_list=sort_str_dict(data_dic)
#     data_list=[data[:] for data in data_list]
#
#     label_list=sort_str_dict(label_dic)
#     label_list=[label[:] for label in label_list]
#
#     frame_number_list=sort_str_dict(frame_number_dic)
#     frame_number_list=[frame_number[:] for frame_number in frame_number_list]
#
#     return data_list,label_list,frame_number_list,start_index,end_index
#
# def create_windowed_arr_per_vid(vids_dict, stride, window_length, img_width, img_height,channels=1):
#     '''
#     Assumes vids_dict is h5py structure, ie. vids_dict = hf['Data_2017/UR/Raw/Split_by_video']
#     '''
#
#     #List number of frames for each video
#
#
#     # num_windowed = sum([int(np.floor(val-window_length)/stride)+1 for val in vid_list])
#     # print('num_windowed', num_windowed)
#     # output_shape = (num_windowed, window_length,img_width, img_height, channels)
#     # print('output_shape', output_shape)
#
#     total = []
#     # print('total.shape', 'num_windowed', 'output_shape', total.shape, num_windowed, output_shape)
#
#     for vid_grp, name in zip(vids_dict.values(), vids_dict.keys()):
#         print('group Name', name)
#
#         vid_list,_,_,_,_ = load_vid_grp(vid_grp)
#
#         vid_windowed_list=[]
#         for vid in vid_list:
#             vid = vid.reshape(len(vid),img_width, img_height,channels)
#             vid_windowed_list.append(create_windowed_arr(vid, stride, window_length))
#         print("Number of sub videos: ",len(vid_windowed_list))
#         vid_windowed=np.concatenate(vid_windowed_list)
#         print('vid_windowed.shape', vid_windowed.shape)
#         total.append(vid_windowed)
#     total=np.concatenate(total)
#     print(total.shape)
#
#
#     return total
#
# def initialize_data_set_master():
#
#     init_videos(img_width = 64, img_height = 64, data_option = 'Option3', path = None, \
#         use_cropped = False, raw = True, dset = 'Thermal')
#
#     init_data_by_class(vid_class = 'NonFall', train_or_test = None, dset = 'Thermal',\
#         raw = False, img_width = 64, img_height = 64, use_cropped = False)
#
#     init_videos(img_width = 256, img_height = 256, data_option = 'Option3', path = None, \
#         use_cropped = False, raw = False, dset = 'Thermal')
#
#
#
#
# def create_windowed_train_data(window_length, dset = 'Thermal', raw = False, img_width = 64, img_height = 64):
#     """
#     Creates windowed data from ADL frames only
#     TODO Delete? Repeat in data_man_main
#     TODO MEM LEAK
#     """
#
#     #path = './H5Data/Data_set_imgdim64x64.h5'
#     path = 'N:/FallDetection/Fall-Data/H5Data/Data_set_imgdim{}x{}.h5'.format(img_width, img_height)
#
#     hf = h5py.File(path)
#     if raw == False:
#         data_dict = hf['Data_2017/'+ dset+'/Processed/Split_by_video']
#     else:
#         data_dict = hf['Data_2017/'+ dset+'/Raw/Split_by_video']
#     data_dict_adl = dict((key,value) for key, value in data_dict.items() if 'adl' in key or 'ADL' in key)
#     print(list(data_dict_adl.keys()))
#     print(len(list(data_dict_adl.keys())))
#
#     adl_win = create_windowed_arr_per_vid(vids_dict = data_dict_adl, \
#                                 stride = 1, \
#                                 window_length=window_length,\
#                                 img_width=img_width,\
#                                 img_height=img_height)
#
#     print('adl_win.shape', adl_win.shape)
#     np.save('./npData/train_data-'+dset+'-NonFalls-proc-windowed_by_vid-win_{}.npy'.format(window_length),adl_win)
#
# def flip_windowed_arr(windowed_data):
#     """
#     windowed_data: of shape (samples, win_len,...)
#
#     returns shape len(windowed_data), win_len, flattened_dim)
#     Note: Requires openCV
#     """
#     win_len = windowed_data.shape[1]
#     flattened_dim = np.prod(windowed_data.shape[2:])
#     #print(flattened_dim)
#     flipped_data_windowed = np.zeros((len(windowed_data), win_len, flattened_dim)) #Array of windows
#     print(flipped_data_windowed.shape)
#     i=0
#     for win_idx in range(len(windowed_data)):
#         window = windowed_data[win_idx]
#         flip_win = np.zeros((win_len, flattened_dim))
#
#         for im_idx in range(len(window)):
#             im = window[im_idx]
#             hor_flip_im = cv2.flip(im,1)
#             #print(hor_flip_im.shape)
#             #print(flip_win[im_idx].shape)
#
#             flip_win[im_idx] = hor_flip_im.reshape(flattened_dim)
#
#         flipped_data_windowed[win_idx] = flip_win
#     return flipped_data_windowed
