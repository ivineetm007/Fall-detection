#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import glob
import time
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1'

from tracker import tracker,detector,helpers
import cv2
import tqdm
import argparse
import config
from data_management import get_mapped_dir_lists,sort_frames
from data_utils import box_to_str


class single_tracker(object):
    def __init__(self,max_age=65,frame_count=0,min_hits=1):
        self.tracker_list=[]
        self.track_id_list= deque([str(i) for i in range(100)])
        self.max_age=max_age
        self.frame_count=frame_count
        self.min_hits=min_hits
        self.detect_track_iou_thres=0.6
        self.box_ios_thres=0.8
        self.otsu_iou_thres=0.5
        self.current_tracker=None



# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

max_age = 20  # no.of consecutive unmatched detection before
             # a track is deleted

min_hits =1  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['1', '2', '3', '4', '5', '6', '7', '7', '8', '9', '10'])

debug = False
'''
Function to track single person

'''
def pipeline_single_tracker(det,img,otsu_box,track,draw=False):
    track.frame_count+=1
    org_im=img.copy()
    img_dim = (img.shape[1], img.shape[0])
    if debug:
        print('\nFrame:', track.frame_count,' \n')
    #Detect person in the image
    detect_box = det.get_localization(img) # measurement
    final_box=[]
    improved=False
    #check for small box
    if len(detect_box)!=0:
        detect_box=helpers.remove_small_box(detect_box[0],height_limit=150,width_limit=150)

    #If Detected
    if len(detect_box)!=0:
        if debug:
            print("Detection found")
        # detect_box=detect_box[0]
        if draw:
           img1= helpers.draw_box_label('Org Det.',img, detect_box, box_color=(255, 0, 0))
        #Tracker alive or not
        if track.current_tracker!=None:
        #If alive
            if debug:
                print("Tracker Alive")
            track_box=track.current_tracker.box
            #Track result matches,detection or not
            #------------------------------------
            #If matches
            if helpers.box_iou2(track_box,detect_box)>track.detect_track_iou_thres:

                #Check abnormal detect box
                #Abnormal, use previous box NOTE can be improved
                # detect_area=helpers.find_area(detect_box)
                # track_area=helpers.find_area(track_box)

                height_d,width_d=helpers.find_dim(detect_box)
                height_t,width_t=helpers.find_dim(track_box)
                delta=0.2
                delta2=0.3
                if height_d<(1-delta)*height_t or width_d<(1-delta)*width_t or height_d>(1+delta2)*height_t or width_d>(1+delta2)*width_t:
                    if debug:
                        print("Detection improved by tracker")
                    improved=True
                    detect_box=track.current_tracker.box
                # detect_box=helpers.union_box(track_box,detect_box)
            #Track box does not matched
            elif otsu_box==True and helpers.box_ios(detect_box,track_box)>track.box_ios_thres and helpers.box_iou2(track_box,helpers.largest_contour(img))>track.otsu_iou_thres:
                if debug:
                    print("Detect box is subset of track. Track box and Otsu are similar.")
                    print("Detection improved by tracker")
                detect_box=track.current_tracker.box
                improved=True
            else:
                if debug:
                    print("Tracker lost deleting the current tracker")
                track.tracker_list.append(track.current_tracker)
                track.current_tracker=None

        #Improve detect_box by Otsu or any other way
        if otsu_box==True:
            ret,detect_box=helpers.detection_otsu(img,detect_box,draw=True,threshold=track.otsu_iou_thres)
        #Update or create tracker
        #Update if exist or matched
        if track.current_tracker!=None:
            final_box = detect_box
            z = np.expand_dims(detect_box, axis=0).T
            track.current_tracker.kalman_filter(z)
            xx = track.current_tracker.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            if improved:
                final_box = xx
            track.current_tracker.box =xx
            track.current_tracker.hits += 1
            track.current_tracker.no_losses =0
        else:
            final_box = detect_box
            z = np.expand_dims(detect_box, axis=0).T
            track.current_tracker = tracker.Tracker() # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            track.current_tracker.x_state = x
            track.current_tracker.predict_only()
            xx = track.current_tracker.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            track.current_tracker.box =xx

            track.current_tracker.id = track.track_id_list.popleft() # assign an ID for the tracker
            if debug:
                print("New Tracker\n ID: ",track.current_tracker.id)

    #Not Detection
    else:
        #Tracker alive or not
        #alive
        if track.current_tracker!=None:
            if debug:
                print("Tracker Alive")
            track.current_tracker.predict_only()
            xx = track.current_tracker.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]

            if otsu_box==True:
            # if False:
                current_state=xx
                flag,current_otsu=helpers.tracker_otsu(img,current_state,draw=True,threshold=track.otsu_iou_thres)
                if not flag:
                    if debug:
                        print("Tracker does not matched with Otsu box, Tracker id",track.current_tracker.id)
                    xx=helpers.remove_small_box(xx,height_limit=150,width_limit=150)

                    if len(xx)==0:
                        if debug:
                            print("Small track box. Deleting...............")
                        track.tracker_list.append(track.current_tracker)
                        track.current_tracker=None
                        final_box = []
                    else:
                        track.current_tracker.no_losses+=1
                        track.current_tracker.box =xx
                        final_box = xx
                else:
                    if debug:
                        print("Tracker box matched with Otsu box, Tracker id",track.current_tracker.id)

                    track.current_tracker.no_losses += 0.5
                    final_box = current_otsu
                    current_otsu = np.expand_dims(current_otsu, axis=0).T

                    track.current_tracker.kalman_filter(current_otsu)
                    xx = track.current_tracker.x_state.T[0].tolist()
                    xx =[xx[0], xx[2], xx[4], xx[6]]

                    track.current_tracker.box =xx
            else:
                if debug:
                    print("No Otsu")
                xx=helpers.remove_small_box(xx,height_limit=150,width_limit=150)
                if len(xx)==0:
                    if debug:
                        print("Small track box. Deleting...............")
                    track.tracker_list.append(track.current_tracker)
                    track.current_tracker=None
                    final_box = []
                else:
                    track.current_tracker.no_losses += 1
                    track.current_tracker.box =xx
                    final_box = xx

            #---------------------
            #Person left the frames or not
                #If left
                #Not left, no detection
        #Not active tracker
        else:
            if debug:
                print("No tracked Box ")

    #Final box
    if track.current_tracker!=None:
        if ((track.current_tracker.hits >= min_hits) and (track.current_tracker.no_losses <=max_age)):
             # final_box = track.current_tracker.box
             if debug:
                 print('updated box: ', final_box)
                 print()
             if draw:
                 img= helpers.draw_box_label("Final",img, final_box,show_label=True) # Draw the bounding boxes on the
                 img= helpers.draw_box_label(track.current_tracker.id,img, track.current_tracker.box,box_color=(255, 255,0),show_label=False) # Draw the bounding boxes on the
        elif track.current_tracker.no_losses >max_age:
            if debug:
                print('Tracker age criteria is not satisfied. Deleting..........')
            track.tracker_list.append(track.current_tracker)
            track.current_tracker=None
        else:
            if debug:
                print('Tracker zero hit')
            if draw:
                img= helpers.draw_box_label("Final",img, final_box,show_label=True) # Draw the bounding boxes on the
    if debug:
        print("Final Box")
        print(final_box)
    return final_box,img

"""
Function to track and save the corner coordinates in csv
input:
    frames:- list of frames path in sorted order
    numbers: the sorted numbes as mentioned in the image name
    Output_path:
    otsu_box:
    visualize:
"""
def tracking_frames_to_csv(detector,frames,numbers,output_path,otsu_box=True,visualize=False):
    track=single_tracker()
    filename=os.path.basename(output_path)
    output_csvname=os.path.join(output_path,filename+'.csv')
    outFile = open(output_csvname, 'w')
    print("----------------------")
    print("csv output path-",output_csvname)
    print("----------------------")
    outFile.write(','.join(['Frame number', 'Track box (y_up x_left y_down x_right)']) + '\n');
    total=len(frames)
    for i in tqdm.tqdm(range(total)):
        frame=frames[i]
        number=numbers[i]
        img=cv2.imread(frame)
        np.asarray(img)

        box,new_img = pipeline_single_tracker(detector,img,otsu_box,track,draw=visualize)
        if len(box)!=0:
            outFile.write(','.join([str(number),box_to_str(box)]) + '\n');



        if visualize:
            cv2.imshow("frame",new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    outFile.close()

    cv2.destroyAllWindows()

"""
Function to track and save the video visualizations
input:
    frames:- list of frames path in sorted order
    Output_path:
    otsu_box:
    visualize:
"""
def tracking_frames_to_video(detector,frames,output_path,frame_rate=10.0,otsu_box=True,visualize=False):
    track=single_tracker()
    output_filename=os.path.join(output_path,'Track.avi')

    print("----------------------")
    print("Video output path-",output_filename)
    print("----------------------")
    # start=time.time()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename,fourcc, frame_rate, (640,480))
    for frame in tqdm.tqdm(frames):
        img=cv2.imread(frame)
        np.asarray(img)

        # new_img = pipeline(det,img,otsu_box)

        _,new_img = pipeline_single_tracker(detector,img,otsu_box,track,draw=True)
        # cv2.imshow("frame",new_img)
        out.write(new_img)
        if visualize:
            cv2.imshow("frame",new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    out.release()

if __name__ == "__main__":

    # print("Available GPU devices")
    parser = argparse.ArgumentParser(description='Person tracking on TSF dataset with Tensorflow API model and Kalman Filtering')
    parser.add_argument('--detection_threshold', default=0.3,
                        help='Threshold for object detection')
    parser.add_argument('--output_type', default='csv',
                             help='Type of output-csv or video')
    parser.add_argument('--otsu_box',default=True,
                help='Set True or False for contour box localization')
    parser.add_argument('--visualize',default=False,
                help='It will show the tracking boxes at run time')
    parser.add_argument('--debug',default=False,
                help='Print the intermediate steps for each image')
    args = parser.parse_args()


    otsu_box=False
    if args.otsu_box=='True':
        print("-------------------------")
        print("Using Otsu Contour box")
        print("-------------------------")
        otsu_box=True

    if args.debug=='True':
        debug=True

    visualize=False
    if args.visualize=='True':
        print("-------------------------")
        print("Viusal mode: All the frames will be shown")
        print("-------------------------")
        visualize=True
        #Initialization of tracker and detector
    detector = detector.PersonDetector(threshold=float(args.detection_threshold),model_path=config.detector_model_path)

    #Output video frame rate
    frame_rate=10.0
    root_drive = config.root_drive
    track_root_folder=config.track_root_folder
    output_type=args.output_type
    dset=config.root_folder#The name of dataset should match the folder name

    output_dir=root_drive+'/'+track_root_folder+'/'+output_type
    #Getting input and output folder mapping
    ADL_list,Fall_list=get_mapped_dir_lists(dset,output_dir=output_dir,d_type='frame')
    if len(ADL_list)==0:
        print("Dataset directory not found")

    #Tracking fall videos
    for input_path,output_path in tqdm.tqdm(Fall_list):
        print("------------------")
        print("input_path:",input_path)
        print("------------------")
        os.makedirs(output_path, exist_ok=True)
        frames = glob.glob(input_path+'/*.jpg') + glob.glob(input_path+'/*.png')
        frames,numbers = sort_frames(frames, dset)
        if output_type=='video':
            tracking_frames_to_video(detector,frames,output_path,frame_rate,otsu_box,visualize)
        elif output_type=='csv':
            tracking_frames_to_csv(detector,frames,numbers,output_path,otsu_box,visualize)
        else:
            print("Invalid output_type argument")
            sys.exit()


    #Tracking ADL videos
    for input_path,output_path in tqdm.tqdm(ADL_list):
        os.makedirs(output_path, exist_ok=True)
        print("------------------")
        print("input_path:",input_path)
        print("------------------")
        frames = glob.glob(input_path+'/*.jpg') + glob.glob(input_path+'/*.png')
        frames,numbers = sort_frames(frames, dset)
        if output_type=='video':
            #tracking_frames_to_video(tracker,detector,frames,output_path,frame_rate=10.0,otsu_box=True,visualize=False)
            tracking_frames_to_video(detector,frames,output_path,frame_rate,otsu_box,visualize)
        elif output_type=='csv':
            tracking_frames_to_csv(detector,frames,numbers,output_path,otsu_box,visualize)
        else:
            print("Invalid output_type argument")
            sys.exit()
