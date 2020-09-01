#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This file contains helpers functions for tracking only.
"""
from skimage.filters import threshold_otsu
import numpy as np
import cv2
'''-------------------------------------'''
'''New function added @author: vineet'''
'''-------------------------------------'''
def draw_box(img,box,color=(0,255,0)):
    # draw the biggest contour (c) in green
    cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),color,2)

def remove_small_box(box,height_limit=150,width_limit=150):
    height_d,width_d=find_dim(box)
    if height_d<height_limit and width_d<width_limit:
         box=[]
    elif height_d<height_limit or width_d<width_limit and width_d*height_d<height_limit*width_limit:
        box=[]
    return box

def largest_contour(img,kernel_size=5,debug=False,offset=20):
    im=img.copy()
    imgray = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2GRAY)
    thresh = threshold_otsu(imgray)

    # Thresholding
    ret, img_thres = cv2.threshold(imgray, thresh, 255, cv2.THRESH_BINARY)


    #opening
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    img_opening = cv2.morphologyEx(img_thres, cv2.MORPH_OPEN, kernel)

    #See description of contours function
    #https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv2.findContours(img_opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    if len(contours) != 0:
        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
    else:
        print("No Contours found")
        return []


    if debug==True:
        print("Number of Contours found = " + str(len(contours)))
        cv2.imshow('Threshold image',img_thres)
        cv2.imshow('Opening image',img_opening)
        cv2.imshow('Contours', im_c)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return [y-offset,x-offset,y+h+offset,x+w+offset] #same convention used in Detection Box

def find_area(box):
    left, top, right, bottom =box[1], box[0], box[3], box[2]
    return abs((top-bottom)*(right-left))
"""
Returns height and width of the given box
"""
def find_dim(box):
    left, top, right, bottom =box[1], box[0], box[3], box[2]
    return abs(bottom-top),abs(right-left)

def select_box(box,detect_box,debug=False):
    area_box=find_area(box)
    area_detect_box=find_area(detect_box)
    if area_box>area_detect_box:
        if debug:
            print("Otsu Box is bigger in area")
        return box
    print("Detect/track Box is bigger in area")
    return detect_box

def union_box(box1,box2):
    left1, top1, right1, bottom1 =box1[1], box1[0], box1[3], box1[2]
    left2, top2, right2, bottom2 =box2[1], box2[0], box2[3], box2[2]
    return [min(top1,top2),min(left1,left2),max(bottom1,bottom2),max(right1,right2)]


def detection_otsu(img,detect_box,threshold=0.5,draw=True,delta=0.25):
    box=largest_contour(img)
    print(detect_box)
    if len(detect_box)==0:
        if draw:
            draw_box(img,box,color=(0,0,255))
        return detect_boxes
    if box_iou2(box,detect_box)>threshold:


        # return [select_box(box,detect_boxes,debug=True)]
        height_o,width_o=find_dim(box)
        height_d,width_d=find_dim(detect_box)
        if height_o<(1-delta)*height_d or width_o<(1-delta)*width_d:
            # print("--------------------------------------------------")
            # print("Detection macthes with otsu but it is small")
            # print("--------------------------------------------------")
            if draw:
                # draw_box(img,box,color=(0,0,255))
                draw_box_label('Otsu NM',img,box,box_color=(0,0,255))
            return 0,detect_box
        if draw:
            # draw_box(img,box,color=(0,255,0))
            draw_box_label('Otsu M',img,box,box_color=(0,255,0))

        #
        # print("--------------------------------------------------")
        # print("Detection box matches with the largest contour box")
        # print("--------------------------------------------------")

        return 1,box
        # return detect_boxes
    else:
        # print("--------------------------------------------------")
        # print("Detection box no match contour box IOU ")
        # print("--------------------------------------------------")
        if draw:
            draw_box_label('Otsu NM',img,box,box_color=(0,0,255))
        # return []
        return -1,detect_box
'''
Single tracker
'''

def tracker_otsu(img,track_box,threshold=0.5,draw=True,delta=0.35):
    box=largest_contour(img)

    if box_iou2(box,track_box)>threshold:
        # print("--------------------------------------------------")
        # print("Detection box matches with the largest contour box")
        # print("--------------------------------------------------")
        height_o,width_o=find_dim(box)
        height_t,width_t=find_dim(track_box)
        if height_o<(1-delta)*height_t or width_o<(1-delta)*width_t:
            print("--------------------------------------------------")
            print("Tracking box no match contour box DIMENSION ")
            print("--------------------------------------------------")
            if draw:
                draw_box_label('Otsu NM',img,box,box_color=(0,0,255))
            return False,track_box
        if draw:
            # draw_box(img,box,color=(0,255,0))
            draw_box_label('Otsu M',img,box,box_color=(0,255,0))
        return True,box
        # return track_box
    else:
        print("--------------------------------------------------")
        print("Tracking box no match contour box IOU ")
        print("--------------------------------------------------")
        if draw:
            draw_box_label('Otsu NM',img,box,box_color=(0,0,255))
        return False,track_box
        # return track_box

def box_ios(a, b):
    '''
    Helper funciton to calculate the ratio between intersection and the area of the smaller box a.
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''

    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    # s_b = (b[2] - b[0])*(b[3] - b[1])

    return float(s_intsec)/s_a
'''-------------------------------------'''


"""
@author: kyleguan

"""
def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);



def box_iou2(a, b):
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''

    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])

    return float(s_intsec)/(s_a + s_b -s_intsec)

def draw_box_label(id,img, bbox_cv2, box_color=(0, 255, 255), show_label=True):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]

    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)

    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        cv2.rectangle(img, (left-2, top-45), (right+2, top), box_color, -1, 1)

        # Output the labels that show the x and y coordinates of the bounding box center.
        text_x= 'id='+str(id)
        cv2.putText(img,text_x,(left,top-25), font, font_size, font_color, 1, cv2.LINE_AA)
        text_y= 'y='+str((top+bottom)/2)
        cv2.putText(img,text_y,(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)

    return img
