from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, f1_score, auc, precision_recall_curve
import glob
import os
import numpy as np
from sklearn.utils import class_weight as cw
import matplotlib.pyplot as plt
# import h5py
from random import randint, seed
from imblearn.metrics import geometric_mean_score
import re
import sys
from data_management import *
seed(0)

#-----------------
#Data Processing
#-----------------
def create_windowed_arr(arr, stride, window_length):
    """
    Create windows of the given array
    """

    img_width, img_height,channels = arr.shape[1], arr.shape[2], arr.shape[3]

    output_length = int(np.floor((len(arr) - window_length) / stride))+1
    output_shape = (output_length, window_length, img_width, img_height, channels)

    total = np.zeros(output_shape)

    i=0
    while i < output_length:
        next_chunk = np.array([arr[i+j] for j in range(window_length)]) #Can use np.arange if want to use time step \
        # ie. np.arrange(0,window_length,dt)

        total[i] = next_chunk

        i = i+stride

    # arr_windowed = total

    return total

def create_diff_mask(mask_windows):
    '''
        Create masks for differenc/optical flow frames.
        Take union of masks of consecutive mask frames
    '''
    mask_shape=list(mask_windows.shape)
    mask_shape[1]=mask_shape[1]-1
    diff_mask_windows=np.zeros(shape=tuple(mask_shape),dtype='int8')
    for i in range(mask_shape[1]):
        diff_mask_windows[:,i,:]=np.bitwise_or(mask_windows[:,i,:],mask_windows[:,i+1,:])
    return diff_mask_windows

#from models import simple_conv
np.set_printoptions(threshold = 200)

def save_multiple_graph(x_list,labels=['train','val'],x_label='Epochs',y_label='Loss',title='Loss Plot',path='',log_plot=False):
    ''''
    Given multiple lists, plot a single graph
    '''
    if log_plot==True:
        plt.yscale('log')
    for i,(y,label) in enumerate(zip(x_list,labels)):
        x=[i+1 for i in range(len(x_list[i]))]
        plt.plot(x,y,label=label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    plt.title(title)
    plt.legend()
    plt.savefig(path)
    # plt.show()
    plt.close()

def create_windowed_labels(labels, stride, tolerance, window_length):
    '''
    Create labels on seq level

    int tolerance: number of fall frames (1's) in a window for it to be labeled as a fall (1). must not exceed window length

    '''
    output_length = int(np.floor((len(labels) - window_length) / stride))+1
    #output_shape = (output_length, window_length, 1)
    output_shape = (output_length, 1)

    total = np.zeros(output_shape)

    i=0
    while i < output_length:
        next_chunk = np.array([labels[i+j] for j in range(window_length)])
        num_falls = sum(next_chunk) #number of falls in the window
        if num_falls >= tolerance:
            total[i] = 1
        else:
            total[i] = 0
        i = i+stride
    labels_windowed = total

    return labels_windowed
#
def gather_auc_avg_per_tol(inwin_mean, inwin_std, labels_list, win_len = 8,tolerance_limit=8):
    '''
    inwin_mean/std are mean over the windows for a video (1,num_windows = vid_length - win_len-1)

    Retruns array of shape (2,win_len = tolerance*2), which are scores for each tolerance in range(win_len),
    *2 for std and mean, one row
    for AUROC, one for AUPR

    tol1_mean, tol1_std, tol2_men, tol2_std.....
    '''
    img_width, img_height = 64,64
    stride= 1

    tol_mat = np.zeros((2,2*tolerance_limit))
    tol_list_ROC = []
    tol_list_PR = []

    # print(tol_mat.shape)
    # print("Window based score.... ")
    tol_keys = [] #For dataframe labels
    for tolerance in range(tolerance_limit):
        # print("Tolerance: ",tolerance+1)
        tolerance +=1 #Start at 1

        windowed_labels = [create_windowed_labels(labels, stride, tolerance, win_len) for labels in labels_list]
        windowed_labels=np.concatenate(windowed_labels)
        # print("Number of window score ",len(inwin_mean))
        # print("Number of windows labels",len(windowed_labels))
	# plt.plot(windowed_labels)
	# plt.show()
        AUROC_mean, conf_mat, g_mean, AUPR_mean = get_output(labels = windowed_labels, \
    	    predictions = inwin_mean, data_option = 'NA', to_plot = False) #single value
        AUROC_std, conf_mat, g_mean, AUPR_std = get_output(labels = windowed_labels, predictions = inwin_std, data_option = 'NA', to_plot = False)

    	#print(AUROC_mean)
        tol_list_ROC.append(AUROC_mean)
    	#tol_mat[0, tolerance-1] = AUROC_mean #mean AUROC note mean refers to inwin_mean, not taking mean of AUROC
        tol_keys.append('tol_{}-mean'.format(tolerance))
        tol_list_ROC.append(AUROC_std)
    	#tol_mat[0, tolerance] = AUROC_std #std AUROC "" std
        tol_keys.append('tol_{}-std'.format(tolerance))

#	 tol_mat[1, tolerance-1] = AUPR_mean #mean AUPR
#	 tol_mat[1, tolerance] = AUPR_std #mean AUPR
        tol_list_PR.append(AUPR_mean)
        tol_list_PR.append(AUPR_std)
    ROCS = np.array(tol_list_ROC)
    PRS = np.array(tol_list_PR)
    tol_mat[0,:] = ROCS
    tol_mat[1,:] = PRS
    return tol_mat, tol_keys
#
def gather_auc_avg_per_tol_Disc(D_scores, labels_list, win_len = 8,tolerance_limit=8):
    '''
    D scores come from discriminaor, are for window
    '''
    img_width, img_height = 64,64
    stride= 1

    tol_mat = np.zeros((2,tolerance_limit))
    tol_list_ROC = []
    tol_list_PR = []

    print(tol_mat.shape)
    tol_keys = [] #For dataframe labels
    for tolerance in range(tolerance_limit):
        tolerance +=1 #Start at 1
        print("Tolerance: ",tolerance)
        windowed_labels = [create_windowed_labels(labels, stride, tolerance, win_len) for labels in labels_list]
        windowed_labels=np.concatenate(windowed_labels)

        print("Number of window score ",len(D_scores))
        print("Number of windows labels",len(windowed_labels))
    	# plt.plot(windowed_labels)
    	# plt.show()
        AUROC, conf_mat, g_mean, AUPR = get_output(labels = windowed_labels,predictions = D_scores, data_option = 'NA', to_plot = False) #single value
    	#print(AUROC_mean)
        tol_list_ROC.append(AUROC)
    	#tol_mat[0, tolerance-1] = AUROC_mean #mean AUROC note mean refers to inwin_mean, not taking mean of AUROC
        tol_keys.append('tol_{}'.format(tolerance))
        #	 tol_mat[1, tolerance-1] = AUPR_mean #mean AUPR
        #	 tol_mat[1, tolerance] = AUPR_std #mean AUPR
        tol_list_PR.append(AUPR)
    ROCS = np.array(tol_list_ROC)
    PRS = np.array(tol_list_PR)
    tol_mat[0,:] = ROCS
    tol_mat[1,:] = PRS
    return tol_mat, tol_keys

def threshold(predictions = None, t = 0.5):
    temp = predictions.copy()
    predicted_classes = temp.reshape(predictions.shape[0])
    for i in range(len(predicted_classes)):
        if predicted_classes[i]<t:
            predicted_classes[i] = 0
        else:
            predicted_classes[i] = 1

    return predicted_classes

def get_output(labels, predictions, data_option = None, t=0.5, to_plot = False, pos_label = 1):
    predicted_classes = threshold(predictions, t)
    true_classes = labels
    conf_mat = confusion_matrix(y_true = true_classes, y_pred = predicted_classes)
    #report = classification_report(true_classes, predicted_classes)
    AUROC = []
    AUPR = []
    # print(np.where(labels==1))
    if np.count_nonzero(labels) > 0 and np.count_nonzero(labels) != labels.shape[0]: #Makes sure both classes present

        fpr, tpr, thresholds = roc_curve(y_true = true_classes, y_score = predictions, pos_label = pos_label)
        #auc1 = roc_auc_score(y_true = labels, y_score = predictions)
        AUROC = auc(fpr, tpr)

        precision, recall, thresholds = precision_recall_curve(true_classes, predictions)
        AUPR = auc(recall, precision)
        # if auc1<0.5:

        #     auc1 = roc_auc_score(y_true = 1-labels, y_score = predictions)
        #print('ROC AUC is ', auc1)
        if to_plot == True:
            plot_ROC_AUC(fpr,tpr, AUROC, data_option)
    else:
        print('only one class present')
        #g_mean = geometric_mean_score(labels, predicted_classes)
    g_mean = geometric_mean_score(labels, predicted_classes)
        # print(report)
        # print("\n")
        #print(conf_mat)

    return AUROC, conf_mat, g_mean, AUPR
#
def MSE(y, t):
    y, t = y.reshape(len(y), np.prod(y.shape[1:])), t.reshape(len(t), np.prod(t.shape[1:]))

    return np.mean(np.power(y-t,2), axis=1)


def generate_vid_keys(vid_base_name, dset):
    #print('dset', dset)

    if dset == 'Thermal' or dset == 'Thermal_pose' or dset == 'Thermal_track':
        if vid_base_name == 'Fall' or vid_base_name =='NFFall':
            num_vids = 35
        elif vid_base_name == 'ADL':
            num_vids = 9
        else:
            print('invalid basename')

    if (dset == 'UR' or dset =='UR-Filled') and vid_base_name == 'ADL':
        keys = ['adl-{num:02d}-cam0-d'.format(num = i+1) for i in range(num_vids)]
    else:
        keys = [vid_base_name + str(i+1) for i in range(num_vids)]
    return keys


def plot_ROC_AUC_tol(fpr, tpr, roc_auc, data_option,tolerance):
    '''
    plots fo rmultiple tolerance
    '''

    #plt.figure()
    lw = 2
    plt.plot(fpr, tpr,\
	 lw=lw, label='tolerance %0.1f (area = %0.4f)'%(tolerance, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for {}'.format(data_option))
    plt.legend(loc="lower right")
    #plt.close()
    #plt.show()

    return plt
#
def make_cross_window_matrix(scores):
    """
    Takes input of form (samples,window_length) corresponding to
    RE averaged accross image dims, and creates matrix of form
    (image_index,cross_window_score)
    """

    win_len = scores.shape[1]
    mat = np.zeros((len(scores)+win_len-1,len(scores)))
    mat[:] = np.NAN
    #print('mat[:,0].shape', mat[:,0].shape)
    #print(mat.shape)
    for i in range(len(scores)):
        #print(i, len(win)+i)
        win = scores[i]
        mat[i:len(win)+i,i] = win

    return mat

def get_cross_window_stats(scores_mat):
    '''
    Assumes scores in form (image_index,cross_window_scores), ie. shape (samples,window_len)
    returns in form (img_index, mean, std, mean+std)
    '''
    scores_final = []
    for i in range(len(scores_mat)):
        #print(i)
        row = scores_mat[i,:]
        #print(row.shape)
        mean = np.nanmean(row, axis= 0)
        std = np.nanstd(row, axis= 0)
        scores_final.append((mean,std, mean+std*10**3))
    # print(len(scores_final))

    scores_final = np.array(scores_final)
    return scores_final

def join_mean_std(AUROC_avg, AUROC_std):
    new_mat = np.zeros(AUROC_avg.shape, dtype = object)
    AUROC_avg = np.around(AUROC_avg, decimals = 2)
    AUROC_std = np.around(AUROC_std, decimals = 2)

    for i in range(len(new_mat)):

        next_val = AUROC_avg[i]
        next_std = AUROC_std[i]
        new_mat[i] = '{}({})'.format(next_val, next_std)
    return new_mat

def agg_window(RE, agg_type):
    '''
    Aggregates window of scores in various ways
    '''

    if agg_type == 'in_mean':
        # print('Calculating in_mean ...................')
        inwin_mean = np.mean(RE, axis =1)
        return inwin_mean

    elif agg_type == 'in_std':
        # print('Calculating in_std ...................')
        # print('inwin_mean', inwin_mean.shape)
        inwin_std = np.std(RE,axis=1)
        return inwin_std
        #inwin_labels = labels_total[win_len-1:]

    elif agg_type == 'x_std':
        # print('Calculating x_std ...................')
        RE_xmat = make_cross_window_matrix(RE)
        stats = get_cross_window_stats(RE_xmat)
        x_std = stats[:,1]
        return x_std

    elif agg_type == 'x_mean':
        # print('Calculating x_mean ...................')
        RE_xmat = make_cross_window_matrix(RE)
        stats = get_cross_window_stats(RE_xmat)
        x_mean = stats[:,0]
        return x_mean

    else:
        print('agg_type not found')
