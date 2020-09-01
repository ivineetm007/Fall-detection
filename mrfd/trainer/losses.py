from keras import backend as K
import numpy as np

#--------------------------------------------
#Keras loss functions for back propagation
#--------------------------------------------

def ROI_mean_squared_error_loss(mask):
    '''
        ROI masked reconstruction loss
    '''
    mask_sum=K.sum(mask)
    def mse_loss(y_true,y_pred):
        return K.sum(K.square(y_true*mask-y_pred*mask))/mask_sum
    return mse_loss

def ROI_diff_temporal_loss(mask,diff_mask):
    """
        ROI masked difference frames reconstruction loss
    """
    diff_mask_sum=K.sum(diff_mask)
    def diff_temporal_loss(y_true,y_pred):
        pred=y_pred*mask
        true=y_true*mask
        pred_diff=pred[:,1:]-pred[:,:-1]
        true_diff=true[:,1:]-true[:,:-1]
        return K.sum(K.square(true_diff-pred_diff))/diff_mask_sum
    return diff_temporal_loss

def ROI_diff_mse_joint_loss(mask,diff_mask,lamdba_S,lamdba_T):
    """
        Combined loss for thermal autoencoder-
    """
    diff_mask_sum=K.sum(diff_mask)
    mask_sum=K.sum(mask)

    def mse_loss(y_true,y_pred):
        return K.sum(K.square(y_true*mask-y_pred*mask))/mask_sum

    def diff_temporal_loss(y_true,y_pred):
        pred=y_pred*mask
        true=y_true*mask
        pred_diff=pred[:,1:]-pred[:,:-1]
        true_diff=true[:,1:]-true[:,:-1]
        return K.sum(K.square(true_diff-pred_diff))/diff_mask_sum

    def joint_loss(y_true,y_pred):
        return lamdba_S*mse_loss(y_true,y_pred)+lamdba_T*diff_temporal_loss(y_true,y_pred)

    return joint_loss

def ROI_diff_mse_joint_loss(mask,diff_mask,lamdba_S,lamdba_T):
    """
        Joint loss for diff ROI 3DCAE
    """
    diff_mask_sum=K.sum(diff_mask)
    mask_sum=K.sum(mask)

    def mse_loss(y_true,y_pred):
        return K.sum(K.square(y_true*mask-y_pred*mask))/mask_sum

    def diff_temporal_loss(y_true,y_pred):
        pred=y_pred*mask
        true=y_true*mask
        pred_diff=pred[:,1:]-pred[:,:-1]
        true_diff=true[:,1:]-true[:,:-1]
        return K.sum(K.square(true_diff-pred_diff))/diff_mask_sum

    def joint_loss(y_true,y_pred):
        return lamdba_S*mse_loss(y_true,y_pred)+lamdba_T*diff_temporal_loss(y_true,y_pred)

    return joint_loss
#--------------------------------------------
#python loss functions for numpy arrays
#--------------------------------------------
def wind_mean_squared_error(y_true,y_pred,win_length, img_height,img_width,channels):
    '''
        Reconstruction error
    '''
    y_pred = y_pred.reshape(len(y_pred),win_length, img_height*img_width*channels)#(samples-win_length+1, 5, wd*ht)
    y_true = y_true.reshape(len(y_true),win_length, img_height*img_width*channels)#(samples-win_length+1, 5, wd*ht)

    MSE = np.mean(np.power(y_true-y_pred, 2), axis = 2)# (samples-win_length+1,win_length)
    return MSE

def wind_ROI_mean_squared_error(mask,y_true,y_pred,win_length, img_height,img_width,channels):
    '''
        ROI masked Reconstruction error
    '''
    #apply mask
    y_true=y_true*mask
    y_pred=y_pred*mask

    y_pred = y_pred.reshape(len(y_pred),win_length, img_height*img_width*channels)#(samples-win_length+1, 5, wd*ht)
    y_true = y_true.reshape(len(y_true),win_length, img_height*img_width*channels)#(samples-win_length+1, 5, wd*ht)
    mask=mask.reshape(len(y_true),win_length, img_height*img_width*channels)


    SE=np.sum(np.power(y_true - y_pred, 2), axis = 2)# (samples-win_length+1,win_length)
    mask_sum=np.sum(mask,axis=2)

    return SE/mask_sum

def wind_ROI_diff_temporal_loss(mask,diff_mask,y_true,y_pred,win_length, img_height,img_width,channels):
    '''
        ROI masked difference frames reconstruction loss
    '''
    #apply mask
    y_true=y_true*mask
    y_pred=y_pred*mask

    y_pred = y_pred.reshape(len(y_pred),win_length, img_height*img_width*channels)#(samples-win_length+1, 5, wd*ht)
    y_true = y_true.reshape(len(y_true),win_length, img_height*img_width*channels)#(samples-win_length+1, 5, wd*ht)
    diff_mask=diff_mask.reshape(len(y_true),win_length-1, img_height*img_width*channels)
    #diff
    pred_diff=y_pred[:,1:]-y_pred[:,:-1]
    true_diff=y_true[:,1:]-y_true[:,:-1]

    SE=np.sum(np.power(true_diff - pred_diff, 2), axis = 2)# (samples-win_length+1,win_length-1)
    diff_mask_sum=np.sum(diff_mask,axis=2)

    return SE/diff_mask_sum
