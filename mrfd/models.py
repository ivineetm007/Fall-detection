# from keras.models import load_model
#Keras import
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, MaxPooling3D, UpSampling3D, Conv3D, Conv2DTranspose, Conv1D, UpSampling1D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Reshape, BatchNormalization
# from keras.layers import Conv3DTranspose as Deconvolution3D
from keras.layers import Deconvolution3D
from keras.optimizers import SGD
from keras import regularizers
from keras.layers import Input, LSTM, RepeatVector, Concatenate,Lambda
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import keras
#Other imports
from trainer.losses import ROI_mean_squared_error_loss,ROI_diff_mse_joint_loss,ROI_diff_temporal_loss,ROI_diff_mse_joint_loss

def diff_ROI_C3D_AE_no_pool(img_width, img_height, win_length, regularizer_list = [],channels=1,lambda_S=1,lambda_T=1,d_type=None):
    """
        diff-ROI-3DCAE model
    """
    def multiply(x):
        image,mask = x #could be K.stack([mask]*3, axis=-1) too
        return mask*image


    input_shape = (win_length, img_width, img_height, channels)
    input_diff_shape = (win_length-1, img_width, img_height, channels)

    input_window = Input(shape = input_shape)
    input_mask=Input(shape = input_shape)
    input_diff_mask=Input(shape = input_diff_shape)


    temp_pool = 2
    temp_depth = 5
    x = Conv3D(16, (5, 3,3), activation='relu', padding='same')(input_window)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    x = Conv3D(8, (5, 3,3), activation='relu', strides = (1,2,2), padding='same')(x)
    if 'Dropout' in regularizer_list:
        x = Dropout(0.25)(x)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    x = Conv3D(8, (5, 3,3), activation='relu', strides = (2,2,2),padding='same')(x)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    encoded = Conv3D(8, (5, 3,3), activation='relu', strides = (2,2,2),padding='same')(x)

    x = Deconvolution3D(8, (temp_depth, 3, 3),strides = (2,2,2), activation='relu', padding='same')(encoded)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    #Double all the dimensions if win_length is even other wise double the H and W, change T to 2*T-1
    if win_length%2==0:
        x = Deconvolution3D(8, (temp_depth, 3, 3), strides = (2,2,2), activation='relu', padding='same')(x)
    else:
        #Deconvolution formula for valid padding
        #out=stride*input-stride+kernel_size
        input_temporal_size=int((win_length+1)/2)
    #         print(input_temporal_size)
        x = Deconvolution3D(8, (input_temporal_size, 2, 2), strides = (1,2,2), activation='relu', padding='valid')(x)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    x = Deconvolution3D(16, (temp_depth, 3, 3), strides = (1,2,2), activation='relu', padding='same')(x)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    layer_name='decoded'
    if d_type!=None:
        layer_name=d_type+'_'+layer_name
    decoded = Conv3D(channels, (5, 3, 3), activation='tanh', padding='same', name = layer_name)(x)
    #Model_name
    model_name="R"
    if d_type!=None:
        model_name=d_type+'_AE'
    autoencoder = Model(inputs=[input_window,input_mask,input_diff_mask], outputs=decoded,name="R")

    autoencoder.compile(optimizer='adadelta', loss=ROI_diff_mse_joint_loss(input_mask,input_diff_mask,lambda_S,lambda_T), \
             metrics=[ROI_mean_squared_error_loss(input_mask),ROI_diff_temporal_loss(input_mask,input_diff_mask)])
    #     autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    model_type = '3Dconv'
    model_name = 'diff_ROI_C3DAE_no_pool'

    for reg in regularizer_list:
        model_name += '-' + reg
    model = autoencoder

    return model, model_name, model_type

def ROI_C3D_AE_no_pool(img_width, img_height, win_length, regularizer_list = [],channels=1,d_type=None):
    """
        ROI-3DCAE model
    """
    def multiply(x):
        image,mask = x #could be K.stack([mask]*3, axis=-1) too
        return mask*image


    input_shape = (win_length, img_width, img_height, channels)

    input_window = Input(shape = input_shape)
    input_mask=Input(shape = input_shape)

    temp_pool = 2
    temp_depth = 5
    x = Conv3D(16, (5, 3,3), activation='relu', padding='same')(input_window)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    x = Conv3D(8, (5, 3,3), activation='relu', strides = (1,2,2), padding='same')(x)
    if 'Dropout' in regularizer_list:
        x = Dropout(0.25)(x)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    x = Conv3D(8, (5, 3,3), activation='relu', strides = (2,2,2),padding='same')(x)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    encoded = Conv3D(8, (5, 3,3), activation='relu', strides = (2,2,2),padding='same')(x)

    x = Deconvolution3D(8, (temp_depth, 3, 3),strides = (2,2,2), activation='relu', padding='same')(encoded)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)
    #Double all the dimensions if win_length is even other wise double the H and W, change T to 2*T-1
    if win_length%2==0:
        x = Deconvolution3D(8, (temp_depth, 3, 3), strides = (2,2,2), activation='relu', padding='same')(x)
    else:
        #Deconvolution formula for valid padding
        #out=stride*input-stride+kernel_size
        input_temporal_size=int((win_length+1)/2)
#         print(input_temporal_size)
        x = Deconvolution3D(8, (input_temporal_size, 2, 2), strides = (1,2,2), activation='relu', padding='valid')(x)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    x = Deconvolution3D(16, (temp_depth, 3, 3), strides = (1,2,2), activation='relu', padding='same')(x)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    #naming layer to be use in stacked
    layer_name='decoded'
    model_name="R"
    if d_type!=None:
        model_name=d_type+'_AE'
        layer_name=d_type+'_decoded'
    decoded = Conv3D(channels, (5, 3, 3), activation='tanh', padding='same', name = layer_name)(x)

    autoencoder = Model(inputs=[input_window,input_mask], outputs=decoded,name=model_name)
    autoencoder.compile(optimizer='adadelta', loss=ROI_mean_squared_error_loss(input_mask))

    model_type = '3Dconv'
    model_name = 'ROI_C3DAE-no_pool'

    for reg in regularizer_list:
        model_name += '-' + reg
    model = autoencoder

    return model, model_name, model_type
def C3D_AE_no_pool(img_width, img_height, win_length, regularizer_list = [],channels=1):
    """
        3DCAE model
    """

    input_shape = (win_length, img_width, img_height, channels)

    input_window = Input(shape = input_shape)

    temp_pool = 2
    temp_depth = 5
    x = Conv3D(16, (5, 3,3), activation='relu', padding='same')(input_window)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    x = Conv3D(8, (5, 3,3), activation='relu', strides = (1,2,2), padding='same')(x)
    if 'Dropout' in regularizer_list:
        x = Dropout(0.25)(x)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    x = Conv3D(8, (5, 3,3), activation='relu', strides = (2,2,2),padding='same')(x)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    encoded = Conv3D(8, (5, 3,3), activation='relu', strides = (2,2,2),padding='same')(x)

    x = Deconvolution3D(8, (temp_depth, 3, 3),strides = (2,2,2), activation='relu', padding='same')(encoded)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)
    #Double all the dimensions if win_length is even other wise double the H and W, change T to 2*T-1
    if win_length%2==0:
        x = Deconvolution3D(8, (temp_depth, 3, 3), strides = (2,2,2), activation='relu', padding='same')(x)
        if 'BN' in regularizer_list:
            x= BatchNormalization()(x)
    else:
        #Deconvolution formula for valid padding
        #out=stride*input-stride+kernel_size
        input_temporal_size=int((win_length+1)/2)
#         print(input_temporal_size)
        x = Deconvolution3D(8, (input_temporal_size, 2, 2), strides = (1,2,2), activation='relu', padding='valid')(x)

    # if 'BN' in regularizer_list:
    #     x= BatchNormalization()(x)

    x = Deconvolution3D(16, (temp_depth, 3, 3), strides = (1,2,2), activation='relu', padding='same')(x)
    if 'BN' in regularizer_list:
        x= BatchNormalization()(x)

    decoded = Conv3D(channels, (5, 3, 3), activation='tanh', padding='same', name = 'decoded')(x)

    autoencoder = Model(input_window, decoded,name="R")
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    model_type = '3Dconv'
    model_name = 'C3DAE-no_pool'

    for reg in regularizer_list:
        model_name += '-' + reg
    model = autoencoder

    return model, model_name, model_type



'''
Deepfall- https://arxiv.org/pdf/1809.00977.pdf
Code-https://github.com/JJN123/Fall-Detection/blob/master/models.py
author- Jacob Nogas, Shehroz S. Khan
'''
def DSTCAE_C3D(img_width, img_height, win_length):
	"""
	int win_length: Length of window of frames
	"""

	input_shape = (win_length, img_width, img_height, 1)

	input_window = Input(shape = input_shape)

	temp_pool = 2
	temp_depth = 5
	x = Conv3D(16, (5, 3,3), activation='relu', padding='same')(input_window)
	x = MaxPooling3D((1,2, 2), padding='same')(x)

	x = Conv3D(8, (5, 3, 3), activation='relu', padding='same')(x)
	x = MaxPooling3D((temp_pool, 2, 2), padding='same')(x) #4
	x = Dropout(0.25)(x)
	x = Conv3D(8, (5, 3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling3D((temp_pool, 2, 2), padding='same')(x) #2

	# at this point the representation is (4, 4, 8) i.e. 128-dimensional

	x = Conv3D(8, (5, 3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling3D((temp_pool, 2, 2))(x) #4
	x = Conv3D(8, (5, 3, 3), activation='relu', padding='same')(x)
	x = UpSampling3D((temp_pool, 2, 2))(x) #8

	x = Conv3D(16, (5, 3, 3), activation='relu', padding = 'same')(x)
	x = UpSampling3D((1, 2, 2))(x)
	decoded = Conv3D(1, (5, 3, 3), activation='tanh', padding='same')(x)



	autoencoder = Model(input_window, decoded)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

	model_type = 'conv'
	model_name = 'DSTCAE_C3D'
	encoder = None
	decoder = None
	model = autoencoder

	return model, model_name, model_type
'''
CONVLSTM- http://individual.utoronto.ca/shehroz/files/ARIALIJCAI.pdf
Code-https://github.com/JJN123/Fall-Detection/blob/master/models.py
author- Jacob Nogas, Shehroz S. Khan
'''
def CLSTM_AE(img_width, img_height, win_len):

	"""
	from https://github.com/yshean/abnormal-spatiotemporal-ae/blob/master/classifier.py
	"""
	from keras.models import Model
	from keras.layers.convolutional import Conv2D, Conv2DTranspose
	from keras.layers.convolutional_recurrent import ConvLSTM2D
	from keras.layers.normalization import BatchNormalization
	from keras.layers.wrappers import TimeDistributed
	from keras.layers.core import Activation
	from keras.layers import Input

	input_tensor = Input(shape=(win_len, img_width, img_height, 1))

	conv1 = TimeDistributed(Conv2D(128, kernel_size=(11, 11), padding='same', strides=(4, 4), name='conv1'),
	                        input_shape=(win_len, 224, 224, 1))(input_tensor)
	conv1 = TimeDistributed(BatchNormalization())(conv1)
	conv1 = TimeDistributed(Activation('relu'))(conv1)

	conv2 = TimeDistributed(Conv2D(64, kernel_size=(5, 5), padding='same', strides=(2, 2), name='conv2'))(conv1)
	conv2 = TimeDistributed(BatchNormalization())(conv2)
	conv2 = TimeDistributed(Activation('relu'))(conv2)

	convlstm1 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm1')(conv2)
	convlstm2 = ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm2')(convlstm1)
	convlstm3 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm3')(convlstm2)

	deconv1 = TimeDistributed(Conv2DTranspose(128, kernel_size=(5, 5), padding='same', strides=(2, 2), name='deconv1'))(convlstm3)
	deconv1 = TimeDistributed(BatchNormalization())(deconv1)
	deconv1 = TimeDistributed(Activation('relu'))(deconv1)

	decoded = TimeDistributed(Conv2DTranspose(1, kernel_size=(11, 11), padding='same', strides=(4, 4), name='deconv2'))(
	    deconv1)

	model =  Model(inputs=input_tensor, outputs=decoded)
	model.compile(optimizer='adadelta', loss='mean_squared_error')

	model_name = 'CLSTM_AE'
	model_type = 'conv'

	return model, model_name, model_type


"""
Discriminator Models
"""
def C3D_no_pool(img_width, img_height, win_length, regularizer_list =[],channels=1):
    """
        3DCNN nodel without pooling
    """
    from keras.layers.normalization import BatchNormalization
    input_shape = (win_length, img_width, img_height, channels)

    input_window = Input(shape = input_shape)

    temp_pool = 2
    temp_depth = 5
    x = Conv3D(16, (5, 3,3), padding='same', strides = (1,2,2))(input_window)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv3D(8, (5, 3, 3), padding='same', strides = (1,2,2))(x)
    x = LeakyReLU(alpha=0.2)(x)
    if 'BN' in regularizer_list:
        x = BatchNormalization()(x)

    x = Conv3D(8, (3, 3, 3), padding='same', strides = (1,2,2))(x)
    x = LeakyReLU(alpha=0.2)(x)
    if 'BN' in regularizer_list:
        x = BatchNormalization()(x)

    encoded = Flatten()(x)
    target_class_likelihood = Dense(1, activation='sigmoid')(encoded)

    model = Model(input_window, target_class_likelihood,name="D")
    sgd = SGD(lr=0.0002, decay=1e-7, momentum=.5)

    model.compile(optimizer=sgd,
          loss='binary_crossentropy')



    model_type = '3Dconv'
    model_name = '3DCNN-no_pool'
    for reg in regularizer_list:
            model_name += '-' + reg

    return model, model_name, model_type


def Fusion_C3D_no_pool(img_width, img_height, win_length, regularizer_list =[],thermal_channels=1,flow_channels=3):
    """
        Fusion Discriminator
    """
    from keras.layers.normalization import BatchNormalization
    #Input shape
    thermal_input_shape = (win_length, img_width, img_height, thermal_channels)
    flow_input_shape = (win_length-1, img_width, img_height, flow_channels)

    temp_pool = 2
    temp_depth = 5
    #------------------------------------------
    #Thermal 3DCNN
    #------------------------------------------
    thermal_input_window = Input(shape = thermal_input_shape)


    thermal_x = Conv3D(16, (5, 3,3), padding='same', subsample = (1,2,2))(thermal_input_window)
    thermal_x = LeakyReLU(alpha=0.2)(thermal_x)

    thermal_x = Conv3D(8, (5, 3, 3), padding='same', subsample = (1,2,2))(thermal_x)
    thermal_x = LeakyReLU(alpha=0.2)(thermal_x)
    if 'BN' in regularizer_list:
        thermal_x = BatchNormalization()(thermal_x)

    thermal_x = Conv3D(8, (3, 3, 3), padding='same', subsample = (1,2,2))(thermal_x)
    thermal_x = LeakyReLU(alpha=0.2)(thermal_x)
    if 'BN' in regularizer_list:
        thermal_x = BatchNormalization()(thermal_x)

    thermal_encoded = Flatten()(thermal_x)
    #------------------------------------------
    #FLow 3DCNN
    #------------------------------------------
    flow_input_window = Input(shape = flow_input_shape)


    flow_x = Conv3D(16, (5, 3,3), padding='same', subsample = (1,2,2))(flow_input_window)
    flow_x = LeakyReLU(alpha=0.2)(flow_x)

    # if 'BN' in regularizer_list:
    #     x = BatchNormalization()(x)

    flow_x = Conv3D(8, (5, 3, 3), padding='same', subsample = (1,2,2))(flow_x)
    flow_x = LeakyReLU(alpha=0.2)(flow_x)
    if 'BN' in regularizer_list:
        flow_x = BatchNormalization()(flow_x)

    flow_x = Conv3D(8, (3, 3, 3), padding='same', subsample = (1,2,2))(flow_x)
    flow_x = LeakyReLU(alpha=0.2)(flow_x)
    if 'BN' in regularizer_list:
        flow_x = BatchNormalization()(flow_x)

    flow_encoded = Flatten()(flow_x)

    #Fusion network
    #-----------------------------------------
    concatenated=Concatenate()([thermal_encoded,flow_encoded])
    target_class_likelihood = Dense(1, activation='sigmoid')(concatenated)

    model = Model(inputs=[thermal_input_window,flow_input_window], outputs=target_class_likelihood,name="D")
    sgd = SGD(lr=0.0002, decay=1e-7, momentum=.5)

    model.compile(optimizer=sgd,
          loss='binary_crossentropy')



    model_type = '3Dconv'
    model_name = 'Fusion_C3D-no_pool'
    for reg in regularizer_list:
            model_name += '-' + reg

    return model, model_name, model_type
