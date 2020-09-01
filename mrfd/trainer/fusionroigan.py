from .losses import ROI_mean_squared_error_loss,wind_ROI_mean_squared_error
from .util import agg_window,create_windowed_arr,save_multiple_graph,get_output,gather_auc_avg_per_tol,join_mean_std,create_diff_mask
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
import numpy as np
import pandas as pd
import os

class Params(object):
    """
        Parameters class to handlde parameters for the Fusion models
    """
    def __init__(self,width=64, height=64,win_length=8,thermal_channels=1,flow_channels=3,dset='Thermal_track',d_type='frame',flow_lambda=1,thermal_lambda=1.0,regularizer_list=None,TR_name=None,FR_name=None,D_name=None,batch_size=32,break_win=10):
        self.width=width
        self.height=height
        self.win_length=win_length
        self.thermal_channels=thermal_channels
        self.flow_channels=flow_channels
        self.dset=dset
        self.d_type=d_type
        self.batch_size = batch_size
        self.flow_lambda=flow_lambda
        self.thermal_lambda=thermal_lambda
        self.regularizer_list=regularizer_list
        self.TR_name=TR_name
        self.FR_name=FR_name
        self.D_name=D_name
        self.gap=break_win

    def create_model_name(self):
        return  self.get_model_type()+ '_T{}_F{}'.format(str(self.thermal_lambda),str(self.flow_lambda))
    def create_hp_name(self):
        return  'lambda_T{}_F{}'.format(str(self.thermal_lambda),str(self.flow_lambda))
    def get_model_type(self):
        TR_name=self.TR_name
        FR_name=self.FR_name
        D_name=self.D_name

        return FR_name+'_'+TR_name+'_'+D_name
    def get_model_dir(self):
        return self.get_root_path()+'/models'
    def get_TR_path(self,epochs_trained):
        return self.get_root_path()+'/models/GAN_T_R_weights_epoch-{}.h5'.format(epochs_trained)
    def get_FR_path(self,epochs_trained):
        return self.get_root_path()+'/models/GAN_F_R_weights_epoch-{}.h5'.format(epochs_trained)
    def get_D_path(self,epochs_trained):
        return self.get_root_path()+'/models/GAN_D_weights_epoch-{}.h5'.format(epochs_trained)
    def get_root_path(self):
        return './{}/{}/{}/{}'.format(self.dset,self.d_type,self.get_model_type(),self.create_hp_name())


class Fusion_ROI_3DCAE_GAN3D(object):
    '''
        Class used to train and test fusion and ROI based adversarial learning
    '''
    def __init__(self, train_par=None,stride=1):

        self.train_par=train_par
        self.stride=stride


    def initialize_model(self,T_Reconstructor, F_Reconstructor, Discriminator ):
        print("Compiling GAN model.")
        self.T_R = T_Reconstructor
        self.F_R = F_Reconstructor
        self.D = Discriminator
        print('Discriminator')
        print(self.D.summary())

        print('Thermal Reconstructor')
        print(self.T_R.summary())
        print('Flow Reconstructor')
        print(self.F_R.summary())
        self.OPTIMIZER = 'adam'



        self.stacked_R_D = self.stacked_R_D()
        loss_weights = {'D':1.0, 'thermal_decoded':self.train_par.thermal_lambda, 'flow_decoded':self.train_par.flow_lambda}


        self.stacked_R_D.compile(loss={'D': 'binary_crossentropy', 'thermal_decoded': ROI_mean_squared_error_loss(self.T_R.input[1]), 'flow_decoded': ROI_mean_squared_error_loss(self.F_R.input[1])},\
         optimizer=self.OPTIMIZER, loss_weights = loss_weights)


    def stacked_R_D(self):
        '''
        Used for training Reconstructor. Dicscriminator is freezed.
        '''

        self.D.trainable = False

        #print(self.R.output)
        model = Model(inputs = self.T_R.input+ self.F_R.input, outputs = [self.T_R.output,self.F_R.output, self.D([self.T_R.output,self.F_R.output])],name='stacked')
        print('stacked')
        print(model.summary())


        return model


    def create_windowed_data(self, videos_dic,stride=1,data_key='FRAME'):
        total = []
        img_width, img_height,win_length=self.train_par.width,self.train_par.height,self.train_par.win_length
        if data_key=='FLOW':
            win_length=win_length-1
        # print('total.shape', 'num_windowed', 'output_shape', total.shape, num_windowed, output_shape)
        for vid_name in videos_dic.keys():
            print('Video Name', vid_name)
            vid_windowed_list=[]
            sub_vid_list=videos_dic[vid_name][data_key]
            for sub_vid in sub_vid_list:
                vid_windowed_list.append(create_windowed_arr(sub_vid, stride, win_length))
            print("Number of sub videos: ",len(vid_windowed_list))
            vid_windowed=np.concatenate(vid_windowed_list)
            total.append(vid_windowed)
        total=np.concatenate(total)
        print("Windowed data shape:")
        print(total.shape)
        return total




    def get_F_RE_all_agg(self, flow_data, flow_masks):
        """
            compute Reconstruction error of flow frames
        """
        img_width, img_height, win_length, channels,model = self.train_par.width, self.train_par.height ,self.train_par.win_length, self.train_par.flow_channels, self.F_R

        recons_seq = model.predict([flow_data,flow_masks]) #(samples-win_length+1, win_length, wd,ht,1)


        RE=wind_ROI_mean_squared_error(flow_masks,flow_data,recons_seq,win_length-1, img_height,img_width,channels)
        print('RE.shape', RE.shape)

        RE_dict = {}

        agg_type_list = ['in_std', 'in_mean']

        for agg_type in agg_type_list:

            RE_dict[agg_type] = agg_window(RE, agg_type)

        return RE_dict, recons_seq
    def get_T_S_RE_all_agg(self, thermal_data, thermal_masks):
        """
            compute Reconstruction error of thermal frames
        """

        img_width, img_height, win_length, channels,model = self.train_par.width, self.train_par.height ,self.train_par.win_length, self.train_par.thermal_channels, self.T_R

        recons_seq = model.predict([thermal_data,thermal_masks]) #(samples-win_length+1, win_length, wd,ht,1)

        RE=wind_ROI_mean_squared_error(thermal_masks,thermal_data,recons_seq,win_length, img_height,img_width,channels)

        RE_dict = {}

        agg_type_list = ['x_std', 'x_mean', 'in_std', 'in_mean']

        for agg_type in agg_type_list:

            RE_dict[agg_type] = agg_window(RE, agg_type)

        return RE_dict, recons_seq

    def train(self, thermal_frame, thermal_mask,flow_frame, flow_mask,epochs= 500,epochs_trained=0, save_interval = 10):
        '''
            Train the adversarial framework
            thermal_frame- window of thermal frames
            thermal_mask- mask of thermal windows
            flow_frame- window of flow frames
            flow_mask- mask of flow windows
        '''

        print('using save root:', self.train_par.get_root_path())
        self.save_root=self.train_par.get_root_path()
        batch_size=self.train_par.batch_size
        print('self.stacked_R_D.metrics_names', self.stacked_R_D.metrics_names)
        print('self.D.metrics_names', self.D.metrics_names)
        num_batches = int(thermal_frame.shape[0]/batch_size)
        print("Train thermal dataset shape",thermal_frame.shape)
        print("Train flow  shape",flow_frame.shape)
        print("Number of batches",num_batches)

        #model save dir
        if not os.path.isdir(self.train_par.get_model_dir()):
            os.makedirs(self.train_par.get_model_dir())
        d_loss_list = [] #m for mean
        r_loss_list_S_RE = [] #Spatial Reconstruction error
        r_loss_list_F_RE = [] #Flow frame  Reconstruction error
        r_loss_list_BCE = [] #Binary cross entropy
        loss_root = self.save_root + '/loss'

        R_loss_root=loss_root +'/R_loss'
        D_loss_root=loss_root +'/D_loss'

        if not os.path.isdir(R_loss_root):
            print("Creating R loss directory ")
            os.makedirs(R_loss_root)

        if not os.path.isdir(D_loss_root):
            print("Creating D loss directory ")
            os.makedirs(D_loss_root)

        print("Loss file status................")
        if os.path.isfile(D_loss_root + '/epoch-{}.npy'.format(epochs_trained)):
            print("D Loss file found")
            d_loss_list=list(np.load(D_loss_root + '/epoch-{}.npy'.format(epochs_trained)))

        if os.path.isfile(R_loss_root + '/S_epoch-{}.npy'.format(epochs_trained)):
            print("R Spatial Loss file found")
            r_loss_list_S_RE=list(np.load(R_loss_root + '/S_epoch-{}.npy'.format(epochs_trained)))

        if os.path.isfile(R_loss_root + '/F_epoch-{}.npy'.format(epochs_trained)):
            print("R Flow Loss file found")
            r_loss_list_F_RE=list(np.load(R_loss_root + '/F_epoch-{}.npy'.format(epochs_trained)))

        if os.path.isfile(R_loss_root + '/BCE_epoch-{}.npy'.format(epochs_trained)):
            print("R BCE Loss file found")
            r_loss_list_BCE=list(np.load(R_loss_root + '/BCE_epoch-{}.npy'.format(epochs_trained)))

        for epoch in range(epochs_trained+1,epochs):
            ## train discriminator

            random_index =  np.random.randint(0, len(thermal_frame) - batch_size)

            permutated_indexes = np.random.permutation(thermal_frame.shape[0])


            for step in range(num_batches):

                batch_indeces = permutated_indexes[step*batch_size:(step+1)*batch_size]
                #Retrieve Batch
                batch_thermal_frame =thermal_frame[batch_indeces]
                batch_thermal_mask=thermal_mask[batch_indeces]
                batch_flow_frame =flow_frame[batch_indeces]
                batch_flow_mask=flow_mask[batch_indeces]
                #AE
                #Thermal
                recons_thermal_images = self.T_R.predict([batch_thermal_frame,batch_thermal_mask])
                #Flow
                recons_flow_images = self.F_R.predict([batch_flow_frame,batch_flow_mask])

                #Combine real and fake
                thermal_combined_batch = np.concatenate((batch_thermal_frame,recons_thermal_images))
                flow_combined_batch = np.concatenate((batch_flow_frame,recons_flow_images))

                y_combined_batch = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))
                # First train Discriminator
                d_loss = self.D.train_on_batch([thermal_combined_batch,flow_combined_batch ], y_combined_batch)

                d_loss_list.append(d_loss)

                # Train Reconstructor

                y_mislabled = np.ones((batch_size, 1))
                r_loss = self.stacked_R_D.train_on_batch([batch_thermal_frame,batch_thermal_mask,batch_flow_frame,batch_flow_mask], {'thermal_decoded':batch_thermal_frame,'flow_decoded':batch_flow_frame,'D':y_mislabled}) #model 2 is D(ie. target class likelihood), dont know how to change name TODO

                r_loss_list_S_RE.append(r_loss[1])
                r_loss_list_F_RE.append(r_loss[2])
                r_loss_list_BCE.append(r_loss[3])



                if step % 10 == 0:

                    print('epoch: {}, step {}, [Discriminator :: d_loss: {}], [ Reconstructor :: S_RE loss, F_RE loss, BCE loss: {}, {}, {}]'.format(epoch, step, d_loss, r_loss[1], r_loss[2], r_loss[3]))






            if epoch % save_interval == 0 or epoch == epochs-1:

                save_string = self.train_par.get_TR_path(epoch)
                self.T_R.save_weights(save_string)

                save_string = self.train_par.get_FR_path(epoch)
                self.F_R.save_weights(save_string)

                save_string = self.train_par.get_D_path(epoch)

                self.D.save_weights(save_string)

                print('saving images')
                np.random.seed(0)
                test_idxs = np.random.choice(len(thermal_frame), 8, replace = False)
                #Saving thermal image
                test_ims =thermal_frame[test_idxs]
                test_masks = thermal_mask[test_idxs]

                self.plot_images_3D(save2file=True, step=epoch, test_window = test_ims,test_masks=test_masks,d_type='thermal')


                #Saving flow image
                test_ims =flow_frame[test_idxs]
                test_masks =flow_mask[test_idxs]

                self.plot_images_3D(save2file=True, step=epoch, test_window = test_ims,test_masks=test_masks,d_type='flow')


                #saving loss values
                np.save(D_loss_root + '/epoch-{}.npy'.format(epoch), np.array(d_loss_list))
                np.save(R_loss_root + '/S_epoch-{}.npy'.format(epoch), np.array(r_loss_list_S_RE))
                np.save(R_loss_root + '/F_epoch-{}.npy'.format(epoch), np.array(r_loss_list_F_RE))
                np.save(R_loss_root + '/BCE_epoch-{}.npy'.format(epoch), np.array(r_loss_list_BCE))

                #D plots
                save_multiple_graph(x_list=[r_loss_list_BCE,d_loss_list],labels=['R_BCE','D_loss'],x_label='Batches',y_label='Losses',title='Loss Plot',path=loss_root+'/D_log_losses.png',log_plot=True)
                #AE RE plots
                total_RE=[self.train_par.thermal_lambda*r_loss_list_S_RE[i]+ self.train_par.flow_lambda*r_loss_list_F_RE[i] for i in range(len(r_loss_list_S_RE))]
                save_multiple_graph(x_list=[r_loss_list_S_RE,r_loss_list_F_RE,total_RE],labels=['S_RE','F_RE','Total_RE'],x_label='Batches',y_label='Losses',title='Loss Plot',path=loss_root+'/R_log_losses.png',log_plot=True)


    def plot_images_3D(self, save2file=False,  samples=16, step=0, test_window = None,test_masks=None,d_type=None):
        '''
            Visualization of input and reconstrcuted sequence. Save or save 4th frame of the input and output windows.
        '''
        test_ims = test_window[:,4,:,:,:]
        masks=test_masks[:,4,:,:,:]
        img_root = self.save_root + "/"+d_type+"_images/"
        if d_type=='thermal':
            channels=self.train_par.thermal_channels
            rec_images = self.T_R.predict([test_window,test_masks])

        elif d_type=='flow':
            channels=self.train_par.flow_channels
            rec_images = self.F_R.predict([test_window,test_masks])


        if not os.path.isdir(img_root):
            os.makedirs(img_root)

        filename = img_root + "/img_{}.png".format(step)

        rec_images = rec_images[:,4,:,:,:]
        masked_rec_images=rec_images*masks
        masks=masks-1
        masked_rec_images=masked_rec_images+masks



        if d_type=='flow' and channels==3:
            rec_images=rec_images[:,:,:,2:3]
            masked_rec_images=masked_rec_images[:,:,:,2:3]
            test_ims=test_ims[:,:,:,2:3]
            channels=1


        plt.figure(figsize=(10,15))

        for i in range(rec_images.shape[0]):
            plt.subplot(6, 4, i+1)
            image = rec_images[i, :, :, :]
            if channels==3:
                image = np.reshape(image, [ self.train_par.height, self.train_par.width, self.train_par.channels])
            else:
                image = np.reshape(image, [ self.train_par.height, self.train_par.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.tight_layout()

        for i in range(masked_rec_images.shape[0]):
            i+=8
            plt.subplot(6, 4, i+1)
            image = masked_rec_images[i-8, :, :, :]
            if channels==3:
                image = np.reshape(image, [ self.train_par.height, self.train_par.width, self.train_par.channels])
            else:
                image = np.reshape(image, [ self.train_par.height, self.train_par.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.tight_layout()

        for i in range(test_ims.shape[0]):
            i+=16
#             print(i)
            plt.subplot(6, 4, i+1)
            image = test_ims[i-16, :, :, :]
            if channels==3:
                image = np.reshape(image, [ self.train_par.height, self.train_par.width, self.train_par.channels])
            else:
                image = np.reshape(image, [ self.train_par.height, self.train_par.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
#         plt.subplots_adjust(wspace=0,hspace=0)

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

    def test(self, test_videos, score_type = 'R', epochs = None,plot=False,tolerance_limit=8):

        '''
        Gets AUC ROC/PR for all videos using RE.

        choose score type from ['T_R_S', 'F_R','R']
        T_R_S- Anomaly scores based on thermal frames reconstrcution error
        F_R- Anomaly scores based on flow frames reconstrcution error
        R- Anomaly scores based on weighted sum of thermal anf fow frames window reconstrcution error
        '''

        dset, img_width, img_height, win_length  = self.train_par.dset,  self.train_par.width, self.train_par.height, self.train_par.win_length
        self.save_root = self.train_par.get_root_path() + '/testing/epochs_{}'.format(epochs)
        stride = self.stride

        model_name = self.train_par.create_model_name()
        model_name += '_{}'.format(score_type)

        print(model_name)

        aucs = []
        std_total = []
        mean_total = []
        labels_total_l = []
        i = 0 #vid index TODO rename


        num_vids = len(test_videos)
        print('num of test vids', num_vids)

        ROC_mat = np.zeros((num_vids,2*tolerance_limit+2)) # 35 is num_vids, 20 scores-Xstd,Xmean,tols std..,tols mean..
        PR_mat = np.zeros((num_vids,2*tolerance_limit+2))
        print('score_type', score_type)



        for vid_name in  test_videos.keys():

            print("--------------------------")
            print("--------------------------")
            print("Processing ",vid_name)
            print("--------------------------")
            print("--------------------------")

            # vid_total, labels_total = restore_Fall_vid(data_dict, Fall_name, NFF_name)
            vid_thermal_list=test_videos[vid_name]['ROI_FRAME']
            vid_flow_list=test_videos[vid_name]['FLOW']
            vid_thermal_mask_list=test_videos[vid_name]['MASK']
            labels_total_list=test_videos[vid_name]['LABELS']
            frame_numbers_list=test_videos[vid_name]['NUMBER']
            start,end=test_videos[vid_name]['START_END']
            display_name = vid_name

            frame_numbers = np.concatenate(frame_numbers_list)
            test_labels = np.concatenate(labels_total_list)

            print("Number of frames",len(frame_numbers))
            print("Number of Labels",len(test_labels))
            #creating windows of thermal frame
            thermal_data_list = [vid.reshape(len(vid), img_width, img_height, self.train_par.thermal_channels) for vid in vid_thermal_list]
            thermal_data_windowed_list = [create_windowed_arr(test_data, stride, win_length) for test_data in thermal_data_list]#create_windowed_arr function in data_management.py
            # creating windows of mask data
            thermal_mask_list = [vid.reshape(len(vid), img_width, img_height, self.train_par.thermal_channels) for vid in vid_thermal_mask_list]

            thermal_mask_windowed_list = [create_windowed_arr(test_data, stride, win_length).astype('int8') for test_data in thermal_mask_list]
            num_sub_videos=len(thermal_data_windowed_list)
            #Create windows of length win_length-1
            flow_data_list = [vid.reshape(len(vid), img_width, img_height, self.train_par.flow_channels) for vid in vid_flow_list]
            flow_data_windowed_list = [create_windowed_arr(test_data, stride, win_length-1) for test_data in flow_data_list]#create_windowed_arr function in data_management.py

            # creating flow mask data
            flow_mask_windowed_list=[create_diff_mask(mask_windows) for mask_windows in thermal_mask_windowed_list]
            flow_mask_windowed_list=[np.concatenate((mask_windowed,mask_windowed,mask_windowed),axis=-1) for mask_windowed in flow_mask_windowed_list]
            # Masking the flow data
            flow_data_masked_windowed_list=[(flow_data_windowed_list[i]*flow_mask_windowed_list[i])+flow_mask_windowed_list[i]-1 for i in range(len(flow_data_windowed_list))]

            num_sub_videos=len(flow_data_masked_windowed_list)
            print("Number of sub flow videos,",num_sub_videos)
            print("Number of sub diff mask videos,",len(flow_mask_windowed_list))

            if score_type == 'F_R':
                in_mean_RE=[]
                in_std_RE=[]


                for index in range(num_sub_videos):
                    test_data_masked_windowed=flow_data_masked_windowed_list[index]
                    test_mask_windowed=flow_mask_windowed_list[index]

                    RE_dict, recons_seq = self.get_F_RE_all_agg(test_data_masked_windowed,test_mask_windowed) #Return dict with value for each score style
                    in_mean_RE.append(RE_dict['in_mean'])
                    in_std_RE.append(RE_dict['in_std'])


                in_mean_RE=np.concatenate(in_mean_RE)
                in_std_RE=np.concatenate(in_std_RE)

                final_in_mean = in_mean_RE
                final_in_std = in_std_RE
            elif score_type == 'R':
                #Thermal spatial RE
                RE_dict, recons_seq = self.get_T_S_RE_all_agg(np.concatenate(thermal_data_windowed_list),thermal_masks=np.concatenate(thermal_mask_windowed_list)) #Return dict with value for each score style
                in_mean_T_S_RE = RE_dict['in_mean']
                in_std_T_S_RE = RE_dict['in_std']

                #Flow RE
                RE_dict, recons_seq = self.get_F_RE_all_agg(np.concatenate(flow_data_masked_windowed_list),flow_masks=np.concatenate(flow_mask_windowed_list)) #Return dict with value for each score style
                in_mean_F_RE = RE_dict['in_mean']
                in_std_F_RE = RE_dict['in_std']
                final_in_mean =  self.train_par.thermal_lambda*in_mean_T_S_RE+self.train_par.flow_lambda*in_mean_F_RE
                final_in_std =  self.train_par.thermal_lambda*in_std_T_S_RE+self.train_par.flow_lambda*in_std_F_RE


            elif score_type == 'T_R_S':
                in_mean_RE=[]
                in_std_RE=[]
                x_std_RE=[]
                x_mean_RE=[]

                for index in range(num_sub_videos):
                    test_data_masked_windowed=thermal_data_windowed_list[index]
                    test_mask_windowed=thermal_mask_windowed_list[index]

                    RE_dict, recons_seq = self.get_T_S_RE_all_agg(test_data_masked_windowed,test_mask_windowed) #Return dict with value for each score style
                    in_mean_RE.append(RE_dict['in_mean'])
                    in_std_RE.append(RE_dict['in_std'])

                    x_std_RE.append(RE_dict['x_std'])
                    x_mean_RE.append(RE_dict['x_mean'])

                in_mean_RE=np.concatenate(in_mean_RE)
                in_std_RE=np.concatenate(in_std_RE)
                x_std_RE=np.concatenate(x_std_RE)
                x_mean_RE=np.concatenate(x_mean_RE)


                final_in_mean = in_mean_RE
                final_in_std = in_std_RE


            #Finding AUC
            #----------------------
            #frame based scores only for thermal frames
            if score_type == 'T_R_S':
                print("Frame based scores")
                auc_x_std, conf_mat, g_mean, ap_x_std = get_output(labels = test_labels,\
                    predictions = x_std_RE, data_option = 'NA', to_plot = False)
                auc_x_mean, conf_mat, g_mean, ap_x_mean = get_output(labels = test_labels,\
                    predictions = x_mean_RE, data_option = 'NA', to_plot = False)

                ROC_mat[i,0] = auc_x_std
                ROC_mat[i,1] = auc_x_mean

                PR_mat[i,0] = ap_x_std
                PR_mat[i,1] = ap_x_mean

            #window based scores
            # print('final_in_mean.shape', final_in_mean.shape, 'final_in_std.shape', final_in_std.shape)
            tol_mat, tol_keys = gather_auc_avg_per_tol(final_in_mean, final_in_std, labels_list = labels_total_list, win_len = win_length,tolerance_limit=tolerance_limit)
            AUROC_tol = tol_mat[0]
            AUPR_tol = tol_mat[1]
            num_scores_tol = tol_mat.shape[1]

            for k in range(num_scores_tol):
                j = k+2 #start at 2, first two were for X_std and X_mean
                ROC_mat[i,j] = AUROC_tol[k]
                PR_mat[i,j] = AUPR_tol[k]


            i += 1

            if plot == True and score_type == 'T_R_S':
                plt.plot(frame_numbers,x_std_RE, label='RE_std',linestyle='--', marker='.')
                plt.plot(frame_numbers,x_mean_RE, label='RE_mean',linestyle='--', marker='.')
                # plt.xticks([i+1 for i in range(max(frame_numbers))])
                plt.xlim(1,max(frame_numbers))
                # plt.ylim(0,1)
                plt.legend()
                plt.axvspan(start,end, alpha = 0.5)

                plot_save_p = self.save_root + '/T_R_S_scores_plots/'
                if not os.path.isdir(plot_save_p):
                    os.makedirs(plot_save_p)
                plt.savefig(plot_save_p + '{}.jpg'.format(vid_name))
                plt.close()


        #    break
        AUROC_avg = np.mean(ROC_mat, axis = 0)
        AUROC_std = np.std(ROC_mat, axis = 0)
        AUROC_avg_std = join_mean_std(AUROC_avg, AUROC_std)
        # print(AUROC_std)
        AUPR_avg = np.mean(PR_mat, axis = 0)
        AUPR_std = np.std(PR_mat, axis = 0)

        AUPR_avg_std = join_mean_std(AUPR_avg, AUPR_std)
        total = np.vstack((AUROC_avg_std, AUPR_avg_std))
        total_no_std = np.vstack((AUROC_avg, AUPR_avg))


        df = pd.DataFrame(data = total, index = ['AUROC','AUPR'], columns = ['X-STD','X-Mean'] + tol_keys)
        df_no_std = pd.DataFrame(data = total_no_std, index = ['AUROC','AUPR'], columns = ['X-STD','X-Mean'] + tol_keys)


        print(df)
        print(df_no_std)



        if not os.path.isdir(self.save_root):
            os.makedirs(self.save_root)

        save_path = self.save_root + '/AUC_{}.csv'.format(score_type)
        save_path_no_std = self.save_root + '/AUC_{}_no_std.csv'.format(score_type)

        print(save_path)
        df.to_csv(save_path)
        df_no_std.to_csv(save_path_no_std)
