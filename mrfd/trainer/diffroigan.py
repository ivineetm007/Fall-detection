from .losses import ROI_mean_squared_error_loss,ROI_diff_mse_joint_loss,ROI_diff_temporal_loss,wind_ROI_mean_squared_error,wind_ROI_diff_temporal_loss
import matplotlib.pyplot as plt
from .util import agg_window,create_windowed_arr,save_multiple_graph,get_output,gather_auc_avg_per_tol,join_mean_std,create_diff_mask
from keras.models import Sequential, Model
import pandas as pd
import numpy as np
import os

class Params(object):
    """
        Parameters class to handlde parameters for the dif ROI based models
    """
    def __init__(self,width=64, height=64,win_length=8,channels=1,dset='Thermal',d_type='frame',lambda_S=1.0,lambda_T=1.0,regularizer_list=None,R_name=None,D_name=None,batch_size=32,break_win=10):
        self.width=width
        self.height=height
        self.win_length=win_length
        self.channels=channels
        self.dset=dset
        self.d_type=d_type
        self.batch_size = batch_size
        self.lambda_S=lambda_S
        self.lambda_T=lambda_T
        self.regularizer_list=regularizer_list
        self.R_name=R_name
        self.D_name=D_name
        self.gap=break_win

    def create_model_name(self):
        return  self.get_model_type()+ '_S{}_T{}'.format(str(self.lambda_S),str(self.lambda_T))

    def create_hp_name(self):
        return  'lambda_S{}_T{}'.format(str(self.lambda_S),str(self.lambda_T))

    def get_model_dir(self):
        return self.get_root_path()+'/models'
    def get_model_type(self):
        R_name=self.R_name
        D_name=self.D_name

        return R_name+'_'+D_name
    def get_R_path(self,epochs_trained):
        return self.get_root_path()+'/models/GAN_R_weights_epoch-{}.h5'.format(epochs_trained)
    def get_D_path(self,epochs_trained):
        return self.get_root_path()+'/models/GAN_D_weights_epoch-{}.h5'.format(epochs_trained)
    def get_root_path(self):
        return './{}/{}/{}/{}'.format(self.dset,self.d_type,self.get_model_type(),self.create_hp_name())




class Diff_ROI_3DCAE_GAN3D(object):
    '''
        Class used to train and test diff ROI based adversarial learning
    '''
    def __init__(self, train_par=None,stride=1):

        self.train_par=train_par
        self.stride=stride
        self.SHAPE = (self.train_par.width, self.train_par.height, train_par.channels)

    def initialize_model(self,Reconstructor , Discriminator ):
        print("Compiling GAN model.")
        self.R = Reconstructor
        self.D = Discriminator
        print('discriminator')
        print(self.D.summary())

        print('Reconstructor')
        print(self.R.summary())
        self.OPTIMIZER = 'adam'



        self.stacked_R_D = self.stacked_R_D()

        R_metrics={'decoded':[ROI_mean_squared_error_loss(self.R.input[1]),ROI_diff_temporal_loss(self.R.input[1],self.R.input[2])]}

        self.stacked_R_D.compile(loss={'D': 'binary_crossentropy', 'decoded': ROI_diff_mse_joint_loss(self.R.input[1],self.R.input[2],self.train_par.lambda_S,self.train_par.lambda_T)},\
         optimizer=self.OPTIMIZER, metrics=R_metrics)


    def stacked_R_D(self):
        '''
        Used for training Reconstructor. Dicscriminator is freezed.
        '''

        self.D.trainable = False

        #print(self.R.output)
        model = Model(inputs = self.R.input, outputs = [self.R.output, self.D(self.R.output)],name="R_D")
        print('stacked')
        print(model.summary())

        return model


    def create_windowed_data(self, videos_dic,stride=1,data_key='FRAME'):
        total = []
        img_width, img_height,channels,win_length=self.train_par.width,self.train_par.height,self.train_par.channels,self.train_par.win_length

        # print('total.shape', 'num_windowed', 'output_shape', total.shape, num_windowed, output_shape)
        for vid_name in videos_dic.keys():
            # print('Video Name', vid_name)
            vid_windowed_list=[]
            sub_vid_list=videos_dic[vid_name][data_key]
            for sub_vid in sub_vid_list:
                vid_windowed_list.append(create_windowed_arr(sub_vid, stride, win_length))
            # print("Number of sub videos: ",len(vid_windowed_list))
            vid_windowed=np.concatenate(vid_windowed_list)
            total.append(vid_windowed)
        total=np.concatenate(total)
        # print("Windowed data shape:")
        print(total.shape)
        return total




    def get_S_RE_all_agg(self, test_data, test_masks,test_diff_masks,type=['frame','window']):
        """
            Anomaly scores based on ROI MSE
        """
        img_width, img_height, win_length, channels,model = self.train_par.width, self.train_par.height ,self.train_par.win_length, self.train_par.channels, self.R

        recons_seq = model.predict([test_data,test_masks,test_diff_masks]) #(samples-win_length+1, win_length, wd,ht,1)
        # print(recons_seq.shape)

        RE=wind_ROI_mean_squared_error(test_masks,test_data,recons_seq,win_length, img_height,img_width,channels)
        # print('RE.shape', RE.shape)

        RE_dict = {}
        agg_type_list=[]
        if 'frame' in type:
            agg_type_list.append('x_std')
            agg_type_list.append('x_mean')
        if 'window' in type:
            agg_type_list.append('in_std')
            agg_type_list.append('in_mean')

        for agg_type in agg_type_list:

            RE_dict[agg_type] = agg_window(RE, agg_type)

        return RE_dict, recons_seq

    def get_T_RE_all_agg(self, test_data, test_masks,test_diff_masks):

        img_width, img_height, win_length, channels,model = self.train_par.width, self.train_par.height ,self.train_par.win_length, self.train_par.channels, self.R

        recons_seq = model.predict([test_data,test_masks,test_diff_masks]) #(samples-win_length+1, win_length, wd,ht,1)
        # print(recons_seq.shape)

        RE=wind_ROI_diff_temporal_loss(test_masks,test_diff_masks,test_data,recons_seq,win_length, img_height,img_width,channels)
        # print('RE.shape', RE.shape)

        RE_dict = {}

        agg_type_list = ['in_std', 'in_mean']

        for agg_type in agg_type_list:

            RE_dict[agg_type] = agg_window(RE, agg_type)

        return RE_dict, recons_seq

    def train(self, X_train_frame, X_train_mask,X_train_diff_mask,epochs= 500,epochs_trained=0, save_interval = 10):
        '''
            Train the adversarial framework
            X_train_frame- window of frames
            X_train_mask- mask of windows
        '''
        print('using save root:', self.train_par.get_root_path())
        self.save_root=self.train_par.get_root_path()
        batch_size=self.train_par.batch_size
        print('self.stacked_R_D.metrics_names', self.stacked_R_D.metrics_names)
        print('self.D.metrics_names', self.D.metrics_names)
        num_batches = int(X_train_frame.shape[0]/batch_size)
        print("Train frame dataset shape",X_train_frame.shape)
        print("Number of batches",num_batches)
        #model save dir
        if not os.path.isdir(self.train_par.get_model_dir()):
            os.makedirs(self.train_par.get_model_dir())

        #Losses stats
        d_loss_list = [] #m for mean
        r_loss_list_S_RE = [] #Spatial Reconstruction error
        r_loss_list_T_RE = [] #Temporal Reconstruction error
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
            print("RE Spatial Loss file found")
            r_loss_list_S_RE=list(np.load(R_loss_root + '/S_epoch-{}.npy'.format(epochs_trained)))

        if os.path.isfile(R_loss_root + '/T_epoch-{}.npy'.format(epochs_trained)):
            print("RE Temporal Loss file found")
            r_loss_list_T_RE=list(np.load(R_loss_root + '/T_epoch-{}.npy'.format(epochs_trained)))

        if os.path.isfile(R_loss_root + '/BCE_epoch-{}.npy'.format(epochs_trained)):
            print("RE BCE Loss file found")
            r_loss_list_BCE=list(np.load(R_loss_root + '/BCE_epoch-{}.npy'.format(epochs_trained)))

        #Training loop
        for epoch in range(epochs_trained+1,epochs):
            ## train discriminator

            random_index =  np.random.randint(0, len(X_train_frame) - batch_size)

            permutated_indexes = np.random.permutation(X_train_frame.shape[0])

            for step in range(num_batches):

                #Random permutation selection
                batch_indeces = permutated_indexes[step*batch_size:(step+1)*batch_size]

                legit_images = X_train_frame[batch_indeces]
                masks=X_train_mask[batch_indeces]
                diff_masks=X_train_diff_mask[batch_indeces]

                #R Input
                recons_images = self.R.predict([legit_images,masks,diff_masks])

                x_combined_batch_size = np.concatenate((legit_images,recons_images))
                y_combined_batch_size = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))

                # First train Discriminator
                d_loss = self.D.train_on_batch(x_combined_batch_size, y_combined_batch_size)

                d_loss_list.append(d_loss)

                # Train Reconstructor
                y_mislabled = np.ones((batch_size, 1))
                r_loss = self.stacked_R_D.train_on_batch([legit_images,masks,diff_masks], {'decoded':legit_images,'D':y_mislabled})

                #Loss order -- ['loss','decoded_loss','D_loss','decoded_mse_loss','decoded_diff_temporal_loss']
                r_loss_list_S_RE.append(r_loss[3])
                r_loss_list_T_RE.append(r_loss[4])
                r_loss_list_BCE.append(r_loss[2])



                if step % 10 == 0:
                    print('epoch: {}, step {}, [Discriminator :: d_loss: {}], [ Reconstructor :: RE loss, RE_S loss, RE_T loss, BCE loss,: {}, {}, {}, {}]'.format(epoch, step, d_loss, r_loss[1],r_loss[3],r_loss[4], r_loss[2]))






            if epoch % save_interval == 0 or epoch == epochs-1:

                save_string = self.train_par.get_R_path(epoch)
                self.R.save_weights(save_string)

                save_string = self.train_par.get_D_path(epoch)
                self.D.save_weights(save_string)
                print('saving images')
                np.random.seed(0)
                test_idxs = np.random.choice(len(X_train_frame), 8, replace = False)

                test_ims = X_train_frame[test_idxs]
                test_masks = X_train_mask[test_idxs]
                diff_masks=X_train_diff_mask[test_idxs]

                print(test_ims.shape)
                self.plot_images_3D(save2file=True, step=epoch, test_window = test_ims,test_masks=test_masks,diff_masks=diff_masks)

                #saving loss values
                np.save(D_loss_root + '/epoch-{}.npy'.format(epoch), np.array(d_loss_list))
                np.save(R_loss_root + '/S_epoch-{}.npy'.format(epoch), np.array(r_loss_list_S_RE))
                np.save(R_loss_root + '/T_epoch-{}.npy'.format(epoch), np.array(r_loss_list_T_RE))
                np.save(R_loss_root + '/BCE_epoch-{}.npy'.format(epoch), np.array(r_loss_list_BCE))

                save_multiple_graph(x_list=[r_loss_list_BCE,d_loss_list],labels=['R_BCE','D_loss'],x_label='Batches',y_label='Losses',title='Loss Plot',path=loss_root+'/D_log_losses.png',log_plot=True)
                r_loss_list_RE= [ self.train_par.lambda_S*r_loss_list_S_RE[i] + self.train_par.lambda_T*r_loss_list_T_RE[i] for i in range(len(r_loss_list_S_RE))]

                save_multiple_graph(x_list=[r_loss_list_S_RE,r_loss_list_T_RE,r_loss_list_RE],labels=['S_RE','T_RE','RE'],x_label='Batches',y_label='Losses',title='Loss Plot',path=loss_root+'/R_log_loss.png',log_plot=True)

    def plot_images_3D(self, save2file=False,  samples=16, step=0, test_window = None,test_masks=None,diff_masks=None):
        '''
            Visualization of input and reconstrcuted sequence. Save or save 4th frame of the input and output windows.
        '''
        test_ims = test_window[:,4,:,:,:]
        masks=test_masks[:,4,:,:,:]
        img_root = self.save_root + "/thermal_images/"
        if not os.path.isdir(img_root):
            os.makedirs(img_root)

        filename = img_root + "/img_{}.png".format(step)
        rec_images = self.R.predict([test_window,test_masks,diff_masks])
        rec_images = rec_images[:,4,:,:,:]
        masked_rec_images=rec_images*masks
        masks=masks-1
        masked_rec_images=masked_rec_images+masks

        plt.figure(figsize=(10,15))

        for i in range(rec_images.shape[0]):
            plt.subplot(6, 4, i+1)
            image = rec_images[i, :, :, :]
            if self.train_par.channels==3:
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
            if self.train_par.channels==3:
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
            if self.train_par.channels==3:
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

    def test(self, test_videos,animate = False, score_type = 'R', epochs = None,plot=False,tolerance_limit=8):

        '''
            Gets AUC ROC/PR for all videos using RE.

            choose score type 'R_S' for thermal image reconstruction scores and 'R_T' for thermal difference frames reconstruction scores
        '''
        dset, img_width, img_height, win_length  = self.train_par.dset,  self.train_par.width, self.train_par.height, self.train_par.win_length
        self.save_root = self.train_par.get_root_path() + '/testing/epochs_{}'.format(epochs)
        stride = self.stride
        # win_length = self.win_length


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
            vid_total_list=test_videos[vid_name]['ROI_FRAME']
            vid_mask_list=test_videos[vid_name]['MASK']
            labels_total_list=test_videos[vid_name]['LABELS']
            frame_numbers_list=test_videos[vid_name]['NUMBER']
            start,end=test_videos[vid_name]['START_END']
            display_name = vid_name

            frame_numbers = np.concatenate(frame_numbers_list)
            test_labels = np.concatenate(labels_total_list)
            print("Number of frames",len(frame_numbers))
            print("Number of Labels",len(test_labels))
            test_data_list = [vid.reshape(len(vid), img_width, img_height, self.train_par.channels) for vid in vid_total_list]
            test_data_windowed_list = [create_windowed_arr(test_data, stride, win_length) for test_data in test_data_list]
            # creating windows of mask data
            test_mask_list = [vid.reshape(len(vid), img_width, img_height, self.train_par.channels) for vid in vid_mask_list]
            test_mask_windowed_list = [create_windowed_arr(test_data, stride, win_length).astype('int8') for test_data in test_mask_list]
            num_sub_videos=len(test_data_windowed_list)

            # creating diff mask data
            test_diff_mask_windowed_list=[create_diff_mask(mask_windows) for mask_windows in test_mask_windowed_list]


            if score_type == 'R_S':
                in_mean_RE=[]
                in_std_RE=[]
                x_std_RE=[]
                x_mean_RE=[]
                for index in range(num_sub_videos):
                    test_data_windowed=test_data_windowed_list[index]
                    test_mask_windowed=test_mask_windowed_list[index]
                    test_diff_mask_windowed=test_diff_mask_windowed_list[index]

                    #Get metric value
                    RE_dict, recons_seq = self.get_S_RE_all_agg(test_data_windowed,test_mask_windowed,test_diff_mask_windowed) #Return dict with value for each score style
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

            elif score_type == 'R_T':
                in_mean_RE=[]
                in_std_RE=[]

                for index in range(num_sub_videos):
                    test_data_windowed=test_data_windowed_list[index]
                    test_mask_windowed=test_mask_windowed_list[index]
                    test_diff_mask_windowed=test_diff_mask_windowed_list[index]

                    #Get metric value
                    RE_dict, recons_seq = self.get_T_RE_all_agg(test_data_windowed,test_mask_windowed,test_diff_mask_windowed) #Return dict with value for each score style
                    in_mean_RE.append(RE_dict['in_mean'])
                    in_std_RE.append(RE_dict['in_std'])

                in_mean_RE=np.concatenate(in_mean_RE)
                in_std_RE=np.concatenate(in_std_RE)

                final_in_mean = in_mean_RE
                final_in_std = in_std_RE

            if score_type == 'R_S':
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
#                    break
            if plot == True and score_type == 'R_S':
                plt.plot(frame_numbers,x_std_RE, label='RE_std',linestyle='--', marker='.')
                plt.plot(frame_numbers,x_mean_RE, label='RE_mean',linestyle='--', marker='.')
                # plt.xticks([i+1 for i in range(max(frame_numbers))])
                plt.xlim(1,max(frame_numbers))
                # plt.ylim(0,1)
                plt.legend()
                plt.axvspan(start,end, alpha = 0.5)

                plot_save_p = self.save_root + '/scores_plots/'
                if not os.path.isdir(plot_save_p):
                    os.makedirs(plot_save_p)
                plt.savefig(plot_save_p + '{}.png'.format(vid_name))
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
