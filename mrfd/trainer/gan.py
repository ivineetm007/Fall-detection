from .losses import wind_mean_squared_error
from .util import agg_window,create_windowed_arr,save_multiple_graph,get_output,gather_auc_avg_per_tol,join_mean_std
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
import numpy as np
import pandas as pd
import os

class Params(object):
    """
    Parameters class to handlde parameters for the ROI based models
    """
    def __init__(self,width=64, height=64,win_length=8,channels=1,dset='Thermal_track',d_type='frame',lambda_=1.0,regularizer_list=None,R_name=None,D_name=None,batch_size=32,break_win=10):
        self.width=width
        self.height=height
        self.win_length=win_length
        self.channels=channels
        self.dset=dset
        self.d_type=d_type
        self.batch_size = batch_size
        self.lambda_=lambda_
        self.regularizer_list=regularizer_list
        self.R_name=R_name
        self.D_name=D_name
        self.gap=break_win

    def create_model_name(self):
        return  self.get_model_type()+ '_{}'.format(str(self.lambda_))
    def create_hp_name(self):
        return  'lambda_{}'.format(str(self.lambda_))
    def get_model_type(self):
        R_name=self.R_name
        D_name=self.D_name

        return R_name+'_'+D_name
    def get_model_dir(self):
        return self.get_root_path()+'/models'
    def get_R_path(self,epochs_trained):
        return self.get_root_path()+'/models/GAN_R_weights_epoch-{}.h5'.format(epochs_trained)
    def get_D_path(self,epochs_trained):
        return self.get_root_path()+'/models/GAN_D_weights_epoch-{}.h5'.format(epochs_trained)
    def get_root_path(self):
        return './{}/{}/{}/{}'.format(self.dset,self.d_type,self.get_model_type(),self.create_hp_name())


class CAE_GAN3D(object):
    '''
        Class used to train and test the base adversarial model
    '''
    def __init__(self, train_par=None,stride=1):

        self.train_par=train_par
        self.stride=stride
        self.SHAPE = (self.train_par.width, self.train_par.height, train_par.channels)

    def initialize_model(self,Reconstructor , Discriminator ):
        print("Compiling GAN model.")
        self.R = Reconstructor
        self.D = Discriminator
        print('Discriminator')
        print(self.D.summary())

        print('Reconstructor')
        print(self.R.summary())
        self.OPTIMIZER = 'adam'



        self.stacked_R_D = self.stacked_R_D()
        loss_weights = {'D':1.0, 'decoded':self.train_par.lambda_}

        self.stacked_R_D.compile(loss={'D': 'binary_crossentropy', 'decoded': 'mean_squared_error'},\
         optimizer=self.OPTIMIZER, loss_weights = loss_weights)



    def stacked_R_D(self):
        '''
        Used for training Reconstructor. Dicscriminator is freezed.
        '''

        self.D.trainable = False
        model = Model(inputs = self.R.input, outputs = [self.R.output, self.D(self.R.output)],name='stacked')
        print('stacked')
        print(model.summary())

        return model


    def create_windowed_data(self, videos_dic,stride=1,data_key='FRAME'):
        '''
            Create windows of frames
        '''
        total = []
        img_width, img_height,channels,win_length=self.train_par.width,self.train_par.height,self.train_par.channels,self.train_par.win_length
        if data_key=='FLOW':
            win_length=win_length-1
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
        # print(total.shape)
        return total


    def get_MSE_all_agg(self, test_data,type=['frame','window']):
        """
            Anomaly scores based on MSE
        """

        img_width, img_height, win_length, channels,model = self.train_par.width, self.train_par.height ,self.train_par.win_length, self.train_par.channels, self.R
        if self.train_par.d_type=='opticalFLow':
            win_length=win_length-1
        recons_seq = model.predict([test_data]) #(samples-win_length+1, win_length, wd,ht,1)
        print(recons_seq.shape)
        RE=wind_mean_squared_error(test_data,recons_seq,win_length, img_height,img_width,channels)
        print('RE.shape', RE.shape)

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

    def train(self, X_train_frame,epochs= 500,epochs_trained=0, save_interval = 10):
        '''
            Train the adversarial framework
            X_train_frame- window of frames
        '''
        print('Using save root:', self.train_par.get_root_path())
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
        d_loss_list = [] # Discriminator loss
        r_loss_list_RE = [] #Reconstruction error
        r_loss_list_BCE = [] #Binary cross entropy
        loss_root = self.save_root + '/loss'

        #Creating loss directory
        if not os.path.isdir(loss_root):
            print("Creating loss directory ")
            os.makedirs(loss_root)
        print("Loss file status................")
        if os.path.isfile(loss_root + '/d_loss_epoch-{}.npy'.format(epochs_trained)):
            print("D Loss file found")
            d_loss_list=list(np.load(loss_root + '/d_loss_epoch-{}.npy'.format(epochs_trained)))
        if os.path.isfile(loss_root + '/r_loss_RE_epoch-{}.npy'.format(epochs_trained)):
            print("RE Loss file found")
            r_loss_list_RE=list(np.load(loss_root + '/r_loss_RE_epoch-{}.npy'.format(epochs_trained)))
        if os.path.isfile(loss_root + '/r_loss_BCE_epoch-{}.npy'.format(epochs_trained)):
            print("BCE Loss file found")
            r_loss_list_BCE=list(np.load(loss_root + '/r_loss_BCE_epoch-{}.npy'.format(epochs_trained)))

        for epoch in range(epochs_trained+1,epochs):
            ## train discriminator

            random_index =  np.random.randint(0, len(X_train_frame) - batch_size)

            permutated_indexes = np.random.permutation(X_train_frame.shape[0])

            for step in range(num_batches):

                batch_indeces = permutated_indexes[step*batch_size:(step+1)*batch_size]

                legit_images = X_train_frame[batch_indeces]
                #R Input
                recons_images = self.R.predict([legit_images])

                x_combined_batch_size = np.concatenate((legit_images,recons_images))
                y_combined_batch_size = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))

                # First train Discriminator
                d_loss = self.D.train_on_batch(x_combined_batch_size, y_combined_batch_size)

                d_loss_list.append(d_loss)

                # Train Reconstructor
                y_mislabled = np.ones((batch_size, 1))
                r_loss = self.stacked_R_D.train_on_batch([legit_images], {'decoded':legit_images,'D':y_mislabled})

                r_loss_list_RE.append(r_loss[1])
                r_loss_list_BCE.append(r_loss[2])

                if step % 10 == 0:

                    print('epoch: {}, step {}, [Discriminator :: d_loss: {}], [ Reconstructor :: RE loss, BCE loss: {}, {}]'.format(epoch, step, d_loss, r_loss[1], r_loss[2]))






            if epoch % save_interval == 0 or epoch == epochs-1:

                save_string = self.train_par.get_R_path(epoch)
                self.R.save_weights(save_string)

                save_string = self.train_par.get_D_path(epoch)
                self.D.save_weights(save_string)
                print('saving images')
                np.random.seed(0)
                test_idxs = np.random.choice(len(X_train_frame), 8, replace = False)

                test_ims = X_train_frame[test_idxs]
                print(test_ims.shape)
                if self.train_par.d_type=='frame':
                    self.plot_images_3D(save2file=True, step=epoch, test_window = test_ims,d_type='thermal')
                elif self.train_par.d_type=='opticalFLow':
                    self.plot_images_3D(save2file=True, step=epoch, test_window = test_ims,d_type='flow')
                else:
                    print("Invalid data type in Params")


                np.save(loss_root + '/d_loss_epoch-{}.npy'.format(epoch), np.array(d_loss_list))
                np.save(loss_root +  '/r_loss_RE_epoch-{}.npy'.format(epoch), np.array(r_loss_list_RE))
                np.save(loss_root + '/r_loss_BCE_epoch-{}.npy'.format(epoch), np.array(r_loss_list_BCE))
                save_multiple_graph(x_list=[r_loss_list_RE,r_loss_list_BCE,d_loss_list],labels=['R_RE','R_BCE','D_loss'],x_label='Batches',y_label='Losses',title='Loss Plot',path=loss_root+'/log_loss.png',log_plot=True)
                save_multiple_graph(x_list=[d_loss_list,r_loss_list_RE,r_loss_list_BCE],labels=['D_loss','R_RE','R_BCE'],x_label='Batches',y_label='Losses',title='Loss Plot',path=loss_root+'/loss.png',log_plot=False)

    def plot_images_3D(self, save2file=False,  samples=16, step=0, test_window = None,d_type=None):
        '''
            Visualization of input and reconstrcuted sequence. Save or save 4th frame of the input and output windows.
        '''
        test_ims = test_window[:,4,:,:,:]
        img_root = self.save_root + "/"+d_type+"_images/"
        channels=self.train_par.channels
        rec_images = self.R.predict([test_window])

        if not os.path.isdir(img_root):
            os.makedirs(img_root)

        filename = img_root + "/img_{}.png".format(step)

        rec_images = rec_images[:,4,:,:,:]

        if d_type=='flow' and channels==3:
            rec_images=rec_images[:,:,:,2:3]
            test_ims=test_ims[:,:,:,2:3]
            channels=1

        plt.figure(figsize=(10,10))
        for i in range(rec_images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = rec_images[i, :, :, :]
            if channels==3:
                image = np.reshape(image, [ self.train_par.height, self.train_par.width, channels])
            else:
                image = np.reshape(image, [ self.train_par.height, self.train_par.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.tight_layout()

        for i in range(test_ims.shape[0]):
            i+=8
            plt.subplot(4, 4, i+1)
            image = test_ims[i-8, :, :, :]
            if channels==3:
                image = np.reshape(image, [ self.train_par.height, self.train_par.width, channels])
            else:
                image = np.reshape(image, [ self.train_par.height, self.train_par.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.tight_layout()


        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()




    def test(self, test_videos, score_type = 'R', epochs = None,plot=False,tolerance_limit=8):

        '''
        Gets AUC ROC/PR for all videos using RE.

        choose score type 'R'
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
            vid_total_list=None
            frame_numbers_list=None
            start,end=None,None
            labels_total_list=test_videos[vid_name]['LABELS']

            display_name = vid_name


            test_labels = np.concatenate(labels_total_list)

            print("Number of Labels",len(test_labels))

            if self.train_par.d_type=='frame':
                vid_total_list=test_videos[vid_name]['FRAME']
                frame_numbers_list=test_videos[vid_name]['NUMBER']
                frame_numbers = np.concatenate(frame_numbers_list)
                print("Number of frames",len(frame_numbers))
                start,end=test_videos[vid_name]['START_END']

                test_data_list = [vid.reshape(len(vid), img_width, img_height, self.train_par.channels) for vid in vid_total_list]
                test_data_windowed_list = [create_windowed_arr(test_data, stride, win_length) for test_data in test_data_list]#create_windowed_arr function in data_management.py

            elif self.train_par.d_type=='opticalFLow':
                vid_total_list=test_videos[vid_name]['FLOW']
                #Create windows of length win_length-1
                test_data_list = [vid.reshape(len(vid), img_width, img_height, self.train_par.channels) for vid in vid_total_list]
                test_data_windowed_list = [create_windowed_arr(test_data, stride, win_length-1) for test_data in test_data_list]#create_windowed_arr function in data_management.py

            else:
                print("Invlaid d_type in the params")
            num_sub_videos=len(test_data_windowed_list)
            print("Number of sub flow videos,",num_sub_videos)




            if score_type == 'R':
                in_mean_RE=[]
                in_std_RE=[]
                x_std_RE=[]
                x_mean_RE=[]
                for index in range(num_sub_videos):
                    test_data_windowed=test_data_windowed_list[index]


                    RE_dict, recons_seq = self.get_MSE_all_agg(test_data_windowed) #Return dict with value for each score style
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

            #frame based scores only for thermal frames
            if self.train_par.d_type=='frame':
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

            if plot == True and score_type == 'R' and self.train_par.d_type=='frame':
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
