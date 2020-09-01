from .util import agg_window,create_windowed_arr,save_multiple_graph,get_output,gather_auc_avg_per_tol,join_mean_std
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
class Params(object):
    """
    Parameters class to handlde parameters for the ROI based models
    """
    def __init__(self,width=64, height=64,win_length=8,channels=1,dset='Thermal_track',d_type='frame',R_name=None,batch_size=32,break_win=10,hp_name='exp1'):
        self.width=width
        self.height=height
        self.win_length=win_length
        self.channels=channels
        self.dset=dset
        self.d_type=d_type
        self.batch_size = batch_size
        self.R_name=R_name
        self.gap=break_win
        self.hp_name=hp_name

    def create_hp_name(self):
        return  self.hp_name
    def get_model_type(self):
        return self.R_name
    def get_model_dir(self):
        return self.get_root_path()+'/models'
    def get_R_path(self,epochs_trained):
        return self.get_root_path()+'/models/R_weights_epoch-{}.h5'.format(epochs_trained)
    def get_root_path(self):
        return './{}/{}/{}/{}'.format(self.dset,self.d_type,self.get_model_type(),self.create_hp_name())



class ConvLSTMAE_trainer(object):
    '''
        Class used to train ConvLSTM model on tracked frames
    '''
    def __init__(self, train_par=None,stride=1):

        self.train_par=train_par
        self.stride=stride
        self.SHAPE = (self.train_par.width, self.train_par.height, train_par.channels)

    def initialize_model(self,Reconstructor):
        self.R = Reconstructor
        print('Reconstructor')
        print(self.R.summary())


    def create_windowed_data(self, videos_dic,stride=1,data_key='FRAME'): #TODO should be called set train data
        total = []
        img_width, img_height,win_length=self.train_par.width,self.train_par.height,self.train_par.win_length

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




    def get_MSE_all_agg(self, test_data):

        img_width, img_height, win_length, channels,model = self.train_par.width, self.train_par.height ,self.train_par.win_length, self.train_par.channels, self.R

        recons_seq = model.predict([test_data]) #(samples-win_length+1, win_length, wd,ht,1)

        recons_seq = recons_seq.reshape(len(recons_seq),win_length, img_height*img_width*channels)#(samples-win_length+1, 5, wd*ht)
        test_data = test_data.reshape(len(test_data),win_length, img_height*img_width*channels)#(samples-win_length+1, 5, wd*ht)

        RE = np.mean(np.power(test_data-recons_seq, 2), axis = 2)# (samples-win_length+1,win_length)



        RE_dict = {}

        agg_type_list = ['x_mean','x_std','in_std', 'in_mean']

        for agg_type in agg_type_list:

            RE_dict[agg_type] = agg_window(RE, agg_type)

        return RE_dict, recons_seq

    def plot_images_3D(self, save2file=False,  samples=16, step=0, test_window = None):

        img_root = self.save_root + "/thermal_images/"
        channels=1


        if not os.path.isdir(img_root):
            os.makedirs(img_root)

        filename = img_root + "/img_{}.png".format(step)

        test_ims = test_window[:,4,:,:,:]
        rec_images = self.R.predict([test_window])
        rec_images = rec_images[:,4,:,:,:]

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

    def train(self,  X_train_frame,epochs= 500, epochs_trained=0,save_interval = 10):
        """
        trains a sequential autoencoder on windowed data. That is, sequeneces of contigous frames
        are reconstucted.
        """
        print('Using save root:', self.train_par.get_root_path())
        self.save_root=self.train_par.get_root_path()
        batch_size=self.train_par.batch_size
        num_batches = int(X_train_frame.shape[0]/batch_size)
        print("Train frame dataset shape",X_train_frame.shape)
        print("Number of batches",num_batches)


        #model save dir
        if not os.path.isdir(self.train_par.get_model_dir()):
            os.makedirs(self.train_par.get_model_dir())

        r_loss_list_RE = [] #Reconstruction error
        loss_root = self.save_root + '/loss'
        #Creating loss directory
        if not os.path.isdir(loss_root):
            print("Creating loss directory ")
            os.makedirs(loss_root)
        print("Loss file status................")
        if os.path.isfile(loss_root + '/r_loss_RE_epoch-{}.npy'.format(epochs_trained)):
            print("RE Loss file found")
            r_loss_list_RE=list(np.load(loss_root + '/r_loss_RE_epoch-{}.npy'.format(epochs_trained)))

        for epoch in range(epochs_trained+1,epochs):
            ## train discriminator

            random_index =  np.random.randint(0, len(X_train_frame) - batch_size)

            permutated_indexes = np.random.permutation(X_train_frame.shape[0])

            for step in range(num_batches):
                batch_indeces = permutated_indexes[step*batch_size:(step+1)*batch_size]

                legit_images = X_train_frame[batch_indeces]
                r_loss=self.R.train_on_batch(legit_images,legit_images)
                r_loss_list_RE.append(r_loss)
                if step % 10 == 0:
                    print('epoch: {}, step {},[ Reconstructor :: RE loss: {}]'.format(epoch, step,  r_loss))

            if epoch % save_interval == 0 or epoch == epochs-1:
                save_string = self.train_par.get_R_path(epoch)
                self.R.save_weights(save_string)
                print('saving images')
                np.random.seed(0)
                test_idxs = np.random.choice(len(X_train_frame), 8, replace = False)

                test_ims = X_train_frame[test_idxs]
                self.plot_images_3D(save2file=True, step=epoch, test_window = test_ims)

                np.save(loss_root +  '/r_loss_RE_epoch-{}.npy'.format(epoch), np.array(r_loss_list_RE))
                save_multiple_graph(x_list=[r_loss_list_RE],labels=['R_RE'],x_label='Batches',y_label='Losses',title='Loss Plot',path=loss_root+'/log_loss.png',log_plot=True)



    def test(self, test_videos, score_type = 'R', epochs = None,tolerance_limit=8,plot=False):

        '''
        Gets AUC ROC/PR for all videos using RE.

        choose score type 'R'
        '''
        dset, img_width, img_height, win_length  = self.train_par.dset,  self.train_par.width, self.train_par.height, self.train_par.win_length
        self.save_root = self.train_par.get_root_path() + '/testing/epochs_{}'.format(epochs)
        stride = self.stride
        # win_length = self.win_length




        aucs = []
        std_total = []
        mean_total = []
        labels_total_l = []
        i = 0 #vid index TODO rename


        num_vids = len(test_videos)
        print('num of test vids', num_vids)
        #When D score is used no mean, no std and frames anomaly score is used

        print('score_type', score_type)
        ROC_mat = np.zeros((num_vids,2*tolerance_limit+2)) # 35 is num_vids, 20 scores-Xstd,Xmean,tols std..,tols mean..
        PR_mat = np.zeros((num_vids,2*tolerance_limit+2))

        for vid_name in  test_videos.keys():
            print("--------------------------")
            print("Processing ",vid_name)
            print("--------------------------")

            # vid_total, labels_total = restore_Fall_vid(data_dict, Fall_name, NFF_name)
            vid_total_list=test_videos[vid_name]['FRAME']
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
            num_sub_videos=len(test_data_windowed_list)


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

            if score_type == 'R':
                auc_x_std, conf_mat, g_mean, ap_x_std = get_output(labels = test_labels,\
                    predictions = x_std_RE, data_option = 'NA', to_plot = False)
                auc_x_mean, conf_mat, g_mean, ap_x_mean = get_output(labels = test_labels,\
                    predictions = x_mean_RE, data_option = 'NA', to_plot = False)

                ROC_mat[i,0] = auc_x_std
                ROC_mat[i,1] = auc_x_mean

                PR_mat[i,0] = ap_x_std
                PR_mat[i,1] = ap_x_mean
                print("------------")
                print("Mean scores")
                print("------------")
                print("AUC ROC ")
                print(auc_x_mean)
                print("AUC PR ")
                print(ap_x_mean)
                print("------------")
                print("STD scores")
                print("------------")
                print("AUC ROC ")
                print(auc_x_std)
                print("AUC PR ")
                print(ap_x_std)


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
            if plot == True and score_type == 'R':
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
