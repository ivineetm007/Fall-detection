This folder stores all the model weights, images and losses at every 10th epoch. It also contains the AUC of the PR and ROC curve along with the graphs for frame based anomaly scores for each fall video.

Directory structure used to store files for training and testing of each network-
```
Thermal_track/<data_type>/<model-type>/<hyper-paramteres combination> where each folder will contain three or four folders- images,loss,models and testing.
```

For Thermal ROI-3DCAE with hyperparamter &lambda;<sub>S</sub> = 0.1, the root foler path will be 
```
Thermal_track/ROIframe/ROI_C3DAE-no_pool-BN_3DCNN-no_pool-BN/lambda_0.1
```
Pre-trained model weights can be found under google drive link. It is recommended to download the entire folder for the respective network as the provided code assumes the above mentioned directory structure.

Drive link- https://drive.google.com/drive/folders/11sp75lDEkFolLoCivF0PLrMsh_lXD0ra?usp=sharing
