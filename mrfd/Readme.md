## Configuration 
Download the TSF dataset and place the video frame folders at ```<root_path>/Thermal/frame/Fall/``` and ```<root_path>/Thermal/frame/NonFall/``` respectively. The labels file ```Labels.csv``` must be place at ```<root_path>/Thermal/```. 

The ```<root_path>``` will contain all the thermal frames, tracked box CSV files, and videos after preprocessing along with TSF dataset. It is set in config.py as ```../dataset```. 
Change the ```root_drive``` variable in config.py if you wish to change the location to save these files.

## Demo and Animation
We have prepared a python notebook to create animations and illustrate all the proposed fall detection method's intermediate steps. See [demo.ipynb](demo.ipynb). 

Download the [pre-trained](#saved-model-weights-and-other-files) weights or train the model and specify the path while running the notebook.

## Data Preprocessing
### Person tracking
The code for kalman filtering and person detector are saved in ```tracker/tracker.py``` and ```tracker/detector.py``` respectively. In this work, we use the pre-trained rfcn_resnet101_coco model checkpoint from tensorflow detection model zoo. 

Download and place the model checkpoint files either from tensorflow model zoo or the google drive in [rfcn_resnet101_coco_2018_01_28](rfcn_resnet101_coco_2018_01_28) folder-
1. https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
2. https://drive.google.com/drive/folders/1vGexPvEVsf0NYRZarWtKrsgOfvB2pyrE?usp=sharing

If you use any other object detector model from the tensorflow detection [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md). Specify the checkpoint path in ```config.py``` by changing the value of ```detector_model_path```.

Run the following command to track and save the coordinates of the tracked person in the .csv files. These files will be saved at ```<root_path>/Thermal_track/csv```
```
python person_tracking.py
```
Check the various option with the following command:
```
python person_tracking.py --help
```
You can use ```python person_tracking.py --visualize True``` to see the images while tracking. It will open an opencv window and shows the frame being tracked along with various boxes. You can save these visualizations in videos by ```python person_tracking.py --output_type video```. These videos will be saved at ```<root_path>/Thermal_track/video```.

We have provided the csv files containing the frame number and the box coordiantes for fair comparison with the approaches on TSF dataset in the future. Downlaod these files from the following google drive link:
[https://drive.google.com/drive/folders/1BJLvY-z0UZxV2G1CuAWQjiPh0NvzrqlJ?usp=sharing](https://drive.google.com/drive/folders/1BJLvY-z0UZxV2G1CuAWQjiPh0NvzrqlJ?usp=sharing)

### Optical flow

## Training and Testing

#### Previous SOTA model

##### ConvLSTMAE
```
python ConvLSTMAE_train.py
python ConvLSTMAE_test.py --epochstrained 50
```

##### DSTCAE_C3D
```
python DSTCAE_C3D_train.py
python DSTCAE_C3D_test.py --epochstrained 300
```

#### 3DCAE

##### Thermal model-Same as 3DCAE-CNN SOTA model
```
python 3DCAE_train.py --datatype thermal --lambda_ 0.1
python 3DCAE_test.py --datatype thermal --epochstrained 299 --lambda_ 0.1
```
##### Flow model
```
python 3DCAE_train.py --datatype flow --lambda_ 0.1
python 3DCAE_test.py --datatype flow --epochstrained 290 --lambda_ 0.1
```

#### ROI-3DCAE

##### Thermal model
```
python ROI-3DCAE_train.py --datatype thermal --lambda_ 0.1
python ROI-3DCAE_test.py  --datatype thermal --epochstrained 299 --lambda_ 0.1
```
##### Flow model
```
python ROI-3DCAE_train.py --datatype flow
python ROI-3DCAE_test.py  --datatype flow --epochstrained 299 --lambda_ 0.1
```

#### Diff-ROI-3DCAE 

```
python diff-ROI-3DCAE_train.py --lambda_S 1 --lambda_T 1
python diff-ROI-3DCAE_test.py --epochstrained 290 --lambda_S 1 --lambda_T 1
```

#### Fusion-ROI-3DCAE 

```
python Fusion-ROI-3DCAE_train.py --lambda_T 0.1 --lambda_F 0.1
python Fusion-ROI-3DCAE_test.py --lambda_T 0.1 --lambda_F 0.1 --epochstrained 299
```

#### Fusion-Diff-ROI-3DCAE 

```
python Fusion-Diff-ROI-3DCAE_train.py --lambda_T_S 1 --lambda_T_T 1 --lambda_F 1
python Fusion-Diff-ROI-3DCAE_test.py --lambda_T_S 1 --lambda_T_T 1 --lambda_F 1 --epochstrained 299
```

## Saved model weights and other files
We saved the model weights, loss lists in .npy format, losses plot, and reconstructed images at intermediate epochs( at every 10th epoch). You can download all these files from the following google drive link:
[https://drive.google.com/drive/folders/1O83xGqtSsxxD5CVATMc8azJslzNblavs?usp=sharing](https://drive.google.com/drive/folders/1O83xGqtSsxxD5CVATMc8azJslzNblavs?usp=sharing)

Download all the folders or the desired one in the same folder structure and place it in the [Thermal_track](Thermal_track) folder.
