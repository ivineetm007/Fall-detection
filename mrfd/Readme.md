## Configuration 
Download the data and place the video frame folders at ```<root_path>/Thermal/frame/Fall/``` and ```<root_path>/Thermal/frame/NonFall/``` respectively. The labels file ```Labels.csv``` must be place at ```<root_path>/Thermal/```. 
The ```<root_path>``` will contain all the thermal frames, tracked box CSV files, and videos after preprocessing along with TSF dataset. It is set in config.py as ```../dataset```. 
Change the ```root_drive``` variable in config.py if you wish to change the location to save these files.

## Demo and Animation
We have prepared a python notebook for the animation of the fall detection See [demo.ipynb](demo.ipynb) 

## Data Preprocessing
### Person tracking

```
python person_tracking.py
```
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

Thermal model-Same as 3DCAE-CNN SOTA model
```
python 3DCAE_train.py --datatype thermal --lambda_ 0.1
python 3DCAE_test.py --datatype thermal --epochstrained 299 --lambda_ 0.1
```
Flow model
```
python 3DCAE_train.py --datatype flow --lambda_ 0.1
python 3DCAE_test.py --datatype flow --epochstrained 290 --lambda_ 0.1
```

#### ROI-3DCAE models

Thermal model
```
python ROI-3DCAE_train.py --datatype thermal --lambda_ 0.1
python ROI-3DCAE_test.py  --datatype thermal --epochstrained 299 --lambda_ 0.1
```
Flow model
```
python ROI-3DCAE_train.py --datatype flow
python ROI-3DCAE_test.py  --datatype flow --epochstrained 299 --lambda_ 0.1
```

#### diff-ROI-3DCAE models

```
python diff-ROI-3DCAE_train.py --lambda_S 1 --lambda_T 1
python diff-ROI-3DCAE_test.py --epochstrained 290 --lambda_S 1 --lambda_T 1
```

#### Fusion-ROI-3DCAE models

```
python Fusion-ROI-3DCAE_train.py --lambda_T 0.1 --lambda_F 0.1
python Fusion-ROI-3DCAE_test.py --lambda_T 0.1 --lambda_F 0.1 --epochstrained 299
```

#### Fusion-Diff-ROI-3DCAE models

```
python Fusion-Diff-ROI-3DCAE_train.py --lambda_T_S 1 --lambda_T_T 1 --lambda_F 1
python Fusion-ROI-3DCAE_test.py --lambda_T 0.1 --lambda_F 0.1 --epochstrained 299
```

## Pre-trained model weights and other files

