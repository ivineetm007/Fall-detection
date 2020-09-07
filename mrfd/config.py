root_drive ='../dataset'
track_root_folder='Thermal_track'
root_folder='Thermal'
label_csv_path=root_drive+'/'+root_folder+'/Labels.csv'
adl_num=9
fall_num=35
#Tracking
detector_model_path='./rfcn_resnet101_coco_2018_01_28'
#Dataset Image dimensions.
WIDTH=640
HEIGHT=480
#Dataset Image dimensions.
LOAD_DATA_SHAPE=(64,64,1)
#Window ceration parameters
WIN_LENGTH=8
SPLIT_GAP=10#Split a video if no person localization for a certain gap
STRIDE=1
BATCH_SIZE=32
flow_dir='./Optical_flow_h-{}_w-{}_win-{}_bw-{}'.format(LOAD_DATA_SHAPE[0],LOAD_DATA_SHAPE[1],WIN_LENGTH,SPLIT_GAP)
