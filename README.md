# Motion and Region Aware Adversarial Learning for Fall Detection with Thermal Imaging

This repository contains the source code for the paper [Motion and Region Aware Adversarial Learning for Fall Detection with Thermal Imaging](https://arxiv.org/abs/2004.08352) by Vineet Mehta, [Abhinav Dhall](https://research.monash.edu/en/persons/abhinav-dhall), Sujata Pal and [Shehroz S. Khan](http://individual.utoronto.ca/shehroz/).

In this work, we formulate fall detection as an anomaly detection within an adversarial framework using thermal imaging. Fall events are detected by using spatio-temporal autoencoders trained in an adversarial manner. A fall event is associated with the person and its interactions with nearby objects. We extract the region around the person by tracking the person in the video using a pre-trained object detector and also extract the optical flow in the region for motion-based discriminative learning. 
### Examples
#### Person Tracking
We use an object detector along with Kalman filters to track the bounding box for the person. We apply contour box localization to improve the overall tracking. It consists of finding the smallest bounding box that surrounds the biggest white blob in a thermal image. The below animation shows the effect of tracking with contour box localization. The left one is generated by Kalman filtering with an object detector, and the right is along with the contour box.

<p align="center">
  <img alt=alt="tracking gif" src="samples/tracking.gif" height="400" >
  <br>
    <em>Left- Tracking without Contour Box, Right- Tracking with Contour Box</em>
</p>

**Dark Blue**- Detection Box, **Sky Blue**- Tracker Prediction Box, **Green** - Contour Box (If Matched with other boxes), **Red** - Otsu Contour (Not matched) and **Yellow**- Final Box

#### Fall detection
Some of the examples of fall detection by our method are shown below. It can be seen that our method not only able to localize the person in the thermal image but also reconstructs only the region around the person, which makes it possible to use in the environment having different background artifacts.    
<p align="center">
  <img alt=alt="fall detection demo1 gif" src="samples/Fall7_mean.gif" height="450" width="450">
  <img alt=alt="fall detection demo1 gif" src="samples/Fall35_mean.gif" height="450" width="450">
  <br>
    <em></em>
</p>

### Dataset
#### TSF (Thermal Simulated Fall)
It contains 9 videos with normal Activities of Daily Living (ADL) and 35 thermal videos containing falls and other normal activities with a spatial resolution of 640x480. 

Please contact Dr. Shehroz S. Khan at [shehroz.khan@uhn.ca](shehroz.khan@uhn.ca) for access to the dataset. Please specify your affiliation and why you need this data in your email. Download and follow the instructions in the Readme file inside the [mrfd](mrfd) folder to perform experiments. 

### Installation
#### Conda
Recreate the conda environment using the provided ```environment.yml``` file
```
conda env create -f environment.yml 
```
#### Others
These are the major libraries along with their verisons.

1. keras=2.3.1
2. tensorflow=1.14.0
3. opencv=4.2.0
4. python=3.7.6

We have used tf1.x for the tensorlfow1.x detection zoo model. The provided code consists of person tracking, data processing, model training and testing. The person tracking code must requires tf1.x. If you are using the pre-tracked boxes, you can run the other code with tf2.x as well. Check the [Readme](mrfd) file inside the mrfd for more details about pre-tracked boxes.

### Additional notes

Citation:

```
@article{mehta2020motion,
  title={Motion and Region Aware Adversarial Learning for Fall Detection with Thermal Imaging},
  author={Mehta, Vineet and Dhall, Abhinav and Pal, Sujata and Khan, Shehroz},
  journal={arXiv preprint arXiv:2004.08352},
  year={2020}
}
```
