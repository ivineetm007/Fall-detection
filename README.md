# Motion and Region Aware Adversarial Learning for Fall Detection with Thermal Imaging

This repository contains the source code for the paper [Motion and Region Aware Adversarial Learning for Fall Detection with Thermal Imaging](https://arxiv.org/abs/2004.08352) by Vineet Mehta, [Abhinav Dhall](https://research.monash.edu/en/persons/abhinav-dhall), Sujata Pal and [Shehroz S. Khan](http://individual.utoronto.ca/shehroz/).

In this work, we formulate fall detection as an anomaly detection within an adversarial framework using thermal imaging. Fall events are detected by using spatio-temporal autoencoders trained in an adversarial manner. A fall event is associated with the person and its interactions with nearby objects. We extract the region around the person by tracking the person in the video using a pre-trained object detector and also extract the optical flow in the region for motion-based discriminative learning. 
### Examples
#### Person Tracking

| ![Screenshot](samples/tracking.gif) | 
|:--:| 
|Dark Blue- Detect box  Sky Blue- Track box   Green- Otsu Contour(Matched)  Red- Otsu Contour(Not matched)|

![Screenshot](samples/tracking.gif)
*Dark Blue- Detect box  Sky Blue- Track box   Green- Otsu Contour(Matched)  Red- Otsu Contour(Not matched)*
### Requirements
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

We have used tf1.x for the tensorlfow1.x detection zoo model. The provided code consists of person tracking, data processing, model training and testing. The person tracking code must requires tf1.x. If you are using the pre-tracked boxes, you can run the other code with tf2.x as well. Check the Readme file inside the mrfd for more details about pre-tracked boxes.

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
