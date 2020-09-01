# Motion and Region Aware Adversarial Learning for Fall Detection with Thermal Imaging

This repository contains the source code for the paper [Motion and Region Aware Adversarial Learning for Fall Detection with Thermal Imaging](https://arxiv.org/abs/2004.08352) by Vineet Mehta, [Abhinav Dhall](https://research.monash.edu/en/persons/abhinav-dhall), Sujata Pal and [Shehroz S. Khan](http://individual.utoronto.ca/shehroz/).

In this work, we formulate fall detection as an anomaly detection within an adversarial framework using thermal imaging. Fall events are detected by using spatio-temporal autoencoders trained in an adversarial manner. A fall event is associated with the person and its interactions with nearby objects. We extract the region around the person by tracking the person in the video using a pre-trained object detector and also extract the optical flow in the region for motion-based discriminative learning. 

## Requirements
#### Conda
Recreate the conda environment using the provided ```environment.yml``` file
``` conda env create -f environment.yml ```
