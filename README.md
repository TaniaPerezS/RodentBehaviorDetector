# RodentBehaviorDetector
A computer-vision project to detect rodents in behavioral taks used in neuroscience. 
# OPEN FIELD RAT DETECTOR MODULES
Sexual Behavior and Plasticity Laboratory, Neurobiology Institute, UNAM.
@autor: Tania Pérez
@Version: 1.0
@Date: April, 2023

This project uses different programs for the pre and post processing and the segmentation.
## Training:
File: OpenFieldTraining.inbpy
Uses the YOLO architecture to train using a previous labeled and augmented dataset of random frames.
The sinaptic weight are directly saved into Drive.

## Pre-processing:
File: SpeedUp.exe
Speed up the videos using a determined percentage of frames.

## Segmentation and post-processing:
File: OpenFieldSegmentation.inbpy
YOLO model segments the rat in the videos and writes a file with the coordinates. After, the coordinates
are readed to calculate the distance walked by the rat, the most common positions and to graph the
movement. The graphs and databases are directly saved into Drive. 


