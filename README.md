## Motivation:

The ability of robotic agents performing everyday activity such as loading a dishwasher and setting a table autonomously in a wide range of context is challenging. Mastering such everyday activities by robotic agents is an important step for robots to become the competent (co-)workers or assistants. Everyday Activity Science and Engineering(EASE) is a research program that investigates the everyday activites of humans and uses this informations for modelling autonomous robotic agents. 



## Objectives:


The main goal of the project was to determine the 3D coordinate of each specific objects that are being moved during the recording session. This project is a small part of EASE research program in which two cameras where used to detect and track objects above the two table setting scenerio named as Kitchen table and Dinner table. The objects are moved by a participant from kitchen to dinner and vice versa. The participant performs the recording session wearing a motion sensor suit which records the coordinate of the body with respect to the origin of the motion capture system.

## GOALS: 

1. *Detect the objects using YOLO-V3 (You Only Look Once)*.
2. *Track the Detected objects using OpenCV Multi-tracker*.
3. *Camera Calibration*.
4. *Combine Motion Capture Data with obtained coordinates from computer vision technique to generate a 3D coordinate of each object*.

## Requirements:

1. OpenCV 4.0.0
2. Darknet YOLO. [Darknet_YOLO-V3](https://github.com/AlexeyAB/darknet)
3. Pandas, Numpy, Imutils
4. Linux OS (Ubuntu 16.04 LTS)
5. Python 3.6.6

## Procedures To use the script:

#### There are six arguments required for using the script *YOLO_MOCAP_TRACKING.py*:

1. **kitchenvdo**: Path to video file of the kitchen scene.
2. **dinnervdo**: Path to video file of the dinner scene.
3. **timestampkitchen**: Path to time stamp file of the kitchen scene.
4. **timestampdinner**: Path to time stamp file of the dinner scene.
5. **optitrackJSON**: Path to mocap .JSON file of the motion capture data.
6. **optitrackCSV**: Path to csv file of the motion capture CSV.


![Detection of Object (Bowl) On Dinner Table Scenerio](https://raw.githubusercontent.com/Sharier/Object-3D-Coordinate/ecc3f1ccd1c6820957cf99b5bade20640c518d16/tracked.png)

## Additional Information:

### Procedures to use Trained YOLO Network for Detection:

A configuration file **config.txt** is used to configure the YOLO network for forward propagation in the main python script.

#### The following parameters are needed for setting the YOLO network:

1. **ClassFile**: The path of the file with extension ”.names” is the file with the class names of the trained YOLO network
2. **ModelConfiguration**: The path of file with extension ”.cfg” is the configuration file of the YOLO network.
3. **ModelWeights**: The path of file with extension ”.weights” is the trained weights of YOLO.
