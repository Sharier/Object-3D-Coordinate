######Motivation:

The ability of robotic agents performing everyday activity such as loading a dishwasher and setting a table autonomously in a wide range of context is challenging. Mastering such everyday activities by robotic agents is an important step for robots to become the competent (co-)workers or assistants. Everyday Activity Science and Engineering(EASE) is a research program that investigates the everyday activites of humans and uses this informations for modelling autonomous robotic agents. 


######Objectives:


The main goal of the project was to determine the 3D coordinate of each specific objects that are being moved during the recording session. This project is a small part of EASE research program in which two cameras where used to detect and track objects above the two table setting scenerio named as Kitchen table and Dinner table. The objects are moved by a participant from kitchen to dinner and vice versa. The participant performs the recording session wearing a motion sensor suit which records the coordinate of the body with respect to the origin of the motion capture system.

######GOALS: 

1. Detect the objects using YOLO (You Only Look Once).
2. Track the Detected objects using OpenCV Multi-tracker.
3. Camera Calibration
4. Combine Motion Capture Data with obtained coordinates from computer vision technique to generate a 3D coordinate of each object.

Requirements:

1. OpenCV 4.0.0
2. Darknet YOLO
3. Pandas, Numpy, imutils



