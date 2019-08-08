# ALL THE REQUIRED LIBRARIES ARE DECLARED 
    
    
import numpy as np
import pandas as pd
import argparse
import imutils
import glob
import cv2
import cv2 as cv
import argparse
import sys
import os.path
from numpy.linalg import inv
from math import sqrt
import argparse

dirpath = os.getcwd()
import json
from numpy import diff
import config

from mocap_combine import initiator,mocap_read


#DECLARATION OF THE TRACKER TYPES


#tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
#tracker_type = tracker_types[2]
    
#if tracker_type == 'BOOSTING':
#        #tracker = cv2.TrackerMIL_create()
#    tracker = cv2.Tracker_create(tracker_type)
#if tracker_type == 'MIL':
#    tracker = cv2.TrackerMIL_create()
#if tracker_type == 'KCF':
#    tracker = cv2.TrackerKCF_create()
#if tracker_type == 'TLD':
#    tracker = cv2.TrackerTLD_create()
#if tracker_type == 'MEDIANFLOW':
#    tracker = cv2.TrackerMedianFlow_create()
#if tracker_type == 'GOTURN':
#    tracker = cv2.TrackerGOTURN_create()
#if tracker_type == 'MOSSE':
#    tracker = cv2.TrackerMOSSE_create()
#if tracker_type == "CSRT":
#    tracker = cv2.TrackerCSRT_create()
    
trackerName = 'csrt'

OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
        }

    

config=config.Config(dirpath+'/'+'config.txt')
classesFile= config.classesFile
modelConfiguration=config.modelConfiguration
modelWeights= config.modelWeights



# Initialize the parameters
confThreshold = config.confThreshold
nmsThreshold = config.nmsThreshold
inpWidth = config.inpWidth
inpHeight = config.inpHeight
#confThreshold = 0.9 #Confidence threshold
#nmsThreshold = 0.7   #Non-maximum suppression threshold#*
#inpWidth = 416       #Width of network's input image
#inpHeight = 416      #Height of network's input image



###########################################################

class coordinate(object):
    
    def __init__(self):
        
        self.net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.classes = None
        #reading the .names file from directory
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
            
        # declare this as intialization for storing the coordinates to a dictionary
        self.coordinate_dict={}
        self.index=0  # this index gives the address to the moved object
        self.tracked_dict={}
  
        #initialize the frame
        self.frame= None
        self.label= None
        self.time=None
        
        #for scene selection either dinner table or kitchen table
        self.scene=None

        #for time stamp of each video
        self.video_time=None
        
        #all variable initialization for the multitracker
        self.trackers = cv2.MultiTracker_create() 
        self.boxes=()
        self.temp_list=[]
        self.tracker_status=False
        

        self.bbox_lst=[]
        self.temp_len=0
        self.ecludian_lst=[]
        self.classIDlst=[]
        self.tracked_classname=[]
        self.j=0 #counter
        
        #variable for tracker initiator
        self.temp_x_orig= 0
        self.temp_y_orig= 0
        
        #self.conf=0
        
    
    
    def create_csv(self,time):
        
    
        dic=self.tracked_dict
    
        row_main=len(time)
        #print(row_main)


        new_list=[] #contains the list of header for each keys

        data_row=0
        
        for m in dic.keys():
            a=len(dic[m])
            new_list.append(m+'_x')
            new_list.append(m+'_y')
            new_list.append(m+'_Time')
    
            if(data_row<a):
                data_row=a
            else:
                pass
            #print('\nempty coloumn')
                #break

        data_col=len(new_list)  #length of the number of coloumns if the dict
        #print(data_col)

        data=np.zeros((row_main,data_col))  #a numpy array for storing dictionary with respect to time

        i=0  #iterator variable for time 

        j=0  #iterator variable for dictionary coloumn

        k=0  #iterator variable for each dictionary row


        while i<row_main:
    
            while k<data_row:            
        
        
            
                while j<data_col: #this represents coloumn of the dictionary
            
                    for key in dic.keys():
        
            
                        try:
                            val=dic[key][k][2]
                            #print('printing value:',val)
                        except:
                            val=-1   #declaring val with negative value assuming time cannot be less than zero
                        #print('printing no values found')
                
                        if (val!=-1):
                            #print('printing both key and j',key,j)
                
                            if ((time[i])-(dic[key][k][2]))==0:
                    
                                #print('\nneed to do somthn')
                    
                                try:
                                    data[i,j]=dic[key][k][0]
                                except:
                                    data[i,j]=0
                                j=j+1
                                try:
                                    data[i,j]=dic[key][k][1]
                                except:
                                    data[i,j]=0
                                j=j+1
                                try:
                                    data[i,j]=dic[key][k][2]
                                except:
                                    data[i,j]=0
                                j=j+1
                                #try:
                                #    data[i,j]=dic[key][k][3]
                                #except:
                                #    data[i,j]=0
                                #j=j+1
            
            
                            else:
                                j=j+3  #each key has four elements
                        else:
                            j=j+3
        
                    #print('going into break')
                    break
            
                k=k+1
                j=0
        
    
            k=0
            i=i+1
        
        
        return data,new_list ###returns a numpy array for data frame and the list for the coloums of data frame


        
     
    def pixeltoworld(self,left,top,right,bottom):
        
        startX=left
        startY=top
        
        endX=right
        endY=bottom
       
    
        X_original=((startX+endX)/2)
        Y_original= ((startY+endY)/2)
        
        z=0.0
        
        image_point=(X_original,Y_original)
        
        dist= np.float64([[ 0.15934017, -0.30815065, -0.00506551, -0.00711125,  0.1484886 ]]) #distortion matrix



        #print(new_camera_matrix)
        dist_coeff= dist

       #defining points for camera calibration

          
        if self.scene==0: #kitchen scene
            
            fx_new=(1280/1920)*1.22627588e+03   #camera was calibrated with 1920*1080 dimension so camera matrix modified for 1280*720 dimension
            fy_new=(720/1080)*1.21279138e+03
        
        #camera matrix

            new_camera_matrix= [[fx_new,0.00000000e+00,9.19037974e+02],[0.00000000e+00,fy_new,5.18150766e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]]

            
            #optitrack points in csl lab
            tnsPoints = np.zeros((4, 3))
            tnsPoints[0] = (-1.080229,0.423667,0) #M6
            tnsPoints[1] = (-1.675380,0.422585,0) #M7
            tnsPoints[2] = (-1.661897,-1.375426,0) #M8
            tnsPoints[3] = (-1.073260,-1.368733,0) #M9
        
            # corresponding pixel coordinates for optitrack points

            imPoints = np.zeros((4,2))
            imPoints[ 0] = (1086,297) #M6
            imPoints[ 1] = (1091,546)#M7
            imPoints[ 2] = (329,542)#M8
            imPoints[ 3] = (336,304)#M9
    
        else:#dinner scene

            fx_new=(1280/1920)*1.19349060e+03   #camera was calibrated with 1920*1080 dimension so camera matrix modified for 1280*720 dimension
            fy_new=(720/1080)*1.17655347e+03
        
        #camera matrix

            new_camera_matrix= [[fx_new,0.00000000e+00,9.04205350e+02],[0.00000000e+00,fy_new,5.13000652e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]]

            #optitrack points in csl lab
            tnsPoints = np.zeros((14, 3)) 
        
            tnsPoints[0] = (0.008897,0.003407,0)  #O1
            tnsPoints[ 1] = (0.412059,-0.016740,0)  #M1
            tnsPoints[ 2] = (0.389876,1.555950,0)   #M2
            tnsPoints[ 3] = (1.162737,-0.026982,0)  #M3
            tnsPoints[ 4] = (1.170563,1.556716,0)   #M4
            tnsPoints[ 5] = (0.984067,0.191943,0)   #M5
            tnsPoints[ 6] = (1.038657,1.312016,0)   #M6   
            tnsPoints[ 7] = (0.563111,0.213965,0)  #M7
            tnsPoints[ 8] = (0.507414,1.284817,0)  #M8
            tnsPoints[ 9] = (0.808466,0.746491,0)  #M9
            tnsPoints[ 10] = (0.410388,0.765601,0)  #M10
            tnsPoints[ 11] = (1.160016,0.786132,0)  #M11
            tnsPoints[ 12] = (0.780790,1.541194,0)  #M12
            tnsPoints[ 13] = (0.781836,-0.013351,0)  #M13


            # corresponding pixel coordinates for optitrack points

    
            imPoints = np.zeros((14,2))
        
            imPoints[ 0] = (386, 613)
            imPoints[ 1] = (293, 530)
            imPoints[ 2] = (882, 519)
            imPoints[ 3] = (292, 237)
            imPoints[ 4] = (869, 237)
            imPoints[ 5] = (374, 311)
            imPoints[ 6] = (785, 285)
            imPoints[ 7] = (383, 469)
            imPoints[ 8] = (782,477)
            imPoints[ 9] = (583, 372)
            imPoints[ 10] = (593, 519)
            imPoints[ 11] = (594, 244)
            imPoints[ 12] = (871, 375)
            imPoints[ 13] = (294, 389)


        retval, rvec, tvec = cv2.solvePnP(tnsPoints,imPoints,np.asarray(new_camera_matrix), np.asarray(dist_coeff))

        rotMat, _ = cv2.Rodrigues(rvec)

        #print(rotMat)

        
        camMat = np.asarray(new_camera_matrix)
        iRot = inv(rotMat)
        iCam = inv(camMat)

        uvPoint = np.ones((3, 1))

        # Image point
        uvPoint[0, 0] = image_point[0]
        uvPoint[1, 0] = image_point[1]

        tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
        tempMat2 = np.matmul(iRot, tvec)

        s = (z + tempMat2[2, 0]) / tempMat[2, 0]
        wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvec))

        # wcPoint[2] will not be exactly equal to z, but very close to it
        
        assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
        wcPoint[2] = z

        return wcPoint

    
    
    
    def getOutputsNames(self, net):
  
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs        
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    
    
    def tracker_reset(self):
        
        
        #clear the coordinate_dict the entire dict and save the moved object to new dictionary
        self.tracked_dict[self.tracked_classname[self.index]]=self.coordinate_dict[self.tracked_classname[self.index]]
        
        self.coordinate_dict.clear()

        
        self.trackers.clear()
        self.trackers = cv2.MultiTracker_create() # global variables
        self.bbox_lst=[]
        self.tracker_status=False  #this variable will be used for yolo activation
        self.boxes=()
        self.temp_len=0
        self.ecludian_lst=[]
        self.classIDlst=[]
        self.tracked_classname=[]
        self.temp_list=[]
        
    
    
    
    def movement_check(self,past_box,present_box):
        
        #check for movement
        if len(past_box)==0:
            print('No tracker initialized so no ecludian distance generated')
            
        elif(len(past_box)==len(present_box)):
            
            for index,val in enumerate(present_box):
                past_X,past_Y=(past_box[index][0],past_box[index][1])
                present_X,present_Y=(val[0],val[1])
                ecludian_dist= sqrt(((present_X-past_X)**2)+((present_Y-past_Y)**2))
                print('ecludian distance with index',ecludian_dist,index)
                
                if ecludian_dist>5:
                    print('Movement detected in index:',index,present_Y)
                    self.index=index
                    
                    # this condition works only for video 2 or kitchen scene video
                    if self.scene==0:
                        if(present_Y<30 or present_X>1140):                               
                            past_box=[]
                            present_box=[]
                            #function is called to reset the tracker and variables
                            self.tracker_reset()
                            print('\nTracker deactivated')
                        
                        else:
                            print('\n Tracker not deactivated still')
                        break
                    # this condition works only for video 1 or dinner scene video
                    else:
                        if (present_Y>670):
                            past_box=[]
                            present_box=[]
                            #function is called to reset the tracker and variables
                            self.tracker_reset()
                            print('\nTracker deactivated')
                        else:
                            print('\n Tracker not deactivated still')
                        break    
                else:
                    print('No movement detected yet')
                    
        elif(len(present_box)>len(past_box)):
            print('Tracker just initialized size difference so no ecludian distance generated')
            
            
    def tacker_initiator(self,classID, conf,left,top,right,bottom):
    
        #initializing classname for tracker
        classname=self.classes
        
    
        
        # bbox redifined for tracker
        bbox=(left,top,right-left,bottom-top) 
        startX=bbox[0]
        startY=bbox[1]
        endX=bbox[2]
        endY=bbox[3]

        
        # defining the midpoint of the box /centre point
        X_original=((startX+endX)/2)
        Y_original= ((startY+endY)/2)


        for i,bb in enumerate(self.bbox_lst):
            

            temp_startX=bb[0]
            temp_startY=bb[1]
            temp_endX=bb[2]
            temp_endY=bb[3]
            self.temp_x_orig= ((temp_startX+temp_endX)/2)
            self.temp_y_orig= ((temp_startY+temp_endY)/2)
            #print(temp_endX)
            
            #define the ecludian distance
        
            ecludian_dist= sqrt(((X_original-self.temp_x_orig)**2)+((Y_original-self.temp_y_orig)**2))
            #print('\nPrinting the ecludian distance:',ecludian_dist)
        
            if 0<=ecludian_dist<=43:  #this works as a threshold for the object movement
                self.ecludian_lst.append(ecludian_dist)
                print('\nMatched Boundary box found from yolo detection so new tracker not initialized')
        
            #break
            else:
                print('\nNew boundary box found, new tracker will be initialized')
            

    
        if len(self.ecludian_lst)>self.temp_len:
            #print('\nmatch found so ecludian list is greater than zero')
            #print('\Printing the len of ecludian list and temp_len',len(self.ecludian_lst),self.temp_len)

            self.temp_len=len(self.ecludian_lst)
        else:
            print('\nTracker initiated now')
            self.bbox_lst.append(bbox[0:4])
            self.classIDlst.append(classID)
            tracker = OPENCV_OBJECT_TRACKERS[trackerName]()
            self.trackers.add(tracker, self.frame, bbox[0:4])
            #self.conf=conf
            self.j=self.j+1   # counter for tracker class name 
        
        #initialization of tracked class name and dictionary
  
        if len(self.tracked_classname)==0:
            temp=classname[classID]+str(self.j)
            self.tracked_classname.append(temp)
            self.coordinate_dict[self.tracked_classname[-1]]=[]
        
        elif len(self.classIDlst)>len(self.tracked_classname):
            temp1=classname[self.classIDlst[-1]]+str(self.j)
            self.tracked_classname.append(temp1)
            self.coordinate_dict[self.tracked_classname[-1]]=[]
    
       
    
        return True

        
        
    def postprocess(self,frame, outs):
        
        #frame from the yolo_activation function
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
       
        classIds = []
        confidences = []
        boxes = []
        
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        
        
        for out in outs:
            for detection in out:
                
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

      
        # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        
        #bbox from the final output of the yolo are sent to the initalizer tracker function 
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            centre_x= (left+(left+width))/2
            centre_y=(top+(top+height))/2
            #function for tracker is called
            #the below condition will change according to the view...which is either kitchen or dinner table
            if self.scene==0:
                
                ######condition for detection only within the table region for kitchen scene
                if top>270:
                    
                    self.tracker_status=self.tacker_initiator(classIds[i], confidences[i], left, top, left + width, top + height)
                else:
                    print('\nobject detected outside the kitchen table region')
            else:
                
                ######condition for detection only within the table region for dinner scene
                if 260<left<900 and 240<top<560:
                    self.tracker_status=self.tacker_initiator(classIds[i], confidences[i], left, top, left + width, top + height)
                else:
                    print('\nobject detected outside the dinner table region')


            
            
    def yolo_activation(self,video,scene,video_time):
        
        
 
        int_fps=0
     
        self.scene=scene
        self.video_time=video_time

        
        #read the video using opencv
        cap = cv.VideoCapture(video)
        
        
        
        #only if needed to show the video will be saved to a directory
        outputFile = dirpath+'/scene'+str(self.scene)+'_YOLO_OUTPUT.avi'
        vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))


        
        #opencv window declared
        #winName = 'Object Tracking using YOLO and OpenCV'
        #cv.namedWindow(winName, cv.WINDOW_NORMAL)
        
        #yolo loaded
        net= self.net
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        
        


        while(cap.isOpened()):
            
                
            
            #get each frame from the video
            hasFrames,self.frame = cap.read()
            
            #check for frames
            if not hasFrames:
                print("Done processing whole video !!!")
                return True
                cv.waitKey(3000)
                break
                
            if self.scene==0: #this works for kitchen table scenerio
                
                #if self.tracker_status==False:
                
                #declaration of opencv dnn
                blob = cv.dnn.blobFromImage(self.frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

                # Sets the input to the network
                net.setInput(blob)

                # Runs the forward pass to get output of the output layers, function getOutputsNames is called
        
                outs = net.forward(self.getOutputsNames(net))    
         

                # Remove the bounding boxes with low confidence
        
                self.postprocess(self.frame, outs)  # calling of the function postprocess
            
            
                #else:
                #    pass
                
            else:
                # this works for dinner table scenerio
                
                #declaration of opencv dnn
                blob = cv.dnn.blobFromImage(self.frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

                # Sets the input to the network
                net.setInput(blob)

                # Runs the forward pass to get output of the output layers, function getOutputsNames is called
        
                outs = net.forward(self.getOutputsNames(net))    
         

                # Remove the bounding boxes with low confidence
        
                self.postprocess(self.frame, outs)  # calling of the function postprocess
            
            
                
            #the tracked bbox are obtained from the tracker
            (success, self.boxes) = self.trackers.update(self.frame)
    
            tracked_list=list(self.boxes)
        
            #this function is used to check the movement of specific object
            self.movement_check(self.temp_list,tracked_list)
    
            self.temp_list=tracked_list
    
            #print('\nprinting the list boxes: ',tracked_list)
    

            for index,value in enumerate(self.temp_list):
                #print(index,value)
                world_cord=self.pixeltoworld(value[0],value[1],value[0]+value[2],value[1]+value[3])
                
                #time from the optitrack will be saved for each object detected
                world_cord[2]=self.video_time[int_fps]
                #world_cord[3]=0
                
                #world_cord[2]=int_fps   #just for testing purpose
                #print(value[0])

                
                #print('\nPrinting the world coordinate:', list(world_cord))
                
                if not len(self.tracked_classname)==0:
                    self.coordinate_dict[self.tracked_classname[index]].append(world_cord)  # saving the values to a dictionary as per keys
                    cv.putText(self.frame, self.tracked_classname[index], (int(value[0]), int(value[1])), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

                else:
                    print('tracker has been fully re-initialized')
            
            # loop over the bounding boxes and draw them on the frame
            for box in self.boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
            
            # show the frame
            #cv.imshow(winName, self.frame)
            
            #this varaible used for time index iterator
            int_fps=int_fps+1
            
            #Recording the video
            vid_writer.write(self.frame.astype(np.uint8))
            
            # Exit if ESC pressed    
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break
                    

        #print('\nprinting the tracked dictionary:', self.tracked_dict)    
        #cap.release()
        #cv2.destroyAllWindows()
        
        
    
    def initiator(self,arg):
        
        
        
        #####################this is for kitchen scene##############################################
        
        vid_status=False
        kit_vdo=arg.kitchenvdo
        
        #first check if the time stamp length is equal to video length
        
        #check for video file directory    
        if not os.path.isfile(arg.kitchenvdo):
            print("Input video file ", arg.kitchenvdo, " doesn't exist")
            sys.exit(1)
        
        
        kit_cap = cv.VideoCapture(arg.kitchenvdo)
        
        vid_length = int(kit_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        kitchen_vdotime=np.load(arg.timestampkitchen)
        
        timestamp_len=len(kitchen_vdotime)

    
        if vid_length!=timestamp_len:
            
            print('\ntime stamp and video length are not equal so they are reshaped')
            
            diff=vid_length-timestamp_len
            d=np.full(diff,kitchen_vdotime[-1])
            kit_timestamp=np.hstack((kitchen_vdotime,d))

        else:
            
            print('\ntime stamp and video same length')
            
            kit_timestamp=kitchen_vdotime
        
        scene=0
        vid_status=self.yolo_activation(kit_vdo,scene,kit_timestamp)
        
        if vid_status==True:
            
            print('\nFirst video of kitchen scene has been fully processed')
            
            ##creating data frame for kitchen
            kitchen_data,kitchen_list=self.create_csv(kit_timestamp)
            
            kitchen_frame = pd.DataFrame(kitchen_data, index=kit_timestamp,columns=kitchen_list)
            
            kitchen_frame.to_csv(dirpath+'/'+'kitchenscene.csv')
            
            #resetting the video status
            vid_status=False
        
        else:
            
            print('\nError occured kitchen scene full video not processed')
            sys.exit(1)
            
        
    
        
        
        #######################################################################################
        self.__init__()
        #self.tracked_dict={}
        ######################for the dinner scene video####################
        
        din_vdo=arg.dinnervdo
        
        #first check if the time stamp length is equal to video length
        
        #check for video file directory    
        if not os.path.isfile(arg.dinnervdo):
            print("Input video file ", arg.dinnervdo, " doesn't exist")
            sys.exit(1)
        
        
        din_cap = cv.VideoCapture(arg.dinnervdo)
        
        vid_length = int(din_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        dinner_vdotime=np.load(arg.timestampdinner)
        
        timestamp_len=len(dinner_vdotime)

    
        if vid_length!=timestamp_len:
            
            print('\ndinner scene time stamp and video length of dinner video are not equal so they are reshaped')
            
            diff=vid_length-timestamp_len
            d=np.full(diff,dinner_vdotime[-1])
            din_timestamp=np.hstack((dinner_vdotime,d))

        else:
            
            print('\ntime stamp and video same length of dinner scene are both equal')
            
            din_timestamp=dinner_vdotime
        
        #yolo is applied on dinner scene video
        scene=1
        vid_status=self.yolo_activation(din_vdo,scene,din_timestamp)
        
        if vid_status==True:
            print('\nSecond video of dinner scene has been fully processed')
            
            ##creating data frame for dinner
            dinner_data,dinner_list=self.create_csv(din_timestamp)
            
            dinner_frame = pd.DataFrame(dinner_data, index=din_timestamp,columns=dinner_list)
            
            dinner_frame.to_csv(dirpath+'/'+'dinnerscene.csv')


            
            #resetting the video status
            
            vid_status=False
        else:
            print('\nError occured in dinner scene, full video not processed')
          
        
        ##############processing mocap and yolo##############
        
        kit_frame=pd.read_csv(dirpath+'/'+'kitchenscene.csv', index_col=0)
        din_frame=pd.read_csv(dirpath+'/'+'dinnerscene.csv', index_col=0)

        kit_frame=kit_frame.replace(0,np.nan)
        din_frame=din_frame.replace(0,np.nan)
        
        
        sampln_rate=120
        mocap_json= arg.optitrackJSON
        mocap_csv= arg.optitrackCSV
        #d="/home/tanbir/MSC_THESIS/Thesis_code/TEST_BENCH/LEFT_HAND_RIGID/Mocap_OptiTrack_17191320190325.meta.json"
        #csv='/home/tanbir/MSC_THESIS/Thesis_code/TEST_BENCH/LEFT_HAND_RIGID/Mocap_OptiTrack_17191320190325.csv'
        opti_frame=mocap_read(mocap_json,mocap_csv)
        final_frame=initiator(kit_frame,din_frame,opti_frame,sampln_rate)#2748.658218360041 :2767.411761561991
        final_frame.to_csv(dirpath+'/'+'yolo_mocap.csv')

        
        

            
        
            
        
        
        
        
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='YOLO DETECTION AND TRACKING FOR CSL LAB')
    parser.add_argument('--kitchenvdo', help='Path to video file of the kitchen scene.')
    parser.add_argument('--dinnervdo', help='Path to video file of the dinner scene.')
    parser.add_argument('--timestampkitchen', help='Path to time stamp file of the kitchen scene.')
    parser.add_argument('--timestampdinner', help='Path to time stamp file of the dinner scene.')
    
    parser.add_argument('--optitrackJSON', help='Path to mocap JSON file of the opti track motion capture.')
    parser.add_argument('--optitrackCSV', help='Path to csv file of the opti-track motion capture.')
    
    
    arg = parser.parse_args()
    
    
    
    
    
    
    #arg ='/home/tanbir/MSC_THESIS/Thesis_code/TEST_PERFORMANCE/converted_video/hand_r6/video.2.190207121121.video.mp4'
    
    starter = coordinate()   #initializing the class parameters
    
    starter.initiator(arg)  # activating the inputs for yolo
    



