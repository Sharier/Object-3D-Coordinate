import numpy as np
import pandas as pd
import json
import os
from numpy import diff



#Function description
#Takes data frame as input
#Data format  is explained below

#Return data: [5337672.796403946,5337686.582796048,5337672.793413234,5337686.582761481,'Bowl6']
#First index gives the starting time of the mocap relative to csv of scene
#Second index gives the end time of the mocap relative to csv of scene
#Third index gives the starting time of the csv of scene
#Fourth index gives the end time of the csv of scene



def start_end(frame,opti_frame):
    #print(frame)
    count=0
    info=[]
    dic_key=[]
    
    for col_name in frame.columns:
        
        if 'x' in col_name:
            
        
             #gets the length of data avalibale for each column  
            
            nan_count=frame[col_name].isnull().sum(axis = 0)   #get the number of nan in each col
            val_count= len(frame[col_name])
            
            data_available= val_count-nan_count      #### gets the length of non 'nan' value available
            
            bool_frame=frame[col_name].isnull()
            
            for time,value in bool_frame.iteritems():
        
                if value!=True:
                   
                    csv_start=time
                    start_point=frame.index.get_loc(csv_start)
 
                    end_index=start_point+data_available    
                    csv_end=frame.index[end_index-1]
        
                    #print(col_name,value,csv_start,csv_end)
               
                    #get the specific class name
                    temp_name=col_name.split('_')
                    temp_name=temp_name[0]     
                    
                    #gets the closest start time of the opti track frame relative to kitchen or dinner csv
                    l=csv_start
                    lst=list(opti_frame.index)
                    opti_start= min(lst, key=lambda x:abs(x-l))
                    
                                        
                    ##gets the closest end time of the opti track frame relative to kitchen or dinner csv
                    l=csv_end
                    opti_end= min(lst, key=lambda x:abs(x-l))
                    
                    d=[opti_start,opti_end,csv_start,csv_end,temp_name]
                    break
            
            dic_key.append(temp_name)
            info.append(d)
            temp_count=0
    
    
    
    return info,dic_key



#find the rising time of all scene
def turningpoints(x,t):
   # print(x,t)
    x=np.gradient(x,t)
    return x



def turning_time(scene,frame,mocap_frame,start_scene,end_scene,id_track,end_mocap):
    
    idx= frame.index

    start_point= idx.get_loc(start_scene)
    end_point= idx.get_loc(end_scene)
    print(start_point)

    point_lst=frame.iloc[start_point:end_point][[id_track+'_x']].values
    point_lst=list(point_lst.reshape(len(point_lst),))

    time=list(frame.index[start_point:end_point].values)
    
    
    a=turningpoints(point_lst,time)
    a=list(a)
    if scene==0: ###kitchen scene
        
        for idx,j in enumerate(a):
            if j>0: #rising slope
                print(idx,j)
                break
    else:
        
        for idx,j in enumerate(a):
            if j<0:   #decreasing slope
                print(idx,j)
                break

        
    turnpoint=start_point+idx
    turnpoint
    turn_x=frame.iloc[turnpoint][[id_track+'_x']].values   ###value at the turning point
    
    turntime_scene=frame.index[turnpoint]   #time of the turining point for kitchen\dinner
    
    #print('prinitng turing point scene x:',turn_x,'\t','prinitng turing point scene time:',turntime_scene)

    
    l=turntime_scene
    timelst=mocap_frame.index.values
    match= min(timelst, key=lambda x:abs(x-l)) #kitc_x is updated with nearest match from the opti frame
    print('printing the match time',match)
    
    #find the rising time of all mocap

    start_mocappoint= mocap_frame.index.get_loc(match)
    
    idx_time= mocap_frame.index[start_mocappoint:]
    
    end_mocappoint= idx_time.get_loc(end_mocap)
    
    temp_minval=0
    
    for col in mocap_frame.columns:
         
        if 'X' in col:    #checking only the x columns
        
            point_lst=mocap_frame.iloc[start_mocappoint:][[col]].values
            point_lst=list(point_lst.reshape(len(point_lst),))

            time=list(idx_time)
            
            g=turningpoints(point_lst,time)
            print('printing g ',g)
    
            if scene==0: #kitchen
                for idx,j in enumerate(g):
            
                    if j>0: #rising slop
                        print('found positive',idx,j)
                        break
        
                turnpoint=start_mocappoint+idx
                mocap_x=mocap_frame.iloc[turnpoint].values
                min_val= min(mocap_x)

        
                if min_val<temp_minval:
                    temp_turnpoint=turnpoint
                    temp_minval=min_val
                    temp_mocapx=mocap_x
                    label=col
                    
            else:#dinner
            
                for idx,j in enumerate(g):
            
                    if j<0: #decreasing slop
                        #print(idx,j)
                        break
        
                turnpoint=start_mocappoint+idx
                mocap_x=mocap_frame.iloc[turnpoint].values
                min_val= max(mocap_x)

        
                if min_val>temp_minval:
                    temp_turnpoint=turnpoint
                    temp_minval=min_val
                    temp_mocapx=mocap_x
                    label=col
    
    
    #for idx,val in enumerate(temp_mocapx):
    #    if val==temp_minval:
    #        label=mocap_frame.columns[idx]
    #        print('matched opti label:', label)
        

    turntime_mocap=mocap_frame.index[temp_turnpoint]
    
    print('prinitng turing point tracker frame x:',turn_x,'\t','prinitng turing point mocap x:',temp_minval,'\n')
    
    print('prinitng turing point tracker frame:',turntime_scene,'\t','prinitng turing point mocap time:',turntime_mocap,'\n')
    print('printing turing point matched label with kitchen label :',label,id_track)
    
    return turn_x,turntime_scene,temp_minval,turntime_mocap,label
    


def till_end(frame):
    
    
    dinner_endframe=frame
    
    ###############this gets the instance or point when the object is placed on the table##############
    
    ##value that needs to be matched, tracker marker value at the corner of the kitchen table
    table_top=-1.675380
    table_bottom=-1.010229

    on_table=[]  #variable stores all the values within the given range
    track=0
    
    for i in dinner_endframe.columns:
        
        if 'X' in i:  #only checks the x coordinate of the mocap data
            
            for val in dinner_endframe[i]:
               # print('column val',val,i)
                
                if val>table_bottom:
                    if track !=0:
                        break
                
                if table_top<val<table_bottom:
                    
                    track=track+1
                    on_table.append(val) #stores all the closest/nearest values in all the coloums of X coordinate 

    #print('on table',on_table)        
    x_table=min(on_table) ##value that needs to be matched 
    #print(x_table)
    
    #####################################getting the data till the end of time using the last coordinate on kitchen table
    
    for time,col_val in dinner_endframe.iterrows():
        
        
        for col_count, value_col in enumerate(col_val):
                
            
            if value_col==x_table:# the position values of opti track markers on the kitchen table
                
                matched_label=dinner_endframe.columns[col_count]  #get the matched label 
                matched_Xlabel=matched_label
                
                matched_label=matched_label.split('X')
                matched_Ylabel=matched_label[0]+'Z'
                matched_Zlabel=matched_label[0]+'Y'
                hand_label=matched_label[0]

                
                end_time=time
                
                tilltable_frame=dinner_endframe.loc[:end_time][[matched_Xlabel,matched_Ylabel,matched_Zlabel]]
                    
                tilltable_data= tilltable_frame.values
                
                
                ###############getting data towards the kitchen table from the moment of no tracking#######
                    
                #first adding a time coloumn to the data
                d=np.array(tilltable_frame.index)
                d=d.reshape((len(d),1))
    
                tilltable_data=np.hstack((d,tilltable_data))
                #print(tilltable_data.shape)
                
        
                ######adding z axis to the data
                        
                #din_z=dinner_endframe.loc[start_time][matched_Zlabel]  #z value when the object on hand
                            
                #z_data=np.full(len(tilltable_frame.index),din_z)
                #z_data=z_data.reshape((len(z_data),1))
                #tilltable_data=np.hstack((tilltable_data,z_data))
                #tilltable_data=tilltable_data.astype(np.float64)
                #print(tilltable_data.shape)
                #second adding a hand coloumn to the data###############
       
                dummy_data=np.full(len(tilltable_frame.index),hand_label)
            
                dummy_data=dummy_data.reshape((len(dummy_data),1))
                
                tilltable_data=np.hstack((tilltable_data,dummy_data))
                #print(tilltable_data.shape)
                
                
                ###########getting till the end of opti track frame######################
                
                
                end_timeframe=dinner_endframe.loc[end_time:][[matched_Xlabel,matched_Ylabel]]
                
                #creating the last values with similar x and y values till the end of time
                
                time_index=np.array(end_timeframe.index)
                time_index=time_index.reshape((len(time_index),1))
    
                
                x_data=np.full(len(end_timeframe.index),value_col)
                x_data=x_data.reshape((len(x_data),1))
                
                y_data=np.full(len(end_timeframe.index),col_val[col_count+1])
                y_data=y_data.reshape((len(y_data),1))
                
                last_data= np.hstack((time_index,x_data))
                last_data=np.hstack((last_data,y_data))
                
                ######adding z axis to the data
                        
                table_z=0.767437  #z value when the marker is placed on the table
                            
                z_tabledata=np.full(len(end_timeframe.index),table_z)
                z_tabledata=z_tabledata.reshape((len(z_tabledata),1))
                
                last_data=np.hstack((last_data,z_tabledata))
                #last_data=last_data.astype(np.float64)
        
                #second adding a hand coloumn to the data
                label='On kitchen table'
       
                hand_data=np.full(len(end_timeframe.index),label)
            
                hand_data=hand_data.reshape((len(hand_data),1))
                
                last_data=np.hstack((last_data,hand_data))
                #print(last_data.shape)
                
                
                #####finally adding both the data,going to the kitchen table and on kitchen table till end of time
                
                
                final_data=np.vstack((tilltable_data,last_data))
                
                return final_data
               

                
                
                



def sample_positions(object_positions, srate):
    
        
    def linearinterpol(t_s, x_1, x_2,t_1,t_2):
        m=(x_2-x_1)/(t_2-t_1)
    
        return m*(t_s-t_1)+x_1    


    
    
    conc_data=np.array([])

    #print(object_positions)
    object_num= {k : v[:,:] for k,v in object_positions.items()}
    tss = [v[:, 0].astype(float) for v in object_num.values()]
    
    first = max([ts[0] for ts in tss])
    
    last = min([ts[-1] for ts in tss])
    #print('printing details',first, last, 1/srate)
    
    sample_points = np.arange(first, last, 1/srate)
    sample_points = sample_points[1:]

    #hand_info={s: b[:,:3].astype(np.string_) for s,b in object_positions

    
    res = {}  
    for name, pos in object_num.items():
    
        _,_,_,_,h=pos.T
        t,x,y,z,_ = pos.T
        x=x.astype(float);y=y.astype(float);t=t.astype(float);z=z.astype(float)
        xs = []; ys = []; zs=[]; ts=[]; hs=[]
        print('do', name,x.shape)
        
        for s in sample_points:
            idx = sum((t-s) < 0)
            #print('printing first last s idx',first, last, s, idx)
            #print((t-s) < 0)
            #print(x[idx-1], x[idx])
            xs.append((x[idx-1], x[idx]))
            ys.append((y[idx-1], y[idx]))
            ts.append((t[idx-1],t[idx]))
            zs.append((z[idx-1],z[idx]))
            
            hs.append((h[idx-1]))
        
        #print('printing xs',xs)
        x_1, x_2 = np.array(xs).T
        y_1, y_2 = np.array(ys).T
        t_1, t_2 = np.array(ts).T
        z_1, z_2 = np.array(zs).T
        h= np.array(hs).T
        

        
        test_t1=t_1
        test_t2=t_2
        
        x_interp = linearinterpol(sample_points, x_1, x_2,test_t1,test_t2)
        y_interp = linearinterpol(sample_points, y_1, y_2,test_t1,test_t2)
        z_interp = linearinterpol(sample_points, z_1, z_2,test_t1,test_t2)
        data= np.array([x_interp, y_interp,z_interp, h],dtype='O')
        
        if conc_data.size==0:
            conc_data=data
        else:
            conc_data=np.vstack((conc_data,data))
        
        
        #print('printinggggggggg shape',data.shape)

        #print('printing x shape',x_interp.shape,'printing hand_info',hand_info.shape)
        #res[name] = np.array([sample_points, x_interp, y_interp,z_interp, h],dtype='O')
    time_data=sample_points.T
    main_data=conc_data.T   
    
    return time_data,main_data





def initiator(kit_frame,din_frame,mocap_frame,srate):
    
    def frame_correction(frame):
        t_m=frame.index.values
        new_t=[]

        for s in t_m:
            s=s-8
            new_t.append(s)
        new_t=np.array(new_t)
        new_t=new_t.T
        kit_val=frame.values    
        kit_val.shape
        new_frame=pd.DataFrame(kit_val,index=new_t,columns=frame.columns)
        return new_frame
    
    global test
    main_dic={}
    main_label=[]
    #opti_frame=opti
    srate=srate

    new_kitframe=frame_correction(kit_frame)
    new_dinframe=frame_correction(din_frame)
    #new_kitframe=sampling_frame(kit_frame,150)
    #new_dinframe=sampling_frame(din_frame,150)

    kitchen_info,kit_keys=start_end(new_kitframe,mocap_frame)
    dinner_info,din_key=start_end(new_dinframe,mocap_frame)
    
    ######creating a dictionary with all the keys
    for i in kitchen_info:
        main_dic[i[4]]=[]
        final_col=[i[4]+'_x',i[4]+'_y',i[4]+'_z',i[4]+'_location']
        main_label.append(final_col)
    
    main_label=np.array(main_label)
    main_label=main_label.reshape((len(main_label)*4),)


    
    
    for i in kitchen_info:
        
        scene=0
        #mocap_frame=mocap_frame
        start_scene=i[2];end_scene=i[3];label=i[4];start_opti=i[0];end_opti=i[1]

        kit_turnx,kit_turntime,mocap_turnx,mocap_turntime,matched_label=turning_time(scene,new_kitframe,mocap_frame,start_scene,end_scene,label,end_opti)
        
        

        class_label_x= i[4]+'_x'
        class_label_y= i[4]+'_y'

        start_x= new_kitframe.loc[i[2]][class_label_x]
        start_y= new_kitframe.loc[i[2]][class_label_y]
        start_z=0.767437   #on kitchen table
        
        start_optipoint =mocap_frame.index.get_loc(start_opti)


        start_endtime=mocap_frame.index.get_loc(mocap_turntime)
        print('end time of object on table at kitchen scene',start_endtime)

        s_length= len(mocap_frame.index[start_optipoint:start_endtime])
    
        start_opti_time=mocap_frame.index[0:start_optipoint+s_length].values
        
        start_length=len(mocap_frame.index[0:start_optipoint+s_length])
        
        
        start_opti_time=start_opti_time.reshape((len(start_opti_time),1))#...............time

#print(start_length)

        x_data= np.full(start_length,start_x)
        x_data=x_data.reshape((len(x_data),1))#...............x
    
        y_data= np.full(start_length,start_y)
        y_data= y_data.reshape((len(y_data),1))#..............y
    
        z_data= np.full(start_length,start_z)
        z_data= z_data.reshape((len(z_data),1))#..............z
    
        loc_data= np.full(start_length,'On kitchen table')
        loc_data=loc_data.reshape((len(loc_data),1))#.............location
    
        start_data=np.hstack((start_opti_time,x_data))
        start_data=np.hstack((start_data,y_data))
        start_data=np.hstack((start_data,z_data))
        start_data=np.hstack((start_data,loc_data))
    
###############for each class from starting of kitchen to end of kitchen####
        din_timelst=[]
        for idx,val in enumerate(dinner_info):
            din_timelst.append(val[2])  #saving the relative starting time of dinner scene to list 
        compare=mocap_turntime
        closest_time= min(din_timelst, key=lambda x:abs(x-compare))
        
        for val in dinner_info:
            if val[2]==closest_time:
                print('closest tiime,kit,din,',label,val[4],val[2])
                dinlabel=val[4]
                scene=1
                start_scene=val[2];end_scene=val[3];label=val[4];end_opti=val[1];start_opti=val[0]
                din_turnx,din_turntime,mocap_dinturnx,mocap_dinturntime,dinmatched_label=turning_time(scene,new_dinframe,mocap_frame,start_scene,end_scene,label,end_opti)
            else:
                pass
            
        mid_starttime=mocap_turntime
        mid_startpoint=mocap_frame.index.get_loc(mid_starttime)

        matched_namelabel=matched_label.split('X')
        matched_Xlabel=matched_namelabel[0]+'X'
        matched_Ylabel=matched_namelabel[0]+'Z'
        matched_Zlabel=matched_namelabel[0]+'Y'


#find mid_end time:
        #for time,col in mocap_frame[mid_starttime:][[matched_Xlabel,matched_Ylabel,matched_Zlabel]].iterrows():    
        #    for val in col:
        #        if val==mocap_dinturnx:
            
        mid_endtime=start_opti
        mid_endpoint=mocap_frame.index.get_loc(mid_endtime)
            
        mid_time=mocap_frame.index[mid_startpoint:mid_endpoint+1].values
            
        mid_data=mid_time.reshape((len(mid_time),1))
            
        opt_data=mocap_frame[mid_starttime:mid_endtime][[matched_Xlabel,matched_Ylabel,matched_Zlabel]].values
            
        mid_data=np.hstack((mid_data,opt_data))
            
        mid_loc=matched_namelabel[0]
            
        hand_data= np.full(len(mid_time),mid_loc)
        hand_data=hand_data.reshape((len(hand_data),1))
        mid_data=np.hstack((mid_data,hand_data))
            
            #############on dinner table
            
                    
        din_endpoint=mocap_frame.index.get_loc(end_opti)

        din_time=mocap_frame.index[mid_endpoint:din_endpoint].values
        din_data=din_time.reshape((len(din_time),1))#...............time

            
        din_x= din_turnx
        din_y= new_dinframe.loc[din_turntime][label+'_y']
        din_z=0.767437
        din_loc='on dinner table'
            
    
        dinx_data= np.full(len(din_time),din_x)
        dinx_data=dinx_data.reshape((len(dinx_data),1))#...............x
    
        diny_data= np.full(len(din_time),din_y)
        diny_data= diny_data.reshape((len(diny_data),1))#..............y
    
        dinz_data= np.full(len(din_time),din_z)
        dinz_data= dinz_data.reshape((len(dinz_data),1))#..............z
        
        dinloc_data= np.full(len(din_time),din_loc)
        dinloc_data= dinloc_data.reshape((len(dinloc_data),1))#.............location
    
        din_data=np.hstack((din_data,dinx_data))
        din_data=np.hstack((din_data,diny_data))
        din_data=np.hstack((din_data,dinz_data))
        din_data=np.hstack((din_data,dinloc_data))

            

        dinmatched_namelabel=dinmatched_label.split('X')
        dinmatched_Xlabel=dinmatched_namelabel[0]+'X'
        dinmatched_Ylabel=dinmatched_namelabel[0]+'Z'
        dinmatched_Zlabel=dinmatched_namelabel[0]+'Y'
            
 #find till object on kitchen table time
        till_kitchen =mocap_frame[end_opti:][[dinmatched_Xlabel,dinmatched_Ylabel,dinmatched_Zlabel]] 

        last_data= till_end(till_kitchen)
        
        
        final_data=start_data
        final_data=np.vstack((final_data,mid_data))
        final_data=np.vstack((final_data,din_data))
        final_data=np.vstack((final_data,last_data))

        
        main_dic[i[4]]=final_data
        #print('printing final data',i[4],final_data)
        
        if i[4]=='Bowl1':
            test=start_data



    print('printing main dic:',main_dic)
    index_time,converted_data=sample_positions(main_dic,srate)
    
    #print('kit_info', kitchen_info)
    #print('din_info',dinner_info)

    main_frame=pd.DataFrame(converted_data,index=index_time,columns=main_label)
    #print(len(main_label))    
    
    return main_frame


#reading the  mocap data
def mocap_read(d,csv):
    
    with open(d) as f1:
        meta = json.load(f1)
    channels = meta['info']["desc"]["channels"]
    original_labels = [d['label'] for d in channels["channel"]]
    labels=original_labels
    
    #####################declare only if an empty coloumn is available#######
    labels=['ts',*labels,'empty']

#labels=['ts',*labels]

    opti_fullframe= pd.read_csv(csv,names=labels, index_col=0)

########################getting only the required coloumns and time frame from the whole panda frame
    #req_col= ['Hand_X','Hand_Z','Hand_Y']
    req_col=['bowl_X','bowl_Y','bowl_Z']
    #################name of the coloumns that needs to be checked########
    #req_col= ['tanbir-left-hand_X','tanbir-left-hand_Z','tanbir-left-hand_Y','tanbir-right-hand_X','tanbir-right-hand_Z','tanbir-right-hand_Y']

    opti_frame= opti_fullframe[req_col]
    
    return opti_frame

