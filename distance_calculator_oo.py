import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from draw_poligons import PolygonDrawer
import pandas as pd
import os



txtname="D:/2_UNAM/INB/D11/IdentificadorRata/prueba_yolov8/predictions/S4D4R5_15step_labels.txt"
videoname='D:/2_UNAM/INB/D11/IdentificadorRata/videos/SpeedUp/S4D4R5_15step.mp4'
prefix="S4D4R5_15step_"

class VideoAdq(): #working with video
    def __init__(self,filename: str,show_ima: bool,is_scaled: bool,scale):
        self.filename=filename #video name
        self.scale=scale #scale percentage
        self.show_ima=show_ima
        self.is_scaled=is_scaled #if we want to scale
        self.cap=cv2.VideoCapture(videoname)#open the video
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS) #fps of the video
        self.FRAME_COUNT = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = (self.FRAME_COUNT/self.FPS)/60
        self.h=0
        self.w=0
        if self.cap.isOpened()==False:
            print("Error opening video stream or file")
    
    def getFrame(self): #Display a random frame
        rand_id=random.sample(range(0,self.FRAME_COUNT),1) #random number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, rand_id[0]) # set the frame
        ret, self.frame = self.cap.read() #read the image
        
        if self.is_scaled==True: # if we eant to scale
            self.h,self.w,_ = self.frame.shape #height and weight
            width = int(self.h * self.scale) #new height and weight
            height = int(self.w * self.scale)
            self.frame = cv2.resize(self.frame, (height,width), interpolation = cv2.INTER_AREA) #re-shaping
        if ret==True: #if the frame is available
            if self.show_ima==True:
                cv2.imshow('Frame',self.frame)
                cv2.waitKey() #wait for any key
                cv2.destroyAllWindows()
        

class DistanceCalculator(object):
    def __init__(self,filename: str,prefix,real_w:float, w: float, h: float, inner_points,outer_points):
        #Run validations for arguments
        #Specify param characteristics
        assert w>0, f"The weight w {w} must be positive." 
        assert h>0, f"The height h {w} must be positive."
        
        #Asigning to self object
        self.filename=filename
        self.prefix=prefix
        self.real_w=real_w
        self.w=w
        self.h=h
        self.x_total=[]
        self.y_total=[]
        self.x_cen=[]
        self.y_cen=[]
        self.dist_acum=[0]
        self.dist_total=[]
        self.inner_points=inner_points
        self.outer_points=outer_points
        self.center_time=0
        self.center_limits=[0,0,0,0]
        self.resting=0
        self.dataframe=pd.DataFrame({"Video":[],"Dist Total (m)":[],"Velocidad (m/min)":[],"Reposo (%)":[]})
        
     
    def getFileData(self):
        file=open(self.filename,"r") #open the labels file
        data=file.read().split("\n") #split in rows
        for line in data:
            label=line.split() #split in columns
            x=[]
            y=[]
            for n in range(1,int((len(label)-1)),2):
                x.append(float(label[n]))#x in odd number of each row
                y.append(float(label[n+1])) #y in even number of each row
            self.x_total.append(np.array(x)) #append all the x coordinates
            self.y_total.append(np.array(y)) #append all the y coordinates
        self.x_total=self.x_total[0:len(self.x_total)-1]
        self.y_total=self.y_total[0:len(self.y_total)-1]
        return self.x_total,self.y_total #size=original label (normalized)
                
    def centroid(self):
        self.x_total,self.y_total=DistanceCalculator.getFileData(self)
        #outer box bordeline
        lim_izq_x=self.outer_points[0][0]/self.h #division into the x distance
        lim_der_x=self.outer_points[1][0]/self.h 
        lim_sup_y=self.outer_points[1][1]/self.w #division into the y distance
        lim_inf_y=self.outer_points[2][1]/self.w
        
        for x,y in zip(self.x_total,self.y_total):            
            #just connsidering centers into the box
            if lim_izq_x<sum(x)/len(x)<lim_der_x and lim_sup_y<sum(y)/len(y)<lim_inf_y:
                self.x_cen.append(sum(x)/len(x))#calculate the x axis centroids 
                self.y_cen.append(sum(y)/len(y)) #calculate the y axis centroids
        self.x_cen=(np.array(self.x_cen)*self.w)/self.real_w#normalize -> pixels -> meters
        self.y_cen=(np.array(self.y_cen)*self.h)/self.real_w
        return self.x_cen,self.y_cen #real world (meters)

    def distance(self): #real world distances
        self.x_cen,self.y_cen=DistanceCalculator.centroid(self)
        for n in range(len(self.x_cen)-1,1,-1): 
            #distance between two points
            dist=np.sqrt((self.x_cen[n]-self.x_cen[n-1])**2+(self.y_cen[n]-self.y_cen[n-1])**2)
            self.dist_total.append(dist)
            self.dist_acum.append(self.dist_acum[-1]+dist)
        self.dist_total=sum(self.dist_total)
        return self.dist_total,self.dist_acum # real world distance (meters)
    
    def areaTime(self):
        self.center_limits[0]=self.inner_points[0][0]*1.25/self.real_w #division into the x distance
        self.center_limits[1]=self.inner_points[1][0]*0.75/self.real_w 
        self.center_limits[2]=self.inner_points[0][1]*1.1/self.real_w #division into the y distance
        self.center_limits[3]=self.inner_points[2][1]*0.9/self.real_w
        count=0
        for x, y in zip(self.x_cen,self.y_cen):
            if self.center_limits[0]<x<self.center_limits[1] and self.center_limits[2]<y<self.center_limits[3]:
                count=count+1
        self.center_time=(count/len(self.x_cen))*100
    
    def restTime(self):
        count=0
        for n in range(len(self.x_cen)-1):
            x=np.abs(self.x_cen[n]-self.x_cen[n+1])
            y=np.abs(self.y_cen[n]-self.y_cen[n+1])
            if x<0.01 and y<0.01:
                count=count+1
        self.resting=count/len(self.x_cen)*100
            
    def saveData(self):
        DistanceCalculator.areaTime(self)
        DistanceCalculator.restTime(self)
        self.dataframe.loc[len(self.dataframe.index)]=[self.prefix,self.dist_total,self.dist_total/60,self.resting]
        name="OpenFieldSeg.csv"
        path="D:/2_UNAM/INB/D11/IdentificadorRata/prueba_yolov8/predictions/Results"
        csvname=path+"/"+name
        header=True
        if os.path.isfile(csvname)==True:
            header=False 
        self.dataframe.to_csv(csvname, mode='a',index=False,header=header)
        return self.dataframe
 
class MotorGraphs(object):
    def __init__(self,filename,x_cen,y_cen,dist_acum, prefix,center):
        self.filename=filename
        self.x_cen=x_cen #idk why x and y are inverted but it works
        self.y_cen=y_cen #real world (m)   
        self.dist_acum=dist_acum
        self.prefix=prefix
    
    def accumulatedDistance(self):
        plt.figure()
        plt.plot(self.dist_acum)
        plt.title(self.prefix+"__Accumulated Distance")
        plt.savefig("D:/2_UNAM/INB/D11/IdentificadorRata/prueba_yolov8/predictions/Results/"+self.prefix+"__AccumulatedDistance.jpg")
    
    def tracking_lines(self):
        self.points=[]
        self.x_cen=(self.x_cen)/np.max(self.x_cen) # real world --> [0,1]
        self.y_cen=((self.y_cen)/np.max(self.y_cen))/2  # real world --> [0,0.5]
        fig,ax=plt.subplots()
        ax.plot(self.x_cen, self.y_cen) #graph the lines
        ax.set_title(self.prefix+"__Tracking")
        
        for x,y in zip(self.x_cen,self.y_cen): #get the centers
            circle=plt.Circle((x,y),0.005,color="r") #graph a circle x center
            ax.add_patch(circle)
        plt.gca().invert_yaxis()
        plt.savefig("D:/2_UNAM/INB/D11/IdentificadorRata/prueba_yolov8/predictions/Results/"+self.prefix+"__Tracking.jpg")
        plt.show()
        
        
    def heat_map(self):
        self.cap = cv2.VideoCapture(self.filename)
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()#filter to substract the background
        first_iteration_indicator = 1
        for i in range(0,int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),5):
            ret, frame = self.cap.read()
            h,w,_ = frame.shape
            if (first_iteration_indicator == 1):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape[:2]
                accum_image = np.zeros((height, width), np.uint8)#void image
                first_iteration_indicator = 0
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
                fgmask = fgbg.apply(gray)  # remove the background
                thresh = 2
                maxValue = 3# the biggest the noisier
                ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)#filter the noise
                accum_image = cv2.add(accum_image, th1)
        
        
        color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)# apply a color map
        color_image = cv2.resize(color_image, (int(w*0.5),int(h*0.5)), interpolation = cv2.INTER_AREA)# re size
        cv2.imshow("HeatMap",color_image)
        cv2.imwrite("D:/2_UNAM/INB/D11/IdentificadorRata/prueba_yolov8/predictions/Results/"+self.prefix+"__HeatMap.jpg", color_image)
        # cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        


if __name__ == "__main__":
    #create a video object
    vid=VideoAdq(videoname,False,True,0.5)
    
    #get an random frame
    vid.getFrame() 
    
    #image to select the base of the box
    poly = PolygonDrawer("Polygon",vid.frame,0.5,(255, 255, 255))
    image,inner_points = poly.run() 
    #image to select the outer bordeline of the box
    poly = PolygonDrawer("Polygon",vid.frame,0.5,(255, 255, 0))
    image,outer_points= poly.run()
    box_w=np.abs(inner_points[3][0]-inner_points[2][0]) #box weight = 1 m

    #Calculate total, accumulated distances, centroids 
    Dist=DistanceCalculator(txtname,prefix,box_w,vid.h,vid.w,inner_points,outer_points)
    dist_total,dist_acum=Dist.distance()
    Dist.saveData()
        
    #Graph the rat's tracking
    graphs=MotorGraphs(videoname,Dist.x_cen,Dist.y_cen,dist_acum,prefix,Dist.center_limits)
    graphs.accumulatedDistance()
    graphs.tracking_lines()
    graphs.heat_map()#heat_map
   
###
#Graficar los caminos
#graficar mapas de calor

