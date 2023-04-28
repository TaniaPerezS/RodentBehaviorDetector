""" OPEN FIELD RAT DETECTOR MODULES
Sexual Behavior and Plasticity Laboratory, Neurobiology Institute, UNAM.
@autor: Tania Pérez
@Version: 1.0
@Date: April, 2023

This file contains all the classes and functions needed for the
post-processing of the rat segmentation in Google Colab.

Classes:
    -PolygonDrawer: Colab Interface to select the box
    -Distance Calculator: Coordinates and distances calculation
    -MotorGraphs: Graphs of the tracking and heat maps.
    
Functions:
    -createLabel: Save all label txt in a single txt file
    -postProcessing: Join all the clases in a main function
    
"""
#Dependencies installation
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from google.colab.output import eval_js
import os
import pandas as pd
import glob


"""
class PolygonDrawer:
    Methods:
        -prepareVideo: Use when extract a frame from a video
        -preprareImage: Use when you are working with a single image
        -drawPoints: Draw the point over the image and whe the coordinates
        -getPoints: Standarize coordinates
    Parameters:
        -filename: Name of the image or video for the background
        -real_box_w: The width in meters of the box   
    Returns:
        -data: [x,y] coordinates of the points (original image size)
"""

class PolygonDrawer(object):
  def __init__(self,filename,real_box_w):
    self.filename=filename
    self.real_box_w=real_box_w
    self.w=0
    self.h=0
    self.data=0
    self.new_name="/usr/local/share/jupyter/nbextensions/background.jpg"

  def prepareVideo(self): #Save a frame in nbextensions
    cap=cv2.VideoCapture(self.filename)#open the video
    if cap.isOpened()==False:
                print("Error opening video stream or file")
    cap.set(cv2.CAP_PROP_POS_FRAMES,50)#50=random frame
    ret,frame = cap.read()
    if ret==True: #if the frame is available 
      self.h,self.w,_=frame.shape
      cv2.imwrite(self.new_name,frame)#write in the new path

  def prepareImage(self): #Create a copy in nbextensions
    shutil.copy(self.filename, self.new_name)
    img = cv2.imread(self.new_name, cv2.IMREAD_COLOR)
    self.h,self.w,_=img.shape

  def drawPoints(self):
    js_code = '''
    <style>
      body {
        background-image: url('/nbextensions/background.jpg');
        background-color:rgb(255, 255, 255);
        background-repeat: no-repeat;
        background-size: 700px;
        }
    </style>

    <body>
        <canvas id="canvas" width="700" height="350"></canvas>
        <div>
          <button id="Finish">Finish</button>
        </div>

    <script >
      var canvas = document.querySelector('canvas')
      var lastClick = [0, 0];
      var currentClick=[];
      var button = document.querySelector('button')

      document.getElementById('canvas').addEventListener('click', drawCircle, false);

      function getCursorPosition(e) {
          var x;
          var y;
          if (e.pageX != undefined && e.pageY != undefined) {
              x = e.pageX;
              y = e.pageY;
          } else {
              x = e.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
              y = e.clientY + document.body.scrollTop + document.documentElement.scrollTop;
          }
          
          return [x, y];
      }
      function drawCircle(e) {
        context = this.getContext('2d');
        x = getCursorPosition(e)[0] - this.offsetLeft;
        y = getCursorPosition(e)[1] - this.offsetTop;

        context.arc(x , y ,5, 0, Math.PI * 2,false);
        context.stroke();
        lastClick = [x, y];
        currentClick=currentClick.concat(lastClick)
    }

      var data = new Promise(resolve=>{
          button.onclick = ()=>{
            resolve(currentClick)
          }
        })
    </script>
    </body>
    '''
    display(HTML(js_code)) #Display image
    self.data = eval_js("data") #Read data from JS
    
  def getPoints(self):#transforms the interface sized coordinates to the original image size
    PolygonDrawer.drawPoints(self)
    self.data=np.reshape(self.data,(int(len(self.data)/2),2))#shape in a 4x2 array
    self.data=np.array(self.data)
    self.data=np.round(self.w*self.data/600) #600= Width of the background image (cte)
    return self.data

"""
class dirFileProcessor:
    Methods:
        -checkDir: check if a folder existis, if not create it.
        -getTxt: get the txt filename using the videoname
    Parameters:
        -videoname: path and name of the video
    Returns:
        None
"""   

class dirFileProcessor(object):
  def __init__(self,videoname):
    self.videoname=videoname
    self.new_path=""
    self.txtname=""
    self.path=""
    self.prefix=""
  
  def getTxt(self):
    videoname=self.videoname.split("/")[-1]
    self.path=self.videoname.replace(videoname,"")
    self.prefix=videoname.split(".")[0]
    self.txtname=self.path+self.prefix+"_labels.txt"
    self.new_path=self.path+"Results/"


  def checkDir(self):
    if os.path.isdir(self.new_path)==False:
      os.mkdir(self.new_path)

"""    
class DistanceCalculator:
    Methods:
        -getFileData: Obtain the coordinates of each segmentation from the txt
        -centroid: Get the centroids of each segmentation (meters)
        -distance: Get the accumulated distance and the total distance (meters)
        -restTime: Calculate the time porcentaje when the rat only moves 1 cm
        -saveData: Save the results in a csv
    Parameters:
        -filename: Name of the image or video for the background
        -w: Width of the box (original image pixels)
        -h: Height of the box (original image pixels)
        -points: Corners of the box (standarized numeration)
        -ima_name: Title displayed in the graphs
    Returns:
        -dist_total: Total meters walked by the rat
        -dist_accum: Meters walked by frame
"""

#Read the data from labels. Calculate distances and centroids
class DistanceCalculator(object):
    def __init__(self,filename: str,prefix,new_path,real_w:float, w: float, h: float, points):
        #Run validations for arguments
        #Specify param characteristics
        assert w>0, f"The weight w {w} must be positive." 
        assert h>0, f"The height h {w} must be positive."
        
        #Asigning to self object
        self.filename=filename
        self.prefix=prefix
        self.new_path=new_path
        self.real_w=real_w
        self.w=w
        self.h=h
        self.x_total=[]
        self.y_total=[]
        self.x_cen=[]
        self.y_cen=[]
        self.dist_acum=[0]
        self.dist_total=[]
        self.points=points
        self.resting1cm=0
        self.resting5cm=0
        self.dataframe=pd.DataFrame({"Video":[],"Dist Total (m)":[],
        "Velocidad (m/min)":[],"Reposo1cm (%)":[]})

     
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
        lim_izq_x=self.points[5][0]/self.w #division into the x distance
        lim_der_x=self.points[6][0]/self.w 
        lim_sup_y=self.points[6][1]/self.h #division into the y distance
        lim_inf_y=self.points[4][1]/self.h
        
        for n in range(len(self.x_total)):            
            x=self.x_total[n]
            y=self.y_total[n]
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
    
    def restTime(self):
        count=0
        for n in range(len(self.x_cen)-1):
            x=np.abs(self.x_cen[n]-self.x_cen[n+1])
            y=np.abs(self.y_cen[n]-self.y_cen[n+1])
            if x<10 and y<10:
                count=count+1
              
        self.resting1cm=count/len(self.x_cen)*100
            
    def saveData(self):
        DistanceCalculator.restTime(self)
        self.dataframe.loc[len(self.dataframe.index)]=[self.prefix,self.dist_total/1000,(self.dist_total/1000)/60,self.resting1cm]
        name="OpenFieldSeg.csv"
        csvname=self.new_path+name
        header=True
        if os.path.isfile(csvname)==True:
            header=False 
        self.dataframe.to_csv(csvname, mode='a',index=False,header=header)

"""    
class MotorGraphs:
    Methods:
        -accumulatedDistance: Plot the distance per frame
        -trackingLines: Show the centroid in red and the paths in blue
        -heatMap: Show the most common locations in bright tones
    Parameters:
        -filename: Name of the image or video for the background
        -x_cen: x coordinates of centroids
        -y_cen: y coordinates of centroids
        -dist_acum: accumulated distances vector
        -ima_name: title of the graphs
    Returns:

"""

#Graph the movement figures
class MotorGraphs(object):
    def __init__(self,filename,new_path,x_cen,y_cen,dist_acum,ima_name):
        self.filename=filename
        self.x_cen=x_cen
        self.y_cen=y_cen #real world (m)  
        self.dist_acum=dist_acum 
        self.ima_name=ima_name
        self.path=new_path

    def accumulatedDistance(self):
        plt.figure()
        plt.plot(self.dist_acum)
        plt.title(self.ima_name+"_Accumulated Distance")
        plt.savefig(self.path+self.ima_name+"_AccumulatedDistance.jpg")
        
        
    def trackingLines(self):
        self.points=[]
        self.x_cen=(self.x_cen)/np.max(self.x_cen) # real world --> [0,1]
        self.y_cen=((self.y_cen)/np.max(self.y_cen))/2  # real world --> [0,0.5]
        fig,ax=plt.subplots()
        ax.plot(self.x_cen, self.y_cen) #graph the lines
        ax.set_title(self.ima_name+"_Tracking")
        for x,y in zip(self.x_cen,self.y_cen): #get the centers
            circle=plt.Circle((x,y),0.005,color="r") #graph a circle x center
            ax.add_patch(circle)
        plt.gca().invert_yaxis()
        plt.savefig(self.path+self.ima_name+"_Tracking.jpg")
        plt.show()
        
    def heatMap(self):
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
        cv2_imshow(color_image)
        cv2.imwrite(self.path+self.ima_name+"_HeatMap.jpg", color_image)
        # cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        


"""
function createLabel:
    Parameters:
        -videoname: path and name of the video
    Returns:
        None
"""
# Save all the labels in a single txt file
def createLabel(videoname):
    import glob
    import os
    prep=dirFileProcessor(videoname)# use a class to extract the path, prefix and new
                                    #path from the video info
    prep.getTxt()
    prep.checkDir()
    new_path=prep.path #path of the video
    prefix=prep.prefix #name of the video 
    path="/content/runs/segment/"
    past_path=sorted(glob.glob(os.path.join(path, '*/')), key=os.path.getmtime)[-1]
    past_path=past_path+"labels/"

    num_labels=len(os.listdir(past_path))
    seg_labels = open(new_path+prefix+"_labels.txt", "w")# create/open de writing file
    for n_file in range(1,num_labels):
      label_file=past_path+prefix+"_"+str(n_file)+".txt"
      if os.path.isfile(label_file)==True:#the labels are not
        seg_file = open(label_file, "r") #open the labels
        FileContent = seg_file.read() #read the labels
        seg_labels.write(FileContent) #group the labels in the writing file
    if os.path.isfile(new_path+prefix+"_labels.txt")==True:
        print("Saved as: "+new_path+prefix+"_labels.txt")
    
"""
function postProcessing:
    Parameters:
        -videoname: Name of the video to obstain the frame
        -txtname: Label txt
        -prefix: id of the rat
        -box_width: real width of the box
    Returns:
        None
"""

def postProcessing(videoname,box_width):

    boxDrawer=PolygonDrawer(videoname,box_width)#create PolygonDrawer object
    boxDrawer.prepareVideo() #create the GUI
    points=boxDrawer.getPoints() #save the corners coordinates

    prep=dirFileProcessor(videoname) #check/create directories
                                        #get filenames
    prep.getTxt()
    prep.checkDir()
    Dist=DistanceCalculator(prep.txtname,prep.prefix,prep.new_path,box_width,boxDrawer.w,boxDrawer.h,points)
    dist_total,dist_acum=Dist.distance() #graph the accumulated distance
    Dist.saveData()

    graphs=MotorGraphs(videoname,prep.new_path,Dist.x_cen,Dist.y_cen,dist_acum,prep.prefix)
    graphs.accumulatedDistance()
    graphs.trackingLines() #tracking of the rat
    graphs.heatMap() #heat_map