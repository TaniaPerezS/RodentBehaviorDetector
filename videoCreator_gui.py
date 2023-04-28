import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk
from tkinter import filedialog as fd
import cv2
import random
import numpy as np
from tkinter.messagebox import showinfo
import time

class videoSpeedUp(object):
    def __init__(self):
      self.filenames=[]
      self.value=0        
      self.videoname=""
     
    def open_video(self):
        self.cap=cv2.VideoCapture(self.filenames[0])#open the video
        if self.cap.isOpened()==False:
            print("Error opening video stream or file")
        
        self.FRAME_COUNT = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rand_id=random.sample(range(0,self.FRAME_COUNT),1) #random number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, rand_id[0]) # set the frame
        
        ret, frame = self.cap.read() #read the image
        if ret==True: #if the frame is available
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            frame=Image.fromarray(frame)
            frame = frame.resize((600,400))
            img=ImageTk.PhotoImage(image=frame)
            disp_img.config(image=img)
            disp_img.image = img
    
    def select_files(self):
        filetypes = (
            ('video files', '*.mp4'),
        )

        self.filenames = fd.askopenfilenames(
            title='Open files',
            initialdir='/',
            filetypes=filetypes)
        
        videoSpeedUp.open_video(self)
        

    def speed_up(self):
        start=time.time()
        self.cap=cv2.VideoCapture(self.filenames[0])#open the video
        if self.cap.isOpened()==False:
            print("Error opening video stream or file")
        self.FRAME_COUNT = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.videoname=self.filenames[0].split("/")[-1]
        path=self.filenames[0].replace(self.videoname,"")
        self.prefix=self.videoname.split(".")[0]
        
        new_videoname=path+self.prefix+"_speedUp.mp4"
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        out = cv2.VideoWriter(new_videoname,
                              cv2.VideoWriter_fourcc('F','M','P','4'),
                              30,
                              (frame_width,frame_height))
        percent=frame_percent.get()
        
        for self.frame_id in range(0,int(self.FRAME_COUNT),int(100/percent)):
           self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)
           ret, self.img = self.cap.read()
           if ret == True: 
               out.write(self.img)
        
        end=time.time()
        self.process_time=end-start
        minutes=int(self.process_time/60)
        secs=int(self.process_time - (minutes*60))
        times=str(minutes)+ " min " + str(secs) + " secs."
        showinfo(message=self.prefix+
                 "_speedUp.mp4 processed in "+ 
                 times)
        out.release()
        
        
        
        
BTN_COLOR="#EAEDED"
IMG_COLOR="#5D6D7E"
INFO_COLOR="#F6DDCC"

#Create main window


window = tk.Tk()
window.geometry("800x500")
window.title("SpeedUpEditor")

global buttons_frame,process_time


#Create all the main containers (frames)
buttons_frame=tk.Frame(window,
                       bg=BTN_COLOR,
                       width=200, 
                       height=500, 
                       pady=3
)

image_frame=tk.Frame(window,
                    bg=IMG_COLOR,
                    width=600, 
                    height=500, 
                    pady=3
)

info_frame=tk.Frame(buttons_frame,
                    bg=INFO_COLOR,
                    width=150, 
                    height=150, 
                    pady=3
)

# layout all of the main containers
window.grid_rowconfigure(0,weight=1)
window.grid_columnconfigure(0,weight=1)

buttons_frame.grid_columnconfigure(0,weight=1)

image_frame.grid_rowconfigure(0,weight=1)
image_frame.grid_columnconfigure(0,weight=1)

info_frame.grid_rowconfigure(0,weight=1)
info_frame.grid_columnconfigure(0,weight=1)

buttons_frame.grid(row=0,column=0,sticky="nsew")
image_frame.grid(row=0,column=1,sticky="nsew")
info_frame.grid(row=0,column=0,sticky="nsew")

#Run the commands
vid=videoSpeedUp()

# Create Widgets

# open button
text_info=tk.Label(
    info_frame,
    text="VIDEO SPEED UP CREATOR",
    bg=INFO_COLOR
)

open_button = ttk.Button(
    buttons_frame,
    text='Open Files',
    command=vid.select_files
)

# speed up button
transform_button = ttk.Button(
    buttons_frame,
    text='Transform',
    command=vid.speed_up
)

#text percentage bar
text_percent=tk.Label(
    buttons_frame,
    text="Frames Percentage",
    bg=BTN_COLOR
)

#percentage bar
frame_percent=tk.Scale(
    buttons_frame,from_=5,
    to=50,
    orient=tk.HORIZONTAL,
)

#image label
disp_img = tk.Label(
    window
    )

#Layout widgets
text_info.grid(row=0,column=0,padx=10, pady=10)

open_button.grid(row=1,column=0,padx=10, pady=20)
transform_button.grid(row=2,column=0,padx=10, pady=30)
text_percent.grid(row=3,column=0)
frame_percent.grid(row=4,column=0,padx=10)

disp_img.grid(row=0,column=1)

window.mainloop()