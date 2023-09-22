# -*- coding: utf-8 -*-
"""
Created on augost 2023

@author: Alfonso Blanco


"""
#######################################################################
# PARAMETERS
######################################################################

########################################
# The program is used for RGB as HSV
# OPTION=0-->RGB  OPTION=1-->HSV
OPTION=0

dir=""
dirname= "Test1"

import time
import cv2
import joblib
import cvzone
import math
import os
import re

Ini=time.time()

from ultralytics import YOLO
# from # https://medium.com/@shaw801796/your-first-object-detection-model-using-yolo-2e841547cc20
# yolov8x.pt is most precise in the series yolov8, but need more resources. Is needed 
modelv8x = YOLO("yolov8x.pt")
class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]

import numpy as np
if OPTION==0:
     from Calc_RGB import Car_RGB
else:          
     from Calc_RGB import Car_RGB_fromHSV

########################################################################
def loadimagesRoboflow (dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     Licenses=[]
    
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                 License=filename[:len(filename)-4]
                 #if License != "PGMN112": continue
                 
                 image = cv2.imread(filepath)
                 # Roboflow images are (416,416)
                 #image=cv2.resize(image,(416,416)) 
                 # kaggle images
                 #image=cv2.resize(image, (640,640))
                 
                           
                 images.append(image)
                 Licenses.append(License)
                 
                 Cont+=1
     
     return images, Licenses

# https://medium.com/@shaw801796/your-first-object-detection-model-using-yolo-2e841547cc20
def DetectCarWithYolov8x (img):
  
   TabcropLicense=[]
   y=[]
   yMax=[]
   x=[]
   xMax=[]

   results = modelv8x(img, stream=True)
   SalvaImg=img.copy()
   for r in results:
        boxes = r.boxes
        for box in boxes:
             x1,y1,x2,y2 = box.xyxy[0]
             x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
             #print(x1,y1,x2,y2)
             cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            
             # crop image to center it           
             xoff= int((x2 - x1)*0.1)
             yoff= int((y2 - y1)*0.1)
             #if int(box[0]) < xoff or int(box[1]) < xoff or int(box[2]) < xoff or int(box[3]) < xoff:
             #    continue

             x1=x1+xoff
             x2=x2-xoff
             y1=y1+ yoff
             y2 = y2 -yoff
             if (x2- x1) < 120: continue
             #print(x1,y1,x2,y2)

             # # https://medium.com/@shaw801796/your-first-object-detection-model-using-yolo-2e841547cc20
             conf = math.ceil((box.conf[0]*100))/100
             #if conf < 0.85: continue
             cls = int(box.cls[0]) #converting the float to int so that the class name can be called
             #if class_names[cls] != "car" and class_names[cls] != "license plate": continue
             #if class_names[cls] != "car"  and class_names[cls] != "truck" and class_names[cls] != "bus": continue
             cvzone.putTextRect(img,f'{class_names[cls]}  {conf} ',(max(0,x1),max(35,y1)),scale=1,thickness=1)

             cv2.imshow("img", img)
             cv2.waitKey()
             cropLicense=SalvaImg[y1:y2,x1:x2]
             
             #cv2.imshow("Crop", cropLicense)
             #cv2.waitKey(0)
             TabcropLicense.append(cropLicense)
             y.append(y1)
             yMax.append(y2)
             x.append(x1)
             xMax.append(x2)
             
      
   return TabcropLicense, y,yMax,x,xMax
# https://github.com/ShoaibBadhe/Color-Classification-Using-Opencv/blob/main/Colors%20Classification.py
import pandas as pd
# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)


# function to calculate minimum distance from all colors and get the most matching color
def getColorName(R, G, B):
    minimum = 10000
    type_order_rgb=Calc_Order_RGB (R, G, B)
    for i in range(len(csv)):
        R_readed=int(csv.loc[i, "R"])
        G_readed= int(csv.loc[i, "G"])
        B_readed= int(csv.loc[i, "B"])
        #
        #  To mantein the order of values of  R G B, not is enough
        # the minimal distance
        #
        type_order_rgb_readed=Calc_Order_RGB (R_readed, G_readed, B_readed)
        if type_order_rgb != type_order_rgb_readed: continue             
        d = abs(R - R_readed) + abs(G - G_readed) + abs(B - B_readed)
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
            R_elected=int(csv.loc[i, "R"])
            G_elected=int(csv.loc[i, "G"])
            B_elected=int(csv.loc[i, "B"])
            Y_elected=i
    return R_elected, G_elected, B_elected, Y_elected, cname

#
# module that receives the RGB values of an image and returns the order according to the values
# for example if R > G > B returns RGB
# if R > B > G returns RBG...see code
def Calc_Order_RGB (R, G, B):

    if R>=G>=B: return "RGB"
    if R>=B>=G: return "RBG"
    if G>=R>=B: return "GRB"
    if G>=B>=R: return "GBR"
    if B>=R>=G: return "BRG"
    if B>=G>=R: return "BGR"
    print( "Raro, caso no considerado R = " +str(R)+ " G = " + str(G) + " B = " + str(B))
    return "---" 
    
    
    
###########################################################
# MAIN
##########################################################


imagesComplete, Licenses=loadimagesRoboflow(dirname)

if OPTION==0:
     FileNameResults="CarColorResults.txt"
else:
     FileNameResults="CarColorResultsHSV.txt"

with open( FileNameResults,"w") as  w:
    for i in range (len(imagesComplete)):
            # solo imagenes .jpg pueden ser referenciadas en formato [:, :, 0]
            cv2.imwrite('pp.jpg',imagesComplete[i])
            img=cv2.imread("pp.jpg")
            #cv2.imshow("img", img)
            #cv2.waitKey()
         
            TabImgSelect, y, yMax, x, xMax =DetectCarWithYolov8x(img)
            #gray=imagesComplete[i]
            
            License=Licenses[i]
            if TabImgSelect==[] :
                 print(License + " NON DETECTED")
                 continue
            for j in range( len(TabImgSelect)):
                 
                 height = TabImgSelect[j].shape[0]
                 width = TabImgSelect[j].shape[1]
               
                 if OPTION==0:
                         R, G, B=Car_RGB(TabImgSelect[j])
                 else:             
                         R, G, B=Car_RGB_fromHSV(TabImgSelect[j])
                 
                 R_elected, G_elected, B_elected, Y_elected, cname =getColorName(R, G, B)

                 
                 rgb= "("+str(R)+","+str(G)+","+ str(B)+")"
                 print(License + " Color code rgb " + rgb)
         


                 rgb_elected= "("+str(R_elected)+","+str(G_elected)+","+ str(B_elected)+")"
                 print(License + " Color code rgb elected " + rgb_elected + " " + cname)

                 
                 print ("")
                 lineaw=[]
                 lineaw.append(License) 
                 lineaw.append(str(R))
                 lineaw.append(str(G))
                 lineaw.append(str(B))
                 lineaw.append(str(R_elected))
                 lineaw.append(str(G_elected))
                 lineaw.append(str(B_elected))
                 lineaw.append(cname)        
                 lineaWrite =','.join(lineaw)
                 lineaWrite=lineaWrite + "\n"
                 w.write(lineaWrite)         

print ("seconds "+ str(round((time.time()-Ini),2)))
 
 
 
