# -*- coding: utf-8 -*-
"""
Created on augost 2023

@author: Alfonso Blanco
"""
#######################################################################
# PARAMETERS
######################################################################
dir=""
dirname= "Test1"

import time
import cv2
import joblib
Ini=time.time()


dirnameYolo="best.pt"
# https://docs.ultralytics.com/python/
from ultralytics import YOLO
model = YOLO(dirnameYolo)
class_list = model.model.names
#print(class_list)



import numpy as np
from CarColor import CarColorImg


import os
import re


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


# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def DetectCarWithYolov8 (img):
  
   TabcropLicense=[]
   y=[]
   yMax=[]
   x=[]
   xMax=[]
   results = model.predict(img)
   for i in range(len(results)):
       # may be several plates in a frame
       result=results[i]
       
       xyxy= result.boxes.xyxy.numpy()
       confidence= result.boxes.conf.numpy()
       class_id= result.boxes.cls.numpy().astype(int)
       # Get Class name
       class_name = [class_list[z] for z in class_id]
       # Pack together for easy use
       sum_output = list(zip(class_name, confidence,xyxy))
       # Copy image, in case that we need original image for something
       out_image = img.copy()
       for run_output in sum_output :
           # Unpack
           #print(class_name)
           label, con, box = run_output
           if label != "vehicle":continue
           
           # crop image to center it           
           xoff=50
           #if int(box[0]) < xoff or int(box[1]) < xoff or int(box[2]) < xoff or int(box[3]) < xoff:
           #    continue
               
           box[0]=int(box[0])+xoff           
           box[1]=int(box[1])+xoff
           box[2]=int(box[2])-xoff
           box[3]=int(box[3])-xoff
           
          
           cropLicense=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
           #cv2.imshow("Crop", cropLicense)
           #cv2.waitKey(0)
           TabcropLicense.append(cropLicense)
           y.append(int(box[1]))
           yMax.append(int(box[3]))
           x.append(int(box[0]))
           xMax.append(int(box[2]))
       
   return TabcropLicense, y,yMax,x,xMax


###########################################################
# MAIN
##########################################################
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import GradientBoostingClassifier
#arr=[]
#arry=[]
arrname=[]
f=open("colors.csv","r")

Conta=0;
for linea in f:
    Conta=Conta+1
    
    lineadelTrain =linea.split(",")
  
 
    #linea_x =[]
    #z=2
    #for x in lineadelTrain:
   
    #    z=z+1
    #    if z==6: break
        
    #    linea_x.append(int(lineadelTrain[z]))
  
    #arr.append(linea_x)
    #arry.append(int(Conta))
    arrname.append(lineadelTrain[1])
   

#X_train=np.array(arr)
#   print(x)
#Y_train=np.array(arry)


Name=np.array(arrname)
 


#rf= RandomForestClassifier()

#modelRf= rf.fit(X_train,Y_train)

#joblib.dump(modelRf, "colors_random_forest.joblib")
modelRf=joblib.load("colors_random_forest.joblib")

imagesComplete, Licenses=loadimagesRoboflow(dirname)

print("Number of imagenes : " + str(len(imagesComplete)))

#print("Number of   licenses : " + str(len(Licenses)))

ContDetected=0
ContNoDetected=0
TotHits=0
TotFailures=0
with open( "CarColorResults.txt" ,"w") as  w:
    for i in range (len(imagesComplete)):
            # solo imagenes .jpg pueden ser referenciadas en formato [:, :, 0]
            cv2.imwrite('pp.jpg',imagesComplete[i])
            img=cv2.imread("pp.jpg")
            #cv2.imshow("img", img)
            #cv2.waitKey()
         
            TabImgSelect, y, yMax, x, xMax =DetectCarWithYolov8(img)
            #gray=imagesComplete[i]
            
            License=Licenses[i]
            if TabImgSelect==[] :
                 print(License + " NON DETECTED")
                 continue
            for j in range( len(TabImgSelect)):
                 #dimensions = TabImgSelect[j].shape
 
                 # height, width, number of channels in image
                 height = TabImgSelect[j].shape[0]
                 width = TabImgSelect[j].shape[1]
                 if height < 1 or width < 1:
                      #print( License + " image too small")
                      continue

                                 
                 R, G, B, NameColor=CarColorImg(TabImgSelect[j], modelRf, Name, License)

                 print ("")
                 lineaw=[]
                 lineaw.append(License) 
                 lineaw.append(str(R))
                 lineaw.append(str(G))
                 lineaw.append(str(B))
                 lineaw.append(NameColor)        
                 lineaWrite =','.join(lineaw)
                 lineaWrite=lineaWrite + "\n"
                 w.write(lineaWrite)         
print ("seconds "+ str(time.time()-Ini))
 
