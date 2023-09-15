# https://www.quora.com/Why-use-an-HSV-image-for-color-detection-rather-than-an-RGB-image#:~:text=The%20reason%20we%20use%20HSV,relatively%20lesser%20than%20RGB%20values.
#The reason we use HSV colorspace for color detection/thresholding over RGB/BGR is that HSV is more robust
# towards external lighting changes. This means that in cases of minor changes in external lighting (such as pale shadows,etc. )
# Hue values vary relatively lesser than RGB values.
import numpy as np
from matplotlib import pyplot as plt
import cv2 
#https://stackoverflow.com/questions/63066842/how-to-convert-hsv-to-rgb-in-python
import colorsys

# https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion
def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))
def ValorMaxHistogram(img):
    
    # https://docs.opencv.org/3.4/d1/db7/tutorial_py_histogram_begins.html
    hist = cv2.calcHist([img],[0],None,[256],[0,256]) # return 256* 1 array
    
    #print("HIST")
    #print(hist)
    OcurrenciasMax=0
    ValorMax=0
    for i in range(len(hist)):
            if i==0: continue
            if hist[i] > OcurrenciasMax:
                OcurrenciasMax=hist[i]
                ValorMax=i

    #print("return max valor " + str(ValorMax))         
    return OcurrenciasMax, ValorMax
        
def CarColorImg_Min_Distance(img, model, TabNames, License):        

         # convert from RGB to HSV format
         img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
          
        
         # https://stackoverflow.com/questions/17185151/how-to-obtain-a-single-channel-value-image-from-hsv-image-in-opencv-2-1
         img0=img[:, :, 0]
         counts, h=ValorMaxHistogram(img0)
        
         img1=img[:, :, 1]
         counts,  s=ValorMaxHistogram(img1)
         
         img2=img[:, :, 2]
         counts,  v=ValorMaxHistogram(img2)
         
         print(" Calculated H S V values h =" + str(h)+ " s = " + str(s) + " v = " + str(v))
         # https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion

         # dvide by max values of h, s, v getting decimal values
         # that colorsys needs
         # https://www.lifewire.com/what-is-hsv-in-design-1078068
         

         # Max value of hue h is 180 if opencv is used
         # https://stackoverflow.com/questions/16685707/why-is-the-range-of-hue-0-180-in-opencv
         h=h/180.0
         s=s/255.0
         v=v/255.0
         #print("h =" + str(h)+ " s = " + str(s) + " v = " + str(v))
         
         #RGB from 0 to 1 NO from 0 to 255
         #r, g, b = colorsys.hsv_to_rgb(h, s, v)
         # https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion
         r, g, b = hsv2rgb(h,s,v)
         

         print("Calculated RGB values R =" + str(r)+ " G = " + str(g) + " B = " + str(b))
         
         x_test=[]
         x_test.append(r)
         x_test.append(g)
         x_test.append(b)
         X_test=[]
         X_test.append(np.array(x_test))
         

         Y_predict_test=model.predict(X_test)

         #print(X_test)
         #print(Y_predict)

         
         
         return r, g, b, Y_predict_test
