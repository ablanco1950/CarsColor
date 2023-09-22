
import numpy as np
from matplotlib import pyplot as plt
import cv2 


def ValorMaxHistogram(img):
    counts, pixels = np.histogram(img, np.array(range(0, 258)))
    #print(pixels)
    #print(counts)
    OcurrenciasMax=0
    ValorMax=0
    for i in range(len(counts)-1):
            if i==0: continue
            if counts[i] > OcurrenciasMax:
                OcurrenciasMax=counts[i]
                ValorMax=pixels[i]
    return counts, pixels, ValorMax
        
def Car_RGB(img):        

         #cv2.imshow("img", img)
         #cv2.waitKey()
         
         img0=img[:, :, 0]
         counts, pixels, B=ValorMaxHistogram(img0)
         img[:, :, 0]=np.where(img[:, :, 0]==B, 255, img[:, :, 0])
         #https://stackoverflow.com/questions/57398643/how-to-extract-individual-channels-from-an-rgb-image
         # valor 0 is b, valor 1 is g , valor 2 is r
         # hay que invertir el orden para que sea r g b

         
        
         """
         print("Valor 0 blue = " + str(B))        
         pixels = pixels[:-1]
         plt.bar(pixels, counts, align='center')
         plt.savefig('histogram0.png')
         plt.xlim(0, 256)
         plt.show()
         """
         img1=img[:, :, 1]
         counts, pixels, G=ValorMaxHistogram(img1)
         img[:, :, 1]=np.where(img[:, :, 1]==G, 255, img[:, :, 1])
         """
         print("Valor 1 Green = " + str(G))        
         pixels = pixels[:-1]
         plt.bar(pixels, counts, align='center')
         plt.savefig('histogram1.png')
         plt.xlim(0, 256)
         plt.show()
         """
         img2=img[:, :, 2]
         counts, pixels, R=ValorMaxHistogram(img2)
         img[:, :, 2]=np.where(img[:, :, 2]==R, 255, img[:, :, 2])
         """
         print("Valor 2 red= " + str(R))        
         pixels = pixels[:-1]
         plt.bar(pixels, counts, align='center')
         plt.savefig('histogram2.png')
         plt.xlim(0, 256)
         plt.show()
         """
         cv2.imshow("ROI", img)
         cv2.waitKey()
         
         
         # web  to see the result of  r g b composition colors
         # https://www.rapidtables.com/web/color/RGB_Color.html

         
         
         return R, G, B

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

def ValorMaxHistogramHSV(img):
    
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

def Car_RGB_fromHSV(img):        

           # convert from RGB to HSV format
         img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
          
        
         # https://stackoverflow.com/questions/17185151/how-to-obtain-a-single-channel-value-image-from-hsv-image-in-opencv-2-1
         img0=img[:, :, 0]
         counts, h=ValorMaxHistogramHSV(img0)
         img[:, :, 0]=np.where(img[:, :, 0]==h, 180, img[:, :, 0])
        
         img1=img[:, :, 1]
         counts,  s=ValorMaxHistogramHSV(img1)
         img[:, :, 1]=np.where(img[:, :, 1]==s, 255, img[:, :, 1])
         
         img2=img[:, :, 2]
         counts,  v=ValorMaxHistogramHSV(img2)
         img[:, :, 2]=np.where(img[:, :, 2]==v, 255, img[:, :, 2])
         print(" Calculated H S V values h =" + str(h)+ " s = " + str(s) + " v = " + str(v))
         # https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion

         # dvide by max values of h, s, v getting decimal values
         # that colorsys needs
         # https://www.lifewire.com/what-is-hsv-in-design-1078068
         
         cv2.imshow("ROI", img)
         cv2.waitKey()
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

         
         
         return r, g, b
            
