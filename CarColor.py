from PIL import Image
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
      
            if counts[i] > OcurrenciasMax:
                OcurrenciasMax=counts[i]
                ValorMax=pixels[i]
    return counts, pixels, ValorMax
   
def CarColorImg(img, modelRf, TabNames, License):
# solo imagenes .jpg pueden ser referenciadas en formato [:, :, 0]
         """
         cv2.imwrite('pp.jpg',img)
         img=cv2.imread("pp.jpg")
         """
         cv2.imshow("img", img)
         cv2.waitKey()
         
         img0=img[:, :, 0]
         counts, pixels, Valor0=ValorMaxHistogram(img0)
         #https://stackoverflow.com/questions/57398643/how-to-extract-individual-channels-from-an-rgb-image
         # valor 0 is b, valor 1 is g , valor 2 is r
         # hay que invertir el orden para que sea r g b
         # web para ver el resultado de  la composicion r g b de colores
         # https://www.rapidtables.com/web/color/RGB_Color.html

         """
         print("Valor 0 blue = " + str(Valor0))        
         pixels = pixels[:-1]
         plt.bar(pixels, counts, align='center')
         plt.savefig('histogram0.png')
         plt.xlim(-1, 256)
         plt.show()
         """
         img1=img[:, :, 1]
         counts, pixels, Valor1=ValorMaxHistogram(img1)
         """
         print("Valor 1 Green = " + str(Valor1))        
         pixels = pixels[:-1]
         plt.bar(pixels, counts, align='center')
         plt.savefig('histogram1.png')
         plt.xlim(-1, 256)
         plt.show()
         """
         img2=img[:, :, 2]
         counts, pixels, Valor2=ValorMaxHistogram(img2)
         """
         print("Valor 2 red= " + str(Valor2))        
         pixels = pixels[:-1]
         plt.bar(pixels, counts, align='center')
         plt.savefig('histogram2.png')
         plt.xlim(-1, 256)
         plt.show()
         """
         # valor 0 is b, valor 1 is g , valor 2 is r
         #
         rgb= "("+str(Valor2)+","+str(Valor1)+","+ str(Valor0)+")"
         print(License + " Color code rgb " + rgb)
         
         x_test=[]
         x_test.append(Valor2)
         x_test.append(Valor1)
         x_test.append(Valor0)
         X_test=[]
         X_test.append(np.array(x_test))

         Y_predict=modelRf.predict(X_test)

         #print(X_test)
         #print(Y_predict)
         
         NameColor=str(TabNames[Y_predict -1])
         print(" Color Name = "+ NameColor)
         return Valor2, Valor1, Valor0, NameColor

