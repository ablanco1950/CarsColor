# CarsColor
Project that from photos of cars, estimates its detailed colors (not basic colors) based on the maximum values of the R G B histograms of the photo. A version that uses the same method with HSV values is also provided.

All necessary packages besides:

ultralytics 

cv2

cvzone

math

can be installed with a simple pip, if you get the message that you cannot import it

The project uses the colors.csv file downloaded from https://github.com/codebrainz/color-names/blob/master/output/colors.csv

Unzip the Test1 file with the test images (obtained from Roboflow and Kaggle)

Execute the python program:

CarsColor_YoloV8n_Min_Distance.py

The photos are shown, to test it, and after close them in console appears the r g b assigned by the max of histograms and the r g b and name of color aproximated in the list of colors in colors.csv.

It also produces the file CarColorResults.txt so that the results can be scored.

Appears the name of the photo  (it matches the license plate of the car) the rgb obtained by applying the maximum histogram of each RGB component and the approximate RGB in the list of colors in color.csv, as well as the name of this color

The results may be tested with  https://www.rapidtables.com/web/color/RGB_Color.html

00004,57,142,217,49,140,231,"Bleu De France"

01702,116,158,44,107,142,35,"Olive Drab (Web) (Olive Drab #3)"

2122267,251,252,249,255,250,250,"Snow"

6662GKS,25,24,28,26,36,33,"Dark Jungle Green" Error: the calculated RGB is correct but when approaching with a minimum distance it gives 
                                               an  incorrect RGB. The true color is a dark blue, although in the photo it appears as black

8544,12,16,19,16,12,8,"Smoky Black"

8544,254,255,255,255,255,255,"White"

8544,254,255,255,255,255,255,"White"         ( Duplicate, the detection recognizes 8544 as car and truck at the same time)

BMW,21,27,34,26,36,33,"Dark Jungle Green"     Error: the calculated RGB is correct but when approaching with a minimum distance it gives 
                                              an incorrect RGB near green dark

CRAIG,249,211,123,248,222,126,"Mellow Yellow"

CY110KS,200,201,203,196,195,208,"Lavender Gray"   (Error, the car looks white, perhaps the shadow)

DRUNK,8,26,100,0,35,102,"Royal Blue (Traditional)"

GCP332,24,131,255,30,144,255,"Dodger Blue"   (Error)

GN64OTP,254,254,254,255,255,255,"White"

GN64OTP,1,1,1,0,0,0,"Black"              (error detecting car)

HF3461,15,14,15,16,12,8,"Smoky Black"

LR33TEE,255,3,5,255,0,0,"Red"

VIPER,214,94,24,210,105,30,"Cocoa Brown"




 Changing the name directory in line 11 of the programa, any directory with any cars fotos can be tested.

==========================================================

HSV version:

Execute the python program:

CarsColor_YoloV8n_Min_DistanceHSV.py

The photos are shown, to test it, and after close them in console appears the r g b assigned by the max of histograms and the r g b and name of color aproximated in the list of colors in colors.csv.

It also produces the file CarColorResultsHSV.txt so that the results can be scored.

Appears the name of the photo  (it matches the license plate of the car) the rgb obtained by applying the maximum histogram of each HSV component and the approximate RGB in the list of colors in color.csv, as well as the name of this color

The results may be tested with  https://www.rapidtables.com/web/color/RGB_Color.html

00004,78,148,217,73,151,208,"Celestial Blue"

01702,105,158,0,86,130,3,"Avocado"

2122267,252,252,250,255,250,250,"Snow"

6662GKS,19,17,28,26,17,16,"Licorice"   Error: the calculated RGB is correct but when approaching with a minimum distance it gives an 
                                       incorrect RGB. The true color is a dark blue, although in the photo it appears as black

8544,13,16,19,16,12,8,"Smoky Black"

8544,253,254,255,255,255,255,"White"

8544,253,254,255,255,255,255,"White"      ( Duplicate, the detection recognizes 8544 as car and truck at the same time)

BMW,23,28,34,26,36,33,"Dark Jungle Green" Error: the calculated RGB is correct but when approaching with a minimum distance it gives an 
                                           incorrect RGB near green dark

CRAIG,249,207,122,255,200,124,"Topaz"

CY110KS,200,201,203,196,195,208,"Lavender Gray"

DRUNK,0,60,255,2,71,254,"Blue (Ryb)"

GCP332,139,197,255,135,206,250,"Light Sky Blue"

GN64OTP,254,178,0,255,179,0,"Ucla Gold"

GN64OTP,1,1,1,0,0,0,"Black"                          ( Error detecting car)

HF3461,13,13,15,16,12,8,"Smoky Black"

LR33TEE,255,0,8,255,0,0,"Red"

VIPER,214,112,86,203,109,81,"Copper Red"

Changing the name directory in line 11 of the programa, any directory with any cars fotos can be tested.

References:

https://medium.com/@shaw801796/your-first-object-detection-model-using-yolo-2e841547cc20  

https://medium.com/@rndayala/image-histograms-in-opencv-40ee5969a3b7

https://github.com/CharansinghThakur/Color-Detection/blob/master/color_detection.py

https://www.rapidtables.com/web/color/RGB_Color.html


=======================================================================================================================

Before 21/08/2023 there was a more complicated procedure that was abandoned after reading the recent article https://medium.com/@shaw801796/your-first-object-detection-model-using-yolo-2e841547cc20  

OPERATION  BEFORE THE  21/08/2023

It  used the best.pt model that serves to frame the cars in the photos (to see details of its creation with yolov8 see the project
https://github.com/ablanco1950/LicensePlate_Yolov8_MaxFilters)

Create the RandomForest model that from R G B values assigns the name of the color in the colors.csv file

run CreateModelColorsRandomForest.py

run the test

Run CarsColor.py, the photos will appear on the screen for your control, and the assigned colors will appear on the console.

It also produces the file CarColorResults.txt so that the results can be scored.

The result has been:

2122267,251,252,249,['"Snow"']

6662GKS,25,24,28,['"Midnight Blue"']

8544,20,31,39,['"Dark Jungle Green"']

8544,254,255,255,['"White"']

BMW,21,27,34,['"Dark Jungle Green"']

CRAIG,249,211,123,['"Mellow Apricot"']

CY110KS,200,201,203,['"Lilac"']

DRUNK,6,26,100,['"Catalina Blue"']

GCP332,0,131,255,['"Azure"']

GN64OTP,254,254,0,['"Laser Lemon"']

HF3461,14,14,16,['"Smoky Black"']

J75665,182,8,8,['"International Orange (Engineering)"']

J75665,8,8,8,['"Smoky Black"']

LR33TEE,255,3,5,['"Red"']

VIPER,214,95,24,['"Chocolate (Web)"']

VIPER,100,104,107,['"Dim Gray"']


The results can be improved, probably the image segmentation method can be improved. Improvements will be introduced in subsequent editions.

For the recognition of colors based on their R G B components, the web can be used https://www.rapidtables.com/web/color/RGB_Color.html




===================================================================


