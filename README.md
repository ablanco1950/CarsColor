# CarsColor
project that from photos of cars, estimates its color based on the maximum values of the R G B histograms of the photo

All necessary packages can be installed with a simple pip, if you get the message that you cannot import

The project uses the colors.csv file downloaded from https://github.com/codebrainz/color-names/blob/master/output/colors.csv

It also uses the best.pt model that serves to frame the cars in the photos (to see details of its creation with yolov8 see the project
https://github.com/ablanco1950/LicensePlate_Yolov8_MaxFilters)

Unzip the Test1 file with the test images (obtained from Roboflow and Kaggle)

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


Other references:



https://medium.com/@rndayala/image-histograms-in-opencv-40ee5969a3b7

https://github.com/CharansinghThakur/Color-Detection/blob/master/color_detection.py
