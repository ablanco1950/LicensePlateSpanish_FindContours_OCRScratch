# -*- coding: utf-8 -*-
"""
Created on Sept 2026

@author: Alfonso Blanco
"""
######################################################################
# PARAMETERS
SwOptionPlot="N"
######################################################################

from OCRScratch_V1 import GetOCR

# Test images
dirname=  "test6Training\\images"

######################################################################

import numpy as np

import cv2

import time
Ini=time.time()

dirnameYolo="best.pt"
# https://docs.ultralytics.com/python/
from ultralytics import YOLO
model = YOLO(dirnameYolo)
class_list = model.model.names
#print(class_list)

import numpy as np

X_resize=220
Y_resize=70

import os
import re

import imutils

X_resize=220
Y_resize=70

import os
import re

import imutils

from scipy.signal import convolve2d

########################################################################

import cv2
import numpy as np


###################################################

from skimage.transform import radon

import numpy
from numpy import  mean, array, blackman, sqrt, square
from numpy.fft import rfft

try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax


def GetRotationImage(image):

   
    I=image
    I = I - mean(I)  # Demean; make the brightness extend above and below zero
    
    
    # Do the radon transform and display the result
    sinogram = radon(I)
   
    
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
      
    # rms_flat does no exist in recent versions
    #r = array([mlab.rms_flat(line) for line in sinogram.transpose()])
    r = array([sqrt(mean(square(line))) for line in sinogram.transpose()])
    rotation = argmax(r)
    #print('Rotation: {:.2f} degrees'.format(90 - rotation))
    #plt.axhline(rotation, color='r')
    
    # Plot the busy row
    row = sinogram[:, rotation]
    N = len(row)
    
    # Take spectrum of busy row and find line spacing
    window = blackman(N)
    spectrum = rfft(row * window)
    
    frequency = argmax(abs(spectrum))
   
    return rotation, spectrum, frequency



# https://medium.com/@garciafelipe03/image-filters-and-morphological-operations-using-python-89c5bbb8dca0
# 5x5 Gaussian Blur
def gaussian_5x5(img):
    
    kernel_gb_5 = (1 / 273) * np.array([[1, 4, 7, 4, 1],
                                        [4, 16, 26, 16, 4],
                                        [7, 26, 41, 26, 7],
                                        [4, 16, 26, 16, 4],
                                        [1, 4, 7, 4, 1]])

    return convolve2d(img, kernel_gb_5, 'valid')    
    

#########################################################################
def FindLicenseNumber (gray, x_offset, y_offset,  License, x_resize, y_resize, \
                       Resize_xfactor, Resize_yfactor, BilateralOption):
#########################################################################

    grayColor=gray
    
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
   
    TotHits=0 
    
    X_resize=x_resize
    Y_resize=y_resize
     
    
    gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
    
    
    # en la mayoria de los casos no es necesaria rotacion
    # pero en algunos casos si (ver TestRadonWithWilburImage.py)
    rotation, spectrum, frquency =GetRotationImage(gray)
    rotation=90 - rotation
    
    if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
        print(License + " rotate "+ str(rotation))
        gray=imutils.rotate(gray,angle=rotation)
    
    
    TabLicensesFounded=[]
    ContLicensesFounded=[]
    
    
    X_resize=x_resize
    Y_resize=y_resize
    print("gray.shape " + str(gray.shape)) 
    Resize_xfactor=1.5
    Resize_yfactor=1.5
   
    
    TotHits=0

    gray1=gaussian_5x5(gray)

    # 1. Convertir directamente a enteros de 8 bits
    imagen_8bit = gray1.astype(np.uint8)

    # 2. Aplicar el umbral de Otsu
    val_umbral, imagen_umbralizada = cv2.threshold(
    imagen_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    text = ocr(imagen_umbralizada)
   
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    
           TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==License:
              print(text + "  Hit with gaussian_5x5 " )
              TotHits=TotHits+1
           else:
               print(License + " detected with Filter gaussian_5x5 "+ text) 
    
    
   
    for z in range(5,6):
       
        
   
    
       kernel = np.array([[0,-1,0], [-1,z,-1], [0,-1,0]])
       gray1 = cv2.filter2D(gray, -1, kernel)

       #if z==5:
       gray1 = cv2.dilate(gray1, (3,3))
       gray1 = cv2.erode(gray1, (3,3))
      
              
       text = ocr(gray1)
       
       text = ''.join(char for char in text if char.isalnum()) 
       text=ProcessText(text)
       if ProcessText(text) != "":
      
           ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==License:
              print(text +  "  Hit with Sharpen filter z= "  +str(z))
              TotHits=TotHits+1
           else:
               print(License + " detected with Sharpen filter z= "  +str(z) + " as "+ text)
               
           # SE DUPLICA EL EFECTO DE ESTE FILTRO
           ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==License:
              print(text +  "  Hit with Sharpen filter z= "  +str(z))
              TotHits=TotHits+1
           else:
               print(License + " detected with Sharpen filter z= "  +str(z) + " as "+ text)   

    
    gray2= cv2.bilateralFilter(gray,3, 75, 75)
    
    for z in range(5,6):
        
        
       kernel = np.array([[-1,-1,-1], [-1,z,-1], [-1,-1,-1]])
       gray1 = cv2.filter2D(gray2, -1, kernel)
      
       text=ocr(gray1)  
       text = ''.join(char for char in text if char.isalnum())
       
       if ProcessText(text) != "": 
       
           ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==Licenses[i]:
              print(text +  "  Hit with Sharpen filter modified z=" +str(z)  )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected with Sharpen filter modifieda z= " + str(z)+ " " + text) 
            
    return TabLicensesFounded, ContLicensesFounded

 ########################################################################
def loadimagesRoboflow (dirname):
 
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
                 
                 # Spanish license plate is NNNNAAA
                 if Detect_Spanish_LicensePlate(License)== -1: continue
                
                 image = cv2.imread(filepath)

                #Color Balance
                #https://blog.katastros.com/a?ID=01800-4bf623a1-3917-4d54-9b6a-775331ebaf05
                
                 img = image
                    
                 r, g, b = cv2.split(img)
                
                 r_avg = cv2.mean(r)[0]
                
                 g_avg = cv2.mean(g)[0]
                
                 b_avg = cv2.mean(b)[0]
                
                 
                 # Find the gain occupied by each channel
                
                 k = (r_avg + g_avg + b_avg)/3
                
                 kr = k/r_avg
                
                 kg = k/g_avg
                
                 kb = k/b_avg
                
                 
                 r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
                
                 g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
                
                 b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
                
                 
                 balance_img = cv2.merge([b, g, r])
                 
                 image=balance_img
                 
                 #image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21) 
                  
                 images.append(image)
                 
                 Licenses.append(License)
                 
                 
                
                 Cont+=1
     
     return images, Licenses




def Detect_Spanish_LicensePlate(Text):
    
    if len(Text) != 7: return -1   
    
    if (Text[0] < "0" or Text[0] > "9" ) : return -1 
    if (Text[1] < "0" or Text[2] > "9" ) : return -1   
    if (Text[2] < "0" or Text[2] > "9" ) : return -1   
    if (Text[3] < "0" or Text[3] > "9" ) : return -1     
    if (Text[4] < "A" or Text[4] > "Z" ) : return -1 
    if (Text[5] < "A" or Text[5] > "Z" ) : return -1 
    if (Text[6] < "A" or Text[6] > "Z" ) : return -1
    
    return 1


def ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text):
    
    SwFounded=0
    for i in range( len(TabLicensesFounded)):
        if text==TabLicensesFounded[i]:
            ContLicensesFounded[i]=ContLicensesFounded[i]+1
            SwFounded=1
            break
    if SwFounded==0:
       TabLicensesFounded.append(text) 
       ContLicensesFounded.append(1)
    return TabLicensesFounded, ContLicensesFounded


def ocr(gray1):
    return GetOCR(gray1,SwOptionPlot)


def ProcessText(text):
    if len(text)  > 7:
       text=text[len(text)-7:] 
    if Detect_Spanish_LicensePlate(text)== -1: 
       return ""
    else:
       return text

# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def DetectLicenseWithYolov8 (img):
  
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
           if label == "vehicle":continue
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


imagesComplete, Licenses=loadimagesRoboflow(dirname)

print("Number of imagenes : " + str(len(imagesComplete)))

print("Number of   licenses : " + str(len(Licenses)))

ContDetected=0
ContNoDetected=0
TotHits=0
TotFailures=0
with open( "LicenseResults.txt" ,"w") as  w:
    for i in range (len(imagesComplete)):
          
            gray=imagesComplete[i]
            
            License=Licenses[i]
            
            TabImgSelect, y, yMax, x, xMax =DetectLicenseWithYolov8(gray)
            
            if TabImgSelect==[]:
                print(License + " NON DETECTED")
                ContNoDetected=ContNoDetected+1 
                continue
            else:
                ContDetected=ContDetected+1
                print(License + " DETECTED ")
            for x in range(len(TabImgSelect)):
                
                if len(TabImgSelect[x]) == 0: continue
                gray=TabImgSelect[x]  
                
                
                x_off=3
                y_off=2
                
                #x_resize=220
                x_resize=215
                y_resize=70
                
                Resize_xfactor=1.78
                Resize_yfactor=1.78
                
                ContLoop=0
                
                SwFounded=0
                
                BilateralOption=0
                TabLicensesFounded=[]
                ContLicensesFounded=[]
                
                TabLicensesFounded, ContLicensesFounded= FindLicenseNumber (gray, x_off, y_off,  License, x_resize, y_resize, \
                                       Resize_xfactor, Resize_yfactor, BilateralOption)
                  
                
                print(TabLicensesFounded)
                print(ContLicensesFounded)
                
                ymax=-1
                contmax=0
                licensemax=""
              
                for z in range(len(TabLicensesFounded)):
                    if ContLicensesFounded[z] > contmax:
                        contmax=ContLicensesFounded[z]
                        licensemax=TabLicensesFounded[z]
                
                if licensemax == License:
                   print(License + " correctly recognized") 
                   TotHits+=1
                else:
                    print(License + " Detected but not correctly recognized")
                    TotFailures +=1
                print ("")  
                lineaw=[]
                lineaw.append(License) 
                lineaw.append(licensemax)
                lineaWrite =','.join(lineaw)
                lineaWrite=lineaWrite + "\n"
                w.write(lineaWrite)
                break # only one plate for image to verify with te image´s name
             
              
print("")           
print("Total Hits = " + str(TotHits ) + " from " + str(len(imagesComplete)) + " images readed")

print("")

print( " Time in seconds "+ str(time.time()-Ini)) 
                 
        
