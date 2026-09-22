
import cv2
from matplotlib import pyplot as plt
import numpy as np

from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential

# Create a new model instance
loaded_model = Sequential()
loaded_model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(MaxPooling2D(pool_size=(4, 4)))
loaded_model.add(Dropout(0.4))
loaded_model.add(Flatten())
loaded_model.add(Dense(128, activation='relu'))
loaded_model.add(Dense(36, activation='softmax'))

# Restore the weights
loaded_model.load_weights('OCR11Hits.weights.h5')



# Match contours to license plate or character template
def find_contours(dimensions, img, SwOptionPlot) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    #cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    Numcntrs=0
    
    for cntr in cntrs :
        #NumNumcntrs=Numcntrs+1
        #print("Numero de contornos =" + str(Numcntrs))
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # cv2.minAreaRect se adapta las letras inclinadas
        #intX, intY, intWidth, intHeight=cv2.minAreaRect(cntr)
        
        X11 = intX 
        X12 = intX + intWidth
        Y11=intY 
        Y12 = intY + intHeight
        
        # COMPROBAR QUE EL CONTORNO NO ESTA INCLUIDO EN OTRO CONTORNO
        SwIncluded=0
        for cntr2 in cntrs:
           
           intX2, intY2, intWidth2, intHeight2 = cv2.boundingRect(cntr2)
           #intX2, intY2, intWidth2, intHeight2 = cv2.minAreaRect(cntr2)
           if intWidth2 > lower_width and intWidth2 < upper_width and intHeight2 > lower_height and intHeight2 < upper_height :
              pp=0
           else:
             
              continue   
           #print("Ancho " + str(intWidth2))
           #if intWidth2 > 200: continue
           X21 = intX2
           X22 = intX2 + intWidth2
           Y21=intY2 
           Y22 = intY2 + intHeight2

           if X11 > X21 and X12 < X22 and Y11 > Y21 and Y12 < Y22:
              SwIncluded=1
              
              break
        
        if SwIncluded == 1: continue 
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours
       
            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
    #print("Numero de contornos =" + str(Numcntrs))

    if SwOptionPlot== "Y":
        plt.imshow(ii, cmap='gray')
        plt.title('Predict Segments')
        plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

# Find characters in the resulting images
def segment_characters(image,SwOptionPlot) :

    plt.imshow(image, cmap='gray')
    plt.title('Original')
    #if SwOptionPlot=="Y":
    #  plt.show()
   
    img_lp=image   
        
   
    #img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY) EN LA VERSION PARA SPANISH YA VIENE EN ESCALA DE GRISES
    img_gray_lp=img_lp
    
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_OTSU)  
    
    
    
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    
    LP_WIDTH = img_gray_lp.shape[0]
    LP_HEIGHT = img_gray_lp.shape[1]
    # Estimations of character contours sizes of cropped license plates
    dimensions = [#LP_WIDTH/6,
                  LP_WIDTH/10,# 1,I case 
                       LP_WIDTH/2 ,  
                       #LP_HEIGHT/10,
                       LP_HEIGHT/13, # letras recortadas
                       2*LP_HEIGHT/3]
                       
    
   
    plt.imshow(img_binary_lp, cmap='gray')
    plt.title('Contour')
    #if SwOptionPlot=="Y":
    #    plt.show()
    
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp, SwOptionPlot)

    return char_list



# Predicting the output
def fix_dimension(img): 
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img
        return new_img
  
def show_results(char):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        #y_ = loaded_model.predict_classes(img, verbose=0)[0] #predicting the class #MOD
        
        cv2.imwrite("caracter" + str(i) + ".jpg",img_)
        #cv2.imshow("caracter" + str(i) + ".jpg", img_)        
        #cv2.waitKey(0)
        predicciones = loaded_model.predict(img, verbose=0)[0] #predicting the class
        y_=np.argmax(predicciones, axis=-1) #MOD
        #print(y_)
        character = dic[y_]
        #print(character)
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number

#image=cv2.imread("Test1/6662GKS.jpg")
#image=cv2.imread("Test/01CC1A0001.png")
#image=cv2.imread("Test/067RAF.png")
#image=cv2.imread("Test/172TMJ.png")
#image=cv2.imread("Test/1268.png")
#image=cv2.imread("Test/8544.png")
#image=cv2.imread("Test1/8314KBY.jpg")


def GetOCR(image, SwOptionPlot):
   

    char_list=segment_characters(image,SwOptionPlot)
    
    #char_list=segment_characters(TabImgSelect[0])

    #print(char_list)

    #print(show_results(char_list))
    #print("RETORNA" + str(show_results(char_list)))
    
    return show_results(char_list)

#TabcropLicense, y,yMax,x,xMax=DetectLicenseWithYolov8 (image)

#print(GetOCR(TabcropLicense[0],SwOptionPlot))
