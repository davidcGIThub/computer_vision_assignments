import numpy as np
import cv2 as cv

def blurred_padding(image,thickness):
    height = np.shape(image)[0]
    width = np.shape(image)[1]
    if np.size(np.shape(image)) == 3:
        depth = np.shape(image)[2]
        new_image = np.zeros((height+thickness*2,width+thickness*2,depth),dtype=np.uint8)
        #top rows
        new_image[0:thickness , thickness:thickness+width , :] = image[0, : , :]
        #bottom rows
        new_image[thickness+height:thickness*2+height , thickness:thickness+width , :] = image[height-1, : , :]
        #center
        new_image[thickness:thickness+height , thickness:thickness+width , : ] = image
        #left column
        new_image[: , 0:thickness , :] = np.reshape(new_image[: , thickness, :],(height+thickness*2,1,3))
        #right column
        new_image[: , thickness+width:thickness*2+width , :] = np.reshape(new_image[: , width+thickness-1 , :],(height+thickness*2,1,3))
        new_image = cv.blur(new_image,(thickness,thickness)) 
        new_image[thickness:thickness+height , thickness:thickness+width , : ] = image
    else:
        new_image = np.zeros((height+thickness*2,width+thickness*2),dtype=np.uint8)
        #top rows
        new_image[0:thickness , thickness:thickness+width] = image[0, :]
        #bottom rows
        new_image[thickness+height:thickness*2+height , thickness:thickness+width] = image[height-1, :]
        #center
        new_image[thickness:thickness+height , thickness:thickness+width ] = image
        #left column
        new_image[: , 0:thickness] = np.reshape(new_image[: , thickness] , (height+thickness*2,1))
        #right column
        new_image[: , thickness+width:thickness*2+width ] = np.reshape(new_image[: , width+thickness-1] , (height+thickness*2,1))
        new_image = cv.blur(new_image,(thickness,thickness)) 
        new_image[thickness:thickness+height , thickness:thickness+width] = image
    return new_image

def convolution_2D(image,kernel):
    height = np.shape(image)[0]
    width = np.shape(image)[1]
    pad_thickness = int(np.round(np.shape(kernel)[0]/2))
    kernel_length = np.shape(kernel)[0]
    padded_image = blurred_padding(image,pad_thickness)
    if np.size(np.shape(image)) == 2: #black and white
        new_image = np.zeros((height,width),dtype=np.uint8)
    else: #color
        new_image = np.zeros((height,width,3),dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if np.size(np.shape(image)) == 2: #black and white
                new_image[i,j] = np.sum(np.multiply(padded_image[i:i+kernel_length,j:j+kernel_length],kernel))
            else: #color
                new_image[i,j,0] = np.sum(np.multiply(padded_image[i:i+kernel_length,j:j+kernel_length,0],kernel))
                new_image[i,j,1] = np.sum(np.multiply(padded_image[i:i+kernel_length,j:j+kernel_length,1],kernel))
                new_image[i,j,2] = np.sum(np.multiply(padded_image[i:i+kernel_length,j:j+kernel_length,2],kernel))
    return new_image




    # def blur_edges(image,thickness):
        

