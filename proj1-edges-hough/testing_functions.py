import cv2 as cv
import numpy as np
import helper_functions as hf
import os

image = cv.imread('/home/david/Pictures/llama.jpg')
padded_image = hf.blurred_padding(image,3)
cv.imshow("padded_image", padded_image)     
cv.waitKey(0)

gray_image = cv.imread('/home/david/Pictures/llama.jpg', cv.IMREAD_GRAYSCALE)
gray_padded_image = hf.blurred_padding(gray_image,3)
kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
x_gradient_image = hf.convolution_2D(gray_image,kernel)
cv.imshow("x_gradient_image", x_gradient_image)
# cv.imshow("gray_padded_image", gray_padded_image)
cv.waitKey(0)

cwd = os.getcwd()
image_path = cwd + '/blocks.png'
print("image path" , image_path)
blocks_image = cv.imread(image_path)
print(type(blocks_image))
cv.imshow("blocks_image" , blocks_image)
cv.waitKey(0)  