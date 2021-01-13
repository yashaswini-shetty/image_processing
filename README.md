# image_processing
1.	Develop a program to display grayscale image using read and write operation.
Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.

Importance of grayscaling –

Dimension reduction: For e.g. In RGB images there are three color channels and has three dimensions while grayscaled images are single dimensional.
Reduces model complexity: Consider training neural article on RGB images of 10x10x3 pixel.The input layer will have 300 input nodes. On the other hand, the same neural network will need only 100 input node for grayscaled images.
functions:
1.the function imread() is used for reading an image
2.the function imwrite() is used to write an image in memory to disk and
3.the function imshow() in conjunction with namedWindow and waitKey is used for displaying an image in memory.
code:
#RGB TO GRAY 
import cv2
imgclr=cv2.imread("blue1.jpg",1)
imggry = cv2.cvtColor(imgclr, cv2.COLOR_BGR2GRAY)
cv2.imshow('imagecolor',imgclr)
cv2.imshow('imagecolor1',imggry)
cv2.imwrite('grayimg.jpg',imggry)
cv2.waitKey()
output:
![p1](https://user-images.githubusercontent.com/72303060/104428017-f13be380-55a9-11eb-860b-6a90adaf6008.png)

2.	Develop a program to perform linear transformations on an image: Scaling and Rotation 
Scaling is just resizing of the image. OpenCV comes with a function cv.resize() for this purpose. The size of the image can be specified manually, or you can specify the scaling factor. Different interpolation methods are used. Preferable interpolation methods are cv.INTER_AREA for shrinking and cv.INTER_CUBIC (slow) & cv.INTER_LINEAR for zooming. By default, the interpolation method cv.INTER_LINEAR is used for all resizing purposes.

Images can be rotated to any degree clockwise or otherwise. We just need to define rotation matrix listing rotation point, degree of rotation and the scaling factor.

functions:
1.To resize an image in Python, you can use cv2.resize() function of OpenCV library cv2.
2.cv.getRotationMatrix2D(	center, angle, scale	)=Calculates an affine matrix of 2D rotation.
  -center	Center of the rotation in the source image.
  -angle	Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
  -scale	Isotropic scale factor.
3.cv.warpAffine()Applies an affine transformation to an image.
code:
#SCALING
import cv2 
imgclr=cv2.imread("imgred.jpg") 
res = cv2.resize(imgclr,(300,300),interpolation=cv2.INTER_CUBIC) 
cv2.imshow('imagecolor',imgclr)
cv2.imshow('imagecolor1',res)
cv2.waitKey()
#ROTATION
import cv2 
imgclr=cv2.imread("colorimg.jpg") 
(row, col) = imgclr.shape[:2] 
M = cv2.getRotationMatrix2D((col / 2, row/ 2), 45, 1)
res = cv2.warpAffine(imgclr, M, (col,row)) 
cv2.imshow('imagecolor',imgclr)
cv2.imshow('imagecolor1',res)
cv2.waitKey()
output:
![p2](https://user-images.githubusercontent.com/72303060/104431007-6c52c900-55ad-11eb-8b32-ed7dc59e17d2.png)
3.	Develop a program to find the sum and mean of a set of images. 
  a.	Create ‘n’ number of images and read them from the directory and perform the operations.
  you can use os.listdir() to get the names of all images in your specified path . Then you can loop over the names of images
   processing a multiple image consists of-
    1.Having the images in a directory e.g. foo/
    2.Getting the list of all images in the foo/ directory.
    3.Loop over the list of images.
    mean:'mean' value gives the contribution of individual pixel intensity for the entire image & variance is normally used to find how each pixel varies from the neighbouring pixel (or centre pixel) and is used in classify into different regions.
    sum:adds the value of each pixel in one of the input images with the corresponding pixel in the other input image and returns the sum in the corresponding pixel of the output image. 
  functions:
  listdir() method in python is used to get the list of all files and directories in the specified directory. If we don't specify any directory, then list of files and directories in the current working directory will be returned
code:
import cv2
import os
path='D:\imageip'
imgs=[]
dirs = os.listdir(path)
for file in dirs :
    fpat=path+"\\"+file
    imgs.append(cv2.imread(fpat))
i=0
sum_img=[]
for sum_img in imgs:
    read_img=imgs[i]
    sum_img=sum_img+read_img
    i = i +1
    cv2.imshow('sum',sum_img)
    print(sum_img)
    cv2.imshow('mean',sum_img/i)
    mean=(sum_img/i)
    print(mean)
    cv2.waitKey()
    cv2.destroyAllWindows()
  



