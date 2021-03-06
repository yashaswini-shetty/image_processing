# image_processing
## 1.	Develop a program to display grayscale image using read and write operation.
Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.

Importance of grayscaling –

Dimension reduction: For e.g. In RGB images there are three color channels and has three dimensions while grayscaled images are single dimensional.
Reduces model complexity: Consider training neural article on RGB images of 10x10x3 pixel.The input layer will have 300 input nodes. On the other hand, the same neural network will need only 100 input node for grayscaled images.
functions:
1.the function imread() is used for reading an image
2.the function imwrite() is used to write an image in memory to disk and
3.the function imshow() in conjunction with namedWindow and waitKey is used for displaying an image in memory.
code:
```python
#RGB TO GRAY 
import cv2
imgclr=cv2.imread("blue1.jpg",1)
imggry = cv2.cvtColor(imgclr, cv2.COLOR_BGR2GRAY)
cv2.imshow('imagecolor',imgclr)
cv2.imshow('imagecolor1',imggry)
cv2.imwrite('grayimg.jpg',imggry)
cv2.waitKey()
```

output:
![p1](https://user-images.githubusercontent.com/72303060/104428017-f13be380-55a9-11eb-860b-6a90adaf6008.png)

## 2.	Develop a program to perform linear transformations on an image: Scaling and Rotation 
Scaling is just resizing of the image. OpenCV comes with a function cv.resize() for this purpose. The size of the image can be specified manually, or you can specify the scaling factor. Different interpolation methods are used. Preferable interpolation methods are cv.INTER_AREA for shrinking and cv.INTER_CUBIC (slow) & cv.INTER_LINEAR for zooming. By default, the interpolation method cv.INTER_LINEAR is used for all resizing purposes.

Images can be rotated to any degree clockwise or otherwise. We just need to define rotation matrix listing rotation point, degree of rotation and the scaling factor.

functions:
1.To resize an image in Python, you can use cv2.resize() function of OpenCV library cv2.
2.cv2.getRotationMatrix2D(	center, angle, scale	)=Calculates an affine matrix of 2D rotation.
  -center	Center of the rotation in the source image.
  -angle	Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
  -scale	Isotropic scale factor.
3.cv2.warpAffine()Applies an affine transformation to an image.
code:
```python#SCALING
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
```

output:
1.![p2](https://user-images.githubusercontent.com/72303060/104431007-6c52c900-55ad-11eb-8b32-ed7dc59e17d2.png)
2.![p1b](https://user-images.githubusercontent.com/72303060/104433714-6dd1c080-55b0-11eb-8f40-c9eddd68cf15.png)

## 3.	Develop a program to find the sum and mean of a set of images. 
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
```python
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
    
```    
 output:
 ![p3](https://user-images.githubusercontent.com/72303060/104434149-e769ae80-55b0-11eb-80b8-29fa4af344e5.png)


## 4.	Develop a program to convert the color image to gray scale and binary image.
Images are composed of Pixels and in Binary image every pixel value is either 0 or 1 i.e either black or white. it is called bi-level or two level image
while in gray scale ; image can have any value between 0 to 255 for 8-bit color(every pixel is represented by 8 bits) i.e it can have transition between pure black or pure white . It only have intensity value.
So, Gray Scale image can have shades of grey varying between Black and white while Binary image can either of two extreme for a pixel value either white or black
Converting an image to black and white with OpenCV can be done with a simple binary thresholding operation.
Converting an image to black and white involves two steps.
  1.Read the source image as grey scale image.
  2.Convert the grey scale image to binary with a threshold of your choice
  functions:
  1.cv2.cvtColor() method is used to convert an image from one color space to another. 
  2.cv2.threshold is used to apply the thresholding. The first argument is the source image, which should be a grayscale image. The second argument is the threshold value which is used to classify the pixel values. The third argument is the maximum value which is assigned to pixel values exceeding the threshold.
  code:
  ```python
  import cv2  
image = cv2.imread('blue1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, bw_img = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) 
cv2.imshow('Original image',image)
cv2.imshow('Gray image', gray)
cv2.imshow("Binary Image",bw_img)
cv2.waitKey(0)
```

output:
![p4](https://user-images.githubusercontent.com/72303060/104435844-dde14600-55b2-11eb-80cf-5c65c761f271.png)

## 5.	Develop a program to convert the given color image to different color spaces. 
Color spaces are different types of color modes, used in image processing and signals and system for various purposes. Some of the common color spaces are:
RGB 
In the RGB model, each color appears in its primary components of red, green and blue.
GRAY
Gray level images use a single value per pixel that is called intensity or brightness
HSL
HSL stands for hue, saturation, and lightness.
HSV
The HSV (Hue, Saturation, Value) model, also known as HSB (Hue, Saturation, Brightness), defines a color space in terms of three constituent components
YUV 
The YUV color model is the basic color model used in analogue color TV broadcasting. 
functions:
cv2.cvtColor() method is used to convert an image from one color space to another. 

code:
```python
import cv2  
image = cv2.imread('dog.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsl=cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
yuv=cv2.cvtColor(image,cv2.COLOR_BGR2YUV)

cv2.imshow('Original image',image)
cv2.imshow('gray image',gray)
cv2.imshow('hsl image',hsl)
cv2.imshow('hsv image',hsv)
cv2.imshow('yuv image',yuv)

cv2.waitKey(0)
```
output:


![p5](https://user-images.githubusercontent.com/72303060/104437750-31ed2a00-55b5-11eb-97f1-898cc79ed738.png)
![p5b](https://user-images.githubusercontent.com/72303060/104437774-374a7480-55b5-11eb-8953-889854a0f509.png)

## 6.	Develop a program to create an image from 2D array (generate an array of random size).
Numpy or Numeric python is a popular library for array manipulation since images are just an array of pixels carrying various color codes.Numpy can be used to convert an array to image.Every array can't be converted into an image because each pixel of an image consists of specific color codes and if the given array is not in a suitable format the libraries wont be able to process it properly
code:
```python
import numpy, cv2
img = numpy.zeros([200,200,3])
img[:,:,0] = numpy.ones([200,200])*255
img[:,:,1] = numpy.ones([200,200])*255
img[:,:,2] = numpy.ones([200,200])*0
cv2.imwrite('color_img.jpg', img)
cv2.imshow('Color image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
output:


![p6](https://user-images.githubusercontent.com/72303060/104438913-8ba22400-55b6-11eb-941d-9da6aff8cf9d.png)

## 7.Write a program to find the sum of neighbour values in a matrix.
The adjacent elements of matrix can be top, down, left, right, diagonal or anti diagonal. The four or more numbers should be adjacent to each other.
8-connected pixels are neighbors to every pixel that touches one of their edges or corners. These pixels are connected horizontally, vertically, and diagonally. In addition to 4-connected pixels, each pixel with coordinates {\displaystyle \textstyle (x\pm 1,y\pm 1)}\textstyle(x\pm1,y\pm1) is connected to the pixel at {\displaystyle \textstyle (x,y)}\textstyle(x,y).
all the adjacent elements are added to get sum of the neighbour values in a matrix except itself.
functions:
np.asarray()-Convert the input to an array. Input data, in any form that can be converted to an array. This includes lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
np.zeros()-Python numpy. zeros() function returns a new array of given shape and type, where the element's value as 0.
shape()-The function "shape" returns the shape of an array. The shape is a tuple of integers. These numbers denote the lengths of the corresponding array dimension.
code:
```python
import numpy as np
M = [[1, 4, 3],
    [9, 8, 6],
    [5, 2, 7]] 
M = np.asarray(M)
N = np.zeros(M.shape)
def sumNeighbors(M,x,y):
    l = []
    for i in range(max(0,x-1),x+2): 
        for j in range(max(0,y-1),y+2):
            try:
                t = M[i][j]
                l.append(t)
            except IndexError: 
                pass
    return sum(l)-M[x][y] 
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        N[i][j] = sumNeighbors(M, i, j)
print ("Original matrix:\n", M)
print ("Summed neighbors matrix:\n", N)
```
output:
```
Original matrix:
 [[1 4 3]
 [9 8 6]
 [5 2 7]]
Summed neighbors matrix:
 [[21. 27. 18.]
 [20. 37. 24.]
 [19. 35. 16.]]
```
## 7.Write a C++ program to perform operator overloading.
In C++, we can make operators to work for user defined classes. This means C++ has the ability to provide the operators with a special meaning for a data type, this ability is known as operator overloading.
For example, we can overload an operator ‘+’ in a class like String so that we can concatenate two strings by just using +.
To overload +, –, * operators, we will create a class named matrix and then make a public function to overload the operators.

To overload operator ‘+’ use prototype:
Return_Type classname :: operator +(Argument list)
{
    // Function Body
}
To overload operator ‘-‘ use prototype:
Return_Type classname :: operator -(Argument list)
{
    // Function Body
}
To overload operator ‘*’ use prototype:
Return_Type classname :: operator *(Argument list)
{
    // Function Body
}

code:
```c++
#include <iostream>
using namespace std;
class matrix
{
 int r1, c1, i, j, a1;
 int a[10][10];

public:int get()
 {
		cout << "Enter the row and column size for the  matrix\n";
		cin >> r1;
		cin >> c1;
			cout	<< "Enter the elements of the matrix\n";
		for (i = 0; i < r1; i++)
		{
			for (j = 0; j < c1; j++)
			{
				cin>>a[i][j];

			}
		}
	
	
 };
 void operator+(matrix a1)
 {
	int	c[i][j];
		
			for (i = 0; i < r1; i++)
			{
				for (j = 0; j < c1; j++)
				{
					c[i][j] = a[i][j] + a1.a[i][j];
				}
			
		}
		cout<<"addition is\n";
		for(i=0;i<r1;i++)
		{
			cout<<" ";
			for (j = 0; j < c1; j++)
			{
				cout<<c[i][j]<<"\t";
			}
			cout<<"\n";
		}

 };

		void operator-(matrix a2)
 {
	int	c[i][j];
		
			for (i = 0; i < r1; i++)
			{
				for (j = 0; j < c1; j++)
				{
					c[i][j] = a[i][j] - a2.a[i][j];
				}
			
		}
		cout<<"subtraction is\n";
		for(i=0;i<r1;i++)
		{
			cout<<" ";
			for (j = 0; j < c1; j++)
			{
				cout<<c[i][j]<<"\t";
			}
			cout<<"\n";
		}
	};

 void operator*(matrix a3)
 {
		int c[i][j];

		for (i = 0; i < r1; i++)
		{
			for (j = 0; j < c1; j++)
			{
				c[i][j] =0;
				for (int k = 0; k < r1; k++)
				{
					c[i][j] += a[i][k] * (a3.a[k][j]);
				}
		}
		}
		cout << "multiplication is\n";
		for (i = 0; i < r1; i++)
		{
			cout << " ";
			for (j = 0; j < c1; j++)
			{
				cout << c[i][j] << "\t";
			}
			cout << "\n";
		}
 };

};

int main()
{
 matrix p,q;
 p.get();
	q.get();
 p + q;
	p - q;
	p * q;
return 0;
}
```
output:
```
Enter the row and column size for the  matrix
2
2
Enter the elements of the matrix
3
2
4
6
Enter the row and column size for the  matrix
2
2
Enter the elements of the matrix
2
1
2
1
addition is
 5      3
 6      7
subtraction is
 1      1
 2      5
mul is
 10     5
 20     10
```
## 8.Write a program to find the neighbour values in a matrix.
The adjacent elements of matrix can be top, down, left, right, diagonal or anti diagonal. The four or more numbers should be adjacent to each other.
8-connected pixels are neighbors to every pixel that touches one of their edges or corners. These pixels are connected horizontally, vertically, and diagonally. In addition to 4-connected pixels, each pixel with coordinates {\displaystyle \textstyle (x\pm 1,y\pm 1)}\textstyle(x\pm1,y\pm1) is connected to the pixel at {\displaystyle \textstyle (x,y)}\textstyle(x,y).
```python
import numpy as np
ini_array = np.array([[1, 2,5, 3], [4,5, 4, 7], [9, 6, 1,0]])
print("initial_array : ", str(ini_array));
def neighbors(radius, rowNumber, columnNumber):
    return[[ini_array[i][j]if i >= 0 and i < len(ini_array) and j >= 0 and j < len(ini_array[0]) else 0
            for j in range(columnNumber-1-radius, columnNumber+radius)]
           for i in range(rowNumber-1-radius, rowNumber+radius)]
neighbors(1, 2, 2)
```
output:
initial_array :  [[1 2 5 3]
 [4 5 4 7]
 [9 6 1 0]]
[[1, 2, 5], [4, 5, 4], [9, 6, 1]]
## 9.develop a program to implement negative transformation of an image
The negative transformation is a point processing function which inverts each pixel in an image and is given by s=L-1-r. The Python example applies the negative transformation to an image and displays the output.
Image is also known as a set of pixels. When we store an image in computers or digitally, it’s corresponding pixel values are stored. So, when we read an image to a variable using OpenCV in Python, the variable stores the pixel values of the image. When we try to negatively transform an image, the brightest areas are transformed into the darkest and the darkest areas are transformed into the brightest.
```python
import cv2
import numpy as np
# Load the image
img = cv2.imread('a21.jpg')
img_neg = 255 - img
# Show the image
cv2.imshow('image',img)
cv2.imshow('negative',img_neg)
cv2.waitKey(0)
```
output:
![negtrans](https://user-images.githubusercontent.com/72303060/105331021-fa0b6580-5bf8-11eb-967f-493e0d2c704f.png)

## 10.Develop a program to implement contrast transformation of an image
Contrast stretching as the name suggests is an image enhancement technique that tries to improve the contrast by stretching the intensity values of an image to fill the entire dynamic range. The transformation function used is always linear and monotonically increasing.
Contrast is the difference in brightness between objects or regions
```python
import cv2
image = cv2.imread('a11.jpg')
alpha = 1.2# Contrast control (1.0-3.0)
beta = 20.0# Brightness control (0-100)
adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
cv2.imshow('original', image)
cv2.imshow('adjusted', adjusted)
cv2.waitKey()
```
output:
![contrsttrans](https://user-images.githubusercontent.com/72303060/105331015-f8da3880-5bf8-11eb-81b5-2715df84471f.png)

## 11.Develop a program to implement thresold transformation of an image
n Image Processing, Thresholding is a kind of Segmentation – it separates pixels into two or more categories.
In its simplest form, a Thresholding operation of an Image involves classification of the pixels into two groups based on a Threshold:
Pixels that exceed a given intensity Threshold.Pixels that do not exceed an intensity Threshold and transforming those two kinds of pixels into two colors, say black and white, typically black for the background and the white for the object(s) identified.Using single threshold to separate pixels and mapping them to two colors is also called as Global Thresholding.Thresholding results in finding objects from the background surrounding them.
```python
import cv2  
import numpy as np  
image1 = cv2.imread('a11.jpg')  
ret, thresh1 = cv2.threshold(image1, 120, 255, cv2.THRESH_TOZERO) 
img3 = cv2.hconcat([image1,thresh1])
cv2.imshow('thresold image', img3) 
cv2.waitKey()
```
output:
![threshtrans](https://user-images.githubusercontent.com/72303060/105331025-fbd52900-5bf8-11eb-8e3b-90eb4aad7679.png)

## 12.Develop a program to implement power law transformation of an image
Power-law (gamma) transformations can be mathematically expressed as s = cr^{\gamma}. Gamma correction is important for displaying images on a screen correctly, to prevent bleaching or darkening of images when viewed from different types of monitors with different display settings. This is done because our eyes perceive images in a gamma-shaped curve, whereas cameras capture images in a linear fashion
```python
import numpy as np
import cv2
# Load the image
img = cv2.imread('a11.jpg')
# Apply Gamma=2.2 on the normalised image and then multiply by scaling constant (For 8 bit, c=255)
gamma_1 = np.array(255*(img/255)**2.2,dtype='uint8')
# Similarly, Apply Gamma=0.4 
gamma_2 = np.array(255*(img/255)**0.4,dtype='uint8')
gamma_3 = np.array(255*(img/255)**0.2,dtype='uint8')
gamma_4 = np.array(255*(img/255)**3.2,dtype='uint8')
# Display the images in subplots
img3 = cv2.hconcat([gamma_1,gamma_2,gamma_3,gamma_4])
cv2.imshow('power law trans',img3)
cv2.waitKey(0)
```
output:
![powerlawtrans](https://user-images.githubusercontent.com/72303060/105331022-fb3c9280-5bf8-11eb-97c2-0340ac92cae3.png)

