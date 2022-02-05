from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


image = cv2.imread('C:/Users/varun/images/test.png')
images = cv2.resize(image, (900,900))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grays = cv2.resize(gray, (900,900))
img1 = Image.open('C:/Users/varun/images/test.png').convert('L')
img1_as_np = np.asarray(img1)
vec1 = img1_as_np.flatten()

image2 = cv2.imread('C:/Users/varun/images/test2.png')
images2 = cv2.resize(image, (900,900))
gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grays2 = cv2.resize(gray, (900,900))
img2 = Image.open('C:/Users/varun/images/test2.png').convert('L')
img2_as_np = np.asarray(img2)
vec2 = img2_as_np.flatten()


image3 = cv2.imread('C:/Users/varun/images/test3.png')
images3 = cv2.resize(image, (900,900))
gray3 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grays3 = cv2.resize(gray, (900,900))
img3 = Image.open('C:/Users/varun/images/test3.png').convert('L')
img3_as_np = np.asarray(img3)
vec3 = img3_as_np.flatten()

image4 = cv2.imread('C:/Users/varun/images/test4.png')
images4 = cv2.resize(image, (900,900))
gray4 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grays4 = cv2.resize(gray, (900,900))
img4 = Image.open('C:/Users/varun/images/test4.png').convert('L')
img4_as_np = np.asarray(img4)
vec4 = img4_as_np.flatten()

A = np.column_stack((vec1,vec2,vec3,vec4))
print(A)
A = A.T
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# define a matrix
#A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# calculate the mean of each column
M = mean(A, axis=0)
print(M)
#C = A - M
#-----------------------------------------------------------------------#

#print(C)

V = cov(A)
g ,h = np.linalg.eig(V)
print('cov')
#V = np.dot(C,(C.T))
#print(V)
#values, vectors = eig(V)
u = np.sort(g)
u2 = h[:,g.argsort()]
A=A.T
fineg = np.dot(A,u2)
print(fineg)

fin1 = fineg[:,0]
#fin2 = fineg[:,1]
#fin3 = fineg[:,2]
fin1 = np.array(fin1)
fin1 = np.reshape(fin1,(3456,3456))
cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)

#resize the window according to the screen resolution
cv2.resizeWindow('Resized Window', 600, 600)
cv2.imshow('Resized Window',fin1.astype("uint8"))

#print(fin1)
#plt.imshow(fin1, cmap="gray")
#plt.show()
#eweig = np.dot((A.T),A)
#print("================")
#print(eweig)
#val ,neweig = eig(eweig)
#print('========================================')
#print(neweig)
#print(type(neweig))
#onedneweig=neweig.flatten()
#print(onedneweig)
#plt.imshow(neweig, cmap="gray")
#plt.show()
#print(vectors)
#neweig = np.dot(vectors,)
#print(values)
#vectors.resize(3456)
#P = vectors.T.dot(C.T)
#print(P.T)
#fin1 = vectors[:,0]
#fin2 = vectors[:,1]
#fin3 = vectors[:,2]
#cv::imshow("test", eigen2cv(fin1));
#im = Image.fromarray(vectors)
#im.show()
