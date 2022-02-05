import pandas as pd
import cv2
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import glob
import re

df=pd.read_csv("C:\\Users\\varun\\Downloads\\Dataset - Sheet1(1).csv",header=0)

grouped=df.groupby('Enrollment Number')
dir_name="D:\\sem 6\\AI\\project\\images\\"
images=[]
fineigenface=[]
finweights=[]
def readdim(path):
    #print(path)
    image = cv2.imread(path)
    images = cv2.resize(image, (900,900))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grays = cv2.resize(gray, (900,900))
    img1 = Image.open(path).convert('L')
    img1_as_np = np.asarray(img1)
    vec1 = img1_as_np.flatten()
    return vec1

def eigenfaces(vec):
    global fineigenface
    global finweights
    A = np.column_stack((vec[0],vec[1],vec[2],vec[3],vec[4],vec[5],vec[6]))
    A = A.T
    # calculate the mean of each column
    M = mean(A, axis=0)
    print(M)
#C = A - M
#-----------------------------------------------------------------------#
#print(C)

    V = cov(A)
    g ,h = np.linalg.eig(V) #returns eigen valus, eigen vectors
    print('cov')
    #V = np.dot(C,(C.T))
    #print(V)
    #values, vectors = eig(V)
    #u = np.sort(g)
    u2 = h[:,g.argsort()]
    A=A.T
    fineg = np.dot(A,u2)
    print(fineg)
    eigenfaces=[]
    for k in range(7):
        r="Resized Window"+str(k)
        fin1 = fineg[:,k]
        #fin1 = np.array(fin1)
        #fin1 = np.reshape(fin1,(3456,3456))
        #print(fin1)
        eigenfaces.append(fin1)
    fineigenface.append(eigenfaces)
    A=np.transpose(A)
    ##print(A)
    ##print(len(A))
    print(len(A[0]))
    for i in range(len(A)):
        weights=[]
        for j in range(len(eigenfaces)):
            E=np.reshape(eigenfaces[j],(11943936,1))
            B=np.reshape(A[i],(1,11943936))
            a5=np.dot(B,E)
            weights.append(a5)
            #print("Weights",a5)
        finweights.append(weights)


'''
take test image(tst) input
for i in range(len(tst)):
        weights=[]
        for j in range(len(eigenfaces)):
            E=np.reshape(eigenfaces[j],(11943936,1))
            B=np.reshape(A[i],(1,11943936))
            a5=np.dot(B,E)
            weights.append(a5)
            #print("Weights",a5)
        tstweight.append(weights)
mindis = 100000
comp(tstweight):
        
            for j in finweights:
                for k in range(len(j)):
                    dis += tstweight[k]-j[k]
                if(dis<mindis):
                    mindis = dis
                    lox = finweights.index(j)
dict = {'J':0;'P':6;'U':12;'Y':18;'V':24;'N':30}
                        




'''
rollno=["16BCP006","16BCP016","16BCP027","16BCP028","16BCP064"]

#rollno=["16BCP028"]
for name,group in grouped:
    if(name in rollno):
        vec=[]
        for i in group['Image_Name']:
            img=""
            #print("gsdb",img)
            img=str(dir_name+i)
            #print(img[7:])
            t=readdim(img)
            #print(t)
            vec.append(t)
        eigenfaces(vec)        
test_image=readdim("D:\\sem 6\\AI\\project\\images\\16BCP064_Female_Sad_RGB.png")

