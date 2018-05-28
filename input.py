#!/usr/bin/env python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('input.jpg') # todo: variable input, also make into function
height, width, _ = img.shape
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,100,500)

thresh = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2) 
out, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#out = cv2.bitwise_not(out)
#cv2.imwrite('output.jpg',out)

plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(out,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

c = max(contours, key=cv2.contourArea) #max contour
f = open('path.svg', 'w+')
f.write('<svg width="'+str(width)+'" height="'+str(height)+'" xmlns="http://www.w3.org/2000/svg">')
f.write('<path d="M')

for i in xrange(len(c)):
    #print(c[i][0])
    x, y = c[i][0]
    print(x)
    f.write(str(x)+  ' ' + str(y)+' ')

f.write('"/>')
f.write('</svg>')
f.close()
