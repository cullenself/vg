#!/usr/bin/env python
import cv2
import numpy as np
import math

scale = 20 
cycles = 3

img = cv2.imread('face.jpg')
height, width, _ = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mini = cv2.resize(gray, None, fx = 1/float(scale), fy = 1/float(scale), interpolation = cv2.INTER_AREA)
out = np.ones(gray.shape, np.uint8)*255

for i in range(0,mini.shape[0]):
    for j in range(0,mini.shape[1]):
        cv2.rectangle(out, (j*scale,i*scale), ((j+1)*scale,(i+1)*scale), int(mini[i,j]),-1)

cv2.imshow('rect',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
