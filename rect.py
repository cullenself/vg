#!/usr/bin/env python
import cv2
import numpy as np
import math
import svgwrite

scale = 20 

img = cv2.imread('Half.jpg')
height, width, _ = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mini = cv2.resize(img, None, fx = 1/float(scale), fy = 1/float(scale), interpolation = cv2.INTER_AREA)
out = np.ones(img.shape, np.uint8)*255

dwg = svgwrite.Drawing('rect.svg', size=(width,height))

for i in range(0,mini.shape[0]):
    for j in range(0,mini.shape[1]):
        col = tuple([int(c) for c in mini[i,j]])
        cv2.rectangle(out, (j*scale,i*scale), ((j+1)*scale,(i+1)*scale), col, 1)
        dwg.add(dwg.rect(insert=(j*scale,i*scale),size=(scale,scale),fill=svgwrite.rgb(col[2],col[1],col[0])))

cv2.imwrite('rect.jpg',out)
dwg.save()
