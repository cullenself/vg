#!/usr/bin/env python
import cv2
import numpy as np
import math

## TODO: convert to gui with sliders for params and variable input
scale = 40 
sin_scale = 5 
cycles = 3 

img = cv2.imread('Half.jpg')
height, width, _ = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mini = cv2.resize(gray, None, fx = 1/float(scale), fy = 1/float(scale), interpolation = cv2.INTER_AREA)
mini_blur = cv2.GaussianBlur(mini, (5,5), 0)
inv_mini_blur = cv2.bitwise_not(mini_blur)
out = np.ones(gray.shape, np.uint8)*255

# Generate Base Curve
x = np.arange(0,cycles*2*math.pi,0.25)
y = np.sin(x)
# Scale to Box
x = x*width/(cycles*2*math.pi*mini.shape[1])
y = y*height/float(4*mini.shape[0])

for j in range(0,mini.shape[0]):
    for i in range(0,mini.shape[1]):
        # scale the wave and draw the line 
        temp = zip(x+i*width/float(mini.shape[1]), y*inv_mini_blur[j,i]*sin_scale/255 + (j+0.5)*height/float(mini.shape[0]))
        pts = np.array(temp).reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(out, [pts], False, 0)
        
#cv2.imshow('rect',mini_blur)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite('waves.jpg',out)
