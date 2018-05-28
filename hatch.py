#!/usr/bin/env python
import cv2
import numpy as np
import math
from random import randint
import svgwrite

## TODO: use polylines to make contours to make a mask to find the average value to color the triangles
# gui to pick number of points and edge algorithm
# color and fill options
maxDensity = 2500000
Q = lambda theta: np.matrix([[math.cos(theta), -math.sin(theta)],[math.sin(theta),math.cos(theta)]])

print('Processing')
img = cv2.imread('input2.jpg')
height, width, _ = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,500)
out = np.ones(gray.shape, np.uint8)*255

# Instantiate SVG Output
dwg = svgwrite.Drawing('hatch.svg', size=(width,height))

corners = cv2.goodFeaturesToTrack(edges,500,0.01,10)
corners = np.int0(corners)
subdiv = cv2.Subdiv2D((0,0,width,height))

for i in corners:
    x, y = i.ravel()
    subdiv.insert((x,y))

triList = subdiv.getTriangleList()
i = 0
print('Drawing')
for t in triList:
    i = i+1
    pts = np.array([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]).reshape((-1,1,2)).astype(np.int32)
    temp = np.zeros(gray.shape, np.uint8)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [pts], 0, 255, -1)
    x,y,w,h = cv2.boundingRect(pts)
    maxDim = max(w,h)
    numLines = int(cv2.mean(gray, mask = mask)[0]/(255*w*h) * maxDensity)
    if (numLines > 1):
        theta = randint(0,90)
        c = np.array([x+w/2,y+h/2])
        verts = np.linspace(y-2*maxDim,y+2*maxDim,6*numLines)
        lines = [[c+np.dot(Q(theta),np.array([x-maxDim,int(y)])-c), c+np.dot(Q(theta),np.array([x+2*maxDim,int(y)])-c)] for y in verts] 
        # todo rotate
        cv2.polylines(out, [pts], True, 0)
        pts = [(int(p[0,0]),int(p[0,1])) for p in pts]
        dwg.add(dwg.polyline(points=pts, stroke='black', fill='none'))
        clip = dwg.defs.add(dwg.clipPath(id='cp_'+str(i)))
        clip.add(dwg.polyline(points=pts))
        for line in lines:
            p1 = (int(line[0][0,0]), int(line[0][0,1]))
            p2 = (int(line[1][0,0]), int(line[1][0,1]))
            cv2.line(temp, p1, p2, 1)
            dwg.add(dwg.line(start=p1,end=p2,stroke='black',clip_path='url(#cp_'+str(i)+')'))
        out = out + cv2.bitwise_and(mask,temp)

#cv2.imshow('rect',gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

print('Saving')
cv2.imwrite('hatch.jpg',out)
dwg.save()
