#!/usr/bin/env python
import cv2
import numpy as np
import math
import svgwrite

## TODO: use polylines to make contours to make a mask to find the average value to color the triangles
# gui to pick number of points and edge algorithm
# color and fill options
img = cv2.imread('Half.jpg')
height, width, _ = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,500)
out = np.ones(img.shape, np.uint8)*255

corners = cv2.goodFeaturesToTrack(edges,500,0.01,10)
corners = np.int0(corners)
subdiv = cv2.Subdiv2D((0,0,width,height))

# Instantiate SVG Output
dwg = svgwrite.Drawing('corners.svg', size=(width,height))

for i in corners:
    x, y = i.ravel()
    subdiv.insert((x,y))

triList = subdiv.getTriangleList()
for t in triList:
    pts = np.array([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]).reshape((-1,1,2)).astype(np.int32)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [pts], 0, 255, -1)
    mean = cv2.mean(img, mask = mask)
    cv2.polylines(out, [pts], True, 0)
    cv2.fillPoly(out, [pts], mean)
    pts = [(int(p[0,0]),int(p[0,1])) for p in pts]
    dwg.add(dwg.polygon(points=pts,fill=svgwrite.rgb(mean[2],mean[1],mean[0])))

#cv2.imshow('rect',edges)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite('corners.jpg',out)
dwg.save()
