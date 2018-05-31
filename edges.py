#!/usr/bin/env python
import cv2
import numpy as np
import svgwrite

def edges(fn,high_edge=500,low_edge=100):
    img = cv2.imread(fn) # todo: variable input, also make into function
    height, width, _ = img.shape
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,high_edge,low_edge)

    thresh = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2) 
    out, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    out = cv2.bitwise_not(out)
    cv2.imwrite('edges.jpg',out)

    dwg = svgwrite.Drawing('edges.svg', size=(width,height))

    for line in contours:
        pts = [(int(p[0,0]),int(p[0,1])) for p in line]
        dwg.add(dwg.polyline(points=pts, stroke='black', fill='none'))

    dwg.save()
