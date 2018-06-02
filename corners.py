#!/usr/bin/env python
import cv2
import numpy as np
import svgwrite

## TODO:
# gui to pick number of points and edge algorithm
# color and fill options
def corners(fn, numcorners=500, color=True):
    img = cv2.imread(fn)
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,500)
    outshape = img.shape if color else gray.shape
    out = np.ones(outshape, np.uint8)*255

    corners = cv2.goodFeaturesToTrack(edges,numcorners,0.01,10)
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
        fill = svgwrite.rgb(mean[2],mean[1],mean[0]) if color else gbr2gray(mean) 
        dwg.add(dwg.polygon(points=pts,fill=fill))

    cv2.imwrite('corners.jpg',out)
    dwg.save()

def gbr2gray(mean):
    mean = [m/float(255) for m in mean]
    Y_lin = 0.2126*mean[2] + 0.7152*mean[1] + 0.0722*mean[0]
    Y_srgb = 12.92*Y_lin if (Y_lin < 0.0031308) else 1.055*Y_lin**(1/2.4) - 0.055
    Y_srgb = int(Y_srgb*255)
    return svgwrite.rgb(Y_srgb, Y_srgb, Y_srgb)
