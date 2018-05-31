#!/usr/bin/env python
import cv2
import numpy as np
import math
import svgwrite

## TODO: convert to gui with sliders for params and variable input
def waves(fn,scale=20,sin_scale=20,pix_per_cycle=10):
    img = cv2.imread(fn)
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mini = cv2.resize(gray, None, fx = 1, fy = 1/float(scale), interpolation = cv2.INTER_AREA)
    mini_blur = cv2.GaussianBlur(mini, (5,5), 0)
    inv_mini_blur = cv2.bitwise_not(mini_blur)
    out = np.ones(gray.shape, np.uint8)*255

    # Instantiat SVG Output
    dwg = svgwrite.Drawing('waves.svg', size=(width,height))

    xs = np.arange(0,mini.shape[1])
    ys = (inv_mini_blur/float(255))*sin_scale*np.sin(xs*2*math.pi/pix_per_cycle) + (np.matrix(range(0,mini.shape[0])).T + 0.5)*height/mini.shape[0]

    for row in ys:
        pts = zip(xs, np.asarray(row).squeeze())
        pts = np.array(pts).reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(out, [pts], False, 0)
        pts = [(int(p[0,0]),int(p[0,1])) for p in pts]
        dwg.add(dwg.polyline(points=pts, stroke='black', fill='none'))

    cv2.imwrite('waves.jpg',out)
    dwg.save()
