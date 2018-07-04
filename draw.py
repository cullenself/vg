import cv2
import numpy as np
import svgwrite
import os.path
import vg
import math
import base64
from random import randint

def readImageFromFileName(fn):
    # cv2.imread does not throw an error if the input is bad
    # Check if the path is at least valid
    if not os.path.isfile(fn):
        raise IOError('Input file not found: \'' + fn + '\'')
    i = {}
    i['img'] = cv2.imread(fn) # this honestly creates a hassle, but makes
    i['h'], i['w'], _ = i['img'].shape # heighth and width easier to access
    return i

def readImageFromFile(f):
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    i = {}
    i['img'] = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    i['h'], i['w'], _ = i['img'].shape
    return i

def cleanOutname(on):
    # Setup output
    base = os.path.splitext(on)[0]
    out = {}
    out['im'] = base + vg.EXT_IMG
    out['svg'] = base + '.svg'
    return out

def gbr2gray(mean):
    # Luminosity weighted grayscale conversion
    mean = [m/float(255) for m in mean]
    Y_lin = 0.2126*mean[2] + 0.7152*mean[1] + 0.0722*mean[0]
    Y_lin = int(Y_lin*255)
    return svgwrite.rgb(Y_lin, Y_lin, Y_lin)

def getTriangles(gray):
    # Get corners and create list of Delaunay triangles
    edges = cv2.Canny(gray, vg.EDGE_LOW, vg.EDGE_HIGH)
    corners = np.int0(cv2.goodFeaturesToTrack(edges,vg.CORNERS_NUM,vg.CORNERS_LOW,vg.CORNERS_HIGH))
    subdiv = cv2.Subdiv2D((0,0,gray.shape[1],gray.shape[0]))
    for i in corners:
        x, y = i.ravel()
        subdiv.insert((x,y))
    return subdiv.getTriangleList()


def edges(img, outname='edges', svgOutput=False, jpgOutput=False, b64Output=False):
    # Only draw the edges of objects, all in grayscale
    outname = cleanOutname(outname)
    
    # Find edges and then make into list of contours
    gray = cv2.cvtColor(img['img'], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, vg.EDGE_HIGH, vg.EDGE_LOW)
    thresh = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY, vg.THRESH_BLOCK, vg.THRESH_C)
    out, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.bitwise_not(out)
    
    if jpgOutput:
        cv2.imwrite(outname['im'], out)

    # Draw the contours onto an SVG
    if svgOutput:
        dwg = svgwrite.Drawing(outname['svg'], size=(img['w'],img['h']))
        for line in contours:
            pts = [(int(p[0,0]),int(p[0,1])) for p in line]
            dwg.add(dwg.polyline(points=pts, stroke='black', fill='none'))
        dwg.save()

    if b64Output:
        _, buff = cv2.imencode('.jpg', out)
        return base64.b64encode(buff)
    else:
        return

def rect(img, color=True, outname='rect', svgOutput=False, jpgOutput=False, b64Output=False):
    # Resample image into pixel blocks
    outname = cleanOutname(outname)

    # Set up base image and output
    gray = cv2.cvtColor(img['img'], cv2.COLOR_BGR2GRAY)
    outshape = img['img'].shape if color else gray.shape
    out = np.ones(outshape, np.uint8)*255

    # Instantiate SVG Outpu
    if svgOutput:
        dwg = svgwrite.Drawing(outname['svg'], size=(img['w'],img['h']))

    # Resample image by shrinking, then draw rectangles
    mini = cv2.resize(img['img'], None, fx = 1/float(vg.RECT_SCALE), fy = 1/float(vg.RECT_SCALE), interpolation = cv2.INTER_AREA)
    for i in range(0,mini.shape[0]):
        for j in range(0,mini.shape[1]):
            col = tuple([int(c) for c in mini[i,j]])
            cv2.rectangle(out, (j*vg.RECT_SCALE,i*vg.RECT_SCALE), ((j+1)*vg.RECT_SCALE,(i+1)*vg.RECT_SCALE), col, -1)
            if svgOutput:
                fill = svgwrite.rgb(col[2],col[1],col[0]) if color else gbr2gray(col)
                dwg.add(dwg.rect(insert=(j*vg.RECT_SCALE,i*vg.RECT_SCALE),size=(vg.RECT_SCALE,vg.RECT_SCALE),fill=fill))

    # Output
    if jpgOutput:
        cv2.imwrite(outname['im'],out)
    if svgOutput:
        dwg.save()

    if b64Output:
        _, buff = cv2.imencode('.jpg', out)
        return base64.b64encode(buff)
    else:
        return

def corners(img, outname='corners', color=True, svgOutput=False, jpgOutput=False, b64Output=False):
    outname = cleanOutname(outname)

    # Set up base image and output
    gray = cv2.cvtColor(img['img'], cv2.COLOR_BGR2GRAY)
    outshape = img['img'].shape if color else gray.shape
    out = np.ones(outshape, np.uint8)*255

    # Instantiate SVG Output
    if svgOutput:
        dwg = svgwrite.Drawing(outname['svg'], size=(img['w'],img['h']))

    # Draw each triangle
    triList = getTriangles(gray)
    for t in triList:
        pts = np.array([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]).reshape((-1,1,2)).astype(np.int32)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [pts], 0, 255, -1)
        mean = cv2.mean(img['img'], mask = mask)
        cv2.polylines(out, [pts], True, 0)
        cv2.fillPoly(out, [pts], mean)
        if svgOutput:
            pts = [(int(p[0,0]),int(p[0,1])) for p in pts]
            fill = svgwrite.rgb(mean[2],mean[1],mean[0]) if color else gbr2gray(mean) 
            dwg.add(dwg.polygon(points=pts,fill=fill))

    # Finish up output
    if jpgOutput:
        cv2.imwrite(outname['im'],out)
    if svgOutput:
        dwg.save()

    if b64Output:
        _, buff = cv2.imencode('.jpg', out)
        return base64.b64encode(buff)
    else:
        return

def hatch(img,outname='hatch',randAngle=True,angle=70,svgOutput=False, jpgOutput=False, b64Output=False):
    # Setup transformation matrix early
    Q = lambda theta: np.matrix([[math.cos(theta), -math.sin(theta)],[math.sin(theta),math.cos(theta)]])
    Q = Q if randAngle else Q(angle)

    outname = cleanOutname(outname)

    # Setup base image and output
    gray = cv2.cvtColor(img['img'], cv2.COLOR_BGR2GRAY)
    out = np.ones(gray.shape, np.uint8)*255

    # Instantiate SVG Output
    if svgOutput:
        dwg = svgwrite.Drawing(outname['svg'], size=(img['w'],img['h']))

    # Process through each triangle
    i = 0 # Unfortunately need a counter for SVG clip path id's
    triList = getTriangles(gray)
    for t in triList:
        i = i+1
        pts = np.array([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]).reshape((-1,1,2)).astype(np.int32)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [pts], 0, 255, -1)
        x,y,w,h = cv2.boundingRect(pts)
        maxDim = max(w,h)
        numLines = int(cv2.mean(gray, mask = mask)[0]/(255*w*h) * vg.HATCH_DENSITY)
        if (numLines > 1):
            temp = np.zeros(gray.shape, np.uint8)
            R = Q(randint(0,90)) if randAngle else Q
            c = np.array([x+w/2,y+h/2])
            verts = np.linspace(y-2*maxDim,y+2*maxDim,6*numLines)
            lines = [[c+np.dot(R,np.array([x-maxDim,int(y)])-c), c+np.dot(R,np.array([x+2*maxDim,int(y)])-c)] for y in verts] 
            cv2.polylines(out, [pts], True, 0)
            if svgOutput:
                pts = [(int(p[0,0]),int(p[0,1])) for p in pts]
                dwg.add(dwg.polyline(points=pts, stroke='black', fill='none'))
                clip = dwg.defs.add(dwg.clipPath(id='cp_'+str(i)))
                clip.add(dwg.polyline(points=pts))
            for line in lines:
                p1 = (int(line[0][0,0]), int(line[0][0,1]))
                p2 = (int(line[1][0,0]), int(line[1][0,1]))
                cv2.line(temp, p1, p2, 1)
                if svgOutput:
                    dwg.add(dwg.line(start=p1,end=p2,stroke='black',clip_path='url(#cp_'+str(i)+')'))
            out = out + cv2.bitwise_and(mask,temp)

    # Output
    if jpgOutput:
        cv2.imwrite(outname['im'],out)
    if svgOutput:
        dwg.save()

    if b64Output:
        _, buff = cv2.imencode('.jpg', out)
        return base64.b64encode(buff)
    else:
        return

def waves(img, outname='waves', svgOutput=False, jpgOutput=False, b64Output=False):
    outname = cleanOutname(outname)

    # Resize and Resample Image
    gray = cv2.cvtColor(img['img'], cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, (img['w'],vg.SIN_COUNT), interpolation = cv2.INTER_AREA)
    inv_mini = cv2.bitwise_not(mini)
    out = np.ones(gray.shape, np.uint8)*255

    # Instantiate SVG Output
    if svgOutput:
        dwg = svgwrite.Drawing(outname['svg'], size=(img['w'],img['h']))

    xs = np.arange(0,mini.shape[1])
    ys = (inv_mini/float(255))*vg.SIN_HEIGHT*np.sin(xs*2*math.pi/vg.SIN_LENGTH) + (np.matrix(range(0,mini.shape[0])).T + 0.5)*img['h']/mini.shape[0]

    for row in ys:
        pts = list(zip(xs, np.asarray(row).squeeze()))
        pts = np.array(pts).reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(out, [pts], False, 0)
        pts = [(int(p[0,0]),int(p[0,1])) for p in pts]
        if svgOutput:
            dwg.add(dwg.polyline(points=pts, stroke='black', fill='none'))

    if jpgOutput:
        cv2.imwrite(outname['im'],out)
    if svgOutput:
        dwg.save()

    if b64Output:
        _, buff = cv2.imencode('.jpg', out)
        return base64.b64encode(buff)
    else:
        return

methods = {"Edges": edges, "Pixilize": rect, "Waves": waves, "Hatch": hatch, "Corners": corners}
