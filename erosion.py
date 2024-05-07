import cv2 as cv
import numpy as np
from PIL import Image

##Grayscales Image already
img = cv.imread('FDA800d_IPA.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('FDA800d_IPA.jpg', cv.IMREAD_COLOR)
ogImg = cv.imread('FDA800d_IPA.jpg', cv.IMREAD_COLOR)
assert img is not None, "file could not be read, check with os.path.exists()"
kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((10,10),np.uint8)

##Apply Simple Thresholding First
ret,thresh1 = cv.threshold(img,1,255,cv.THRESH_BINARY)
def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True

##Apply Opening to get rid of excess dots (erosion->dilation)
##Apply Closing to fill in black spaces (dilation->erosion)
##*(Combined the two processes together)
firstOpening = cv.erode(thresh1,kernel2,iterations = 1)
opening = cv.dilate(firstOpening,kernel,iterations = 30)
closing = cv.erode(opening,kernel,iterations = 50)
##Finds Contours based on the thresholding models
contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

##Converts original image back to rgb
backtorgb = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
drawnContours = cv.drawContours(img2,contours,-1, (255,0,0), 3)
##**To visualize the contours on the closing threshold model use:
    ##backtorgb = cv.cvtColor(closing,cv.COLOR_GRAY2RGB)
    ##drawnContours = cv.drawContours(backtorgb,contours,-1, (255,0,0), 3)

##Approximate contours to polygons + get bounding rects and circles
contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
centers = [None]*len(contours)
radius = [None]*len(contours)
for i, c in enumerate(contours):
    contours_poly[i] = cv.approxPolyDP(c, 3, True)
    boundRect[i] = cv.boundingRect(contours_poly[i])
    centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
count = 0
for i in range(len(contours)):

    color = (0,255,0)
    if (i ==61 ):
        color = (255,0,0)
    elif (i==2):
        color = (0,0,255)
    x1 = int(boundRect[i][0])
    y1 = int(boundRect[i][1])
    x2 = int((boundRect[i][0]+boundRect[i][2]))
    y2 = int((boundRect[i][1]+boundRect[i][3]))
    ##Increases the scale of the rectangle
    xfactor = int((x2-x1)*0.14)
    yfactor = int((y2-y1)*0.14)
    radiusC = int(radius[i]*1.1)
    
    cv.rectangle(drawnContours, (x1-xfactor,y1-yfactor), \
    (x2+xfactor,y2+yfactor), color, 2)
    cv.circle(drawnContours, (int(centers[i][0]), int(centers[i][1])), radiusC, color, 2)

    ##Actually Cutting out the images
    if (radiusC>600):
            color=(0,0,255)
            roi = ogImg[y1-yfactor:y2+yfactor,x1-xfactor:x2+xfactor]
            cv.imwrite("image"+str(54-count)+".jpg",roi)
            count = count+1
print(count)
cv.imwrite('contours.jpg',drawnContours)


