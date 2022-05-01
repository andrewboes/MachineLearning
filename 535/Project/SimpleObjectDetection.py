import cv2 #pip install opencv-python
#import matplotlib.image as img
from matplotlib import pyplot as plt
import numpy as np
import math
#import torch

def parseVideo():
  vidcap = cv2.VideoCapture('2022-04-16T07_00_00.487491-07_00.mp4')
  success,image = vidcap.read()
  count = 0
  while success:
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', count)
    count += 1

def main():
  #parseVideo()
  for i in range(7200):
    print(i)
    image = cv2.imread("./frames/frame%d.jpg" %(i))
    
    #img = cv2.imread("digitbox.jpg", 0)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.ones_like(img) * 255
    boxes = []
    
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            hull = cv2.convexHull(contour)
            cv2.drawContours(mask, [hull], -1, 0, -1)
            x,y,w,h = cv2.boundingRect(contour)
            boxes.append((x,y,w,h))
    
    boxes = sorted(boxes, key=lambda box: box[0])
    
    mask = cv2.dilate(mask, np.ones((5,5),np.uint8))
    
    img[mask != 0] = 255
    
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    showImage = False
    
    for n,box in enumerate(boxes):
        x,y,w,h = box
        diag = math.sqrt(w**2+h**2)
        if diag > 170 and h > 50 and w < 1150:
          showImage = True
          print(diag)
          print(h)
          print(w)
        cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(result, str(n),(x+5,y+17), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0),2,cv2.LINE_AA)
    
    if showImage:
      cv2.imshow("resut", result)
      cv2.waitKey(0)
    
    #cv2.imwrite('digitbox_result.png', result)
    
  cv2.destroyAllWindows()
    
if __name__=="__main__":
  main()
