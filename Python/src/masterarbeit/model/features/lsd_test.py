import cv2
import numpy as np
from skimage.morphology import remove_small_objects

image1 = cv2.imread('C:\\Users\\chris\\Desktop\\DSC_6532_cropped.jpg')
image1 = cv2.resize(image1, (int(image1.shape[1]/3), int(image1.shape[0]/3)))
image2 = cv2.imread('C:\\Users\\chris\\Desktop\\DSC_6532_cropped2.jpg')

gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
background_mask = (gray == 255).astype(np.uint8) * 255
background_mask = cv2.GaussianBlur(background_mask, (0,0), 5)
background_mask = (background_mask != 0)

for sigma in [0, 1, 2]:
  if sigma > 0:
    blurred = cv2.GaussianBlur(gray, (0,0), sigma)
  else:
    blurred = gray
  edges = cv2.Canny(blurred, 0, 20, 3)            
  edges[background_mask] = 0
  if sigma == 2:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = remove_small_objects(edges, min_size = 1000)
    minLineLength = 10
    maxLineGap = 1
    line_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(edges,1,np.pi/180, 30,minLineLength,maxLineGap)
    if lines is not None:
      for line in lines:
        for x1,y1,x2,y2 in line:
          cv2.line(line_img, (x1,y1), (x2,y2), (0,255,0), thickness=1)
          cv2.imshow('hough lines', line_img)
  
  cv2.imshow('{}'.format(sigma), edges)
  edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
        

cv2.waitKey()