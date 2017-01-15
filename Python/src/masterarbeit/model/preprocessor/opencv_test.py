import cv2

filename = 'D:/Eigene Dateien/Dokumente/OneDrive/Dokumente/Studium/Master HTW/Masterarbeit/Repository/Python/src/masterarbeit/DSC_5827.JPG'
image = cv2.imread(filename, cv2.IMREAD_COLOR)
height, width = image.shape[:2]
image = cv2.resize(image,(int(width/4), int(height/4)))
image = cv2.GaussianBlur(image, (5,5), 0)
shape = image.shape
shape = (shape[0] / 4, shape[1] / 4, shape[2])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
edged = cv2.Canny(gray, 30, 200)
im2,contours,hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#for c in contours:
    #print(len(c))
contours = [c for c in contours if len(c) > 500]    
cv2.drawContours(image,contours,-1,(0,255,0),3)
cv2.imshow('',image)
cv2.waitKey(0)
cv2.destroyAllWindows()