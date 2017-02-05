import cv2
import numpy as np

def _contours(filename):
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
    
def _hough_lines(filename):
    
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        
    cv2.imshow('',img)
    cv2.waitKey(0)   
    cv2.destroyAllWindows()   

if __name__ == '__main__':    
    filename = 'D:/Eigene Dateien/Dokumente/OneDrive/Dokumente/Studium/Master HTW/Masterarbeit/Repository/Python/src/masterarbeit/DSC_5827.JPG'    
    _hough_lines(filename)