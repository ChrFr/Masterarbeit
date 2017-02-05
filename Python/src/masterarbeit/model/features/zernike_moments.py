from mahotas.features import zernike_moments
from skimage import measure
from masterarbeit.model.features.feature import Feature
import numpy as np
import cv2

class ZernikeMoments(Feature):
    label = 'Zernike-Moments'
    columns = list(np.arange(25))
    radius = 10
    
    def describe(self, binary, steps={}):
        #moments = cv2.HuMoments(cv2.moments(binary))
        # in case range is [0,255]
        #clipped = binary.clip(max=1)
        #radius = binary.shape[1]/ 2
        shape = list(binary.shape) + [3]
        kernel = np.ones((40,40),np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        steps['closed'] = closed
    
        im, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            self.values = np.zeros(len(self))
            return
        (x,y),radius = cv2.minEnclosingCircle(contours[0])
        m = zernike_moments(binary, radius)
        cont_img = np.zeros((binary.shape[0], binary.shape[1], 3))
        cv2.drawContours(cont_img, contours, 0, (0,255,0), 3)
        steps['contour'] = cont_img
        sk_cont = measure.find_contours(binary, 0.8)
        self.values = m 
    
    