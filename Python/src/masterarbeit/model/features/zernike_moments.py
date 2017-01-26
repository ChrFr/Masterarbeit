from mahotas.features import zernike_moments
from masterarbeit.model.features.feature import Feature
import numpy as np
import cv2

class ZernikeMoments(Feature):
    label = 'Zernike-Moments'
    columns = list(np.arange(25))
    radius = 10
    
    def extract(self, binary):
        #moments = cv2.HuMoments(cv2.moments(binary))
        # in case range is [0,255]
        #clipped = binary.clip(max=1)
        #radius = binary.shape[1]/ 2
        shape = list(binary.shape) + [3]
        kernel = np.ones((40,40),np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
        contours = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[1]
        if len(cnt) == 0:
            self.values = np.zeros(self.length)
            return
        (x,y),radius = cv2.minEnclosingCircle(cnt[0])
        m = zernike_moments(binary, radius)
        self.values = m
    
    