'''
contains features to describe the geometry-moments of binary images

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

from mahotas.features import zernike_moments
from masterarbeit.model.features.feature import Feature
from skimage import measure
import numpy as np
import cv2

class HuMoments(Feature):
    label = 'Hu-Moments'
    dim = 7
    binary_input = True
    
    def _describe(self, binary, steps=None):
        clipped = binary.clip(max=1)
        m = measure.moments(clipped)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        central = measure.moments_central(clipped, cr, cc)
        normalized = measure.moments_normalized(central)
        moments = measure.moments_hu(normalized)
        # nan determines, that moment could not be described, 
        # but is hard to handle in prediction, set to zero instead
        moments[np.isnan(moments)] = 0
        return moments 

class ZernikeMoments(Feature):
    label = 'Zernike-Moments'
    dim = 25
    binary_input = True
    
    def _describe(self, binary, steps=None):
        shape = list(binary.shape) + [3]
        kernel = np.ones((40,40),np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
        im, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        # no contours if binary is empty
        if len(contours) == 0:
            return np.zeros(self.dim)
        # compute moments around center of binary
        center, radius = cv2.minEnclosingCircle(contours[0])
        moments = zernike_moments(binary, radius, cm=center)
        if steps is not None:
            steps['closed'] = closed
            cont_img = np.zeros((binary.shape[0], binary.shape[1], 3))
            cv2.drawContours(cont_img, contours, 0, (0, 255, 0), 3)
            steps['contour'] = cont_img
        return moments 
        
    