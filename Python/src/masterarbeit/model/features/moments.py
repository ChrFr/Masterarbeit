from mahotas.features import zernike_moments
from masterarbeit.model.features.feature import Feature
from skimage import measure
import numpy as np
import cv2

class HuMoments(Feature):
    label = 'Hu-Moments'
    columns = list(np.arange(7))
    
    def describe(self, binary, steps=None):
        clipped = binary.clip(max=1)
        m = measure.moments(clipped)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        central = measure.moments_central(clipped, cr, cc)
        normalized = measure.moments_normalized(central)
        moments = measure.moments_hu(normalized)
        self.values = moments

class ZernikeMoments(Feature):
    label = 'Zernike-Moments'
    columns = list(np.arange(25))
    
    def describe(self, binary, steps=None):
        shape = list(binary.shape) + [3]
        kernel = np.ones((40,40),np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
        im, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        # no contours if binary is empty
        if len(contours) == 0:
            self.values = np.zeros(len(self))
            return
        # compute moments around center of binary
        center, radius = cv2.minEnclosingCircle(contours[0])
        moments = zernike_moments(binary, radius, cm=center)
        if steps is not None:
            steps['closed'] = closed
            cont_img = np.zeros((binary.shape[0], binary.shape[1], 3))
            cv2.drawContours(cont_img, contours, 0, (0, 255, 0), 3)
            steps['contour'] = cont_img
        self.values = moments 
    
    