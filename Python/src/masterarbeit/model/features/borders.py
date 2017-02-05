import cv2
from skimage.filters import gaussian
import numpy as np
from masterarbeit.model.features.feature import Feature
from masterarbeit.model.segmentation.segmentation_opencv import Binarize

class Borders(Feature):
    label = 'Borders'
    columns = ['0', '1', '2', '3', '4', '5', '6']
    
    def describe(self, binary, steps={}):        
        if binary.max() <= 1:
            binary *= 255
        self.values = np.zeros(len(self))
        blur = cv2.blur(binary,(50,50))        
        steps['blur'] = blur
        ret, binary_blur = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)      
        binary_blur[0] = 0
        binary_blur[-1] = 0
        binary_blur[:, 0] = 0
        binary_blur[:, -1] = 0        
        steps['blur b'] = binary_blur
        im, binary_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,
                                                          cv2.CHAIN_APPROX_SIMPLE)     
        im, blurred_contours, hierarchy = cv2.findContours(binary_blur, cv2.RETR_TREE,
                                                           cv2.CHAIN_APPROX_SIMPLE)    
        cont_img = np.zeros((binary.shape[0], binary.shape[1], 3))
        cv2.drawContours(cont_img, binary_contours, -1, (0,255,0), 3)
        cv2.drawContours(cont_img, blurred_contours, -1, (255,0,0), 3)
        steps['contours'] = cont_img
        
        
        
    
    