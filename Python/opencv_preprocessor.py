import cv2
import numpy as np
from collections import OrderedDict

from preprocessor import PreProcessor

class OpenCVPreProcessor(PreProcessor):
    
    def read_file(self, filename):     
        """
        read image from file
        
        Parameters
        ----------
        filename : str, name of the file to read       
        """   
        pixel_array = cv2.imread(filename, cv2.IMREAD_COLOR) 
        cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB, pixel_array)
        self.source_pixels = pixel_array  
        return self.source_pixels
            
    def binarize(self):
        grey = cv2.cvtColor(self.source_pixels, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(grey, (5, 5), 0)
        ret, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)         
        #binarized = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 6)    
        #binarized = cv2.Canny( grey, 1, 200, 100 );
               
        #binarized = cv2.pyrMeanShiftFiltering(self.source_pixels, 30, 30, 3);
        
        #green = self.source_pixels[:, :, 1]
        #ret, binarized = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)           
        return binary
    
    def segment(self):
        #binarized = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 6)        # gar nicht schlecht für adern evtl.
        return self.source_pixels
    
    def process(self):
        processed_images = OrderedDict()
        binary = self.binarize()
        self.binary_mask = np.clip(binary, 0, 1)
        masked_source = self.source_pixels.copy()
        
        for i in range(3):
            masked_source[:, :, i] = np.multiply(self.source_pixels[:, :, i], self.binary_mask)      
            
        processed_images['binary'] = binary
        processed_images['masked source'] = masked_source
        
        return processed_images