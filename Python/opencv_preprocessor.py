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
    
    def _mask(self, source, binary):
        self.binary_mask = np.clip(binary, 0, 1)
        masked_source = source.copy()
        
        # colored (3 values per pixel)
        if len(source.shape) == 3:
            for i in range(3):
                masked_source[:, :, i] = np.multiply(masked_source[:, :, i], self.binary_mask)
        # black/white or grey
        else:
            masked_source = np.multiply(masked_source, self.binary_mask) 
            
        return masked_source
    
    def _segment_veins(self):
        grey = cv2.cvtColor(self.source_pixels, cv2.COLOR_BGR2GRAY)
        #grey = cv2.GaussianBlur(grey,(5,5),0)
        #segmented = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 6)
        #segmented = cv2.Canny(self.source_pixels, 1, 50, 3)
        
        #lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        #lines = lsd.detect(grey)
        #segmented = lsd.drawSegments(np.empty(grey.shape), lines[0])
        
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(grey, kernel, iterations = 1)
        dilation = cv2.dilate(erosion, kernel, iterations = 1)
        segmented = cv2.Canny(dilation, 30, 50, 3)
        #dilate(image,dst,element);        
        
        return segmented
    
    def segment(self):
        return self.source_pixels
    
    def process(self):
        processed_images = OrderedDict()
        binary = self.binarize()
        masked_source = self._mask(self.source_pixels, binary)
        veins = self._segment_veins()
        
        processed_images['binary'] = binary
        processed_images['masked source'] = masked_source
        processed_images['veins'] = veins
        
        return processed_images