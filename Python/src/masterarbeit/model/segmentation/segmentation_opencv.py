import numpy as np
import os
import cv2
from scipy import ndimage as ndi

from masterarbeit.model.segmentation.segmentation import Segmentation
from skimage.morphology import remove_small_objects   

class OpenCVSegmentation(Segmentation):
    
    def read(self, filename):     
        """
        read image from file
        
        Parameters
        ----------
        filename : str, name of the file to read       
        """           
        pixel_array = cv2.imread(filename, cv2.IMREAD_COLOR) 
        cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB, pixel_array)
        return pixel_array    
      
    def write(self, image, filename):
        pixels = image.copy()
        cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR, pixels)
        success = cv2.imwrite(filename, pixels)
        return success
        

class Binarize(OpenCVSegmentation):    
    
    label = 'Binarize image with Otsu-Thresholding (OpenCV)'    
    
    
    def process(self, image, steps=None):
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(grey, 220, 255, 
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)     
        return binary
    
class SegmentVeinsGabor(OpenCVSegmentation):   
    label = 'Segment leaf veins with Gabor (OpenCV)'    
     
    def gabor(self, image):
    
        def build_filters():
            filters = []
            ksize = 31
            for theta in np.arange(0, np.pi, np.pi / 16):
                kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                kern /= 1.5*kern.sum()
                filters.append(kern)
            return filters
    
        def process(img, filters):
            accum = np.zeros_like(img)
            for kern in filters:
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
                np.maximum(accum, fimg, accum)
            return accum        
    
        filters = build_filters()
    
        res1 = process(image, filters)
    
        return res1     
       
    def process(self, image, steps=None):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    
        gabor = self.gabor(gray)        
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = lsd.detect(gabor)
        line_img = lsd.drawSegments(np.empty(gray.shape), lines[0])
        #masked_source = self._mask(image)
        if steps is not None:
            steps['gabor'] = gabor
            steps['lines'] = line_img
            #steps['masked source'] = masked_source    
        #cropped = scale_to_bounding_box(binary.copy(), masked_source)
        return gabor    
    
class KMeansBinarize(Segmentation):
    label = 'clustered k-means Binarization (OpenCV)'     
    process_width = 2000
    def process(self, image, steps=None):
        res_factor = self.process_width / image.shape[1]
        new_shape = (int(image.shape[0] * res_factor), int(image.shape[1] * res_factor))
        resized = cv2.resize(image, new_shape)
        
        reshaped_img = resized.reshape((-1,3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10)
        # for binarization we need the color space clustered into 2 labels
        n_clusters = 2
        ret, label, center = cv2.kmeans(reshaped_img, n_clusters, None,
            criteria, 1, cv2.KMEANS_PP_CENTERS)    
        
        if steps is not None:   
            res = center[label.flatten()]
            res2 = res.reshape((resized.shape))
            steps['clustered colors'] = res2               
                        
        width = resized.shape[0]
        height = resized.shape[1]
        labels_reshaped = label.reshape(
            (width, height)).astype(np.float32)      
        
        # sometimes are in different order as expected
        # label 0 is supposed to be our background
        border_sum = (labels_reshaped[1, :].sum() + 
                      labels_reshaped[-1, :].sum() + 
                      labels_reshaped[:, 1].sum() + 
                      labels_reshaped[:, -1].sum())
        
        # if the border labels if they have label 1 by majority -> revert
        if  border_sum > (width + height):
            labels_reshaped = (labels_reshaped - 1) * -1
        
        # remove noise           
        labels_reshaped = cv2.GaussianBlur(labels_reshaped, (0, 0), 2)        
        retval, binary = cv2.threshold(labels_reshaped, .5, 255.0, 
                                       cv2.THRESH_BINARY)
        #kernel = np.ones((20, 20),np.uint8)
        #closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)    
        closed = (ndi.binary_fill_holes(
            binary, structure=np.ones((10,10)))).astype(np.uint8) 
        
        rem = remove_small_objects(closed.astype(bool), min_size=5000)
        rem = rem.astype(np.uint8)
        o_size_binary = cv2.resize(rem, (image.shape[1], image.shape[0]))
        
        return o_size_binary.astype(np.uint8)
    
class KMeansHSVBinarize(KMeansBinarize):
    label = 'clustered k-means Binarization in HSV space (OpenCV)'    
    def process(self, image, steps=None):
        hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 0] = 0
        #hsv_image[:, :, 2] = 0
        if steps is not None:
            steps['hsv without hue'] = hsv_image
        binary = super(KMeansHSVBinarize, self).process(hsv_image, steps)

        return binary

def scale_to_bounding_box(binary, image):
    contours = cv2.findContours(binary, 1, 2)
    x,y,w,h = cv2.boundingRect(contours[0])
    cropped = image[y: y + h, x: x + w]
    return cropped.copy()    
