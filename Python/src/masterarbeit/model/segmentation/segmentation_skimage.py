import numpy as np
from collections import OrderedDict
import os
from skimage import color
from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, dilation, remove_small_objects
from skimage.morphology import disk, square
from skimage.transform import resize
from scipy import ndimage as ndi
from masterarbeit.model.segmentation.common import mask
from skimage.filters import gabor_kernel

from masterarbeit.model.segmentation.segmentation import Segmentation

class SkimageSegmentation(Segmentation): 
    process_width = 1280
    
    def read(self, filename):    
        pixel_array = io.imread(filename)
        return pixel_array
    
    def write(self, image, filename):
        try:
            io.imsave(filename, image)
        # possible exceptions: file not found, no permission, 
        # malformed image array (not specifically handled here)
        except: 
            return False
        return True        
    
    def scale(self, image, shape):      
        '''
        resize image to given shape
        '''
        shape = list(shape)
        for i in range(2, len(image.shape)):
            shape.append(image.shape[i])
        resized = resize(image.copy(), shape, preserve_range=True)
        # if image was binary -> back to binary range [0,1] by rounding
        # (scaling interpolates)
        if ((image==0) | (image==1)).all():  
            resized = np.rint(resized)
        # back to original dtype (scaling returns floats)
        resized = resized.astype(image.dtype)
        return resized
    
class BinarizeHSV(SkimageSegmentation):     
    
    label = 'Binarize image with Otsu-Thresholding in HSV space (Skimage)'  
        
    def binarize_otsu(self, axis):        
        threshold = threshold_otsu(axis)
        otsu = axis <= threshold
        binary = np.zeros(otsu.shape, dtype=np.uint8)
        binary[-otsu] = 1
        return binary   
    
    def process(self, image, steps=None):
        # scale image to a uniform size, so common structured elements can be used
        # (and to reduce calc. time)
        o_height = image.shape[0]
        o_width = image.shape[1]
        res_factor = self.process_width / o_width
        new_shape = (int(o_height * res_factor), int(o_width * res_factor))
        resize = self.scale(image, new_shape)
        
        hsv = color.rgb2hsv(resize)      
        # take values only
        s = hsv[:,:,1]
        res_binary = self.binarize_otsu(s) 
        # remove holes in leaf (caused by reflections or actual holes)
        wo_holes = (ndi.binary_fill_holes(
            res_binary,structure=np.ones((5,5)))).astype(np.uint8)
        # rescale
        binary = self.scale(wo_holes, (o_height, o_width))  
        
        #TODO: maybe do removal of small holes on masked_binary instead of stem (remove fragment border pixels)
        
        if steps is not None:
            steps['hsv'] = hsv
            steps['binary (resized)'] = res_binary
            steps['binary without holes (resized)'] = wo_holes
        
        return binary        
    
class SegmentGabor(SkimageSegmentation): 
    
    label = 'Segment leaf veins with Gabor (Skimage)'  
        
    def process(self, image, steps=None):
        def compute_feats(image, kernels):
            feats = np.zeros((len(kernels), 2), dtype=np.double)
            for k, kernel in enumerate(kernels):
                filtered = ndi.convolve(image, kernel, mode='wrap')
                feats[k, 0] = filtered.mean()
                feats[k, 1] = filtered.var()
            return feats
        
    
        def match(feats, ref_feats):
            min_error = np.inf
            min_i = None
            for i in range(ref_feats.shape[0]):
                error = np.sum((feats - ref_feats[i, :])**2)
                if error < min_error:
                    min_error = error
                    min_i = i
            return min_i
        
        
        # prepare filter bank kernels
        kernels = []
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                  sigma_x=sigma, sigma_y=sigma))
                    kernels.append(kernel)
                    
        # prepare reference features
        ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
        gray = color.rgb2gray(image)
        steps['grau'] = gray
        feats = compute_feats(gray, kernels)  
        
        def power(image, kernel):
            # Normalize images for better comparison.
            image = (image - image.mean()) / image.std()
            return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                           ndi.convolve(image, np.imag(kernel), mode='wrap')**2)       
        
        # Plot a selection of the filter bank kernels and their responses.
        for theta in (0, 1):
            theta = theta / 4. * np.pi
            for frequency in (0.1, 0.4):
                kernel = gabor_kernel(frequency, theta=theta)
                params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
                # Save kernel and the power image for each image
                p = power(gray, kernel)   
                norm_power = p * 255 / p.max()
                steps[params] = norm_power
                
        
        return gray