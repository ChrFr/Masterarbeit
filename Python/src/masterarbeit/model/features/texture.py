'''
contains features to describe textures

(c) 2017, Christoph Franke

this file is part of the master thesis 
"Computergestuetzte Identifikation von Pflanzen anhand ihrer Blattmerkmale"
'''
__author__ = "Christoph Franke"

from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
import numpy as np
import cv2
import itertools

from masterarbeit.model.features.feature import Feature
from masterarbeit.model.segmentation.helpers import crop
from masterarbeit.model.segmentation.helpers import simple_binarize

  
class LocalBinaryPattern(Feature):
    label = 'Local Binary Pattern Detection'
    radius = 3
    n_points = 8 * radius        
    
    def _describe(self, image, steps=None):
        # scale here, because radius depends on number of pixels
        self._common_scale(image, downscale=2)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
        #gray = gabor(gray)
        lbp = local_binary_pattern(gray, self.n_points, self.radius, 
                                   method='uniform')
        if steps is not None:
            lbp_img = (lbp * 255 / self.n_points).astype(np.uint8)
            cv2.imshow('', lbp_img)
        histogram, _ = np.histogram(lbp.ravel(), normed=True,
                                    bins=np.arange(0, self.n_points + 2),
                                    range=(0, self.n_points + 2)) 
        #normed = normalize(histogram.reshape(1, -1))[0]                
        return histogram
    
    
class LocalBinaryPatternCenterPatch(LocalBinaryPattern):
    label = 'Local Binary Pattern Detection on extracted texture patch'
    # sufficient ratio of information in texture patch
    texture_ratio = 1.0

    def _describe(self, image, steps=None):
        patch = get_circular_patch(image, texture_ratio=self.texture_ratio, 
                                   steps=steps)
        if steps is not None:
            steps['patch'] = patch
        return super(LocalBinaryPatternCenterPatch, self)._describe(image, 
                                                                    steps=steps)
    
    
class LocalBinaryPatternPatches(LocalBinaryPattern):

    label = 'Local Binary Pattern Matrix-Patches'
    
    def _describe(self, image, steps=None):        
        patches = get_matrix_patches(image, 300, steps=steps)
        histogram = None
        for patch in patches:     
            sub_hist = super(LocalBinaryPatternPatches, self)._describe(
                patch, steps=None)
            if histogram is None:
                histogram = sub_hist
            else:
                histogram += sub_hist
        histogram = histogram / len(patches)
        #histogram = normalize(histogram.reshape(1, -1))[0]           
        return histogram 
    
class LeafvenationMorph(Feature):
    label = 'Leaf Veins Morphology'
    
    def _describe(self, image, steps=None):  
        scaled = self._common_scale(image)
        
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (200, 200))     
        background_mask = (gray == 255).astype(np.uint8) * 255
        background_mask = cv2.GaussianBlur(background_mask, (0,0), 100)
        background_mask = (background_mask != 0)
        # area of foreground in pixels
        leaf_area = gray.size - background_mask.sum()
        gabor_img = gabor_image(gray)      
        if steps is not None:
            cv2.imshow('k', gabor_img)
        
        def segment_veins(img, kernel_size, threshold=125):
            disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (kernel_size, kernel_size))
            closed = cv2.morphologyEx(gabor_img, cv2.MORPH_CLOSE, disk)
            veins = gabor_img - closed
            # binarize (simple version)
            veins[veins < threshold] = 0
            veins = np.clip(veins, 0, 1)
            # invert
            veins = 1 - veins 
            veins[background_mask] = 0
            return veins        
        
        histogram = []
        for kernel_size in [10, 30, 50]:
            veins = segment_veins(gabor_img, kernel_size)
            if steps is not None:
                # sometimes ui crashes when trying to make pixmap out of this
                #steps['{}'.format(kernel_size)] = veins*255
                # -> deactivated and imshow instead
                cv2.imshow('kernel {}'.format(kernel_size), veins * 255)
                
                # extract hough lines, only for visualization at the moment
                lines = cv2.HoughLinesP(veins,1,np.pi/180,100, 100, 10)
                line_img = np.zeros(scaled.shape)
                for line in lines:
                    for x1,y1,x2,y2 in line:
                        cv2.line(line_img, (x1, y1), (x2, y2), (0,255,0), 2)
                steps['lines kernel {}'.format(kernel_size)] = line_img
                
            perc_veins = veins.sum() / leaf_area
            histogram.append(perc_veins)            
        
        return np.array(histogram)          


class GaborFilterBank(Feature):
    label = 'Gabor filters'
    binary_input = False
    
    def _describe(self, image, steps=None):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
        thetas = np.arange(0, np.pi, np.pi / 8)
        sigmas = range(1, 4)
        lambdas = [5, 10]
        kernels = build_gabor_filterbank(thetas, sigmas, lambdas)
                    
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            # convolve with filter
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var() 
        normed = normalize(feats.flatten().reshape(1, -1))[0]
        return normed
    
    
class GaborFilterBankPatches(GaborFilterBank):
    label = 'Gabor filter Matrix-Patches'
    def _describe(self, image, steps=None): 
        patches = get_matrix_patches(image, 400, steps=steps)
        histogram = np.zeros(8 * 3 * 2 * 2)
        for patch in patches:     
            histogram += super(GaborFilterBankPatches, self)._describe(
                patch, steps=None)
        return histogram / histogram.max()
    
    
class GaborFilterBankCenterPatch(GaborFilterBank):
    label = 'Gabor filter Center Patch'
    def _describe(self, image, steps=None): 
        patch = get_circular_patch(image, steps=steps)        
        histogram = super(GaborFilterBankCenterPatch, self)._describe(
                patch, steps=steps)
        if steps is not None:
            steps['patch'] = patch
        return histogram / histogram.max()
    
def build_gabor_filterbank(thetas, sigmas, lambdas):
    filters = []
    for theta, sigma, lambd in itertools.product(thetas, sigmas, lambdas):
        kernel = cv2.getGaborKernel(None, sigma, theta, 
                                    lambd, 0.5, 0, ktype=cv2.CV_32F)
        filters.append(kernel)
    return filters
                
def gabor_image(image):
    # 16 orientations
    thetas = np.arange(0, np.pi, np.pi / 16)
    filters = build_gabor_filterbank(thetas, [4.0], [10.0])

    accumulated = np.zeros(image.shape, dtype=np.uint8)
    for kernel in filters:
        # normalize kernel
        kernel /= 1.5 * kernel.sum()
        # convolve with filter
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        
        # accumulate the energys        
        accumulated = np.maximum(accumulated, filtered)
        
    return accumulated          

def get_center_patch(image, texture_ratio=0.95):
    binary = simple_binarize(image)

    im2, contours, hierarchy = cv2.findContours(binary, 
                                                cv2.RETR_TREE, 
                                                cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    height, width = binary.shape
    # there should be only one contour, if segmentation was correctly done
    contour_points = max(contours,key=len)
    rect = cv2.minAreaRect(contour_points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # center of mass
    moments = cv2.moments(contour_points)
    mass_center = (int(moments['m10']/moments['m00']),
                   int(moments['m01']/moments['m00']))
    #img = cv2.circle(img, mass_center, 20, (255, 0, 0), thickness=10)
    #steps['rotated bbox'] = img

    center = rect[0]
    angle = rect[2]
    if angle < -45:
        angle += 90.0

    rot = cv2.getRotationMatrix2D(mass_center, angle, 1)
    trans = cv2.warpAffine(image, rot, (width, height))
    #steps['trans'] = trans
    min_size = int(min(trans.shape[1], trans.shape[0]))
    trans[trans==255] = 0
    trans_center = (int(trans.shape[1] / 2), int(trans.shape[0] / 2))
    #cv2.circle(trans, trans_center, 20, (255, 0, 0), thickness=10)
    for scale in np.arange(0, 0.9, 0.1):
        scale = 1.0 - scale
        patch_size = int(min_size * scale)
        patch = cv2.getRectSubPix(trans, (patch_size, patch_size), 
                                  trans_center)
        #steps['patch {}'.format(np.round(scale, decimals = 1))] = patch
        information_count = np.count_nonzero(patch)
        if information_count >= patch.size * texture_ratio:
            break    
    return patch

def get_circular_patch(image, texture_ratio=1, steps=None):
    binary = simple_binarize(image) * 255

    im2, contours, hierarchy = cv2.findContours(binary, 
                                                cv2.RETR_TREE, 
                                                cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    contour = max(contours, key=len)
    center, enclosing_radius = cv2.minEnclosingCircle(contour) 

    # starting with size of 70% of enclosing circle
    for scale in np.arange(0.3, 0.9, 0.1):
        scale = 1.0 - scale
        radius = enclosing_radius * scale
        encircled_pixels = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(encircled_pixels, 
                   (int(center[0]), int(center[1])), 
                   int(radius), 
                   (255, 255, 255), thickness=-1) 
        circle_mask = encircled_pixels > 0
        encircled_pixels.fill(255)
        encircled_pixels[circle_mask] = image[circle_mask] 
        if ((encircled_pixels == 0).sum() <= 
            circle_mask.sum() * (1 - texture_ratio)):
            break

    if steps is not None:
        cv2.circle(image, 
                   (int(center[0]), int(center[1])), 
                   int(radius), (0, 255, 0), thickness=10)
        steps['cutout'] = image 
    return crop(encircled_pixels)

def get_matrix_patches(image, n_cells, pick=None, steps=None):
    '''
    image is divided in n cells
    '''
    # find a suitable size for patches, so that roughly n cells may fit  in
    patch_size = int(np.sqrt(image.size / n_cells))
    image = image.copy()
    s_image = image.copy()
    # unify background to black
    image[image == 255] = 0
    half_size = int(patch_size / 2)
    patches = []
    max_black = patch_size * patch_size * 0.05
    for y, x in itertools.product(
        np.arange(half_size, image.shape[0] - half_size, patch_size), 
        np.arange(half_size, image.shape[1] - half_size, patch_size)):
        patch = cv2.getRectSubPix(image, (patch_size, patch_size), (x, y))
        if (patch == 0).sum() < max_black:
            patches.append(patch)
            if steps is not None:
                cv2.rectangle(s_image, (x - half_size, y - half_size), 
                              (x + half_size, y + half_size),
                              (0, 255, 0), thickness=10)
    if steps is not None:
        steps['cutout'] = s_image
        
    # leaf too small for patches, take whole leaf instead
    if len(patches) == 0:
        patches.append(image)         
    return patches                    
