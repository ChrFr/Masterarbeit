from skimage.feature import local_binary_pattern, multiblock_lbp, draw_multiblock_lbp
from skimage.transform import integral_image
from sklearn.preprocessing import normalize
from scipy import ndimage as ndi
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.feature import greycomatrix
from mahotas.features import haralick
import numpy as np
import cv2
import itertools

from masterarbeit.model.features.feature import Feature, UnsupervisedFeature
from masterarbeit.model.features.codebook import KMeansCodebook
from masterarbeit.model.segmentation.segmentation_opencv import Binarize
from masterarbeit.model.segmentation.helpers import crop
from masterarbeit.model.segmentation.helpers import simple_binarize

class Sift(UnsupervisedFeature):
    label = 'Sift Keypoints'
    histogram_length = 50
    columns = np.arange(0, histogram_length)
    codebook_type = KMeansCodebook
    
    def _describe(self, image, steps=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
        gray[gray==255] = 0
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        if steps is not None:
            img_keys = cv2.drawKeypoints(gray, kp, None, 
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            steps['keypoints'] = img_keys
            
        return des
        
class SiftPatch(Sift):
    label = 'Sift Keypoints with gabor'
    
    def _describe(self, image, steps=None):
        patch = get_center_patch(image)
        gabor_patch = gabor(patch)
        if steps is not None:
            steps['patch'] = patch
            steps['gabor'] = gabor_patch
        return super(SiftPatch, self)._describe(patch, steps=steps)      
    
    
class Surf(UnsupervisedFeature):
    label = 'Surf Keypoints'
    histogram_length = 50
    codebook_type = KMeansCodebook
    
    def _describe(self, image, steps=None):    
        surf = cv2.xfeatures2d.SURF_create()
        kp, des = surf.detectAndCompute(image, None)
        if steps is not None:
            img_keys = cv2.drawKeypoints(image, kp, None, 
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            steps['keypoints'] = img_keys
            
        return des
    
    
class SurfPatch(Surf):
    label = 'Surf Keypoints with gabor'
    
    def _describe(self, image, steps=None):
        #patch = get_center_patch(image, 1)
        #gabor_patch = gabor(patch)
        #if steps is not None:
            #steps['patch'] = patch
            #steps['gabor'] = gabor_patch
        patches = get_matrix_patches(image, 100)
        desc = None
        for patch in patches:
            patch_desc = super(SurfPatch, self)._describe(patch, steps=None)  
            if patch_desc is not None:
                if desc is None:
                    desc = patch_desc
                else:
                    desc = np.concatenate((desc, patch_desc))
        return desc
                
def gabor(image):

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


class LocalBinaryPattern(Feature):
    label = 'Local Binary Pattern Detection'
    radius = 3
    n_points = 8 * radius        
    
    def _describe(self, image, steps=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
        #gray = gabor(gray)
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method='uniform')
        histogram, _ = np.histogram(lbp.ravel(),
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
        return super(LocalBinaryPatternCenterPatch, self)._describe(image, steps=steps)
    
    
class LocalBinaryPatternPatches(LocalBinaryPattern):

    label = 'Local Binary Pattern Multi Patches'
    
    def _describe(self, image, steps=None):        
        #patches = get_random_patches(image, 100, 100)
        patches = get_matrix_patches(image, 100, steps=steps)
        histogram = None
        for patch in patches:     
            sub_hist = super(LocalBinaryPatternPatches, self)._describe(
                patch, steps=None)
            if histogram is None:
                histogram = sub_hist
            else:
                histogram += sub_hist
        histogram = histogram / len(patches)
        return normalize(histogram.reshape(1, -1))[0]
    
    
class LocalBinaryPatternKMeans(UnsupervisedFeature):
    label = 'Local Binary Pattern Detection KMeans'
    radius = 3
    n_points = 8 * radius        
    histogram_length = 50
    codebook_type = KMeansCodebook
    
    def _describe(self, image, steps=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method='uniform')             
        return lbp
    
    
class Leafvenation(Feature):
    label = 'Leaf Veins Sceleton'
    
    def _describe(self, image, steps=None):  
        scaled = self._common_scale(image)
        
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        gray[gray == 255] = 0
        binary = np.clip(gray, 0, 1) * 255        
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (200, 200))     
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, disk)
        background_mask = binary == 0
        # area of foreground in pixels
        leaf_area = gray.size - background_mask.sum()
        gabor_img = gabor(gray)        
        gabor_img[background_mask] = 0     
        #steps['gabor'] = gabor_img        
        #binary = Binarize().process(gabor_img)  
        #steps['binary'] = binary
        
        def segment_veins(img, kernel_size):
            disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (kernel_size, kernel_size))
            opened = cv2.morphologyEx(gabor_img, cv2.MORPH_CLOSE, disk)
            veins = gabor_img - opened
            veins[veins < 125] = 0
            veins = np.clip(veins, 0, 1)
            veins = 1 - veins 
            veins[background_mask] = 0
            return veins
        
        histogram = []
        for kernel_size in[10, 30, 50]:
            veins = segment_veins(gabor_img, kernel_size)
            if steps is not None:
                # sometimes ui crashes when trying to make pixmap 
                # -> imshow instead
                #steps['{}'.format(i)] = veins*255
                cv2.imshow('kernel {}'.format(kernel_size), veins * 255)
            perc_veins = veins.sum() / leaf_area
            histogram.append(perc_veins)
        
        return np.array(histogram)
    
        
class Haralick(Feature):
    label = 'Haralick Features'
    
    def _describe(self, image, steps=None):   
        #patch = get_center_patch(image) 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        gray[gray==255] = 0
        histogram = haralick(gray, ignore_zeros=True)
        return np.array([histogram.mean()])
       
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
    #img=cv2.drawContours(image.copy(),cnt,0,(0,0,255),2)   
    #img=cv2.drawContours(image.copy(),[box],0,(0,0,255),10)  
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
    binary = simple_binarize(image)
    binary[binary > 0] = 255
                 
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
        encircled_pixels = np.zeros(image.shape)
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
        
def get_matrix_patches(image, patch_size, pick=None, steps=None):
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
    return patches                    

def get_random_patches(image, patch_size, n):
    binary = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary[binary==255] = 0
    binary = np.clip(binary, 0, 1)
    
    
    # make a window with the minimum window size to find
    kernel = np.ones((patch_size, patch_size),dtype=np.uint8)
    # filter the image with this kernel
    res = cv2.filter2D(binary, cv2.CV_16S, kernel)   
    # find the maximum 
    m = np.amax(res)
    # if the maximum is the area of your window at least
    # one location has been found
    patches = []
    if m == patch_size * patch_size:
        #mask_color = cv2.cvtColor(binary*255, cv2.COLOR_RGB2GRAY)
        # show each location where we found the maximum
        # i.e. a possible location for a patch
        
        res[res < m] = 0        
        points = np.transpose(np.nonzero(res))
        idx = np.random.choice(len(points), n)
        for y, x in points[idx]:
            # if you are only interested in the first hit use this:
            # find firs index and convert it to coordinates of the mask
            #y, x = np.unravel_index(np.argmax(res), binary.shape)
            # could do now other things with this location, 
            # but for now just show it in another color
            patch = cv2.getRectSubPix(image, (patch_size, patch_size), 
                                      (x, y))
            patches.append(patch)
            #cv2.imshow('{}'.format(x), patch)
    return patches
        
#class MultiblockLocalBinaryPattern()
    
   
#def process(self, image, steps=None):
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    
    #gabor = self.gabor(gray)        
    #lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    #lines = lsd.detect(gabor)
    #line_img = lsd.drawSegments(np.empty(gray.shape), lines[0])
    ##masked_source = self._mask(image)
    #if steps is not None:
        #steps['gabor'] = gabor
        #steps['lines'] = line_img
        ##steps['masked source'] = masked_source    
    ##cropped = scale_to_bounding_box(binary.copy(), masked_source)
    #return gabor    
    
    
class GaborFilterBank(Feature):
    label = 'Gabor filters'
    binary_input = False
    columns = np.arange(32)
    
    def _describe(self, image, steps=None):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    
        kernels = []
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                  sigma_x=sigma, sigma_y=sigma))
                    kernels.append(kernel)        
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(gray, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var() 
        normed = normalize(feats.flatten().reshape(1, -1))[0]
        return normed
    
class GaborFilterBankPatches(GaborFilterBank):
    label = 'Gabor filters Multi Patches'
    def _describe(self, image, steps=None):        
        #patches = get_random_patches(image, 100, 50)
        patches = get_matrix_patches(image, 100, steps=steps)
        histogram = np.zeros(32)
        for patch in patches:     
            histogram += super(GaborFilterBankPatches, self)._describe(
                patch, steps=None)
        return histogram / histogram.max()
    
class GaborFilterBankCenterPatch(GaborFilterBank):
    label = 'Gabor filters Center Patch'
    def _describe(self, image, steps=None): 
        patch = get_center_patch(image)
        histogram = super(GaborFilterBankCenterPatch, self)._describe(
                patch, steps=None)
        return histogram
    
    