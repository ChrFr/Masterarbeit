import numpy as np
import scipy as sp, scipy.spatial
import cv2
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import euclidean
import itertools
import math
from scipy.sparse.csgraph import floyd_warshall
from sklearn.preprocessing import normalize

from masterarbeit.model.features.feature import UnsupervisedFeature
from masterarbeit.model.segmentation.helpers import simple_binarize
from masterarbeit.model.features.codebook import (DictLearningCodebook, 
                                                  KMeansCodebook)
distance = euclidean
shortest_path = floyd_warshall

def get_points_on_line(p1, p2, n=10):
    points = np.zeros((n, 2))
    x = np.linspace(p1[0], p2[0], n).astype(np.int)
    y = np.linspace(p1[1], p2[1], n).astype(np.int)
    points[:, 0] = x
    points[:, 1] = y
    ## take unique points only
    #points_on_line = []    
    #for i in range(1, n):
        #if (points[i] == points[i-1]).sum() < 2:
            #points_on_line.append(points[i])        
    return points

class IDSC(UnsupervisedFeature):
    '''
    subclass this, no dictionary and histo length defined
    '''    
    label = 'Inner Distance Shape Context'
    codebook_type = None
    histogram_length = None
    n_contour_points = 300
    n_angle_bins = 8
    n_distance_bins = 8
    n_levels = 1
    binary_input = True
    
    def _describe(self, binary, steps=None):
        # maximum distance is the from upper left to lower right pixel,
        # so all points lie within distance
        self.max_distance = distance((0, 0), binary.shape)
        contour_points = self._sample_contour_points(binary, 
                                                     self.n_contour_points)
        if len(contour_points) == 0:
            print('contours missing in IDSC {}'.format(self.id))
            return np.zeros(self.histogram_length)
        dist_matrix = self._build_distance_matrix(binary, contour_points)
        context = self._build_shape_context(dist_matrix, contour_points)        
        
        ### Visualisation ###
        
        if steps is not None:
            img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            for p in contour_points:
                img= cv2.circle(binary, tuple(p), 2, thickness=20, 
                                color=(0, 255, 0))
            steps['picked points'] = img   
                 
        return context
        
    def _sample_contour_points(self, binary, n):
        im2, contours, hierarchy = cv2.findContours(binary.copy(), 
                                                    cv2.RETR_TREE, 
                                                    cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return []
        # there should be only one contour, if segmentation was correctly done
        # if there are more, caused by noise, take the longest one
        contour_points = max(contours,key=len)
        # find contour points adds unnessecary dim., 
        # remove second dim of length 1
        contour_points = np.reshape(contour_points, (len(contour_points), 2))
        # take evenly distributed points along the contour
        idx = np.linspace(0, len(contour_points) - 1, num=n).astype(np.int)
        contour_points = contour_points[idx]     
        return contour_points
     
    def _build_distance_matrix(self, binary, contour_points):        
        dist_matrix = np.zeros((len(contour_points), len(contour_points)))        
        
        # fill the distance matrix pairwise
        for i, p1 in enumerate(contour_points):
            for j, p2 in enumerate(contour_points[i+1:]):
                line_points = get_points_on_line(p1, p2, 10)
                # check, if all points in between are inside shape (indicated 
                # by binary pixels, where row is y and column is x)
                for point in line_points:
                    if binary[int(point[1]), int(point[0])] == 0:
                        break 
                # no break -> all points in between are inside the shape
                else:
                    # if all points on line in shape -> calculate distance 
                    # and store in matrix (mirrored)
                    dist = distance(p1, p2)
                    # ignore distances beyond the defined maximum 
                    # (extension for gauss pyramid IDSC, can never be ful-
                    # filled in normal IDSC)
                    if dist > self.max_distance:
                        break
                    dist_matrix[j + i, i] = dist
                    dist_matrix[i, j + i] = dist
        return dist_matrix
        
    def _build_shape_context(self, distance_matrix, contour_points, 
                             skip_distant_points=False):
        histogram = []
        max_log_distance = np.log2(self.max_distance)
        # steps between assigned bins
        dist_step = max_log_distance / self.n_distance_bins
        angle_step = np.pi * 2 / self.n_angle_bins
        # find shortest paths in distance matrix (distances as weights)
        graph = shortest_path(distance_matrix, directed=False)
    
        # iterate all points on contour
        for i, (x0, y0) in enumerate(contour_points):
            hist = np.zeros((self.n_angle_bins, self.n_distance_bins))
    
            # calc. contour tangent from previous to next point
            # to determine angles to all other contour points
            (prev_x, prev_y) = contour_points[i - 1]
            (next_x, next_y) = contour_points[(i + 1) % len(contour_points)]
            tangent = np.arctan2(next_y - prev_y, 
                                 next_x - prev_x)
    
            # inspect relationship to all other points (except itself)
            # direction and distance are logarithmic partitioned into n bins
            for j, (x1, y1) in enumerate(contour_points):
                if j == i: 
                    continue
                dist = graph[i, j]
                # 0 or infinity determine, that there is no path to point 
                if dist != 0 and dist != np.inf:                
                    log_dist = np.log2(dist)
                # ignore unreachable points, if requested
                elif skip_distant_points:
                    continue
                # else unreachable point is put in last dist. bin
                else:
                    log_dist = max_log_distance
                angle = (tangent - np.arctan2(y1 - y0, x1 - x0)) % (2 * np.pi)
                # calculate bins, the inspected point belongs to
                dist_idx = int(min(np.floor(log_dist / dist_step), 
                                   self.n_distance_bins - 1))
                angle_idx = int(min(angle / angle_step, 
                                    self.n_angle_bins - 1))    
                # point fits into bin
                hist[angle_idx, dist_idx] += 1

            # L1 norm
            if hist.sum() > 0:
                hist = hist / hist.sum()
            histogram.append(hist.flatten())
    
        return np.array(histogram)    
    
    
class IDSCGaussians(IDSC):
    '''
    subclass this, no dictionary attached
    '''
    label = 'Gaussian Inner Distance Shape Context'
    # points to take per level, from finest to coarsest (up the pyramid)
    n_contour_points = [400, 200, 40]
    tau = 4
    n_levels = len(n_contour_points)
    
    def _describe(self, binary, steps={}):   
        # add a border, otherwise blurred shapes may melt with image margin
        border_size = int(math.sqrt(binary.size) * 0.1)
        binary = cv2.copyMakeBorder(binary, top=border_size, bottom=border_size, 
                                    left=border_size, right=border_size, 
                                    borderType= cv2.BORDER_CONSTANT, value=[0])
            
        max_distance = distance((0, 0), binary.shape)                
        level_contexts = []        
        
        # grayscale image needed for thresholding gauss
        if binary.max() == 1:
            binary = binary.copy() * 255
                
        # from finest level to coarsest level
        for level in range(self.n_levels):
            # sigma increases with level (going up in gauss pyramid)
            sigma = np.power(8, level)
            # maximum distance decreases exponentially with increasing detail 
            self.max_distance = max_distance / np.power(
                self.tau, self.n_levels - level - 1)
                        
            if level == 0:
                # keep input image at finest level
                pyr_binary = binary
                # don't skip points as well (-> do original IDSC)
                skip_distant_points = False
            else:
                gaussian = cv2.GaussianBlur(binary, (0,0), sigma)
                # remove blurred outlines by thresholding
                ret, thresh = cv2.threshold(gaussian, 20, 255, 
                                            cv2.THRESH_BINARY)
                pyr_binary = thresh
                skip_distant_points = True
            
            # individual contour points for each level (ToDo: take same points
            # on every level)  
            contour_points = self._sample_contour_points(
                pyr_binary, self.n_contour_points[level])
            
            # leaf structures too thin > set to zero (serves as indicator as well)
            if contour_points is None:
                print('contours missing in IDSC {} at gauss level {}'.format(
                    gauss_level, self.id))
                context = np.zeros(len(self.histogram_length))  
                    
            else:
                dist_matrix = self._build_distance_matrix(pyr_binary,
                                                          contour_points)   
                
                # get the shape context in max_distance neighbourhood
                context = self._build_shape_context(
                    dist_matrix, contour_points, 
                    skip_distant_points=skip_distant_points)  
                
            level_contexts.append(context)
            
            ### Visualisation of gauss levels ###
                
            if steps is not None:
                img = cv2.cvtColor(pyr_binary, cv2.COLOR_GRAY2BGR)
                for p in contour_points:
                    cv2.circle(img, tuple(p), 2, 
                               thickness=int(20 / (level + 1)), 
                               color=(0, 255, 0))
                # take an example point to visualize the max distance
                ex_point = contour_points[int(len(contour_points) / 2)]
                cv2.circle(img, tuple(p), 2, thickness=20, 
                           color=(255, 0, 0))        
                cv2.circle(img, tuple(p), int(self.max_distance), 
                           thickness=20, color=(255, 0, 0))                
                steps['pyramid {}'.format(level)] = img
                
        return level_contexts

        
### the callable classes with defined codebook types ###
    
class IDSCKMeans(IDSC):
    codebook_type = KMeansCodebook
    histogram_length = 100
    columns = np.arange(0, histogram_length).astype(np.str)
    
    
class IDSCDict(IDSC):
    codebook_type = DictLearningCodebook
    histogram_length = 100
    columns = np.arange(0, histogram_length).astype(np.str)
    

class IDSCGaussiansKMeans(IDSCGaussians):
    label = 'Gaussian Inner Distance Shape Context'
    histogram_length = 150
    codebook_type = KMeansCodebook
    columns = np.arange(0, histogram_length).astype(np.str)


class IDSCGaussiansDict(IDSCGaussians):
    label = 'Gaussian Inner Distance Shape Context'
    histogram_length = 150
    codebook_type = DictLearningCodebook
    columns = np.arange(0, histogram_length).astype(np.str)
    
#class IDSCPolyKMeans(IDSCPoly):
    #label = 'Polygon Inner Distance Shape Context'
    #histogram_length = 100
    #codebook_type = DictLearningCodebook
    #columns = np.arange(0, histogram_length).astype(np.str)
    