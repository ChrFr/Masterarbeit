import numpy as np
import scipy as sp, scipy.spatial
import cv2
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import euclidean
from scipy.sparse.csgraph import floyd_warshall
from masterarbeit.model.features.feature import UnsupervisedFeature
from masterarbeit.model.features.codebook import (DictLearningCodebook, 
                                                  KMeansCodebook)
import math
distance = euclidean
shortest_path = floyd_warshall

def get_points_on_line(p1, p2, n=10):
    x = np.linspace(p1[0], p2[0], n+2)
    y = np.linspace(p1[1], p2[1], n+2)    
    return zip(x, y)

class IDSC(UnsupervisedFeature):
    '''
    subclass this, no dictionary and histo length defined
    '''    
    label = 'Inner Distance Shape Context'
    codebook_type = None
    histogram_length = None
    n_contour_points = 300
    n_angle_buckets = 8
    n_distance_buckets = 8
    columns = np.arange(n_angle_buckets).astype(np.str)
    n_levels = 1
    
    def _describe(self, binary, steps=None):
        if len(binary.shape) > 2:
            raise Exception('IDSC Features can only describe binary images')
        # maximum distance is the from upper left to lower right pixel,
        # so all points lie within distance
        self.max_distance = distance((0, 0), binary.shape)
        contour_points = self._sample_contour_points(binary, 
                                                     self.n_contour_points)
        dist_matrix = self._build_distance_matrix(binary, contour_points)
        context = self._build_shape_context(dist_matrix, contour_points)        
        
        ### Visualisation ###
        
        if steps is not None:
            img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            for p in contour_points:
                img = cv2.circle(img, tuple(p), 2, thickness=20, 
                                 color=(0, 255, 0))
            steps['picked points'] = img
            
        # flatten the histograms        
        return context
        
    def _sample_contour_points(self, binary, n):
        im2, contours, hierarchy = cv2.findContours(binary.copy(), 
                                                    cv2.RETR_TREE, 
                                                    cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return None
        # there should be only one contour, if segmentation was correctly done
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
                    if dist > self.max_distance:
                        break
                    dist_matrix[j + i, i] = dist
                    dist_matrix[i, j + i] = dist
        return dist_matrix
        
    def _build_shape_context(self, distance_matrix, contour_points, 
                             skip_distant_points=False):
        histogram = []
        max_log_distance = np.log2(self.max_distance)
        # steps between assigned buckets
        dist_step = max_log_distance / self.n_distance_buckets
        angle_step = np.pi * 2 / self.n_angle_buckets
        # find shortest paths in distance matrix (distances as weights)
        graph = shortest_path(distance_matrix, directed=False)
    
        # iterate all points on contour
        for i, (x0, y0) in enumerate(contour_points):
            hist = np.zeros((self.n_angle_buckets, self.n_distance_buckets))
    
            # calc. contour tangent from previous to next point
            # to determine angles to all other contour points
            (prev_x, prev_y) = contour_points[i - 1]
            (next_x, next_y) = contour_points[(i + 1) % len(contour_points)]
            contourTangent = np.arctan2(next_y - prev_y, 
                                        next_x - prev_x)
    
            # inspect relationship to all other points (except itself)
            # direction and distance are logarithmic partitioned into n buckets
            for j, (x1, y1) in enumerate(contour_points):
                if j == i: 
                    continue
                dist = graph[i, j]
                # 0 determines, that there is no path to point 
                if dist != 0:                
                    log_dist = np.log2(dist)
                # ignore unreachable points, if requested
                elif skip_distant_points:
                    continue
                # else unreachable point is put in last dist. bucket
                else:
                    log_dist = max_log_distance
                angle = (contourTangent - 
                         np.arctan2(y1 - y0, x1 - x0)) % (2 * np.pi)
                # calculate buckets, the inspected point belongs to
                dist_idx = int(min(np.floor(log_dist / dist_step), 
                                   self.n_distance_buckets - 1))
                angle_idx = int(min(angle / angle_step, 
                                    self.n_angle_buckets - 1))    
                # increase bin in bucket
                hist[angle_idx, dist_idx] += 1
    
            histogram.append(hist)
    
        return np.array(histogram)    
    
    
class IDSCGaussians(IDSC):
    '''
    subclass this, no dictionary attached
    '''
    label = 'Gaussian Inner Distance Shape Context'
    resolution = 2000000
    # gauss kernels, coarsest to finest, determines number of gauss levels
    gauss_kernels = [(501, 501), (101, 101), None]
    # points to take per level
    n_contour_points = [40, 200, 400]
    columns = np.arange(len(gauss_kernels))
    tau = 4
    n_levels = len(gauss_kernels)
    
    def _describe(self, binary, steps={}):
        if len(binary.shape) > 2:
            raise Exception('IDSC Features can only describe binary images')
        
        resolution = binary.size
        scale = math.sqrt(self.resolution / resolution)
        new_shape = np.array(binary.shape) * scale
        resized = cv2.resize(binary, tuple(new_shape.astype(np.int)))
                
        max_distance = distance((0, 0), binary.shape)        
        
        n_levels = len(self.gauss_kernels)
        level_contexts = []        
        # alternative to gauss: polygon approximation?
        #epsilon = 0.01 * cv2.arcLength(contour_points, True)
        #approx = cv2.approxPolyDP(contour_points,epsilon,True)    
        #appr_contour_points = approx.reshape(approx.shape[0], approx.shape[2])    
        
        # grayscale image needed for thresholding gauss
        if binary.max() == 1:
            binary = binary.copy() * 255
        for i, gauss_kernel in enumerate(self.gauss_kernels):
            # starting with the coarsest level
            gauss_level = n_levels - i
            # maximum distance decreases exponentially with increasing detail 
            self.max_distance = max_distance / np.power(4, 
                                                        n_levels - gauss_level)
            
            if gauss_kernel is not None:
                gaussian = cv2.GaussianBlur(binary, gauss_kernel, 0)
                # cut off small blurred values (else shape would be blown up)
                ret, thresh = cv2.threshold(gaussian, 20, 255, 
                                            cv2.THRESH_BINARY)
                
                ## leaves with very thin structures may be 'blurred away', 
                ## take more color information as shape
                #if thresh.sum() == 0:
                    #ret, thresh = cv2.threshold(gaussian, 10, 255, 
                                                #cv2.THRESH_BINARY)                    
                filtered = thresh
            else:
                filtered = binary
          
            # individual contour points for each level (ToDo: take same points
            # on every level)  
            contour_points = self._sample_contour_points(
                filtered, self.n_contour_points[i])
            
            dist_matrix = self._build_distance_matrix(filtered,
                                                      contour_points)   
            
            # get the shape context in max_distance (skip distant points)
            context = self._build_shape_context(dist_matrix, contour_points,
                                                skip_distant_points=True)  
            level_contexts.append(context)
            
            ### Visualisation of gauss levels ###
                
            if steps is not None:
                thickness = int(20 / (i + 1))
                img = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
                for p in contour_points:
                    img = cv2.circle(img, tuple(p), 2, thickness=thickness, 
                                     color=(0, 255, 0))
                # take an example point to visualize the max distance
                ex_point = contour_points[int(len(contour_points) / 2)]
                img = cv2.circle(img, tuple(p), 2, thickness=20, 
                                 color=(255, 0, 0))        
                img = cv2.circle(img, tuple(p), int(self.max_distance), 
                                 thickness=20, 
                                 color=(255, 0, 0))                
                steps['gauss level {}'.format(gauss_level)] = img
                
        return level_contexts
        
        
### the callable classes with defined codebook types ###
    
class IDSCKMeans(IDSC):
    codebook_type = KMeansCodebook
    histogram_length = 50
    columns = np.arange(0, histogram_length).astype(np.str)
    
    
class IDSCDict(IDSC):
    codebook_type = DictLearningCodebook
    histogram_length = 100
    columns = np.arange(0, histogram_length).astype(np.str)
    

class IDSCGaussiansKMeans(IDSCGaussians):
    label = 'Gaussian Inner Distance Shape Context'
    histogram_length = 100
    codebook_type = KMeansCodebook
    columns = np.arange(0, histogram_length).astype(np.str)


class IDSCGaussiansDict(IDSCGaussians):
    label = 'Gaussian Inner Distance Shape Context'
    histogram_length = 100
    codebook_type = DictLearningCodebook
    columns = np.arange(0, histogram_length).astype(np.str)
    
#class IDSCPolyKMeans(IDSCPoly):
    #label = 'Polygon Inner Distance Shape Context'
    #histogram_length = 100
    #codebook_type = DictLearningCodebook
    #columns = np.arange(0, histogram_length).astype(np.str)
    