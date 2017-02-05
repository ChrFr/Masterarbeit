import numpy as np
import scipy as sp, scipy.spatial
import cv2
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from scipy.sparse.csgraph import floyd_warshall
from masterarbeit.model.features.feature import MultiFeature
from masterarbeit.model.features.dictionary import MiniBatchDictionary
dist_func = sp.spatial.distance.euclidean

# http://www.cs.mun.ca/~rod/2500/notes/numpy-arrays/numpy-arrays.html
def seg_intersect(a1, a2, b1, b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom)*db + b1

def points_on_line(p1, p2, n):
    x = np.linspace(p1[0], p2[0], n+2)
    y = np.linspace(p1[1], p2[1], n+2)
    
    return zip(x, y)

class IDSC(MultiFeature):
    label = 'Inner-distance Shape Context'
    dictionary_type = MiniBatchDictionary    
    columns = np.arange(0, dictionary_type.n_atoms).astype('str')
    
    #def __init__(self, category):
        #super(IDSC, self).__init__(category)       
    
    def describe(self, binary, steps={}):
        if len(binary.shape) > 2:
            raise Exception('IDSC Features can only describe binary images')
        
        contour_points = self._sample_contour_points(binary, 100)
        cv2.drawContours(binary, [contour_points], 0, (255, 0, 0))
        dist_matrix = self._build_distance_matrix(contour_points, binary)
        raw_features = self._build_shape_context(dist_matrix, contour_points, 
                                              binary)
        # flatten the histograms
        self._raw = raw_features
    
    def decompose(self, dictionary):
        pass
    
    def _sample_contour_points(self, binary, n):
        im2, contours, hierarchy = cv2.findContours(binary.copy(), 
                                                    cv2.RETR_TREE, 
                                                    cv2.CHAIN_APPROX_SIMPLE)
        contour_points = contours[0]
        # find contour points adds unnessecary dim., 
        # remove second dim of length 1
        contour_points = np.reshape(contour_points, (len(contour_points), 2))
        # take evenly distributed points
        idx = np.linspace(0, len(contour_points) - 1, num=n).astype(np.int)
        contour_points = contour_points[idx]     
        return contour_points
     
    def _build_distance_matrix(self, contour_points, binary):
        #dist_matrix = pairwise_distances(contour_points, metric='euclidean')
        
        dist_matrix = np.zeros((len(contour_points), len(contour_points)))        
        
        for i, p1 in enumerate(contour_points):
            for j, p2 in enumerate(contour_points[i+1:]):
                line_points = points_on_line(p1, p2, 10)
                for point in line_points:
                    # point on line is not in shape
                    # row is y and col is x
                    if binary[int(point[1]), int(point[0])] == 0:
                        break
    
                else:
                    d = dist_func(p1, p2)
                    dist_matrix[j + i, i] = d
                    dist_matrix[i, j + i] = d
        return dist_matrix
        
    def _build_shape_context(self, distance_matrix, contour_points, binary):
        histogram = []
        max_log_distance = np.log2(dist_func((0, 0), binary.shape))
        distances = floyd_warshall(distance_matrix, directed=False)
    
        for i, (x0, y0) in enumerate(contour_points):
            hist = np.zeros((8, 8))
    
            # Calculate the contour tangent
            (px, py) = contour_points[i-1]
            (nx, ny) = contour_points[(i+1) % len(contour_points)]
            contourTangent = np.arctan2(ny-py, nx-px)
    
            for j, (x1, y1) in enumerate(contour_points):
                if j == i: continue
                distance = distances[i, j]
                if distance!=0:                
                    log_dist = np.log2(distance)  
                else:
                    log_dist = max_log_distance
                angle = (contourTangent - 
                         np.arctan2(y1-y0, x1-x0)) % (2 * np.pi)
                dist_bucket = int(min(np.floor(log_dist / 
                                                  (max_log_distance/8)), 7))
                angle_bucket = int(min(angle / (np.pi/4), 7))
    
                hist[angle_bucket, dist_bucket] += 1
    
            histogram.append(hist)
    
        return np.array(histogram)
    