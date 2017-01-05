import unittest
import numpy as np
import os
from masterarbeit.model.preprocessor.opencv_preprocessor import OpenCVPreProcessor
from masterarbeit.model.features.hu_moments import HuMoments
from masterarbeit.model.backend.hdf5_data import HDF5Data
from masterarbeit.model.features.feature import Feature

class TestPreprocessing(unittest.TestCase):
    preprocessor = OpenCVPreProcessor()
    testfile = 'test.h5'
    
    def test_extract_feature(self):
        source_pixels = self.preprocessor.read_file('DSC_5820.JPG')
        binary = self.preprocessor.binarize()
        feature = HuMoments().describe(binary)        
        
    def test_write_feature(self):
        feature = Feature('Testfeature')
        feature.values = np.random.rand(5)
        h5 = HDF5Data()
        h5.open(self.testfile)
        h5.add_feature('Testspecies', feature)
        h5.close()
    
    @classmethod
    def tearDownClass(cls):
        os.remove(cls.testfile)

if __name__ == '__main__':
    unittest.main()