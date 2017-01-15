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
    test_species = ['Testspecies1', 'Testspecies2']
    
    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.testfile):
            os.remove(cls.testfile)
    
    def test_extract_feature(self):
        source_pixels = self.preprocessor.read_file('DSC_5820.JPG')
        binary = self.preprocessor.binarize()
        feature = HuMoments().describe(binary)        
        
    def test1_write_features(self):
        feature = Feature('Testfeature')
        h5 = HDF5Data()
        h5.open(self.testfile)        
        
        for s in self.test_species:            
        
            feature.values = np.random.rand(5)        
            h5.add_feature(s, feature)
            
            feature.values = np.random.rand(5)      
            h5.add_feature(s, feature)
            
        h5.close()
            
    def test2_read_features(self):        
        h5 = HDF5Data()
        h5.open(self.testfile)        
        species_stored = h5.get_species()
        
        # checks if both lists contain same items
        # (not just item count, as name might suggest)
        self.assertCountEqual(species_stored, self.test_species)
        
        h5.close()
        
    
if __name__ == '__main__':
    unittest.main()