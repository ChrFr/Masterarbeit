import numpy as np
import pytest
from masterarbeit.model.features.hu_moments import HuMoments
from masterarbeit.model.features.zernike_moments import ZernikeMoments
from masterarbeit.model.features.leaf_veins import LeafVeins
from masterarbeit.model.backend.hdf5_data import HDF5Pandas

_image_shape = (200, 200)
_features = [ZernikeMoments, HuMoments]
_data = {HDF5Pandas: 'hdf5_test.h5'}

@pytest.fixture(params=_data)
def data(request):    
    Data = request.param
    data = Data()
    source = _data[Data]
    data.open(source)    
    yield data
    data.close()
    
@pytest.fixture(scope='module')
def binaries():
    zeros = np.zeros(_image_shape, dtype=np.uint8)
    square = zeros.copy()
    square[30:100, 30:100] = 1
    return [zeros, square]

@pytest.fixture(scope='module', params=_features)
def features(request, binaries):
    Feature = request.param
    features = []    
    for i in range(len(binaries)):
        category = 'Test{}'.format(i)
        feat = Feature(category)
        features.append(feat)
    return features

@pytest.mark.order1
def test_extract_features(features, binaries, data):
    z = zip(features, binaries)
    for feature, binary in z:
        feature.extract(binary)

@pytest.mark.order2
def test_save_features(features, data):
    data.add_features(features)  

@pytest.mark.order3
def test_read_features(data):
    for Feature in _features:
        data.read_features(Feature)  
        
    categories = data.get_categories()
    # 2 different binaries defined as different categories
    assert len(categories) == 2
    
    ## checks if both lists contain same items
    ## (not just item count, as name might suggest)
    #self.assertCountEqual(categories, self.test_categories)

    #feature_frame = h5.read_feature(HuMoments)
    #self.assertEqual(len(feature_frame.index), len(self.test_species) * 2)    

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
