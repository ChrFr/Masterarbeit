import numpy as np
import pytest
import os
from masterarbeit.model.features.moments import HuMoments
from masterarbeit.model.features.moments import ZernikeMoments
from masterarbeit.model.features.idsc import IDSCGaussiansKMeans
from masterarbeit.model.backend.hdf5_data import HDF5Pandas

_image_shape = (200, 200, 3)
_features = [ZernikeMoments, HuMoments]
_data = {HDF5Pandas: 'hdf5_test.h5'}

@pytest.fixture(scope='module', params=_data)
def data(request):    
    Data = request.param
    data = Data()
    source = _data[Data]
    if os.path.exists(source):
        os.remove(source)
    data.open(source)    
    yield data
    data.close()
    
@pytest.fixture(scope='module')
def images():
    bgr_only = np.empty(_image_shape, dtype=np.uint8)    
    bgr_only.fill(255)
    square_img = bgr_only.copy()
    square_img[30:100, 30:100] = 125   
    return [bgr_only, square_img]

@pytest.fixture(scope='module', params=_features)
def features(request, images):
    Feature = request.param
    features = []    
    for i in range(len(images)):
        category = 'Test{}'.format(i)
        feat = Feature(category)
        features.append(feat)
    return features

@pytest.mark.order1
def test_extract_features(features, images, data):
    z = zip(features, images)
    for feature, image in z:
        feature.describe(image)

@pytest.mark.order2
def test_save_features(features, data):
    data.add_features(features)

@pytest.mark.order3
def test_read_features(data, features):    
    feat_type = type(features[0])   
    stored_features = data.get_features(feat_type)
    for i, feature in enumerate(features):
        assert (stored_features[i]._v != feature._v).sum() == 0
        
    categories = data.get_categories()
    # 2 different binaries defined as different categories
    assert len(categories) == 2
    
    for category in categories:
        count = data.get_feature_count(category, feat_type)
        # we extracted 1 feature per feature type and species
        assert count == 1
    
    ## checks if both lists contain same items
    ## (not just item count, as name might suggest)
    #self.assertCountEqual(categories, self.test_categories)

    #feature_frame = h5.read_feature(HuMoments)
    #self.assertEqual(len(feature_frame.index), len(self.test_species) * 2)  

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
