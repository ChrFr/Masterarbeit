from masterarbeit.model.preprocessor import preprocessor_opencv as pocv
from masterarbeit.model.preprocessor import preprocessor_skimage as psk
from masterarbeit.model.features.hu_moments import HuMoments
from masterarbeit.model.features.zernike_moments import ZernikeMoments

IMAGE_FILTER = 'Images (*.png, *.jpg)'
ALL_FILES_FILTER = 'All Files(*.*)'

PRE_PROCESSORS = (pocv.Binarize, psk.BinarizeHSV, psk.SegmentGabor, 
                  pocv.SegmentVeinsGabor)
FEATURES = (HuMoments, ZernikeMoments)