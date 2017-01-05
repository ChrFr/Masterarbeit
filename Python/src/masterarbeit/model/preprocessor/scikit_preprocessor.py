from skimage import data, io, filters
import numpy as np
from collections import OrderedDict
import os

from masterarbeit.model.preprocessor.preprocessor import PreProcessor

class OpenCVPreProcessor(PreProcessor):
    
    def __init__(self):
        self.source_pixels = None
        self.processed_images = OrderedDict()