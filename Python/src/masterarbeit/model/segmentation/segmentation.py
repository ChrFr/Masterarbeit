class Segmentation():
    
    label = 'None'    
     
    def read(self, path):
        return np.zeros((1024, 768))
           
    def write(self, path):
        return False
         
    def process(self, image, steps=None):
        return np.zeros((1024, 768))