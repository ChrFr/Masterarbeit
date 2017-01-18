class PreProcessor():
    
    label = 'None'    
     
    def read(self, path):
        return np.zeros((1024, 768))
           
    def write(self, path):
        return False
         
    def process(self, image, steps_dict=None):
        return np.zeros((1024, 768))