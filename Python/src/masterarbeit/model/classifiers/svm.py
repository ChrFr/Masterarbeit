from masterarbeit.model.classifiers.classifier import Classifier
from sklearn.svm import LinearSVC

class SVM(Classifier):   
    
    label = 'Support Vector Machine'
    
    def __init__(self, name):
        super(SVM, self).__init__(name)
        self.meta = {}
        
    def setup_model(self, input_dim, n_classes):    
        self.input_dim = input_dim
        self.model = LinearSVC() 

    def _train(self, values, classes, n_classes):
        input_dim = values.shape[1]
    
        self.setup_model(input_dim, n_classes)        
        self.model.fit(values, classes)

    def _predict(self, values):
        classes = self.model.predict(values)
        return classes