from masterarbeit.model.classifiers.classifier import Classifier
from masterarbeit.model.backend.data import load_class, class_to_string
from sklearn.svm import LinearSVC
import pickle

class SVM(Classifier):   
    
    label = 'Support Vector Machine'
    
    def __init__(self, name, seed=None):
        super(SVM, self).__init__(name, seed=seed)
        self.meta = {}
        
    def setup_model(self, input_dim, n_classes):    
        self.input_dim = input_dim
        # if random_state is None, no predefined seed will be used by SVM
        self.model = LinearSVC(random_state=self.seed) 

    def _train(self, values, classes, n_classes):
        input_dim = values.shape[1]
    
        self.setup_model(input_dim, n_classes)        
        self.model.fit(values, classes)

    def _predict(self, values):
        classes = self.model.predict(values)
        return classes
    
    def serialize(self):
        pickled = pickle.dumps(self.model)
        trained_features = [class_to_string(ft) for ft in self.trained_features]
        return [[pickled], trained_features, self.trained_categories]
    
    def deserialize(self, serialized):
        self.model = pickle.loads(serialized[0][0])
        self.trained_features = [load_class(ft) for ft in serialized[1]]
        self.trained_categories = serialized[2] 