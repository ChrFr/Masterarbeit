from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

class ConfusionMatrix():
    def __init__(self, real, predicted):
        self.metrics = confusion_matrix(real, predicted)  
        self.classes = np.unique(real)
        
    def normalize(self):
        self.metrics = (self.metrics.astype('float') / 
                        self.metrics.sum(axis=1)[:, np.newaxis])
    
    def plot(self, title='Confusion matrix', decimals=2):
        plt.figure()
        metrics = np.round(self.metrics, decimals=decimals)     
        plot_confusion_matrix(metrics, self.classes, title)        
        plt.show()
        
def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    """
    original code taken from 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')            
    plt.xlabel('Predicted label')    