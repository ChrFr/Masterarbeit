import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import copy
import seaborn as sns

def mlp_evaluation():
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    dataframe = pandas.read_csv("iris.csv", header=None)
    dataset = dataframe.values
    X = dataset[:,0:4].astype(float)
    Y = dataset[:,4]    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
        model.add(Dense(3, init='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model    
    
    estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)

    KerasClassifier.get_params = get_params_fix
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))    
    
def get_params_fix(classifier, deep=False):
    res = copy.deepcopy(classifier.sk_params)
    res.update({'build_fn': classifier.build_fn})
    return res 
    
def mlp():
    iris = sns.load_dataset("iris")
    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(iris, hue="species")
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load dataset
    dataframe = pandas.read_csv("iris.csv", header=None)
    dataset = dataframe.values
    X = dataset[:,0:4].astype(float)
    Y = dataset[:,4]
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    # define baseline model
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
        model.add(Dense(3, init='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
    
    KerasClassifier.get_params = get_params_fix
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)
    estimator.fit(X_train, Y_train)
    predictions = estimator.predict(X_test)
    print(predictions)
    print(encoder.inverse_transform(predictions))    
    print('done')
    

if __name__ == '__main__':
    mlp()