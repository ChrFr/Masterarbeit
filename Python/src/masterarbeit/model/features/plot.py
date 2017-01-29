import seaborn as sns
sns.set(style="ticks", color_codes=True)

from masterarbeit.model.backend.hdf5_data import HDF5Pandas
from masterarbeit.model.features.hu_moments import HuMoments

def plot(features):
    #species = feature_frame['species']
    #del feature_frame['species']
    #feature_frame = (feature_frame - feature_frame.mean()) / (feature_frame.max() - feature_frame.min())
    #feature_frame['species'] = species
    g = sns.pairplot(feature_frame, hue="category")
    sns.plt.show()

if __name__ == '__main__':      
    h5 = HDF5Pandas()
    h5.open('../../batch_test.h5')     
    features = h5.read_feature(HuMoments)
    plot(features)