import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
labels = ['love', 'hate', 'happiness']
data = np.array([[0, 1000, 20],
                 [1000, 0, 1000],
                 [10, 1000 , 0]])
distances = pdist(data)
distance_matrix = squareform(distances, checks=True)
mds = MDS(n_components=3, dissimilarity='precomputed', verbose=10)
coordinates = mds.fit_transform(distance_matrix)
crds_w_labs = zip(labels, coordinates)
for crd in crds_w_labs:
    print(crd)

