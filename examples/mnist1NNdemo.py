# Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance

import numpy as np

from keras.datasets import mnist
(TrainIms, TrainLabels), (TestIms, TestLabels) = mnist.load_data()

TrainIms = TrainIms.reshape(TrainIms.shape[0], -1)
TestIms = TestIms.reshape(TestIms.shape[0], -1)

def PredictandError(testims, testlabs):
    from sklearn.metrics.pairwise import euclidean_distances
    Distances = euclidean_distances(TrainIms, testims, squared=True)

    predictions = TrainLabels[np.argmin(Distances, axis=0)]

    error = 1 - np.mean(np.equal(predictions, testlabs))
    return (error * 100)

import time
t0 = time.time()

# error: 3.0900000000000025
# Time taken: 19.416749477386475
    
BucketSize = 1000
errors = [PredictandError(TestIms[i:(i+BucketSize)], TestLabels[i:(i+BucketSize)])
          for i in range(0, len(TestLabels), BucketSize)]

# error: 3.090000000000004
# Time taken: 80.45218276977539    

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=1, algorithm='brute', p=2, n_jobs=-1)
#clf.fit(TrainIms, TrainLabels)
#errors = [(1 - clf.score(TestIms, TestLabels)) * 100]

t1 = time.time()

print('error: {}'.format(np.mean(errors)))
print('Time taken: {}'.format(t1 - t0))