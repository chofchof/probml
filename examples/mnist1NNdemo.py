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

def run_main(TestSize, BucketSize):

    BucketSize = min(BucketSize, TestSize)

    import time
    t0 = time.time()

    # error: 3.0900000000000025
    # Time taken: 19.416749477386475
    errors = [PredictandError(TestIms[i:(i+BucketSize)], TestLabels[i:(i+BucketSize)])
              for i in range(0, TestSize, BucketSize)]

    # error: 3.090000000000004
    # Time taken: 80.45218276977539    
    #from sklearn.neighbors import KNeighborsClassifier
    #clf = KNeighborsClassifier(n_neighbors=1, algorithm='brute', p=2, n_jobs=-1)
    #testims = TrainIms[:TestSize]
    #testlabs = TrainLabels[:TestSize]
    #clf.fit(testims, testlabs)
    #errors = [(1 - clf.score(testims, testlabs)) * 100]

    t1 = time.time()

    print("error: {}".format(np.mean(errors)))
    print("Time taken: {}".format(t1 - t0))

def parse_args():

    default_test_size = len(TestLabels)
    default_bucket_size = 1000
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-size', type=int, default=default_test_size,
                        help='size of the test images')
    parser.add_argument('--bucket-size', type=int, default=default_bucket_size,
                        help='size of the test bucket')
    args = parser.parse_args()
    
    assert (args.test_size >= default_bucket_size), \
        'Too small test size (< {}).'.format(default_bucket_size)
    assert (args.bucket_size >= 100), 'Too small bucket size (< 100).'
    
    return args.test_size, args.bucket_size
    
if __name__ == '__main__':
    test_size, bucket_size = parse_args()
    run_main(test_size, bucket_size)
