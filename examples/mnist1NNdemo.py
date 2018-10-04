# Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance

import numpy as np

DEFAULT_TEST_SIZE = None
DEFAULT_BUCKET_SIZE = 1000
MINIMUM_BUCKET_SIZE = 100


def PredictandError(trainims, trainlabs, testims, testlabs):

    from sklearn.metrics.pairwise import euclidean_distances
    Distances = euclidean_distances(trainims, testims, squared=True)

    predictions = trainlabs[np.argmin(Distances, axis=0)]
    return np.not_equal(predictions, testlabs).sum()


def run_main(TestSize, BucketSize):

    from keras.datasets import mnist
    (TrainIms, TrainLabels), (TestIms, TestLabels) = mnist.load_data()

    TrainIms = TrainIms.reshape(TrainIms.shape[0], -1)
    TestIms = TestIms.reshape(TestIms.shape[0], -1)

    if TestSize is DEFAULT_TEST_SIZE:
        TestSize = len(TrainLabels)
    BucketSize = min(BucketSize, TestSize)

    import time
    t0 = time.time()

    # error: 3.0900000000000025
    # Time taken: 19.416749477386475
    errors = [
        PredictandError(TrainIms, TrainLabels,
                        TestIms[i:min(i+BucketSize, TestSize)],
                        TestLabels[i:min(i+BucketSize, TestSize)])
        for i in range(0, TestSize, BucketSize)
    ]

    # error: 3.090000000000004
    # Time taken: 80.45218276977539    
    #from sklearn.neighbors import KNeighborsClassifier
    #clf = KNeighborsClassifier(n_neighbors=1, algorithm='brute', p=2, n_jobs=-1)
    #testims = TrainIms[:TestSize]
    #testlabs = TrainLabels[:TestSize]
    #clf.fit(testims, testlabs)
    #errors = [(1 - clf.score(testims, testlabs)) * 100]

    t1 = time.time()

    print("error: {:.3f} %".format(sum(errors) / TestSize * 100))
    print("Time taken: {:.3f} sec".format(t1 - t0))


def parse_args():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--test-size', type=int, default=DEFAULT_TEST_SIZE,
                        help='size of the test images')
    parser.add_argument('--bucket-size', type=int, default=DEFAULT_BUCKET_SIZE,
                        help='size of the test bucket')
    args = parser.parse_args()

    assert (args.test_size is DEFAULT_TEST_SIZE or args.test_size >= DEFAULT_BUCKET_SIZE), \
        'Too small test size (< {}).'.format(DEFAULT_BUCKET_SIZE)
    assert (args.bucket_size >= MINIMUM_BUCKET_SIZE), \
        'Too small bucket size (< {}).'.format(MINIMUM_BUCKET_SIZE)

    return args.test_size, args.bucket_size


if __name__ == '__main__':
    test_size, bucket_size = parse_args()
    run_main(test_size, bucket_size)
