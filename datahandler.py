# call this datahandler.py

# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class helps to handle the data.

"""

import os
import random
import logging
import tensorflow as tf
import numpy as np




class Data(object):
    """
    If the dataset can be quickly loaded to memory self.X will contain np.ndarray
    Otherwise we will be reading files as we train. In this case self.X is a structure:
        self.X.paths        list of paths to the files containing pictures
        self.X.dict_loaded  dictionary of (key, val), where key is the index of the
                            already loaded datapoint and val is the corresponding index
                            in self.X.loaded
        self.X.loaded       list containing already loaded pictures
    """
    def __init__(self, opts, X, paths=None, dict_loaded=None, loaded=None,is_train=False):
        """
        X is either np.ndarray or paths
        """

        self.X = None
        self.normalize = opts['input_normalize_sym']
        self.paths = None
        self.dict_loaded = None
        self.loaded = None
        if isinstance(X, np.ndarray):
            self.X = X
            self.shape = X.shape
        

    def __len__(self):
        if isinstance(self.X, np.ndarray):
            return len(self.X)
        else:
            # Our dataset was too large to fit in the memory
            return len(self.paths)
    def __getitem__(self, key):
        if isinstance(self.X, np.ndarray):
            return self.X[key]
        else:
            # Our dataset was too large to fit in the memory
            if isinstance(key, int):
                keys = [key]
            elif isinstance(key, list):
                keys = key
            elif isinstance(key, np.ndarray):
                keys = list(key)
            elif isinstance(key, slice):
                start = key.start
                stop = key.stop
                step = key.step
                start = start if start is not None else 0
                if start < 0:
                    start += len(self.paths)
                stop = stop if stop is not None else len(self.paths) - 1
                if stop < 0:
                    stop += len(self.paths)
                step = step if step is not None else 1
                keys = range(start, stop, step)
            else:
                print(type(key))
                raise Exception('This type of indexing yet not supported for the dataset')
            res = []
            new_keys = []
            new_points = []
            for key in keys:
                if key in self.dict_loaded:
                    idx = self.dict_loaded[key]
                    res.append(self.loaded[idx])
                else:
                    """
                    if self.dataset_name == 'celebA':
                        point = self._read_celeba_image(self.data_dir, self.paths[key])
                    else:
                        raise Exception('Disc read for this dataset not implemented yet...')
                    """
                    if self.normalize:
                        point = (point - 0.5) * 2.
                    res.append(point)
                    new_points.append(point)
                    new_keys.append(key)
            n = len(self.loaded)
            cnt = 0
            for key in new_keys:
                self.dict_loaded[key] = n + cnt
                cnt += 1
            self.loaded.extend(new_points)
            return np.array(res)


    

class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """

    DATASET_NAMES=['mnist','2d_ring','2d_grid','hd']

    def __init__(self, opts):
        self.data_shape = None
        self.num_points = None
        self.data = None
        self.test_data = None
        self.labels = None
        self.test_labels = None
        self._load_data(opts)


    def _load_data(self, opts):
        """Load a dataset and fill all the necessary variables.

        """
        logging.debug("loading data")
        if opts['dataset'] == 'mnist':
            self._load_mnist(opts)
        elif opts['dataset'] == 'small_mnist':
            self._load_small_mnist(opts)
        elif opts['dataset']=='2d_ring':
            self._load_ring(opts)
        elif opts['dataset']=='small_2d_ring':
            self._load_small_ring(opts)
        elif opts['dataset']=='2d_grid':
            self._load_grid(opts)
        elif opts['dataset']=='small_2d_grid':
            self._load_small_grid(opts)
        elif opts['dataset']=='hd':
            logging.info("loading hd")
            self._load_hd(opts)
        elif opts['dataset']=='small_hd':
            logging.info("loading small hd")
            self._load_small_hd(opts)
        else:
            raise ValueError('Unknown %s' % opts['dataset'])


        



    def _load_mnist(self,opts):
        """Load data from MNIST or ZALANDO files.

        """
        
        print('Loading MNIST')

        # pylint: disable=invalid-name
        # Let us use all the bad variable names!
        tr_X = None
        tr_Y = None
        te_X = None
        te_Y = None

        #tr_X=np.load('mnist/train-images-idx3-ubyte')
        #tr_Y=np.load('mnist/train-labels-idx1-ubyte')
        #te_X=np.load('mnist/t10k-images-idx3-ubyte')
        #te_Y=np.load('mnist/t10k-labels-idx1-ubyte')

        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        tr_X = np.concatenate([mnist.train.images, mnist.validation.images])
        te_X = mnist.test.images
        tr_Y = np.concatenate([mnist.train.labels, mnist.validation.labels])
        te_Y = mnist.test.labels

        tr_Y = np.asarray(tr_Y)
        te_Y = np.asarray(te_Y)


        X = np.concatenate((tr_X, te_X), axis=0)
        y = np.concatenate((tr_Y, te_Y), axis=0)
        X = X / 255.

        X=np.reshape(X,(X.shape[0],28,28,1))

        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        np.random.seed()

        self.data_shape = (28,28, 1)
        test_size = 10000
        # test_size=10

        
        self.data = Data(opts, X[:-test_size])
        self.test_data = Data(opts, X[-test_size:])
        self.labels = y[:-test_size]
        self.test_labels = y[-test_size:]
        self.num_points = len(self.data)

        logging.debug('Loading Done.')


    def _load_small_mnist(self,opts):
        """Load data from MNIST or ZALANDO files.

        """
        
        """Load data from MNIST or ZALANDO files.

        """
        
        print('Loading SMALL MNIST')

        # pylint: disable=invalid-name
        # Let us use all the bad variable names!
        tr_X = None
        tr_Y = None
        te_X = None
        te_Y = None

        #tr_X=np.load('mnist/train-images-idx3-ubyte')
        #tr_Y=np.load('mnist/train-labels-idx1-ubyte')
        #te_X=np.load('mnist/t10k-images-idx3-ubyte')
        #te_Y=np.load('mnist/t10k-labels-idx1-ubyte')

        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        tr_X = np.concatenate([mnist.train.images, mnist.validation.images])
        te_X = mnist.test.images
        tr_Y = np.concatenate([mnist.train.labels, mnist.validation.labels])
        te_Y = mnist.test.labels

        tr_Y = np.asarray(tr_Y)
        te_Y = np.asarray(te_Y)

        X = np.concatenate((tr_X, te_X), axis=0)
        y = np.concatenate((tr_Y, te_Y), axis=0)
        X = X / 255.

        X=np.reshape(X,(X.shape[0],28,28,1))


        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        np.random.seed()

        self.data_shape = (28,28,1)
        # test_size=10

        
        self.data = Data(opts, X[0:20])
        self.test_data = Data(opts, X[20:40])
        self.labels = y[0:20]
        self.test_labels = y[20:40]
        self.num_points = len(self.data)

        logging.debug('Loading Done.')



    def _load_ring(self,opts):
        logging.debug('Loading ring dataset')

        # set all the relevant params
        train_data=np.load('synthetic_datasets/2d_ring.npy')
        test_data=np.load('synthetic_datasets/2d_ring_test.npy')

        logging.debug("shapes")
        logging.debug(train_data.shape)
        logging.debug(test_data.shape)

        train_data=np.reshape(train_data,(train_data.shape[0],2,1,1))
        test_data=np.reshape(test_data,(test_data.shape[0],2,1,1))

        self.data=Data(opts,train_data)
        self.test_data=Data(opts,test_data)
        self.data_shape=(2,1,1)
        self.num_points=len(self.data)



    def _load_small_ring(self,opts):
        logging.debug('Loading small ring dataset')

        # set all the relevant params
        train_data=np.load('synthetic_datasets/2d_ring.npy')[0:20]
        test_data=np.load('synthetic_datasets/2d_ring_test.npy')[0:20]


        train_data=np.reshape(train_data,(train_data.shape[0],2,1,1))
        test_data=np.reshape(test_data,(test_data.shape[0],2,1,1))


        self.data=Data(opts,train_data)
        self.test_data=Data(opts,test_data)
        self.data_shape=(2,1,1)
        self.num_points=len(self.data)

    def _load_grid(self,opts):
        logging.debug('Loading grid dataset')

        # set all the relevant params
        train_data=np.load('synthetic_datasets/2d_grid.npy')
        test_data=np.load('synthetic_datasets/2d_grid_test.npy')

        train_data=np.reshape(train_data,(train_data.shape[0],2,1,1))
        test_data=np.reshape(test_data,(test_data.shape[0],2,1,1))

        self.data=Data(opts,train_data)
        self.test_data=Data(opts,test_data)
        self.data_shape=(2,1,1)
        self.num_points=len(self.data)


    def _load_small_grid(self,opts):
        logging.debug('Loading grid dataset')

        # set all the relevant params
        train_data=np.load('synthetic_datasets/2d_grid.npy')
        test_data=np.load('synthetic_datasets/2d_grid_test.npy')

        train_data=train_data[0:20]
        test_data=test_data[0:20]

        train_data=np.reshape(train_data,(train_data.shape[0],2,1,1))
        test_data=np.reshape(test_data,(test_data.shape[0],2,1,1))

        self.data=Data(opts,train_data)
        self.test_data=Data(opts,test_data)
        self.data_shape=(2,1,1)
        self.num_points=len(self.data)

    def _load_hd(self,opts):
        logging.info('Loading high dimensional dataset')

        # set all the relevant params
        train_data=np.load('synthetic_datasets/hd_train.npy')
        test_data=np.load('synthetic_datasets/hd_test.npy')

        num_dim=train_data.shape[1]
        train_data=np.reshape(train_data,(train_data.shape[0],num_dim,1,1))
        test_data=np.reshape(test_data,(test_data.shape[0],num_dim,1,1))

        self.data=Data(opts,train_data)
        self.test_data=Data(opts,test_data)
        self.data_shape=(num_dim,1,1)
        self.num_points=len(self.data)


    def _load_small_hd(self,opts):
        logging.info('Loading small high dimensional dataset')

        # set all the relevant params
        train_data=np.load('synthetic_datasets/hd_train.npy')
        test_data=np.load('synthetic_datasets/hd_test.npy')

        train_data=train_data[0:20]
        test_data=test_data[0:20]

        num_dim=train_data.shape[1]
        train_data=np.reshape(train_data,(train_data.shape[0],num_dim,1,1))
        test_data=np.reshape(test_data,(test_data.shape[0],num_dim,1,1))

        self.data=Data(opts,train_data)
        self.test_data=Data(opts,test_data)
        self.data_shape=(num_dim,1,1)
        self.num_points=len(self.data)






