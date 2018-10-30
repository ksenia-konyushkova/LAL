import numpy as np
import scipy
import scipy.io as sio
import pandas as pd

from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split


class DatasetSimulated:

    def __init__(self, sizeTrain, n_dim):
        # now let's have unbalanced datasets
        cl1_prop = np.random.rand()
        # we want the proportion of class 1 to vary from 10% to 90%
        cl1_prop = (cl1_prop-0.5)*0.8+0.5
        trainSize1 = int(sizeTrain*cl1_prop)
        trainSize2 = sizeTrain-trainSize1
        
        testSize1 = trainSize1*10
        testSize2 = trainSize2*10
        
        all_data_for_lal = np.array([[]])
        all_labels_for_lal = np.array([[]])
        
        
        # generate parameters of datasets
        mean1 = scipy.random.rand(n_dim)
        cov1 = scipy.random.rand(n_dim,n_dim)-0.5
        cov1 = np.dot(cov1,cov1.transpose())
        mean2 = scipy.random.rand(n_dim)
        cov2 = scipy.random.rand(n_dim,n_dim)-0.5
        cov2 = np.dot(cov2,cov2.transpose())
        
        # generate data 2 dimensional features X1 and X2 and their labels Y1 and Y2
        # first class
        trainX1 = np.random.multivariate_normal(mean1, cov1, trainSize1)
        trainY1 = np.ones((trainSize1,1))
        # second class
        trainX2 = np.random.multivariate_normal(mean2, cov2, trainSize2)
        trainY2 = np.zeros((trainSize2,1))

        # let's generate the test data for AL evaluation
        testX1 = np.random.multivariate_normal(mean1, cov1, testSize1)
        testY1 = np.ones((testSize1,1))
        # second class 
        testX2 = np.random.multivariate_normal(mean2, cov2, testSize2)
        testY2 = np.zeros((testSize2,1))
        
        # all data
        self.trainData = np.concatenate((trainX1, trainX2), axis=0)
        self.trainLabels = np.concatenate((trainY1, trainY2))
        self.testData = np.concatenate((testX1, testX2), axis=0)
        self.testLabels = np.concatenate((testY1, testY2))