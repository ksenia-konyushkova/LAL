import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import time


class ActiveLearner:
    '''This is the base class for active learning models'''

    def __init__(self, dataset, nEstimators, name):
        '''input: dataset -- an object of class Dataset or any inheriting classes
                  nEstimators -- the number of estimators for the base classifier, usually set to 50
                  name -- name of the method for saving the results later'''
        
        self.dataset = dataset
        self.indicesKnown = dataset.indicesKnown
        self.indicesUnknown = dataset.indicesUnknown
        # base classification model
        self.nEstimators = nEstimators
        self.model = RandomForestClassifier(self.nEstimators, n_jobs=8)
        self.name = name
        
        
    def reset(self):
        
        '''forget all the points sampled by active learning and set labelled and unlabelled sets to default of the dataset'''
        self.indicesKnown = self.dataset.indicesKnown
        self.indicesUnknown = self.dataset.indicesUnknown
        
        
    def train(self):
        
        '''train the base classification model on currently available datapoints'''
        trainDataKnown = self.dataset.trainData[self.indicesKnown,:]
        trainLabelsKnown = self.dataset.trainLabels[self.indicesKnown,:]
        trainLabelsKnown = np.ravel(trainLabelsKnown)
        self.model = self.model.fit(trainDataKnown, trainLabelsKnown)
        
        
    def evaluate(self, performanceMeasures):
        
        '''evaluate the performance of current classification for a given set of performance measures
        input: performanceMeasures -- a list of performance measure that we would like to estimate. Possible values are 'accuracy', 'TN', 'TP', 'FN', 'FP', 'auc' 
        output: performance -- a dictionary with performanceMeasures as keys and values consisting of lists with values of performace measure at all iterations of the algorithm'''
        performance = {}
        test_prediction = self.model.predict(self.dataset.testData)   
        m = metrics.confusion_matrix(self.dataset.testLabels,test_prediction)
        
        if 'accuracy' in performanceMeasures:
            performance['accuracy'] = metrics.accuracy_score(self.dataset.testLabels,test_prediction)
            
        if 'TN' in performanceMeasures:
            performance['TN'] = m[0,0]
        if 'FN' in performanceMeasures:    
            performance['FN'] = m[1,0]
        if 'TP' in performanceMeasures:    
            performance['TP'] = m[1,1]
        if 'FP' in performanceMeasures:
            performance['FP'] = m[0,1]
            
        if 'auc' in performanceMeasures:
            test_prediction = self.model.predict_proba(self.dataset.testData)  
            test_prediction = test_prediction[:,1]
            performance['auc'] = metrics.roc_auc_score(self.dataset.testLabels, test_prediction)
            
        return performance
    
        
class ActiveLearnerRandom(ActiveLearner):
    '''Randomly samples the points'''
    
    def selectNext(self):
                
        self.indicesUnknown = np.random.permutation(self.indicesUnknown)
        self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([self.indicesUnknown[0]])]));            
        self.indicesUnknown = self.indicesUnknown[1:]
        
        
class ActiveLearnerUncertainty(ActiveLearner):
    '''Points are sampled according to uncertainty sampling criterion'''
    
    def selectNext(self):
                
        # predict for the rest the datapoints
        unknownPrediction = self.model.predict_proba(self.dataset.trainData[self.indicesUnknown,:])[:,0]
        selectedIndex1toN = np.argsort(np.absolute(unknownPrediction-0.5))[0]
        selectedIndex = self.indicesUnknown[selectedIndex1toN]
                
        self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([selectedIndex])]))
        self.indicesUnknown = np.delete(self.indicesUnknown, selectedIndex1toN)  
        
        
class ActiveLearnerLAL(ActiveLearner):
    '''Points are sampled according to a method described in K. Konyushkova, R. Sznitman, P. Fua 'Learning Active Learning from data'  '''
    
    def __init__(self, dataset, nEstimators, name, lalModel):
        
        ActiveLearner.__init__(self, dataset, nEstimators, name)
        self.model = RandomForestClassifier(self.nEstimators, oob_score=True, n_jobs=8)
        self.lalModel = lalModel
    
    
    def selectNext(self):
        
        unknown_data = self.dataset.trainData[self.indicesUnknown,:]
        known_labels = self.dataset.trainLabels[self.indicesKnown,:]
        n_lablled = np.size(self.indicesKnown)
        n_dim = np.shape(self.dataset.trainData)[1]
        
        # predictions of the trees
        temp = np.array([tree.predict_proba(unknown_data)[:,0] for tree in self.model.estimators_])
        # - average and standard deviation of the predicted scores
        f_1 = np.mean(temp, axis=0)
        f_2 = np.std(temp, axis=0)
        # - proportion of positive points
        f_3 = (sum(known_labels>0)/n_lablled)*np.ones_like(f_1)
        # the score estimated on out of bag estimate
        f_4 = self.model.oob_score_*np.ones_like(f_1)
        # - coeficient of variance of feature importance
        f_5 = np.std(self.model.feature_importances_/n_dim)*np.ones_like(f_1)
        # - estimate variance of forest by looking at avergae of variance of some predictions
        f_6 = np.mean(f_2, axis=0)*np.ones_like(f_1)
        # - compute the average depth of the trees in the forest
        f_7 = np.mean(np.array([tree.tree_.max_depth for tree in self.model.estimators_]))*np.ones_like(f_1)
        # - number of already labelled datapoints
        f_8 = np.size(self.indicesKnown)*np.ones_like(f_1)
        
        # all the featrues put together for regressor
        LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7], [f_8]), axis=0)
        LALfeatures = np.transpose(LALfeatures)
            
        # predict the expercted reduction in the error by adding the point
        LALprediction = self.lalModel.predict(LALfeatures)
        # select the datapoint with the biggest reduction in the error
        selectedIndex1toN = np.argmax(LALprediction)
        # retrieve the real index of the selected datapoint    
        selectedIndex = self.indicesUnknown[selectedIndex1toN]
            
        self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([selectedIndex])]))
        self.indicesUnknown = np.delete(self.indicesUnknown, selectedIndex1toN)  