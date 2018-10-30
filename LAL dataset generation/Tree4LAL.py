import numpy as np
import scipy
import GPy
from scipy import stats
import scipy.io as sio
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

class Tree4LAL:

    def __init__(self, criterion, dataset, lalModels, method):
                
        self.dataset = dataset
        self.criterion = criterion
        self.lalModels = lalModels
        # let's decide that the number of estimators for any classifier is 50
        self.n_estimators = 50
        self.method = method
        
    def generateTree(self, depth):
            
        # first get 1 positive and 1 negative point
        cl1 = np.nonzero(self.dataset.trainLabels==1)[0]
        indeces1 = np.random.permutation(cl1)
        self.indecesKnown = np.array([indeces1[0]])
        
        cl2 = np.nonzero(self.dataset.trainLabels==0)[0]
        indeces2 = np.random.permutation(cl2)
        self.indecesKnown = np.concatenate(([self.indecesKnown, np.array([indeces2[0]])]))
        self.indecesUnknown = np.concatenate(([indeces1[1:], indeces2[1:]]))
        self.indecesUnknown = np.random.permutation(self.indecesUnknown)
        
        if depth>2:
            if self.criterion=='random':
                # combine all the rest of the indeces that were not sampled yet
                #indecesRestAll = np.concatenate(([indeces1[1:], indeces2[1:]]))
                # permute them
                #indecesRestAll = np.random.permutation(indecesRestAll)
       
                # if we need more than 2 datapoints (and yes, we need 3)
            
                self.indecesKnown = np.concatenate(([self.indecesKnown, self.indecesUnknown[0:depth-2]]))
                self.indecesUnknown = self.indecesUnknown[depth-2:]

            
            elif self.criterion=='iterative': 
            
                # we build tree iteratively based on previous training iterations
                          
                iteration2simulate = np.arange(2,depth,1)
                
                for it in iteration2simulate:
                    
                    known_data = self.dataset.trainData[self.indecesKnown,:]
                    known_labels = self.dataset.trainLabels[self.indecesKnown]
                    unknown_data = self.dataset.trainData[self.indecesUnknown,:]
                    unknown_labels = self.dataset.trainLabels[self.indecesUnknown]
                    
                    # train a model every time we grow the tree
                    self.model = RandomForestClassifier(self.n_estimators, oob_score=True, n_jobs=8)
                    known_labels = np.ravel(known_labels)
                    self.model = self.model.fit(known_data, known_labels)
                    
                    # it should modify the list of known and unknown indeces
                    self._selectNext(self.model, self.lalModels[it-2], it, known_data, known_labels, unknown_data, unknown_labels)            

    def getLALdatapoints(self, n_points_per_experiment):
        
        # train a model based on all known data
        known_data = self.dataset.trainData[self.indecesKnown,:]
        known_labels = self.dataset.trainLabels[self.indecesKnown]
        unknown_data = self.dataset.trainData[self.indecesUnknown,:]
        unknown_labels = self.dataset.trainLabels[self.indecesUnknown]
       
        # train a model every time we grow the tree
        self.model = RandomForestClassifier(self.n_estimators, oob_score=True, n_jobs=8)
        known_labels = np.ravel(known_labels)
        self.model = self.model.fit(known_data, known_labels)
        
        nFeatures = 8
        # get my features
        # now we need the number of training datapoints as a feature
        feature_vector = self._getFeaturevector4LAL(self.model, unknown_data[0:n_points_per_experiment,:], known_labels, nFeatures)
        
        # predict on test data to evaluate the classifier quality

        if self.method=='error':
            test_prediction = self.model.predict(self.dataset.testData)  
            quality_0_error = metrics.zero_one_loss(self.dataset.testLabels, test_prediction)
        elif self.method=='auc':
            test_prediction = self.model.predict_proba(self.dataset.testData)  
            test_prediction = test_prediction[:,1]
            quality_0_auc = metrics.roc_auc_score(self.dataset.testLabels, test_prediction)
            
        # sample n_points_per_experiment samples that we will add to the training dataset and check the change in error
        gains_quality = np.zeros((n_points_per_experiment))
        for i in range(n_points_per_experiment):
            # try to add it to the labelled data
            new_known_data = np.concatenate((known_data,[unknown_data[i,:]]))
            new_known_labels = np.concatenate((known_labels,unknown_labels[i]))

            # train updated model - model_i
            m_i = RandomForestClassifier(self.n_estimators, n_jobs=8)
            new_known_labels = np.ravel(new_known_labels)
            m_i = m_i.fit(new_known_data, new_known_labels)
                    
            if self.method=='error':
                # predict on test data
                test_prediction = m_i.predict(self.dataset.testData)   
                quality_i_error = metrics.zero_one_loss(self.dataset.testLabels, test_prediction)
                # how much the quality has changed
                gains_quality[i]=(quality_0_error-quality_i_error)

            elif self.method=='auc':
                test_prediction = m_i.predict_proba(self.dataset.testData)  
                test_prediction = test_prediction[:,1]
                quality_i_auc = metrics.roc_auc_score(self.dataset.testLabels, test_prediction)
                gains_quality[i]=(quality_i_auc-quality_0_auc)
        
        #print(quality_0_error)
        #plt.plot(feature_vector[:,0], gains_quality, '.')
        #plt.show(block=False)
                  
        return feature_vector, gains_quality    
            
            
            
            
    # ---------------------------PRIVATE FUNCTIONS-------------------------
    # ---------------------------------------------------------------------
    def _selectNext(self, model, lalModel, it, known_data, known_labels, unknown_data, unknown_labels):
                    
        LALfeatures = self._getFeaturevector4LAL(model, unknown_data, known_labels, 7)
            
        LALprediction = lalModel.predict(LALfeatures)
        selectedIndex1toN = np.argmax(LALprediction)
        selectedIndex = self.indecesUnknown[selectedIndex1toN]
        
        self.indecesKnown = np.concatenate(([self.indecesKnown, np.array([selectedIndex])]))
        self.indecesUnknown = np.delete(self.indecesUnknown, selectedIndex1toN) 

        
    def _getFeaturevector4LAL(self, model, unknown_data, known_labels, nFeatures):
        
        # - predicted mean (but only for n_points_per_experiment datapoints)
        prediction_unknown = model.predict_proba(unknown_data)
        
        # features are in the following order:
        # 1: prediction probability
        # 2: prediction variance
        # 3: proportion of positive class
        # 4: oob score
        # 5: coeficiant of variance of feature importance
        # 6: variance of forest
        # 7: average depth of trees
        # 8: number of datapoints in training
        
        f_1 = prediction_unknown[:,0]
        # - predicted standard deviation 
        # need to call each tree of a forest separately to get a prediction because it is not possible to get them all immediately
        f_2 = np.std(np.array([tree.predict_proba(unknown_data)[:,0] for tree in model.estimators_]), axis=0)
        # - proportion of positive points
        # check np.size(self.indecesKnown)
        f_3 = (sum(known_labels>0)/np.size(self.indecesKnown))*np.ones_like(f_1)
        # the score estimated on out of bag estimate
        f_4 = model.oob_score_*np.ones_like(f_1)
        # - coeficient of variance of feature importance
        # check if this is the number of features!
        f_5 = np.std(model.feature_importances_/self.dataset.trainData.shape[1])*np.ones_like(f_1)
        # - estimate variance of forest by looking at avergae of variance of some predictions
        f_6 = np.mean(np.std(np.array([tree.predict_proba(unknown_data)[:,0] for tree in model.estimators_]), axis=0))*np.ones_like(f_1)
        # - compute the average depth of the trees in the forest
        f_7 = np.mean(np.array([tree.tree_.max_depth for tree in model.estimators_]))*np.ones_like(f_1)            
        LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7]), axis=0)
        
        if nFeatures>7:
            # the same as f_3, check np.size(self.indecesKnown)
            f_8 = np.size(self.indecesKnown)*np.ones_like(f_1)
            LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7], [f_8]), axis=0)
        
        LALfeatures = np.transpose(LALfeatures)        
        
        return LALfeatures
            