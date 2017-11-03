from sklearn.ensemble import RandomForestRegressor
import numpy as np
import math

class LALmodel:
    ''' Class for the regressor that predicts the expected error reduction caused by adding datapoints'''
    
    def __init__(self, all_data_for_lal, all_labels_for_lal):
        
        self.all_data_for_lal = all_data_for_lal
        self.all_labels_for_lal = all_labels_for_lal
        
    def crossValidateLALmodel(self, possible_estimators, possible_depth, possible_features):
        ''' Cross-validate the regressor model.
        input: possible_estimators -- list of possible number of estimators (trees) in Random Forest regression
        possible_depth -- list of possible maximum depth of the tree in RF regressor
        possible_features -- list of possible maximum number of features in a split of tree in RF regressor'''
            
        best_score = -math.inf

        self.best_est = 0
        self.best_depth = 0
        self.best_feat = 0
    
        print('start cross-validating..')
        for est in possible_estimators:
            for depth in possible_depth:
                for feat in possible_features:
                    model = RandomForestRegressor(n_estimators = est, max_depth=depth, max_features=feat, oob_score=True, n_jobs=8)
                    model.fit(self.all_data_for_lal[:,:], np.ravel(self.all_labels_for_lal))
                    if model.oob_score_>best_score:
                        self.best_est = est
                        self.best_depth = depth
                        self.best_feat = feat
                        self.model = model
                        best_score = model.oob_score_
                    print('parameters tested = ', est, ', ', depth, ', ', feat, ', with the score = ', model.oob_score_)
        # now train with the best parameters
        print('best parameters = ', self.best_est, ', ', self.best_depth, ', ', self.best_feat, ', with the best score = ', best_score)
        return best_score
    
    
    def builtModel(self, est, depth, feat):
        ''' Fits the regressor with the parameters identifier as an input '''
            
        self.model = RandomForestRegressor(n_estimators = est, max_depth=depth, max_features=feat, oob_score=True, n_jobs=8)
        self.model.fit(self.all_data_for_lal, np.ravel(self.all_labels_for_lal))
        print('oob score = ', self.model.oob_score_)
