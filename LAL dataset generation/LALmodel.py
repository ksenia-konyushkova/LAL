from sklearn.ensemble import RandomForestRegressor
import numpy as np
import math

class LALmodel:

    def __init__(self, all_data_for_lal, all_labels_for_lal):
        
        self.all_data_for_lal = all_data_for_lal
        self.all_labels_for_lal = all_labels_for_lal
        
    def crossValidateLALmodel(self):
            
        possible_estimators = [500, 1000, 5000]
        possible_depth = [5, 10, 20]
        possible_features =[3, 5, 7]
        small_number = 0.0001
    
        best_score = -math.inf

        self.best_est = 0
        self.best_depth = 0
        self.best_feat = 0
    
        print('start cross-validating..')
        for est in possible_estimators:
            for depth in possible_depth:
                for feat in possible_features:
                    model = RandomForestRegressor(n_estimators = est, max_depth=depth, max_features=feat, oob_score=True, n_jobs=8)
                    model.fit(self.all_data_for_lal[:,:-1], np.ravel(self.all_labels_for_lal))
                    if model.oob_score_>best_score+small_number:
                        self.best_est = est
                        self.best_depth = depth
                        self.best_feat = feat
                        self.model = model
                        best_score = model.oob_score_
        # now train with the best parameters
        print('best parameters = ', self.best_est, ', ', self.best_depth, ', ', self.best_feat, ', with the best score = ', best_score)
        return best_score
