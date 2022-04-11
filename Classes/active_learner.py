from multiprocessing import Pool

import numpy as np
from Classes.models import Model, PyTorchModel
from sklearn import metrics
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from Classes.svi import GaussianSVI
import torch


class ActiveLearner:
    '''This is the base class for active learning models'''

    def __init__(self, dataset, name, model: Model):
        '''input: dataset -- an object of class Dataset or any inheriting classes
                  name -- name of the method for saving the results later
                  model -- the scikit-learn model that will be doing the learning'''
        
        self.dataset = dataset
        self.indicesKnown = dataset.indicesKnown
        self.indicesUnknown = dataset.indicesUnknown
        # base classification model
        self.model = model
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
        test_prediction = self.model.predict(self.dataset.testData).flatten()
        m = metrics.confusion_matrix(self.dataset.testLabels, test_prediction)
        
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
        self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([self.indicesUnknown[0]])]))
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
    def __init__(self, dataset, name, model: RandomForestClassifier, lalModel):
        ActiveLearner.__init__(self, dataset, name, model)
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

class ActiveLearnerPNML(ActiveLearner):
    def selectNext(self):
        # 1. For each unlabelled point x
        #   i. For each label t
        #       a. Add the (x, t) pair to the data and fit
        #       b. Get p(t|x) and store
        #   ii. Normalize p(t | x) for all t
        #   iii. Compute uncertainty using some metric
        #   iv. Update index of most uncertain point if necessary
        labels = np.unique(self.dataset.trainLabels)
        max_uncertainity = float("-inf")
        selectedIndex = -1
        selectedIndex1toN = -1
        for i, unknown_index in enumerate(self.indicesUnknown):
            label_probabilities = []
            for j, t in enumerate(labels):
                temp_model = self.model.clone()
                train_data = np.concatenate(
                    (
                        self.dataset.trainData[self.indicesKnown, :],
                        self.dataset.trainData[(unknown_index,), :]
                    )
                )
                train_labels = np.concatenate(
                    (
                        self.dataset.trainLabels[self.indicesKnown, :],
                        np.array([[t]])
                    )
                )
                train_labels = np.ravel(train_labels)
                temp_model = temp_model.fit(train_data, train_labels)
                # Get the probability predicted for class t
                pred = temp_model.predict_proba(self.dataset.trainData[(unknown_index,), :])[:, j]
                label_probabilities.append(pred)
            label_probabilities = np.array(label_probabilities)
            # stats.entropy() will automatically normalize
            entropy = stats.entropy(label_probabilities)
            if entropy > max_uncertainity:
                max_uncertainity = entropy
                selectedIndex = unknown_index
                selectedIndex1toN = i

        self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([selectedIndex])]))
        self.indicesUnknown = np.delete(self.indicesUnknown, selectedIndex1toN)


class ActiveLearnerACNML(ActiveLearner):
    def __init__(self, dataset, name, model: PyTorchModel):
        super().__init__(dataset, name, model)

    def log_prior(self, latent):
        normal = torch.distributions.normal.Normal(0, 1)
        return torch.sum(normal.log_prob(latent), axis=-1)

    def log_likelihood(self, latent):
        batch_size = latent.shape[0]
        result = np.zeros(batch_size)
        for n in range(batch_size):
            self.model.set_parameters(latent[n, :])
            probabilities = torch.Tensor(self.model.predict_proba(self.dataset.trainData))
            log_prob = torch.sum(torch.log(torch.maximum(torch.gather(probabilities, dim=1, index=torch.Tensor(self.dataset.trainLabels).long()), torch.Tensor([1e-9]))))
            result[n] = log_prob
        return torch.Tensor(result)

    def log_joint(self, latent):
        return self.log_likelihood(latent) + self.log_prior(latent)

    def get_approximate_posterior(self):
        print(self.model.total_params)

        # Hyperparameters
        n_iters = 800
        num_samples_per_iter = 5

        svi = GaussianSVI(true_posterior=self.log_joint, num_samples_per_iter=num_samples_per_iter)

        # Set up optimizer.
        D = self.model.total_params
        init_mean = torch.randn(D)
        init_mean.requires_grad = True
        init_log_std  = torch.randn(D)
        init_log_std.requires_grad = True
        init_params = (init_mean, init_log_std)

        params = init_params

        def callback(params, t):
            if t % 25 == 0:
                print("Iteration {} lower bound {}".format(t, svi.objective(params)))

        def update(params):
            loss = svi.objective(params)
            loss.backward()
            optim = torch.optim.SGD(params, lr=1e-5, momentum=0.9)
            optim.step()
            return params

        # Main loop.
        print("Optimizing variational parameters...")
        for i in range(n_iters):
            params = update(params)
            callback(params, i)

        return params

    def selectNext(self):
        # 1. For each unlabelled point x
        #   i. For each label t
        #       a. Add the (x, t) pair to the data and fit
        #       b. Get p(t|x) and store
        #   ii. Normalize p(t | x) for all t
        #   iii. Compute uncertainty using some metric
        #   iv. Update index of most uncertain point if necessary
        labels = np.unique(self.dataset.trainLabels)
        max_uncertainity = float("-inf")
        selectedIndex = -1
        selectedIndex1toN = -1

        (svi_mean, svi_log_std) = self.get_approximate_posterior()

        for i, unknown_index in enumerate(self.indicesUnknown):
            label_probabilities = []
            for j, t in enumerate(labels):
                temp_model = self.model.clone()
                train_data = np.concatenate(
                    (
                        self.dataset.trainData[self.indicesKnown, :],
                        self.dataset.trainData[(unknown_index,), :]
                    )
                )
                train_labels = np.concatenate(
                    (
                        self.dataset.trainLabels[self.indicesKnown, :],
                        np.array([[t]])
                    )
                )
                train_labels = np.ravel(train_labels)
                temp_model.set_parameters(svi_mean)
                # Get the probability predicted for class t
                pred = temp_model.predict_proba(self.dataset.trainData[(unknown_index,), :])[:, j]
                label_probabilities.append(pred + torch.exp(self.log_joint(svi_mean.unsqueeze(dim=0))).item())
            label_probabilities = np.array(label_probabilities)
            # stats.entropy() will automatically normalize
            entropy = stats.entropy(label_probabilities)
            if entropy > max_uncertainity:
                max_uncertainity = entropy
                selectedIndex = unknown_index
                selectedIndex1toN = i

        self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([selectedIndex])]))
        self.indicesUnknown = np.delete(self.indicesUnknown, selectedIndex1toN)