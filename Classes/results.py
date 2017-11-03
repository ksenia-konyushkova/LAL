import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors


class Results:
    '''The class that that saves, load and plots the results.'''
    
    def __init__(self, experiment = None, nExperiments = None):
        
        # the performances measures that can be computed
        self.existingMetrics = ['accuracy', 'auc', 'IoU', 'dice', 'f-measure']
        
        if experiment is not None:
            experiment.dtstname = experiment.dataset.__class__.__name__
            self.nIterations = experiment.nIterations
            self.nEstimators = experiment.nEstimators
            self.performanceMeasures = experiment.performanceMeasures
            self.dataset = experiment.dataset
            self.alearners = []
            for alearner in experiment.alearners:
                self.alearners.append(alearner.name)
            self.comment = experiment.comment
            self.nExperiments = nExperiments
            
            self.performances = dict()
            for alearner in self.alearners:
                self.performances[alearner] = dict()
                for performanceMeasure in self.performanceMeasures:
                    self.performances[alearner][performanceMeasure] = np.array([[]])
        
    
    def addPerformance(self, performance):
        '''This function adds performance measures of new experiments'''
        for alearner in performance:
            for performanceMeasure in performance[alearner]:
                if np.size(self.performances[alearner][performanceMeasure])==0:
                    self.performances[alearner][performanceMeasure] = np.array([performance[alearner][performanceMeasure]])
                else:
                    self.performances[alearner][performanceMeasure] = np.concatenate((self.performances[alearner][performanceMeasure], np.array([performance[alearner][performanceMeasure]])), axis=0)
                    
                    
    def saveResults(self, filename):
        '''Save the current results to a file filename in ./exp folder'''
        state = self.__dict__.copy()
        pkl.dump(state, open( './exp/'+filename+'.p', "wb" ) )    
        
    
    def readResult(self, filename):
        '''Read the results from filename from ./exp folder'''
        state = pkl.load( open ('./exp/'+filename+'.p', "rb") )
        self.__dict__.update(state)
        
    
    def plotResults(self, metrics = None):
        '''Plot the performance in the metrics, if metrics is not specified, plot all the metrics that were saved'''
        # add small epsilon to the denominator to avoid division by zero
        small_eps = 0.000001
        col = self._get_cmap(len(self.alearners)+1)
        if metrics is None:
            for performanceMeasure in self.performanceMeasures:
                plt.figure()
                i = 0
                for alearner in self.alearners:
                    avResult = np.mean(self.performances[alearner][performanceMeasure], axis=0)
                    plt.plot(avResult, color=col(i), label=alearner)
                    i = i+1
                plt.xlabel('# labelled points')
                plt.ylabel(performanceMeasure)
                lgd = plt.legend(loc='lower right')
        else:
            for performanceMeasure in metrics:
                if performanceMeasure in self.existingMetrics:
                    plt.figure()
                    i = 0
                    for alearner in self.alearners:
                        if performanceMeasure=='accuracy' or performanceMeasure=='auc':
                            avResult = np.mean(self.performances[alearner][performanceMeasure], axis=0)
                        if performanceMeasure=='auc':
                            avResult =np.mean(self.performances[alearner]['auc'],axis=(0))
                        if performanceMeasure=='IoU':
                            avResult =np.mean((self.performances[alearner]['TP']/(self.performances[alearner]['TP']+self.performances[alearner]['FP']+self.performances[alearner]['FN']+small_eps)),axis=(0))
                        elif performanceMeasure=='dice':
                            avResult = np.mean((2*self.performances[alearner]['TP']/(2*self.performances[alearner]['TP']+self.performances[alearner]['FP']+self.performances[alearner]['FN']+small_eps)),axis=(0))
                        elif performanceMeasure=='f-measure':
                            avResult = np.mean((2*self.performances[alearner]['TP']/(2*self.performances[alearner]['TP']+self.performances[alearner]['FP']+self.performances[alearner]['FN']+small_eps)),axis=(0))
                            
                        plt.plot(avResult, color=col(i), label=alearner)
                        i = i+1
                    plt.xlabel('# labelled points')
                    plt.ylabel(performanceMeasure)
                    lgd = plt.legend(loc='lower right')
                else:
                    print('This metric is not implemented, existing metrics = ', self.existingMetrics)
                
    
    def _get_cmap(self, N):
        '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
        RGB color.'''
        color_norm  = colors.Normalize(vmin=0, vmax=N-1)
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
        def map_index_to_rgb_color(index):
            return scalar_map.to_rgba(index)
        return map_index_to_rgb_color