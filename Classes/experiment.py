class Experiment:
    '''The class that runs active learning experiment'''
    
    def __init__(self, nIterations, nEstimators, performanceMeasures, dataset, alearners, comment=''):
        
        self.nIterations = nIterations
        self.nEstimators = nEstimators
        self.performanceMeasures = performanceMeasures
        self.dataset = dataset
        self.alearners = alearners
        self.comment = comment
        self.performances = dict()
        for alearner in self.alearners:
            self.performances[alearner.name] = dict()
            for performanceMeasure in self.performanceMeasures:
                self.performances[alearner.name][performanceMeasure] = []

        
    def run(self):
        '''Run the experiment for nIterations for all alearners and return performances'''
        for it in range(self.nIterations):
            print('.', end="")
            for alearner in self.alearners:
                alearner.train()
                perf = alearner.evaluate(self.performanceMeasures)
                for key in perf:
                    self.performances[alearner.name][key].append(perf[key])
                alearner.selectNext()
        return self.performances
    
    
    def reset(self):
        '''Reset the experiment: reset the starting datapoint of the dataset, reset alearners and performances'''
        self.dataset.setStartState(self.dataset.nStart)
        for alearner in self.alearners:
            alearner.reset()
            
        self.performances = dict()
        for alearner in self.alearners:
            self.performances[alearner.name] = dict()
            for performanceMeasure in self.performanceMeasures:
                self.performances[alearner.name][performanceMeasure] = []
