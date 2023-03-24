import numpy as np
import unittest
from SignalDetection import SignalDetection
import scipy.stats

class Metropolis:
    def __init__(self, logTarget, initialState):
        self.logTarget = logTarget
        self.initialState = initialState
        self.currentState = initialState
        self.sd = 1
        self.mu = 0
        self.samples = []
    def __accept(self, proposal):
        a = min(0, self.logTarget(proposal) - self.logTarget(self.currentState))
        if np.log(np.random.uniform()) < a:
            self.currentState = proposal
            yesno = True
            return yesno
        else:
            yesno = False
            return yesno
    def adapt(self, blockLengths):
        for i in range(len(blockLengths)):
            r = [None] * blockLengths[i]
            for j in range(blockLengths[i]):
                Proposed = np.random.normal(loc = self.mu, scale = self.sd)
                if Metropolis.__accept(self, Proposed) == True:
                    r[j] = 1
                    self.currentState = Proposed
                else:
                    r[j] = 0
            rk = sum(r)/len(r)
            self.mu = self.currentState
            self.sd = self.sd * (rk/0.4)**1.1
        return self
    def sample(self, nSamples):
        samples = [None] * nSamples
        for i in range(nSamples):
            Proposed = np.random.normal(loc=self.currentState, scale=self.sd)
            if Metropolis.__accept(self, Proposed) == True:
                self.currentState = Proposed
            samples[i] = Proposed
        self.samples = samples
        return self
    def summary(self):
        summ = dict({'mean': np.mean(self.samples), 'c025' : np.percentile(self.samples, 2.5), 'c975': np.percentile(self.samples, 97.5)})
        return summ

class TestMetropolis(unittest.TestCase):
    """
    Unit test suite for Metropolis class.
    """
    
    def test_sd(self):
        """
        Test standard deviation calculation to confirm it changes after inputting data.
        """
        sdtList = SignalDetection.simulate(dprime       = 1,
                                           criteriaList = [-1, 0, 1],
                                           signalCount  = 40,
                                           noiseCount   = 40)
        def loglik(a):
            return -SignalDetection.rocLoss(a, sdtList) + scipy.stats.norm.logpdf(a, loc = 0, scale = 10)

        # Create a Metropolis sampler object and adapt it to the target distribution
        sampler = Metropolis(logTarget = loglik, initialState = 0)
        sampler = sampler.adapt(blockLengths = [200]*3)

        # Sample from the target distribution
        sampler = sampler.sample(nSamples = 400)
        sd   = sampler.sd
        expected = 1
        obtained = sd
        self.assertNotEqual(obtained, expected)
    def test_corruption(self):
        sdtList = SignalDetection.simulate(dprime       = 1,
                                           criteriaList = [-1, 0, 1],
                                           signalCount  = 40,
                                           noiseCount   = 40)
        def loglik(a):
            """
            Test internal corruption of the object by changing initial state
            """
            return -SignalDetection.rocLoss(a, sdtList) + scipy.stats.norm.logpdf(a, loc = 0, scale = 10)

        # Create a Metropolis sampler object and adapt it to the target distribution
        sampler = Metropolis(logTarget = loglik, initialState = 0)
        exData = sampler.adapt(blockLengths = [200]*3)
        expected = exData.sd
        sampler.initialState = 2
        obData = sampler.adapt(blockLengths = [200]*3)
        obtained = obData.sd
        self.assertNotEqual(obtained, expected)
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
