import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import unittest
import matplotlib.pyplot as plt

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections
    def hitRate(self):
        return self.hits / (self.hits+self.misses)
    def falseAlarmRate(self):
        return self.falseAlarms / (self.falseAlarms + self.correctRejections)
    def __add__(self, other):
        return SignalDetection(self.hits + other.hits, self.misses + other.misses, self.falseAlarms + 
                other.falseAlarms, self.correctRejections + other.correctRejections)
    def __mul__(self, k):
        return SignalDetection(self.hits *k, self.misses *k, self.falseAlarms *k, self.correctRejections * k)
    def d_prime(self):
        return (norm.ppf(self.hitRate()) - norm.ppf(self.falseAlarmRate()))
    def criterion(self):
        return -0.5 * (norm.ppf(self.hitRate()) + norm.ppf(self.falseAlarmRate()))
    @staticmethod
    def simulate(dprime, criteriaList, signalCount, noiseCount):
        sdtList = [None] * len(criteriaList)
        for i in range(len(criteriaList)):
            hRate = norm.cdf((dprime - 2*criteriaList[i])/2)
            faRate = norm.cdf((- dprime - 2*criteriaList[i])/2)
            # Hit and False alarm rates calculated given dprime and criterion values
            sdtList[i] = SignalDetection(np.random.binomial(signalCount, hRate), np.random.binomial(signalCount, 1-hRate), np.random.binomial(noiseCount, faRate), np.random.binomial(noiseCount, 1 - faRate))
        return sdtList
    def plot_sdt(self):
        plt.axvline((self.d_prime() / 2) + self.criterion(), color = 'yellow')
        plt.axhline(y = 0.4, color = 'g', xmin = 0.5, xmax = (self.d_prime() + 5)/10)
        x_axis = np.arange(-5, 5, 0.01)
        plt.plot(x_axis, norm.pdf(x_axis, 0, 1), color = 'r' ,label = "Noise")
        plt.plot(x_axis, norm.pdf(x_axis, self.d_prime(), 1), color = 'b', label = "Signal")
        plt.ylabel('Probability Density')
        plt.xlabel('Signal Strength')
        plt.title('Signal Detection Theory Plot')
        plt.legend(loc="upper left")
        plt.show()
    def nLogLikelihood(self, hitRate, falseAlarmRate):
        ell = - (self.hits * np.log(hitRate) + self.misses * np.log(1-hitRate) + self.falseAlarms * np.log(falseAlarmRate) + self.correctRejections * np.log(1-falseAlarmRate))
        return ell
    @staticmethod
    def rocCurve(falseAlarmRate, a):
        hitRate = norm.cdf(a + norm.ppf(falseAlarmRate))
        return hitRate
    @staticmethod
    def rocLoss(a, sdtList):
        L = 0
        for i in range(len(sdtList)):
            HitRate = sdtList[i].rocCurve(sdtList[i].falseAlarmRate(), a)
            L += sdtList[i].nLogLikelihood(HitRate, sdtList[i].falseAlarmRate())
        return L
    @staticmethod
    def plot_roc(sdtList):
        x = [0] * (len(sdtList))
        # Done so 0 and 1 are included in the plot
        for i in range(len(sdtList)):
            x[i] = sdtList[i].falseAlarmRate()
        y = [0] * (len(sdtList))
        for i in range(len(sdtList)):
            y[i] = sdtList[i].hitRate()
        plt.plot(sorted(x), sorted(y), marker="o", linestyle="None", color ='black')
        plt.plot(np.arange(0,1,0.01), np.arange(0,1,0.01), linestyle='dashed', color = 'black')
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Hit Rate')
        plt.title('ROC Plot')
    @staticmethod
    def fit_roc(sdtList):
        def roc_Loss(a):
            return SignalDetection.rocLoss(a, sdtList)
        result = minimize(fun = roc_Loss, x0 = np.random.randn())
        aHat = result.x[0]
        x_line = np.arange(0,1,0.01)
        y_line = SignalDetection.rocCurve(x_line, aHat)
        SignalDetection.plot_roc(sdtList)
        plt.plot(x_line, y_line, color ='r')
        plt.show()
        return aHat
    
class TestSignalDetection(unittest.TestCase):
    """
    Test suite for SignalDetection class.
    """

    def test_d_prime_zero(self):
        """
        Test d-prime calculation when hits and false alarms are 0.
        """
        sd   = SignalDetection(15, 5, 15, 5)
        expected = 0
        obtained = sd.d_prime()
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_d_prime_nonzero(self):
        """
        Test d-prime calculation when hits and false alarms are nonzero.
        """
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.421142647060282
        obtained = sd.d_prime()
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_zero(self):
        """
        Test criterion calculation when hits and false alarms are both 0.
        """
        sd   = SignalDetection(5, 5, 5, 5)
        expected = 0
        obtained = sd.criterion()
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_nonzero(self):
        """
        Test criterion calculation when hits and false alarms are nonzero.
        """
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.463918426665941
        obtained = sd.criterion()
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_addition(self):
        """
        Test addition of two SignalDetection objects.
        """
        sd = SignalDetection(1, 1, 2, 1) + SignalDetection(2, 1, 1, 3)
        expected = SignalDetection(3, 2, 3, 4).criterion()
        obtained = sd.criterion()
        self.assertEqual(obtained, expected)

    def test_multiplication(self):
        """
        Test multiplication of a SignalDetection object with a scalar.
        """
        sd = SignalDetection(1, 2, 3, 1) * 4
        expected = SignalDetection(4, 8, 12, 4).criterion()
        obtained = sd.criterion()
        self.assertEqual(obtained, expected)

    def test_simulate_single_criterion(self):
        """
        Test SignalDetection.simulate method with a single criterion value.
        """
        dPrime       = 1.5
        criteriaList = [0]
        signalCount  = 1000
        noiseCount   = 1000
        
        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 1)
        sdt = sdtList[0]
        
        self.assertEqual(sdt.hits             , sdtList[0].hits)
        self.assertEqual(sdt.misses           , sdtList[0].misses)
        self.assertEqual(sdt.falseAlarms      , sdtList[0].falseAlarms)
        self.assertEqual(sdt.correctRejections, sdtList[0].correctRejections)

    def test_simulate_multiple_criteria(self):
        """
        Test SignalDetection.simulate method with multiple criterion values.
        """
        dPrime       = 1.5
        criteriaList = [-0.5, 0, 0.5]
        signalCount  = 1000
        noiseCount   = 1000
        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 3)
        for sdt in sdtList:
            self.assertLessEqual (sdt.hits              ,  signalCount)
            self.assertLessEqual (sdt.misses            ,  signalCount)
            self.assertLessEqual (sdt.falseAlarms       ,  noiseCount)
            self.assertLessEqual (sdt.correctRejections ,  noiseCount)
   
    def test_nLogLikelihood(self):
        """
        Test case to verify nLogLikelihood calculation for a SignalDetection object.
        """
        sdt = SignalDetection(10, 5, 3, 12)
        hit_rate = 0.5
        false_alarm_rate = 0.2
        expected_nll = - (10 * np.log(hit_rate) +
                           5 * np.log(1-hit_rate) +
                           3 * np.log(false_alarm_rate) +
                          12 * np.log(1-false_alarm_rate))
        self.assertAlmostEqual(sdt.nLogLikelihood(hit_rate, false_alarm_rate),
                               expected_nll, places=6)
        
    def test_rocLoss(self):
        """
        Test case to verify rocLoss calculation for a list of SignalDetection objects.
        """
        sdtList = [
            SignalDetection( 8, 2, 1, 9),
            SignalDetection(14, 1, 2, 8),
            SignalDetection(10, 3, 1, 9),
            SignalDetection(11, 2, 2, 8),
        ]
        a = 0
        expected = 99.3884
        self.assertAlmostEqual(SignalDetection.rocLoss(a, sdtList), expected, places=4)
        
    def test_integration(self):
        """
        Test case to verify integration of SignalDetection simulation and ROC fitting.
        """
        dPrime  = 1
        sdtList = SignalDetection.simulate(dPrime, [-1, 0, 1], 1e7, 1e7)
        aHat    = SignalDetection.fit_roc(sdtList)
        self.assertAlmostEqual(aHat, dPrime, places=2)
        plt.close()
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
