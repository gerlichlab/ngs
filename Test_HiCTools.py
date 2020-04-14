import unittest
import pandas as pd
from NGS import HiCTools as HT
from pandas.testing import assert_frame_equal
import numpy as np
from scipy.stats import multivariate_normal

# define functions

def generate2dGauss(mean, variance, gridsize=10, spacing=0.2):
    # define grid
    x, y = np.mgrid[-gridsize:gridsize:spacing, -gridsize:gridsize:spacing]
    # get position vector
    pos = np.dstack((x, y))
    # get multivariate_normal distribution with given mean and covariance matrix
    rv1 = multivariate_normal(mean, variance)
    # get matrix and blank out diagonal
    rvmat1 = rv1.pdf(pos)
    rvmat1[np.diag_indices_from(rvmat1)] = np.nan
    return rvmat1

# define tests
class TestResources(unittest.TestCase):
    def testArms(self):
        check = pd.read_csv("./testFiles/arms.csv")
        arms = HT.getArmsHg19()
        assert_frame_equal(check, arms)


class TestSlidingDiamond(unittest.TestCase):
    def setUp(self):
        self.gaussian = generate2dGauss(mean=[0, 0], variance=[[3, 0], [0, 3]], gridsize=15)
    def testCenterEnrichment(self):
        """Tests center enrichment of sliding diamond"""
        x, y = HT.slidingDiamond(self.gaussian, sideLen=6)
        centerMean = np.mean(y[np.where(np.abs(x) < 1)])
        borderMean = np.mean(y[:5])
        self.assertTrue(centerMean > 5 * borderMean)

def suiteAll():
    suite = unittest.TestSuite()
    suite.addTest(TestResources("testArms"))
    suite.addTest(TestSlidingDiamond("testCenterEnrichment"))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suiteAll())
