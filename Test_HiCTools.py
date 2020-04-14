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
        self.testMatrix = np.array([[1, 2, 3, 4, 5, 6],
                                    [6, 5, 4, 3, 2, 1],
                                    [2, 4, 6, 8, 10, 12],
                                    [1, 2, 2, 3, 3, 4],
                                    [5, 5, 4, 4, 3, 3],
                                    [1, 1, 1, 1, 1, 1]], dtype=np.float64)
    def testCenterEnrichment(self):
        """Tests center enrichment of sliding diamond"""
        x, y = HT.slidingDiamond(self.gaussian, sideLen=6)
        centerMean = np.mean(y[np.where(np.abs(x) < 1)])
        borderMean = np.mean(y[:5])
        self.assertTrue(centerMean > 5 * borderMean)
    def testEvenDiamond(self):
        """Tests sliding a diamond of even sidelength"""
        x, y = HT.slidingDiamond(self.testMatrix, sideLen=2, centerX=False)
        xCheck = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        yCheck = np.array([3.5, 4.75, 4.75, 3.25, 2.0])
        self.assertTrue(all(np.isclose(x, xCheck)))
        self.assertTrue(all(np.isclose(y, yCheck)))
    def testEvenDiamondXNorm(self):
        """Tests sliding a diamond of even sidelength
        with x normalization"""
        x, y = HT.slidingDiamond(self.testMatrix, sideLen=2, centerX=True)
        xCheck = np.array([-2, -1, 0, 1, 2])
        yCheck = np.array([3.5, 4.75, 4.75, 3.25, 2.0])
        self.assertTrue(all(np.isclose(x, xCheck)))
        self.assertTrue(all(np.isclose(y, yCheck)))
    def testOddDiamond(self):
        """Tests sliding a diamond of even sidelength"""
        x, y = HT.slidingDiamond(self.testMatrix, sideLen=3, centerX=False)
        xCheck = np.array([1, 2, 3, 4])
        yCheck = np.array([3.666666666, 4.11111111, 4.77777777, 2.55555555])
        self.assertTrue(all(np.isclose(x, xCheck)))
        self.assertTrue(all(np.isclose(y, yCheck)))
    def testOddDiamondXNorm(self):
        """Tests sliding a diamond of even sidelength"""
        x, y = HT.slidingDiamond(self.testMatrix, sideLen=3, centerX=True)
        xCheck = np.array([-1.5, -0.5, 0.5, 1.5])
        yCheck = np.array([3.666666666, 4.11111111, 4.77777777, 2.55555555])
        self.assertTrue(all(np.isclose(x, xCheck)))
        self.assertTrue(all(np.isclose(y, yCheck)))


def suiteAll():
    suite = unittest.TestSuite()
    suite.addTest(TestResources("testArms"))
    suite.addTest(TestSlidingDiamond("testCenterEnrichment"))
    suite.addTest(TestSlidingDiamond("testEvenDiamond"))
    suite.addTest(TestSlidingDiamond("testEvenDiamondXNorm"))
    suite.addTest(TestSlidingDiamond("testOddDiamond"))
    suite.addTest(TestSlidingDiamond("testOddDiamondXNorm"))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suiteAll())
