import unittest
import pandas as pd
from NGS import HiCTools as HT
from pandas.testing import assert_frame_equal
import numpy as np
from scipy.stats import multivariate_normal
import cooler
from pandas.testing import assert_frame_equal

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
        self.gaussian = generate2dGauss(
            mean=[0, 0], variance=[[3, 0], [0, 3]], gridsize=15
        )
        self.testMatrix = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
                [2, 4, 6, 8, 10, 12],
                [1, 2, 2, 3, 3, 4],
                [5, 5, 4, 4, 3, 3],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=np.float64,
        )

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


class TestGetExpected(unittest.TestCase):
    def setUp(self):
        self.cooler1 = cooler.Cooler("testFiles/test1.mcool::/resolutions/5000000")
        self.cooler2 = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        self.arms1 = pd.read_csv("testFiles/arms.csv")
        self.arms2 = pd.DataFrame(
            {"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0]
        )

    def test_ignore0_diag(self):  # calculated using HT.getExpected
        result = HT.getExpected(self.cooler1, self.arms1, proc=1, ignoreDiagonals=0)
        checkFrame = pd.read_csv("testFiles/test1_expected_ignore_0_diag.csv")
        assert_frame_equal(result, checkFrame)

    def test_ignore1_diag(self):  # calculated using HT.getExpected
        result = HT.getExpected(self.cooler1, self.arms1, proc=1, ignoreDiagonals=1)
        checkFrame = pd.read_csv("testFiles/test1_expected_ignore_1_diag.csv")
        assert_frame_equal(result, checkFrame)

    def test_ignore2_diag(self):  # calculated using HT.getExpected
        result = HT.getExpected(self.cooler1, self.arms1, proc=1, ignoreDiagonals=2)
        checkFrame = pd.read_csv("testFiles/test1_expected_ignore_2_diag.csv")
        assert_frame_equal(result, checkFrame)

    def test_synthetic_data(self):  # calculated by hand
        result = HT.getExpected(self.cooler2, self.arms2, proc=1, ignoreDiagonals=0)
        check = pd.read_csv("testFiles/test_expected_chrSyn.csv")
        assert_frame_equal(result, check)


class TestAssignRegions(unittest.TestCase):
    def setUp(self):
        self.arms = pd.read_csv("testFiles/arms.csv")

    def test_case1(self):
        bedFile = pd.read_csv("testFiles/testSmall.bed", sep="\t")
        result = HT.assignRegions(
            window=500000,
            binsize=50000,
            chroms=bedFile["chrom"],
            positions=bedFile["pos"],
            arms=self.arms,
        )
        expected = pd.read_csv("testFiles/testAssignRegions.csv")
        assert_frame_equal(result, expected)

    def test_case2(self):
        bedFile = pd.read_csv("testFiles/testSmall_2.bed", sep="\t")
        result = HT.assignRegions(
            window=500000,
            binsize=50000,
            chroms=bedFile["chrom"],
            positions=bedFile["pos"],
            arms=self.arms,
        )
        expected = pd.read_csv("testFiles/testAssignRegions_2.csv")
        assert_frame_equal(result, expected)


class TestPileupICCF(unittest.TestCase):
    def test_no_collapse(self):  # Simulated cooler
        """tests extracting individual windows
        without averaging."""
        positionFrame = pd.read_csv("testFiles/posPileups.csv")
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        assigned = HT.assignRegions(
            50000, 10000, positionFrame["chrom"], positionFrame["pos"], arms
        )
        c = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        result = HT.doPileupICCF(c, assigned, proc=1, collapse=False)
        expected = np.load("testFiles/test_pileups_iccf_noCollapse.npy")
        self.assertTrue(np.allclose(result, expected))

    def test_collapse(self):  # Simulated cooler
        positionFrame = pd.read_csv("testFiles/posPileups.csv")
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        assigned = HT.assignRegions(
            50000, 10000, positionFrame["chrom"], positionFrame["pos"], arms
        )
        c = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        result = HT.doPileupICCF(c, assigned, proc=1, collapse=True)
        expected = np.load("testFiles/test_pileups_iccf_collapse.npy")
        self.assertTrue(np.allclose(result, expected))


class TestPileupObsExp(unittest.TestCase):
    def test_no_collapse(self):  # Simulated cooler
        """tests extracting individual windows
        without averaging."""
        positionFrame = pd.read_csv("testFiles/posPileups.csv")
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        assigned = HT.assignRegions(
            50000, 10000, positionFrame["chrom"], positionFrame["pos"], arms
        )
        c = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        expF = pd.read_csv("testFiles/test_expected_chrSyn.csv")
        result = HT.doPileupObsExp(c, expF, assigned, proc=1, collapse=False)
        expected = np.load("testFiles/test_pileups_obsExp_noCollapse.npy")
        self.assertTrue(np.allclose(result, expected))

    def test_collapse(self):  # Simulated cooler
        positionFrame = pd.read_csv("testFiles/posPileups.csv")
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        assigned = HT.assignRegions(
            50000, 10000, positionFrame["chrom"], positionFrame["pos"], arms
        )
        c = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        expF = pd.read_csv("testFiles/test_expected_chrSyn.csv")
        result = HT.doPileupObsExp(c, expF, assigned, proc=1, collapse=True)
        expected = np.load("testFiles/test_pileups_obsExp_collapse.npy")
        self.assertTrue(np.allclose(result, expected))


class TestPairingScoreObsExp(unittest.TestCase):
    def test_specificRegions(self):  # Simulated cooler
        positionFrame = pd.read_csv("testFiles/posPileups.csv")
        positionFrame.loc[:, "mid"] = positionFrame["pos"]
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        c = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        expF = pd.read_csv("testFiles/test_expected_chrSyn.csv")
        pairingScore = HT.getPairingScoreObsExp(
            c, expF, 50000, regions=positionFrame, arms=arms, norm=False
        )
        expected = pd.read_csv("testFiles/test_pairingScore_obsExp_specificRegions.csv")
        assert_frame_equal(pairingScore, expected)


if __name__ == "__main__":
    res = unittest.main(verbosity=3, exit=False)

