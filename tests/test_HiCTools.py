"""Tests for HiCTools"""
import unittest
from functools import partial
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from scipy.stats import multivariate_normal
import cooler
import cooltools
from ngs import HiCTools as HT

# define functions


def generate_2d_gauss(mean, variance, gridsize=10, spacing=0.2):
    """Helper function to create 2d-Gaussian for use in
    sliding diamond testing."""
    # define grid
    x_coord, y_coord = np.mgrid[-gridsize:gridsize:spacing, -gridsize:gridsize:spacing]
    # get position vector
    pos = np.dstack((x_coord, y_coord))
    # get multivariate_normal distribution with given mean and covariance matrix
    rv1 = multivariate_normal(mean, variance)
    # get matrix and blank out diagonal
    rvmat1 = rv1.pdf(pos)
    rvmat1[np.diag_indices_from(rvmat1)] = np.nan
    return rvmat1


# define tests
class TestResources(unittest.TestCase):
    """Test whether resources accessed by bioframe are available
    and have not changed."""

    def test_arms(self):
        """Tests whether fetching and generating
        of supports for chromosomal arms of hg19 works."""
        check = pd.read_csv("./testFiles/arms.csv")
        arms = HT.get_arms_hg19()
        assert_frame_equal(check, arms)


class TestSlidingDiamond(unittest.TestCase):
    """Tests for sliding diamond function."""

    def setUp(self):
        """Setting up common resources."""
        self.gaussian = generate_2d_gauss(
            mean=[0, 0], variance=[[3, 0], [0, 3]], gridsize=15
        )
        self.test_matrix = np.array(
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

    def test_center_enrichment(self):
        """Tests center enrichment of sliding diamond
        of a synthetically generated gaussian."""
        x_values, y_values = HT.sliding_diamond(self.gaussian, side_len=6)
        center_mean = np.mean(y_values[np.where(np.abs(x_values) < 1)])
        border_mean = np.mean(y_values[:5])
        self.assertTrue(center_mean > 5 * border_mean)

    def test_even_diamond(self):
        """Tests sliding a diamond of even sidelength
        for small fixed matrix."""
        x_values, y_values = HT.sliding_diamond(
            self.test_matrix, side_len=2, center_x=False
        )
        x_check = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        y_check = np.array([3.5, 4.75, 4.75, 3.25, 2.0])
        self.assertTrue(all(np.isclose(x_values, x_check)))
        self.assertTrue(all(np.isclose(y_values, y_check)))

    def test_even_diamond_xnorm(self):
        """Tests sliding a diamond of even sidelength
        with x normalization (center is set to 0) for
        small fixed matrix.."""
        x_values, y_values = HT.sliding_diamond(
            self.test_matrix, side_len=2, center_x=True
        )
        x_check = np.array([-2, -1, 0, 1, 2])
        y_check = np.array([3.5, 4.75, 4.75, 3.25, 2.0])
        self.assertTrue(all(np.isclose(x_values, x_check)))
        self.assertTrue(all(np.isclose(y_values, y_check)))

    def test_odd_diamond(self):
        """Tests sliding a diamond of odd sidelength
        for small fixed matrix."""
        x_values, y_values = HT.sliding_diamond(
            self.test_matrix, side_len=3, center_x=False
        )
        x_check = np.array([1, 2, 3, 4])
        y_check = np.array([3.666666666, 4.11111111, 4.77777777, 2.55555555])
        self.assertTrue(all(np.isclose(x_values, x_check)))
        self.assertTrue(all(np.isclose(y_values, y_check)))

    def test_odd_diamond_xnorm(self):
        """Tests sliding a diamond of odd sidelength
        with x normalization for small fixed matrix."""
        x_values, y_values = HT.sliding_diamond(
            self.test_matrix, side_len=3, center_x=True
        )
        x_check = np.array([-1.5, -0.5, 0.5, 1.5])
        y_check = np.array([3.666666666, 4.11111111, 4.77777777, 2.55555555])
        self.assertTrue(all(np.isclose(x_values, x_check)))
        self.assertTrue(all(np.isclose(y_values, y_check)))


class TestGetExpected(unittest.TestCase):
    """Tests for getting expected diagonal counts from
    a Hi-C matrix."""

    def setUp(self):
        """Setting up common resources"""
        self.cooler = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        self.arms = pd.DataFrame(
            {"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0]
        )

    def test_synthetic_data(self):
        """Tests expected counts for synthetic Hi-C data.
        Known values were provided and expected counts for each diagonal
        calculated."""
        result = HT.get_expected(self.cooler, self.arms, proc=1, ignore_diagonals=0)
        check = pd.read_csv("testFiles/test_expected_chrSyn.csv")
        assert_frame_equal(result, check)

    def test_synthetic_data_mult_chroms(self):
        """Tests expected counts for synthetic Hi-C data
        with multiple chromosomal arms. Known values were
        provided and expected counts for each diagonal
        calculated."""
        arms = pd.DataFrame(
            {
                "chrom": ["chrSyn", "chrSyn"],
                "start": [0, 2000000],
                "end": [2000000, 4990000],
            }
        )
        result = HT.get_expected(self.cooler, arms, proc=1, ignore_diagonals=0)
        check = pd.read_csv("testFiles/test_expected_multiple_chroms.csv")
        assert_frame_equal(result, check)

    @unittest.skipIf(
        cooltools.__version__ == "0.2.0", "bug in the old cooltools version"
    )
    def test_expected_real_data(self):
        """Tests expected counts for real Hi-C data
        with multiple chromosomal arms. Known values were
        provided and expected counts for each diagonal
        calculated."""
        arms = HT.get_arms_hg19()
        cooler_file = cooler.Cooler(
            "testFiles/test3_realdata.mcool::/resolutions/50000"
        )
        result = HT.get_expected(cooler_file, arms, proc=1, ignore_diagonals=0)
        result_sorted = result.sort_values(by=["chrom", "start"]).drop(
            columns="count.sum"
        )
        check = pd.read_csv("testFiles/test_expected_realdata.csv")
        assert_frame_equal(result_sorted, check)


class TestAssignRegions(unittest.TestCase):
    """Tests assigning of chromosomal positions
    to Hi-C bins and chromosomal supports."""

    def setUp(self):
        """Setting up common resources"""
        self.arms = pd.read_csv("testFiles/arms.csv")

    def test_case1(self):
        """Test assignment of small, synthetic set of regions."""
        bed_file = pd.read_csv("testFiles/testSmall.bed", sep="\t")
        result = HT.assign_regions(
            window=500000,
            binsize=50000,
            chroms=bed_file["chrom"],
            positions=bed_file["pos"],
            arms=self.arms,
        )
        expected = pd.read_csv("testFiles/testAssignRegions.csv")
        assert_frame_equal(result, expected)

    def test_case2(self):
        """Test assignment of a different small,
        synthetic set of regions."""
        bed_file = pd.read_csv("testFiles/testSmall_2.bed", sep="\t")
        result = HT.assign_regions(
            window=500000,
            binsize=50000,
            chroms=bed_file["chrom"],
            positions=bed_file["pos"],
            arms=self.arms,
        )
        expected = pd.read_csv("testFiles/testAssignRegions_2.csv")
        assert_frame_equal(result, expected)


class TestPileupICCF(unittest.TestCase):
    """Tests pileup of iteratively corrected counts (ICCF)"""

    def test_no_collapse(self):
        """Tests pileup of synthetic Hi-C data, without collapsing results."""
        position_frame = pd.read_csv("testFiles/posPileups.csv")
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        assigned = HT.assign_regions(
            50000, 10000, position_frame["chrom"], position_frame["pos"], arms
        )
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        result = HT.do_pileup_iccf(cooler_file, assigned, proc=1, collapse=False)
        expected = np.load("testFiles/test_pileups_iccf_noCollapse.npy")
        self.assertTrue(np.allclose(result, expected))

    def test_collapse(self):
        """Tests pileup of synthetic Hi-C data, with collapsing results."""
        position_frame = pd.read_csv("testFiles/posPileups.csv")
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        assigned = HT.assign_regions(
            50000, 10000, position_frame["chrom"], position_frame["pos"], arms
        )
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        result = HT.do_pileup_iccf(cooler_file, assigned, proc=1, collapse=True)
        expected = np.load("testFiles/test_pileups_iccf_collapse.npy")
        self.assertTrue(np.allclose(result, expected))


class TestPileupObsExp(unittest.TestCase):
    """Tests pileup of Obs/Exp values."""

    def test_no_collapse(self):
        """Tests pileup of synthetic Hi-C data, without collapsing results."""
        position_frame = pd.read_csv("testFiles/posPileups.csv")
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        assigned = HT.assign_regions(
            50000, 10000, position_frame["chrom"], position_frame["pos"], arms
        )
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        exp_f = pd.read_csv("testFiles/test_expected_chrSyn.csv")
        result = HT.do_pileup_obs_exp(
            cooler_file, exp_f, assigned, proc=1, collapse=False
        )
        expected = np.load("testFiles/test_pileups_obsExp_noCollapse.npy")
        self.assertTrue(np.allclose(result, expected))

    def test_collapse(self):
        """Tests pileup of synthetic Hi-C data, with collapsing results."""
        position_frame = pd.read_csv("testFiles/posPileups.csv")
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        assigned = HT.assign_regions(
            50000, 10000, position_frame["chrom"], position_frame["pos"], arms
        )
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        exp_f = pd.read_csv("testFiles/test_expected_chrSyn.csv")
        result = HT.do_pileup_obs_exp(
            cooler_file, exp_f, assigned, proc=1, collapse=True
        )
        expected = np.load("testFiles/test_pileups_obsExp_collapse.npy")
        self.assertTrue(np.allclose(result, expected))


class TestPairingScoreObsExp(unittest.TestCase):
    """Tests for extracting Obs/Exp pairing score"""

    def test_specific_regions(self):
        """Tests functionality to extract Obs/Exp values
        at specific regions from synthetic Hi-C data."""
        position_frame = pd.read_csv("testFiles/posPileups.csv")
        position_frame.loc[:, "mid"] = position_frame["pos"]
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        exp_f = pd.read_csv("testFiles/test_expected_chrSyn.csv")
        pairing_score = HT.get_pairing_score_obs_exp(
            cooler_file, exp_f, 50000, regions=position_frame, arms=arms, norm=False
        )
        expected = pd.read_csv("testFiles/test_pairingScore_obsExp_specificRegions.csv")
        assert_frame_equal(pairing_score, expected)

    def test_genome_wide_not_norm(self):
        """Tests functionality to extract Obs/Exp values
        genome-wide without median normalization from synthetic Hi-C data."""
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        exp_f = pd.read_csv("testFiles/test_expected_chrSyn.csv")
        pairing_score = HT.get_pairing_score_obs_exp(
            cooler_file, exp_f, 50000, arms=arms, norm=False
        )
        expected = pd.read_csv(
            "testFiles/test_pairingScore_obsExp_genomeWide.csv",
            dtype={
                name: pairing_score.dtypes[name] for name in pairing_score.dtypes.index
            },
        )
        assert_frame_equal(pairing_score, expected)

    def test_genome_wide_norm(self):
        """Tests functionality to extract Obs/Exp values
        genome-wide with median normalization from synthetic Hi-C data."""
        position_frame = pd.read_csv("testFiles/posPileups.csv")
        position_frame.loc[:, "mid"] = position_frame["pos"]
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        exp_f = pd.read_csv("testFiles/test_expected_chrSyn.csv")
        pairing_score = HT.get_pairing_score_obs_exp(
            cooler_file, exp_f, 50000, arms=arms, norm=True
        )
        expected = pd.read_csv(
            "testFiles/test_pairingScore_obsExp_genomeWide_Norm.csv",
            dtype={
                name: pairing_score.dtypes[name] for name in pairing_score.dtypes.index
            },
        )
        assert_frame_equal(pairing_score, expected)

    def test_wrong_parameters(self):
        """Tests raising of error when specific
        region pileup is done with the norm parameter set to True."""
        position_frame = pd.read_csv("testFiles/posPileups.csv")
        position_frame.loc[:, "mid"] = position_frame["pos"]
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        exp_f = pd.read_csv("testFiles/test_expected_chrSyn.csv")
        bad_call = partial(
            HT.get_pairing_score_obs_exp,
            cooler_file,
            exp_f,
            50000,
            regions=position_frame,
            arms=arms,
            norm=True,
        )
        self.assertRaises(ValueError, bad_call)


class TestPairingScore(unittest.TestCase):
    """Tests for extracting ICCF pairing score"""

    def test_specific_regions_with_diag(self):
        """Tests functionality to extract ICCF values
        at specific regions from synthetic Hi-C data
        without blanking the main diagonal."""
        position_frame = pd.read_csv("testFiles/posPileups.csv")
        position_frame.loc[:, "mid"] = position_frame["pos"]
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        pairing_score = HT.get_pairing_score(
            cooler_file,
            50000,
            regions=position_frame,
            arms=arms,
            norm=False,
            blank_diag=False,
        )
        expected = pd.read_csv(
            "testFiles/test_pairingScore_ICCF_specificRegions_withDiag.csv"
        )
        assert_frame_equal(pairing_score, expected)

    def test_specific_regions_without_diag(self):  # Simulated cooler
        """Tests functionality to extract ICCF values
        at specific regions from synthetic Hi-C data
        with blanking the main diagonal."""
        position_frame = pd.read_csv("testFiles/posPileups.csv")
        position_frame.loc[:, "mid"] = position_frame["pos"]
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        pairing_score = HT.get_pairing_score(
            cooler_file,
            50000,
            regions=position_frame,
            arms=arms,
            norm=False,
            blank_diag=True,
        )
        expected = pd.read_csv(
            "testFiles/test_pairingScore_ICCF_specificRegions_withoutDiag.csv"
        )
        assert_frame_equal(pairing_score, expected)

    def test_genome_wide_not_norm_with_diag(self):
        """Tests functionality to extract ICCF values
        genome-wide without median normalization from synthetic Hi-C data
        without blanking the main diagonal."""
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        pairing_score = HT.get_pairing_score(
            cooler_file, 50000, arms=arms, norm=False, blank_diag=False
        )
        expected = pd.read_csv(
            "testFiles/test_pairingScore_ICCF_genomeWide_withDiag.csv",
            dtype={
                name: pairing_score.dtypes[name] for name in pairing_score.dtypes.index
            },
            index_col=0,
        )
        assert_frame_equal(pairing_score, expected)

    def test_genome_wide_not_norm_without_diag(self):
        """Tests functionality to extract ICCF values
        genome-wide without median normalization from synthetic Hi-C data
        with blanking the main diagonal."""
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        pairing_score = HT.get_pairing_score(
            cooler_file, 50000, arms=arms, norm=False, blank_diag=True
        )
        expected = pd.read_csv(
            "testFiles/test_pairingScore_ICCF_genomeWide_withoutDiag.csv",
            dtype={
                name: pairing_score.dtypes[name] for name in pairing_score.dtypes.index
            },
            index_col=0,
        )
        assert_frame_equal(pairing_score, expected)

    def test_genome_wide_norm_with_diag(self):
        """Tests functionality to extract ICCF values
        genome-wide with median normalization from synthetic Hi-C data
        without blanking the main diagonal."""
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        pairing_score = HT.get_pairing_score(
            cooler_file, 50000, arms=arms, norm=True, blank_diag=False
        )
        expected = pd.read_csv(
            "testFiles/test_pairingScore_ICCF_genomeWide_withDiag_norm.csv",
            dtype={
                name: pairing_score.dtypes[name] for name in pairing_score.dtypes.index
            },
            index_col=0,
        )
        assert_frame_equal(pairing_score, expected)

    def test_genome_wide_norm_without_diag(self):
        """Tests functionality to extract ICCF values
        genome-wide with median normalization from synthetic Hi-C data
        with blanking the main diagonal."""
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        pairing_score = HT.get_pairing_score(
            cooler_file, 50000, arms=arms, norm=True, blank_diag=True
        )
        expected = pd.read_csv(
            "testFiles/test_pairingScore_ICCF_genomeWide_withoutDiag_norm.csv",
            dtype={
                name: pairing_score.dtypes[name] for name in pairing_score.dtypes.index
            },
            index_col=0,
        )
        assert_frame_equal(pairing_score, expected)

    def test_wrong_parameters(self):
        """Tests raising of error when specific
        region pileup is done with the norm
        parameter set to True."""
        arms = pd.DataFrame({"chrom": "chrSyn", "start": 0, "end": 4990000}, index=[0])
        position_frame = pd.read_csv("testFiles/posPileups.csv")
        position_frame.loc[:, "mid"] = position_frame["pos"]
        cooler_file = cooler.Cooler("testFiles/test2.mcool::/resolutions/10000")
        bad_call = partial(
            HT.get_pairing_score,
            cooler_file,
            50000,
            regions=position_frame,
            arms=arms,
            norm=True,
        )
        self.assertRaises(ValueError, bad_call)


if __name__ == "__main__":
    res = unittest.main(verbosity=3, exit=False)
