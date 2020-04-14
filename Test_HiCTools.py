import unittest
import pandas as pd
from NGS import HiCTools as HT
from pandas.testing import assert_frame_equal

class TestResources(unittest.TestCase):
    def testArms(self):
        check = pd.read_csv("./testFiles/arms.csv")
        arms = HT.getArmsHg19()
        assert_frame_equal(check, arms)


def suiteAll():
    suite = unittest.TestSuite()
    suite.addTest(TestResources("testArms"))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suiteAll())
