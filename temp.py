from NGS import HiCTools as HT


arms = HT.getArmsHg19()

arms.to_csv("./testFiles/arms.csv", index=False)