import setuptools

setuptools.setup(
    name="ngs",
    version="0.3",
    author="Michael Mitter",
    author_email="michael_mitter@hotmail.com",
    description="Scripts for NGS analysis",
    long_description="Scripts for NGS analysis",
    long_description_content_type="",
    url="https://github.com/gerlichlab/ngs",
    packages=setuptools.find_packages(),
    install_requires=[
        "multiprocess",
        "cooltools==0.4.0",
        "pandas==1.2.4",
        "bioframe==v0.2.0",
        "cooler",
        "pairtools",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
