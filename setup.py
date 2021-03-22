import setuptools

setuptools.setup(
    name="ngs",
    version="0.2",
    author="Michael Mitter",
    author_email="michael_mitter@hotmail.com",
    description="Scripts for NGS analysis",
    long_description="Scripts for NGS analysis",
    long_description_content_type="",
    url="https://github.com/gerlichlab/ngs",
    packages=setuptools.find_packages(),
    install_requires=[
        "multiprocess",
        "cooltools",
        "pandas",
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
