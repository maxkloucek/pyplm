from setuptools import setup, find_packages

setup(
    name='pseudolm',
    version='1.0.0',
    description='Python package to perfrom psuedo-likelihood maximisation inference on Ising models',
    classifiers=['Topic :: Scientific/Engineering :: Physics',],
    packages=find_packages(
        # where='inference',
        include=['pyplm', 'pyplm.*']),
    install_requires=[
        'numpy>=1.23.5',
        'pandas>=1.5.2',
        'h5py>=3.7.0',
        'joblib>=1.1.1',
        'scikit-learn>=1.1.3',
        'tabulate>0.8.10',
        'numba>=0.56.4',
        'matplotlib>=3.6.2'
    ]
)
