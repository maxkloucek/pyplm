from setuptools import setup, find_packages

setup(
    name='pyplm',
    version='0.1.0',
    packages=find_packages(
        # where='inference',
        include=['pyplm', 'pyplm.*']),
    install_requires=[
        # 'cycler>=0.10.0',
        # 'joblib>=1.0.1',
        # 'kiwisolver>=1.3.1',
        # 'llvmlite>=0.36.0',
        # 'matplotlib>=3.4.1',
        # 'numba>=0.53.1',
        # 'numpy>=1.20.2',
        # 'Pillow>=8.2.0',
        # 'pyparsing>=2.4.7',
        # 'python-dateutil>=2.8.1',
        # 'scikit-learn>=0.24.1',
        # 'scipy>=1.6.2',
        # 'six>=1.15.0',
        # 'threadpoolctl>=2.1.0',
        'h5py>=3.2.1'
    ]
)
