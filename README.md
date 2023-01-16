## PyPLM: a Pseudo-Likelihood Maximization (PLM) python package

This python package implements the Pseudo-Likelihood Maximization (PLM) technique to solve the inverse Ising problem (equivalently known as pairwise Maximum Entropy modelling). The method allows the interaction network of systems of binary variables to be reconstructed, and relies on performing logistic regression.

PyPLM is structured around a single pipeline (the data_pipeline), which allows data to be generated and inferred. PyPLM also implements Firth's logistic regression penalty.

### Installation

```
git clone https://github.com/maxkloucek/pyplm.git
cd pyplm
pip install .
```

### Usage Examples

Please see example_2DIsing.py for an overview of how to use the pyplm data_pipeline. All data is stored using the hdf5 file format. To perform some elementary analysis, first run `python example_2DIsing.py`, and then see the  `example_2DIsing_analysis.ipynb` notebook (requires jupyter notebook to be installed).

If you use this package for any scientific purpose then please cite: 
https://arxiv.org/abs/2301.05556
