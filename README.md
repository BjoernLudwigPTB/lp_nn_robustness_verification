# Neural network robustness verification via Linear Programming

[![pipeline status](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/badges/main/pipeline.svg)](https://gitlab1.ptb.de/ludwig10_masters_thesis/lp_nn_robustness_verification/-/commits/main)
[![Latest Release](https://img.shields.io/github/v/release/BjoernLudwigPTB/lp_nn_robustness_verification?label=Latest%20release)](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/releases/latest)
[![DOI](https://zenodo.org/badge/587113361.svg)](https://zenodo.org/badge/latestdoi/587113361)

This is the code written in conjunction with the second part of my Master's thesis on 
GUM-compliant neural network robustness verification. The code was written for 
_Python 3.10_.

The final submission date is 23. January 2023. Until then, this code base will be 
subject to constant change.

## Getting started

The [INSTALL guide](INSTALL.md) assists in installing the required packages.
After that you might want to have a look at our [optimization script
](./src/lp_nn_robustness_verification/optimize.py) or the provided [notebook
](./src/lp_nn_robustness_verification/examples/linear_inclusion.ipynb) to get a 
feeling for how to use the software.

## Documentation

To locally build the HTML or pdf documentation first the required dependencies need 
to be installed into your virtual environment (check the [INSTALL guide](INSTALL.md) 
first and upon completion execute the following):

```shell
(venv) $ python -m piptools sync docs-requirements.txt
(venv) $ sphinx-build docs/ docs/_build
sphinx-build docs/ docs/_build
Running Sphinx v5.3.0
loading pickled environment... done
[...]
The HTML pages are in docs/_build.
```

After that the documentation can be viewed by opening the file
_docs/\_build/index.html_ in any browser.

## Roadmap

- check what improvements are made by switching to optimizable variables for the $r_i$ s