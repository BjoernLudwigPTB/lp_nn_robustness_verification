# Installation

Before you can use lp_nn_robustness_verification, you need a working installation of 
the [SCIP Optimization suite](https://www.scipopt.org/). Please review [SCIP's 
license restrictions](https://scipopt.org/index.php#license) before installing 
lp_nn_robustness_verification.

After successfully completing all steps in [their official installation
guide](https://scipopt.org/doc/html/INSTALL.php), you can proceed here.

There is a [quick way](#quick-setup-not-recommended) to get started, but we advise 
setting up a virtual environment and guide through the process in the section
[Proper Python setup with virtual environment
](#proper-python-setup-with-virtual-environment--recommended)

## Quick setup (**not recommended**)

If you just want to use the software, the easiest way is to run from your
system's command line in the root folder of the project

```shell
pip install --user .
```

This will install the current version into your local folder of third-party libraries. 
Note that lp_nn_robustness_verification runs with **Python 
version 3.10**. Usage in any Python environment on your computer is then possible by

```python
import lp_nn_robustness_verification
```

or, for example, for the custom activation function QuadLU:

```python
from lp_nn_robustness_verification import linear_program
```

### Updating to the newest version

Updates can be installed via the same command

```shell
pip install --user .
```

## Proper Python setup with virtual environment  (**recommended**)

The setup described above allows the quick and easy use of
lp_nn_robustness_verification, but it also has its downsides. 
When working with Python we should rather always work in so-called virtual 
environments, in which our project specific dependencies are satisfied without 
polluting or breaking other projects' dependencies and to avoid breaking all your 
dependencies in case of an update of our Python distribution.

### Set up a virtual environment

If you are not familiar with [Python virtual environments
](https://docs.python.org/3/glossary.html#term-virtual-environment) you can get the
motivation and an insight into the mechanism in the
[official docs](https://docs.python.org/3/tutorial/venv.html).

You have the option to set up lp_nn_robustness_verification using the Python 
built-in tool `venv`. The commands differ slightly between [Windows
](#create-a-venv-python-environment-on-windows) and [Mac/Linux
](#create-a-venv-python-environment-on-mac--linux).

#### Create a `venv` Python environment on Windows

In your Windows PowerShell execute the following to set up a virtual environment in
a folder of your choice.

```shell
PS C:> cd C:\LOCAL\PATH\TO\ENVS
PS C:\LOCAL\PATH\TO\ENVS> py -3 -m venv lp_nn_robustness_verification_venv
PS C:\LOCAL\PATH\TO\ENVS> lp_nn_robustness_verification_venv\Scripts\activate
```

Proceed to [the next step
](#install-lp_nn_robustness_verification-via-pip).

#### Create a `venv` Python environment on Mac & Linux

In your terminal execute the following to set up a virtual environment in a folder
of your choice.

```shell
$ cd /LOCAL/PATH/TO/ENVS
$ python3 -m venv lp_nn_robustness_verification_venv
$ source lp_nn_robustness_verification_venv/bin/activate
```

Proceed to [the next step
](#install-lp_nn_robustness_verification-via-pip).

### Install lp_nn_robustness_verification via `pip`

Once you activated your virtual environment, you can install
lp_nn_robustness_verification via:

```shell
pip install .
```

```shell
Collecting lp_nn_robustness_verification
[...]
Successfully installed lp_nn_robustness_verification-[...] [...]
```

That's it!

### Install known to work dependencies' versions

In case errors arise within lp_nn_robustness_verification, 
the first thing you can try is installing the known to work configuration of 
dependencies against which we run our test suite. This you can easily achieve with 
our requirements file. This is done with the following sequence of commands after 
activating:

```shell
(lp_nn_robustness_verification_venv) $ pip install --upgrade pip-tools
Collecting pip-tools
[...]
Successfully installed pip-tools-6.11.0
(lp_nn_robustness_verification_venv) $ python --version
Python 3.10.7
(lp_nn_robustness_verification_venv) $ python -m piptools sync requirements.txt 
requirements.txt
Collecting [...]
[...]
Successfully installed [...]
(lp_nn_robustness_verification_venv) $
```

### Optional dependencies

For the different stages of using or developing lp_nn_robustness_verification we 
have provided several requirements files with pinned versions. Instead of 
`requirements.txt` in the above command, you can choose one of the following 
depending on your task:

- `[all-requirements.txt](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/blob/main/all-requirements.txt)`:
  if you want to cover all cases and simply install all optional dependencies in a
  known to work configuration
- `[examples-requirements.txt](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/blob/main/examples-requirements.txt)`: 
  for experimenting with the [provided
  examples](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/tree/main/src/lp_nn_robustness_verification/examples)
- `[dev-requirements.txt](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/blob/main/dev-requirements.txt)`:
  for starting to work on the code. Please contact
  [us](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/graphs/contributors),
  if you have any questions.
- `[docs-requirements.txt](https://github.com/BjoernLudwigPTB/lp_nn_robustness_verification/blob/main/docs-requirements.txt)`:
  if you want to build the documentation locally
