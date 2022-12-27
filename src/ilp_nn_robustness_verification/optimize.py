"""This module provides the CLI for the application

At the moment it is only possible to call it via:

.. code-block:: shell

    $ pip install .
    Processing /home/bjorn/code/ilp_nn_robustness_verification
      Installing build dependencies ... done
    [...]
    Successfully installed ilp_nn_robustness_verification-[...]
    $ python -m ilp_nn_robustness_verification.optimize

We might add command line parameters at a later time. For now please edit the main
function at the very bottom of this file to change inputs.
"""
import numpy as np
from zema_emc_annotated.dataset import provide_zema_samples  # type: ignore[import]

from ilp_nn_robustness_verification.data_acquisition.uncertain_inputs import (
    UncertainInputs,
)
from ilp_nn_robustness_verification.data_types import NNParams, UncertainArray
from ilp_nn_robustness_verification.ilp import RobustnessVerification
from ilp_nn_robustness_verification.pre_processing import LinearInclusion

if __name__ == "__main__":
    zema_data = provide_zema_samples()
    for (values, uncertainties) in zip(zema_data.values, zema_data.uncertainties):
        linear_inclusion = LinearInclusion(
            uncertain_inputs=UncertainInputs(UncertainArray(values, uncertainties)),
            nn_params=NNParams(
                np.zeros_like(values)[np.newaxis, ...],
                np.eye(len(uncertainties))[np.newaxis, ...],
            ),
        )
        optimization_model = RobustnessVerification(linear_inclusion)
        print(optimization_model.solve())
