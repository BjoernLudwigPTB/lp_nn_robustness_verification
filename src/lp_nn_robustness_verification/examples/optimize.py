"""This module provides the CLI for the application

At the moment it is only possible to call it via:

.. code-block:: shell

    $ pip install .
    Processing /home/bjorn/code/lp_nn_robustness_verification
      Installing build dependencies ... done
    [...]
    Successfully installed lp_nn_robustness_verification-[...]
    $ python -m lp_nn_robustness_verification.optimize

We might add command line parameters at a later time. For now please edit the main
function at the very bottom of this file to change inputs.
"""
import yappi  # type: ignore[import]
from zema_emc_annotated.dataset import ZeMASamples  # type: ignore[import]

from lp_nn_robustness_verification.data_acquisition.activation_functions import Sigmoid
from lp_nn_robustness_verification.data_acquisition.generate_nn_params import (
    construct_out_features_counts,
    generate_weights_and_biases,
)
from lp_nn_robustness_verification.data_acquisition.uncertain_inputs import (
    UncertainInputs,
)
from lp_nn_robustness_verification.data_types import UncertainArray
from lp_nn_robustness_verification.linear_program import RobustnessVerification
from lp_nn_robustness_verification.pre_processing import LinearInclusion

if __name__ == "__main__":
    SAMPLES_PER_SENSOR = 10
    DEPTH = 1
    zema_data = ZeMASamples(size_scaler=SAMPLES_PER_SENSOR, normalize=True)
    nn_params = generate_weights_and_biases(
        len(zema_data.values[0]),
        construct_out_features_counts(len(zema_data.values[0]), depth=DEPTH),
        seed=0,
    )
    for (values, uncertainties) in zip(zema_data.values, zema_data.uncertainties):
        yappi.start()
        linear_inclusion = LinearInclusion(
            uncertain_inputs=UncertainInputs(UncertainArray(values, uncertainties)),
            activation=Sigmoid,
            nn_params=nn_params,
        )
        optimization = RobustnessVerification(linear_inclusion)
        optimization.solve()
        yappi.stop()
        with open("timings.txt", "a", encoding="utf-8") as timings_file:
            timings_file.write(
                f"\n==================================================================="
                f"===================\n"
                f"Timings for {SAMPLES_PER_SENSOR * 11} inputs and {DEPTH} "
                f"{'layers' if DEPTH > 1 else 'layer'}"
                f"\n==================================================================="
                f"===================\n"
            )
            yappi.get_func_stats().print_all(
                out=timings_file, columns={0: ("name", 180), 3: ("ttot", 8)}
            )