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
from lp_nn_robustness_verification.linear_program import RobustVerifier
from lp_nn_robustness_verification.pre_processing import LinearInclusion


def optimize() -> None:
    """Solve one specific hard coded instance and time the process"""
    samples_per_sensor = 10
    depth = 1
    zema_data = ZeMASamples(size_scaler=samples_per_sensor, normalize=True)
    nn_params = generate_weights_and_biases(
        len(zema_data.values[0]),
        construct_out_features_counts(len(zema_data.values[0]), depth=depth),
        seed=0,
    )
    for (values, uncertainties) in zip(zema_data.values, zema_data.uncertainties):
        yappi.start()
        linear_inclusion = LinearInclusion(
            uncertain_inputs=UncertainInputs(UncertainArray(values, uncertainties)),
            activation=Sigmoid,
            nn_params=nn_params,
        )
        optimization = RobustVerifier(linear_inclusion)
        optimization.solve()
        yappi.stop()
        with open(
            (
                f"{samples_per_sensor * 11}_inputs_and_"
                f"{depth}_layers_with_sample_0_and_seed_0_timings.txt"
            ),
            "a",
            encoding="utf-8",
        ) as timings_file:
            timings_file.write(
                f"\n==================================================================="
                f"===================\n"
                f"Timings for {samples_per_sensor * 11} inputs and "
                f"{depth} {'layers' if depth > 1 else 'layer'} with sample 0 and seed 0"
                f"\n==================================================================="
                f"===================\n"
            )
            yappi.get_func_stats().print_all(
                out=timings_file, columns={0: ("name", 180), 3: ("ttot", 8)}
            )


if __name__ == "__main__":
    optimize()
