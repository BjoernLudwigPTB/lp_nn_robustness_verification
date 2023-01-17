"""An implementation of a parallelized search for valid samples and seeds"""

import sys
from time import sleep

from tqdm import trange
from zema_emc_annotated.dataset import ZeMASamples  # type: ignore[import]

from lp_nn_robustness_verification.data_acquisition.activation_functions import (
    Sigmoid,
)
from lp_nn_robustness_verification.data_acquisition.generate_nn_params import (
    construct_out_features_counts,
    generate_weights_and_biases,
)
from lp_nn_robustness_verification.data_acquisition.uncertain_inputs import (
    UncertainInputs,
)
from lp_nn_robustness_verification.data_types import (
    UncertainArray,
    ValidCombinationForZeMA,
)
from lp_nn_robustness_verification.linear_program import RobustnessVerification
from lp_nn_robustness_verification.pre_processing import LinearInclusion


def find_seeds_and_samples() -> None:
    """Iterate over all possible parameter choices to find valid examples"""
    valid_seeds: dict[ValidCombinationForZeMA, int] = {}
    size_scalers: list[int] = [2000]
    depths: list[int] = [8]
    for size_scaler in size_scalers:
        zema_data = ZeMASamples(4766, size_scaler, True)
        for depth in depths:
            for idx_start in range(1):
                uncertain_inputs = UncertainInputs(
                    UncertainArray(
                        zema_data.values[idx_start], zema_data.uncertainties[idx_start]
                    )
                )
                print(
                    f"Trying to find valid seed for "
                    f"{ValidCombinationForZeMA(size_scaler, depth, idx_start)}"
                )
                for seed in trange(1):
                    linear_inclusion = LinearInclusion(
                        uncertain_inputs,
                        Sigmoid,
                        generate_weights_and_biases(
                            len(uncertain_inputs.values),
                            construct_out_features_counts(
                                len(uncertain_inputs.values), depth=depth
                            ),
                            seed,
                        ),
                    )
                    optimization = RobustnessVerification(linear_inclusion)
                    optimization.model.hideOutput()
                    optimization.solve()
                    if optimization.model.getSols():
                        valid_seeds[
                            ValidCombinationForZeMA(size_scaler, depth, idx_start)
                        ] = seed
                        break
    sleep(0.5)
    print(f"valid seeds: {valid_seeds}")


if __name__ == "__main__":
    find_seeds_and_samples()
