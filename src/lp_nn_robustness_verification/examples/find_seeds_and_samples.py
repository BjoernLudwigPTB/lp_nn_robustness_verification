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
    IndexAndSeed,
    UncertainArray,
    ValidCombinationForZeMA,
)
from lp_nn_robustness_verification.linear_program import RobustnessVerification
from lp_nn_robustness_verification.pre_processing import LinearInclusion


def find_seeds_and_samples(task_id: int) -> None:
    """Iterate over all possible parameter choices to find valid examples"""
    seeds = trange(90000 * task_id // 144, 90000 * (task_id + 1) // 144)
    valid_seeds: dict[ValidCombinationForZeMA, IndexAndSeed] = {}
    size_scalers: list[int] = [100]
    depths: list[int] = [3]
    for size_scaler in size_scalers:
        zema_data = ZeMASamples(4766, size_scaler, True)
        for depth in depths:
            for idx_start in range(1):
                uncertain_inputs = UncertainInputs(
                    UncertainArray(
                        zema_data.values[idx_start], zema_data.uncertainties[idx_start]
                    )
                )
                if (
                    valid_seeds.get(ValidCombinationForZeMA(size_scaler, depth))
                    is not None
                ):
                    print(f"valid seeds: {valid_seeds}")
                    break
                for seed in seeds:
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
                            ValidCombinationForZeMA(size_scaler, depth)
                        ] = IndexAndSeed(
                            idx_start,
                            seed,
                        )
                        break


if __name__ == "__main__":
    find_seeds_and_samples(int(sys.argv[1]))
