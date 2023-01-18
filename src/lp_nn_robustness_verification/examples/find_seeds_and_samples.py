"""An implementation of a parallelized search for valid samples and seeds"""
import fcntl
import sys

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


def find_seeds_and_samples(task_id: int, proc_id: int) -> None:
    """Iterate over all possible parameter choices to find valid examples

    Parameters
    ----------
    task_id : int
        expected to lie between 0 and 6 each included
    proc_id : int
        expected to lie between 0 and 27 each included
    """
    valid_seeds: dict[ValidCombinationForZeMA, IndexAndSeed] = {}
    size_scalers: list[int] = [1]
    depths: list[int] = [1, 3, 5, 8]
    for size_scaler in size_scalers:
        zema_data = ZeMASamples(100, size_scaler, True)
        for depth in depths:
            print(
                f"Trying to find seed for {ValidCombinationForZeMA(size_scaler, depth)}"
            )
            for idx_start in range(100):
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
                    with open(f"{size_scaler}_{depth}.txt", "w") as valid_seeds_file:
                        fcntl.flock(valid_seeds_file, fcntl.LOCK_EX)
                        valid_seeds_file.write(
                            f"{ValidCombinationForZeMA(size_scaler, depth)}\n"
                        )
                        fcntl.flock(valid_seeds_file, fcntl.LOCK_UN)
                    break
                for seed in range(
                    100000 // (28 * 7) * (task_id * 28 + proc_id),
                    100000 // (28 * 7) * (task_id * 28 + proc_id + 1),
                ):
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
                        ] = IndexAndSeed(idx_start, seed)
                        break


if __name__ == "__main__":
    find_seeds_and_samples(int(sys.argv[1]), int(sys.argv[2]))
