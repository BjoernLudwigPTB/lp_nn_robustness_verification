"""An implementation of a parallelized search for valid instances and their results"""
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
    UncertainArray,
)
from lp_nn_robustness_verification.examples.data_types import (
    IndexAndSeed,
    ScalerAndLayers,
)
from lp_nn_robustness_verification.linear_program import RobustVerifier
from lp_nn_robustness_verification.pre_processing import LinearInclusion


def find_seeds_and_samples(task_id: int, proc_id: int) -> None:
    """Iterate over all possible parameter choices to find valid examples

    Parameters
    ----------
    task_id : int
        parameter to parallelize workload, expected to lie between 0 and 4 each included
    proc_id : int
        parameter to parallelize workload, expected to lie between 0 and 3 each included
    """
    valid_seeds: dict[ScalerAndLayers, IndexAndSeed] = {}
    size_scalers: list[int] = [1, 10, 100, 1000, 2000]
    size_scaler = size_scalers[task_id]
    depths: list[int] = [1, 3, 5, 8]
    depth = depths[proc_id]
    if size_scaler * 11 - depth >= 100:
        out_features = 100
    elif size_scaler * 11 - depth < 10:
        out_features = size_scaler * 11 - depth
    else:
        out_features = 10
    zema_data = ZeMASamples(100, size_scaler, True)
    print(f"Trying to solve for {ScalerAndLayers(size_scaler, depth)}")
    for idx_start in range(100):
        uncertain_inputs = UncertainInputs(
            UncertainArray(
                zema_data.values[idx_start], zema_data.uncertainties[idx_start]
            )
        )
        if valid_seeds.get(ScalerAndLayers(size_scaler, depth)) is not None:
            print(f"valid seeds: {valid_seeds}")
            with open(
                f"{size_scaler}_{depth}.txt", "a", encoding="utf-8"
            ) as valid_seeds_file:
                valid_seeds_file.write(str(valid_seeds) + "\n")
                break
        for seed in range(100):
            linear_inclusion = LinearInclusion(
                uncertain_inputs,
                Sigmoid,
                generate_weights_and_biases(
                    len(uncertain_inputs.values),
                    construct_out_features_counts(
                        len(uncertain_inputs.values),
                        out_features=out_features,
                        depth=depth,
                    ),
                    seed,
                ),
            )
            optimization = RobustVerifier(linear_inclusion)
            optimization.solve()
            if optimization.model.getSols():
                valid_seeds[ScalerAndLayers(size_scaler, depth)] = IndexAndSeed(
                    idx_start, seed
                )
                optimization.model.writeProblem(
                    filename=f"solved_problem_for_{size_scaler}_inputs_and_"
                    f"{depth}_layers_with_sample_{idx_start}_and_seed_{seed}.cip"
                )
                optimization.model.writeProblem(
                    filename=f"solved_transformed_problem_for_{size_scaler}_inputs_and_"
                    f"{depth}_layers_with_sample_{idx_start}_and_seed_{seed}.cip",
                    trans=True,
                )
                optimization.model.writeBestSol(
                    filename=f"best_solution_for_{size_scaler}_inputs_and_"
                    f"{depth}_layers_with_sample_{idx_start}_and_seed_{seed}.sol"
                )
                optimization.model.writeBestTransSol(
                    filename=f"best_solution_for_transformed_problem"
                    f"_{size_scaler}_inputs_and_"
                    f"{depth}_layers_with_sample_{idx_start}_and_seed_{seed}.sol"
                )
                break


if __name__ == "__main__":
    find_seeds_and_samples(int(sys.argv[1]), int(sys.argv[2]))
