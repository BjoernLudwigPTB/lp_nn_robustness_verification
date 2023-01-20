"""An implementation of a parallelized search for valid instances and their results"""
import sys

import yappi  # type: ignore[import]
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
from lp_nn_robustness_verification.data_types import UncertainArray
from lp_nn_robustness_verification.linear_program import RobustVerifier
from lp_nn_robustness_verification.pre_processing import LinearInclusion


def solve_and_store_timed_solutions(task_id: int, proc_id: int) -> None:
    """Iterate over all possible parameter choices to find valid examples

    Parameters
    ----------
    task_id : int
        parameter to parallelize workload, expected to lie between 0 and 7 each included
    proc_id : int
        parameter to parallelize workload, expected to lie between 0 and 1 each included
    """
    size_scalers: list[int] = [1000, 2000]
    size_scaler = size_scalers[task_id // 4]
    depths: list[int] = [1, 3, 5, 8]
    depth = depths[task_id % 4]
    if size_scaler * 11 - depth >= 100:
        out_features = 100
    elif size_scaler * 11 - depth < 10:
        out_features = size_scaler * 11 - depth
    else:
        out_features = 10
    zema_data = ZeMASamples(100, size_scaler, True)
    print(
        f"Trying to solve for {size_scaler * 11} inputs and {depth} "
        f"{'layers' if depth > 1 else 'layer'}"
    )
    solved = False
    for idx_start in range(100):
        if solved:
            break
        uncertain_inputs = UncertainInputs(
            UncertainArray(
                zema_data.values[idx_start], zema_data.uncertainties[idx_start]
            )
        )
        for seed in range(100):
            yappi.start()
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
            yappi.stop()
            if optimization.model.getSols():
                solved = True
                with open(
                    f"{size_scaler * 11}_inputs_and_{depth}_layers_with_sample_"
                    f"{idx_start}_and_seed_{seed}_"
                    f"timings.txt",
                    "a",
                    encoding="utf-8",
                ) as timings_file:
                    timings_file.write(
                        f"\n==========================================================="
                        f"===========================\n"
                        f"Timings for {size_scaler * 11} inputs and {depth} "
                        f"{'layers' if depth > 1 else 'layer'} with sample 0 and seed 0"
                        f"\n==========================================================="
                        f"===========================\n"
                    )
                    yappi.get_func_stats().print_all(
                        out=timings_file, columns={0: ("name", 180), 3: ("ttot", 8)}
                    )
                optimization.model.writeProblem(
                    filename=(
                        f"{size_scaler * 11}_inputs_and_{depth}_layers_with_sample_"
                        f"{idx_start}_and_seed_{seed}_"
                        f"solved_problem.cip"
                    )
                )
                optimization.model.writeProblem(
                    filename=(
                        f"{size_scaler * 11}_inputs_and_{depth}_layers_with_sample_"
                        f"{idx_start}_and_seed_{seed}_"
                        f"solved_transformed_problem.cip"
                    ),
                    trans=True,
                )
                optimization.model.writeBestSol(
                    filename=(
                        f"{size_scaler * 11}_inputs_and_{depth}_layers_with_sample_"
                        f"{idx_start}_and_seed_{seed}_"
                        f"best_solution.sol"
                    )
                )
                optimization.model.writeBestTransSol(
                    filename=(
                        f"{size_scaler * 11}_inputs_and_{depth}_layers_with_sample_"
                        f"{idx_start}_and_seed_{seed}_"
                        f"best_solution_for_transformed_problem.sol"
                    )
                )
                break


if __name__ == "__main__":
    solve_and_store_timed_solutions(int(sys.argv[1]), int(sys.argv[2]))
