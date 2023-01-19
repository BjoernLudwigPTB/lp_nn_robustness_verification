import os
from glob import glob
from itertools import chain
from typing import Callable, Generator

import pytest

from lp_nn_robustness_verification.examples.solve_instances_in_parallel import (
    solve_and_store_timed_solutions,
)
from lp_nn_robustness_verification.examples.solve_one_instance import optimize


@pytest.fixture(scope="module")
def file_deleter() -> Callable[[tuple[str, ...]], None]:
    def deleter(endings: tuple[str, ...]) -> None:
        for file in chain(
            *(glob(f"*{ending}") for ending in endings)
        ):
            try:
                os.remove(file)
            except FileNotFoundError:
                pass

    return deleter


@pytest.fixture
def cleanup_txt_sol_and_cip_after_run(
    file_deleter: Callable[[tuple[str, ...]], None]
) -> Generator[None, None, None]:
    yield
    file_deleter(("*_timings.txt", ".sol", ".cip"))


@pytest.fixture
def cleanup_txt_after_run(
    file_deleter: Callable[[tuple[str, ...]], None]
) -> Generator[None, None, None]:
    yield
    file_deleter(("*_timings.txt",))


def test_solve_and_store_timed_solutions(
    cleanup_txt_sol_and_cip_after_run: Generator[None, None, None]
) -> None:
    solve_and_store_timed_solutions(0, 0)


def test_optimize(cleanup_txt_after_run: Generator[None, None, None]) -> None:
    optimize()
