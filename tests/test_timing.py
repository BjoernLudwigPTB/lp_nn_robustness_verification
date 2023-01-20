import os
from inspect import signature
from pathlib import Path
from typing import Generator

import pytest
import yappi  # type: ignore[import]

from lp_nn_robustness_verification import timing
from lp_nn_robustness_verification.timing import write_current_timing_stats


@pytest.fixture
def tmp_file_name(tmp_path: Path) -> str:
    temporary_file_name = str(tmp_path.joinpath("test_timings.txt"))
    assert not os.path.exists(temporary_file_name)
    return temporary_file_name


@pytest.fixture
def measure_something() -> Generator[None, None, None]:
    yappi.start()
    print("Do something")
    yappi.stop()
    yield
    yappi.clear_stats()


@pytest.fixture(scope="session")
def any_string() -> str:
    return "This is pretty arbitrary."


def test_write_current_timing_stats_in_all() -> None:
    assert write_current_timing_stats.__name__ in timing.__all__


def test_write_current_timing_stats_expects_filename() -> None:
    assert "filename" in signature(write_current_timing_stats).parameters


def test_write_current_timing_stats_expects_filename_to_be_str() -> None:
    assert (
        signature(write_current_timing_stats).parameters["filename"].annotation is str
    )


def test_write_current_timing_stats_filename_default_is_timings_txt() -> None:
    assert (
        signature(write_current_timing_stats).parameters["filename"].default
        == "timings.txt"
    )


def test_write_current_timing_stats_expects_msg() -> None:
    assert "msg" in signature(write_current_timing_stats).parameters


def test_write_current_timing_stats_expects_msg_to_be_str() -> None:
    assert signature(write_current_timing_stats).parameters["msg"].annotation is str


def test_write_current_timing_stats_msg_default_is_timings_txt() -> None:
    assert signature(write_current_timing_stats).parameters["msg"].default == ""


def test_write_current_timing_stats_filename_states_to_return_nothing() -> None:
    assert signature(write_current_timing_stats).return_annotation is None


def test_print_current_timing_stats_creates_file_if_started(
    tmp_file_name: str, measure_something: Generator[None, None, None]
) -> None:

    write_current_timing_stats(tmp_file_name)
    assert os.path.exists(tmp_file_name)


def test_print_current_timing_stats_does_not_create_file_if_not_started(
    tmp_file_name: str,
) -> None:
    write_current_timing_stats(tmp_file_name)
    assert not os.path.exists(tmp_file_name)


def test_print_current_timing_stats_does_not_create_file_with_msg_if_not_started(
    tmp_file_name: str, any_string: str
) -> None:
    write_current_timing_stats(tmp_file_name, any_string)
    assert not os.path.exists(tmp_file_name)


def test_print_current_timing_stats_creates_file_with_message(
    tmp_file_name: str, measure_something: Generator[None, None, None], any_string: str
) -> None:
    write_current_timing_stats(tmp_file_name, any_string)
    with open(tmp_file_name, "r") as read_file:
        assert any_string in read_file.read()


def test_print_current_timing_stats_creates_file_with_stats(
    tmp_file_name: str, measure_something: Generator[None, None, None]
) -> None:
    write_current_timing_stats(tmp_file_name)
    with open(tmp_file_name, "r") as read_file:
        assert (
            "Clock type: CPU\nOrdered by: totaltime, desc\n\nname" in read_file.read()
        )
