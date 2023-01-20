import os
from glob import glob
from itertools import chain
from typing import Callable

import pytest


@pytest.fixture(scope="session")
def file_deleter() -> Callable[[tuple[str, ...]], None]:
    def deleter(endings: tuple[str, ...]) -> None:
        for file in chain(*(glob(f"*{ending}") for ending in endings)):
            try:
                os.remove(file)
            except FileNotFoundError:
                pass

    return deleter
