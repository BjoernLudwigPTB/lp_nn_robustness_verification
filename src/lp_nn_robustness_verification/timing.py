"""Time the current progress of model preprocessing setup or any part of the process"""

__all__ = ["write_current_timing_stats"]

from io import StringIO

import yappi  # type: ignore[import]


def write_current_timing_stats(
    filename: str = "timings.txt", msg: str = "", mode: str = "a"
) -> None:
    """Write current content of ``yappi.YStat`` object to file

    Only useful, if in the current execution context ``yappi.start()`` was called
    beforehand, so no file is created otherwise and nothing is written anywhere.

    Parameters
    ----------
    filename : str, optional
        destination file name, can be relative to current working directory,
        defaults to "timings.txt"
    msg : str, optional
        an additional message to be written to the file before the stats are written,
        defaults to empty string
    mode : str, optional
        mode for opening and writing to file, same as with :func:`open()`, defaults
        to "a"
    """
    out = StringIO()
    yappi.get_func_stats().print_all(
        out=out, columns={0: ("name", 170), 3: ("ttot", 8)}
    )
    if out.getvalue():
        with open(filename, mode, encoding="utf-8") as timings_file:
            timings_file.write(
                f"\n==========================================================="
                f"===========================\n\n"
                f"Timings for {filename}\n{msg}"
                f"\n==========================================================="
                f"===========================\n"
            )
            timings_file.write(out.getvalue())
