import time

import pytest

from ncu_salsa_rt4 import ScanSet

from .data import data_archives


def test_file_loading():
    for filename in data_archives:
        start_time = time.time()
        scan_set = ScanSet(archive_filename=filename, on_off=False, debug=True)
        end_time = time.time()
        print(f"Loaded {filename} in {end_time - start_time:.2f} seconds.")
        assert scan_set.noOfScans > 0


def test_file_isostrings():
    for filename in data_archives:
        scan_set = ScanSet(archive_filename=filename, on_off=False, debug=False)
        for scan in scan_set.scans:
            print(scan.isotime)


if __name__ == "__main__":
    test_file_loading()
    test_file_isostrings()
