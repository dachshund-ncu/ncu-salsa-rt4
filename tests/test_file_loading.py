import pytest
from ncu_salsa_rt4 import ScanSet
from .data import test_data_filenames
import time

def test_file_loading():
    for filename in test_data_filenames:
        start_time = time.time()
        scan_set = ScanSet(archive_filename=filename, on_off=False, debug=True)
        end_time = time.time()
        print(f"Loaded {filename} in {end_time - start_time:.2f} seconds.")
        assert (scan_set.noOfScans == 20)

if __name__ == "__main__":
    test_file_loading()