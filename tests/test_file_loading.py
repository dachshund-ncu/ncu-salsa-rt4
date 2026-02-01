import pytest
from ncu_salsa_rt4 import ScanSet
from .data import test_data_filenames

def test_file_loading():
    for filename in test_data_filenames:
        scan_set = ScanSet(archive_filename=filename, on_off=False)
        assert (scan_set.noOfScans == 20)

if __name__ == "__main__":
    test_file_loading()