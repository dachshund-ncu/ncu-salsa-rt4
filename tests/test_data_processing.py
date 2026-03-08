import json
import os
import pytest
import time
from ncu_salsa_rt4 import ScanSet
from .data import data_archives


def test_data_accuracy() -> None:
    """
    This util simply compares old implementation of the utils (used in simpleSingleDishDataReductor
    and the new, optimized one used here. Both should return approx same results
    """

    for archive in data_archives:
        print(f"---> Archive: {os.path.basename(archive)}")
        # load data from archive
        scan_set_old = ScanSet(archive_filename=archive, on_off=False, debug=False, use_optimized_methods=False, use_multithreaded_utils=False)
        scan_set_new = ScanSet(archive_filename=archive, on_off=False, debug=False, use_optimized_methods=True)

        # compare scan data
        for scan_old, scan_new in zip(scan_set_old.scans, scan_set_new.scans):
            new_data_full = scan_new.spectr_bbc_final.tolist()
            target_data_full = scan_old.spectr_bbc_final.tolist()
            for bbc_index in range(len(new_data_full)):
                new_data_single_bbc = new_data_full[bbc_index]
                target_data_single_bbc = target_data_full[bbc_index]
                assert target_data_single_bbc == pytest.approx(
                    new_data_single_bbc
                )


def test_processing_performance():
    old_timings = []
    for archive in data_archives:
        start_time = time.time()
        scan_set = ScanSet(archive_filename=archive, on_off=False, debug=False, use_optimized_methods=False, use_multithreaded_utils=False)
        end_time = time.time()
        old_timings.append(end_time - start_time)

    new_timings = []
    for archive in data_archives:
        start_time = time.time()
        scan_set = ScanSet(archive_filename=archive, on_off=False, debug=False, use_optimized_methods=True, use_multithreaded_utils=True)
        end_time = time.time()
        new_timings.append(end_time - start_time)

    for old_time, new_time in zip(old_timings, new_timings):
        print(f"---> Old: {old_time}, New: {new_time}")

if __name__ == "__main__":
    test_data_accuracy()
    test_processing_performance()
