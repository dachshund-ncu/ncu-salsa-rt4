import json
import os

import pytest

from ncu_salsa_rt4 import ScanSet

from .data import target_data_filenames, test_data_filenames


def test_data_accuracy():
    for archive, json_target in zip(test_data_filenames, target_data_filenames):
        # load data from archive
        scan_set = ScanSet(archive_filename=archive, on_off=False, debug=False)
        # load data from json file
        with open(json_target, "r+") as fle:
            data_list = json.load(fle)

        # compare scan data
        for i in range(len(scan_set.scans)):
            generated_data_full = scan_set.scans[i].spectr_bbc_final.tolist()
            target_data_full = data_list[i]
            for bbc_no in range(len(generated_data_full)):
                generated_data_single_bbc = generated_data_full[bbc_no]
                target_data_single_bbc = target_data_full[f"bbc_{bbc_no + 1}"]
                assert target_data_single_bbc == pytest.approx(
                    generated_data_single_bbc
                )
        print(f"--> {os.path.basename(archive)}, {os.path.basename(json_target)}")


if __name__ == "__main__":
    test_data_accuracy()
