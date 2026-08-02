import os
import pytest
import time
from ncu_salsa_rt4 import FitsSpectrum, SetOfFitsSpectra
from .data import fits_files, fits_files_directory
import glob

DE_CAT = os.path.dirname(os.path.abspath(__file__))


def test_header_loading() -> None:
    for fits_file in fits_files:
        spec = FitsSpectrum(fits_file)
        for key, item in spec.get_header_data().items():
            assert isinstance(item, str)
            print(f"{key}: {item}")
        assert spec.get_header_data()["Molecule"] == "CH3OH 6668"

def test_content_loading() -> None:
    for fits_file in fits_files:
        spec = FitsSpectrum(fits_file)
        assert len(spec.iTab) == 2048
        assert len(spec.lhcTab) == 2048
        assert len(spec.rhcTab) == 2048
        assert len(spec.vTab) == 2048

        for i in range(25):
            assert spec.iTab[i] == 0
            assert spec.iTab[-i] == 0

            assert spec.lhcTab[i] == 0
            assert spec.lhcTab[-i] == 0

            assert spec.rhcTab[i] == 0
            assert spec.rhcTab[-i] == 0

            assert spec.vTab[i] == 0
            assert spec.vTab[-i] == 0


def test_check_data_sorting() -> None:
    # load data
    set_of_spec = SetOfFitsSpectra(
        cat_with_source=fits_files_directory
    )
    # check if data is sorted properly
    last_mjd = 0
    for sp in set_of_spec.spectra:
        assert sp.mjd > last_mjd
        last_mjd = sp.mjd

def test_data_flagging() -> None:
    # load data
    set_of_spec = SetOfFitsSpectra(
        cat_with_source=fits_files_directory
    )
    flagged_filename = "g32p74_61097242.fits"
    filenames_loaded = [os.path.basename(sp.filename) for sp in set_of_spec.spectra]
    filenames_all = glob.glob(os.path.join(fits_files_directory, "*.fits"))
    assert flagged_filename not in filenames_loaded
    assert len(set_of_spec.spectra) == len(filenames_all)-1

if __name__ == "__main__":
    test_header_loading()
    test_content_loading()
    test_check_data_sorting()
    test_data_flagging()