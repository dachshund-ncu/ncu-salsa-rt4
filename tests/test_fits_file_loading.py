import os
import pytest
import time
from ncu_salsa_rt4 import FitsSpectrum, SetOfFitsSpectra
from .data import fits_files, fits_files_directory
import glob
import io
import numpy as np
from werkzeug.datastructures import FileStorage

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
        assert len(spec.i_tab) == 2048
        assert len(spec.lhc_tab) == 2048
        assert len(spec.rhc_tab) == 2048
        assert len(spec.v_tab) == 2048

        for i in range(25):
            assert spec.i_tab[i] == 0
            assert spec.i_tab[-i] == 0

            assert spec.lhc_tab[i] == 0
            assert spec.lhc_tab[-i] == 0

            assert spec.rhc_tab[i] == 0
            assert spec.rhc_tab[-i] == 0

            assert spec.v_tab[i] == 0
            assert spec.v_tab[-i] == 0


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
    filenames_loaded = [os.path.basename(sp.fits_input) for sp in set_of_spec.spectra]
    filenames_all = glob.glob(os.path.join(fits_files_directory, "*.fits"))
    assert flagged_filename not in filenames_loaded
    assert len(set_of_spec.spectra) == len(filenames_all)-1

def test_flask_filestorage_loading() -> None:
    """
    Tests FITS file reading, simulating passing it as:
    - BytesIO binary stream
    - FileStorage object from flask
    """
    for fits_path in fits_files:
        # -- read standard way --
        spec_og = FitsSpectrum(fits_path)
        og_header = spec_og.get_header_data()

        # -- read file as a binary stream --
        with open(fits_path, "rb") as f:
            file_bytes = f.read()
        # ----------------------------------

        # 1. Testing on a BytesIO object
        binary_stream = io.BytesIO(file_bytes)
        spec_from_stream = FitsSpectrum(binary_stream)
        stream_header = spec_from_stream.get_header_data()
        assert len(spec_og.lhc_tab) == len(spec_from_stream.lhc_tab)
        assert len(spec_og.rhc_tab) == len(spec_from_stream.rhc_tab)
        assert len(spec_og.i_tab) == len(spec_from_stream.i_tab)
        assert len(spec_og.v_tab) == len(spec_from_stream.v_tab)
        # -- validate loaded data --
        np.testing.assert_array_equal(spec_og.lhc_tab, spec_from_stream.lhc_tab)
        np.testing.assert_array_equal(spec_og.rhc_tab, spec_from_stream.rhc_tab)
        np.testing.assert_array_equal(spec_og.i_tab, spec_from_stream.i_tab)
        np.testing.assert_array_equal(spec_og.v_tab, spec_from_stream.v_tab)
        np.testing.assert_array_equal(spec_og.velocity_table, spec_from_stream.velocity_table)

        # 2. Testing on FileStorage flask object
        flask_file = FileStorage(
            stream=io.BytesIO(file_bytes),
            filename=os.path.basename(fits_path),
            content_type="application/fits",
        )
        spec_from_flask = FitsSpectrum(flask_file)
        flask_header = spec_from_flask.get_header_data()
        assert len(spec_og.lhc_tab) == len(spec_from_flask.lhc_tab)
        assert len(spec_og.rhc_tab) == len(spec_from_flask.rhc_tab)
        assert len(spec_og.i_tab) == len(spec_from_flask.i_tab)
        assert len(spec_og.v_tab) == len(spec_from_flask.v_tab)
        # -- validate loaded data --
        np.testing.assert_array_equal(spec_og.lhc_tab, spec_from_flask.lhc_tab)
        np.testing.assert_array_equal(spec_og.rhc_tab, spec_from_flask.rhc_tab)
        np.testing.assert_array_equal(spec_og.i_tab, spec_from_flask.i_tab)
        np.testing.assert_array_equal(spec_og.v_tab, spec_from_flask.v_tab)
        np.testing.assert_array_equal(spec_og.velocity_table, spec_from_flask.velocity_table)
        # BONUS: validate headers
        for key in og_header.keys():
            assert og_header[key] == flask_header[key]
            assert og_header[key] == stream_header[key]


if __name__ == "__main__":
    test_header_loading()
    test_content_loading()
    test_check_data_sorting()
    test_data_flagging()
    test_flask_filestorage_loading()