__doc__ = "This file contains simple set of classes that allow simple fits_files reading. There are two classes:\
    - Spectrum, which allows for reading single fits file\
    - setOfSpec, which reads a bunch of the .fits files\
For performance reasons, classes below utilize fitsio package for reading fits files."

import fitsio as fits
import numpy as np
from astropy.time import Time
import pandas as pd
import copy
from typing import BinaryIO, Any
from werkzeug.datastructures import FileStorage
import tempfile
import os

class FitsSpectrum:
    def __init__(
            self,
            fits_input: str | BinaryIO | FileStorage):
        """
        initializes the class instance
        """
        self.fits_original_fname = self._get_original_filename(fits_input)
        self.fits_input: str = self._preprocess_input(fits_input)
        self.read_data_from_header(self.__read_data_header(self.fits_input))

    def _get_original_filename(self, fits_input: str | BinaryIO | FileStorage) -> str:
        if hasattr(fits_input, "filename") and fits_input.filename:
            fits_file_name = os.path.basename(fits_input.filename)
        elif isinstance(fits_input, str):
            fits_file_name = os.path.basename(fits_input)
        elif hasattr(fits_input, "name") and isinstance(fits_input.name, str):  # Standardowy file object
            fits_file_name = os.path.basename(fits_input.name)
        else:
            fits_file_name = "spectrum.fits"
        return fits_file_name

    def _preprocess_input(self, fits_input: str | BinaryIO | FileStorage) -> str:
        if isinstance(fits_input, str):
            return fits_input
        # -- handle data stream file (create temporary file) --
        if hasattr(fits_input, "stream"):
            fits_input = fits_input.stream
        if hasattr(fits_input, "read"):
            fits_input.seek(0)
            self._tmp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".fits")
            self._tmp_file.write(fits_input.read())
            self._tmp_file.flush()
            fits_path = self._tmp_file.name
        else:
            raise TypeError(f"Provided data type ({type(fits_input)}) is not supported!")
        return fits_path

    def __read_data_header(self, filename: str) -> Any:
        """
        reads and returns data header
        """
        return fits.FITS(filename)[1]

    def read_data_from_header(self, data_section_of_fits_file: Any):
        """
        reads data from FITS file header
        assumes, that data header was read before
        """
        header = data_section_of_fits_file.read_header()
        self.lhc_tab, self.rhc_tab = self.__read_pol_data(data_section_of_fits_file.read())
        # --- reading key values ---
        self.sourcename = header['OBJECT']
        # -- DOPPLER TRACKING --
        self.v_lsr = header['VSYS']  # systemic velocity
        self.__freqRang = header['FRQ_RANG']  # frequency range
        self.__restFreq = header['FREQ'] / 1000000.0  # rest frequency (in FITS file it is in Hz, we convert it to MHz)

        try:  # TSYS
            self.__tsys1 = header['TSYS1']
            self.__tsys2 = header['TSYS2']
        except KeyError:
            self.__tsys1 = header['TSYS']
            self.__tsys2 = header['TSYS']
        self.tsys = np.mean([self.__tsys1, self.__tsys2])

        # ADDITIONAL INFORMATIONS (should be mostly 0.0)
        self.__dopp_vto = header['DOPP_VTO']

        # -- DATE AND TIME --
        self.isotime = header['DATE-OBS']
        t = Time(self.isotime, format='isot', scale='utc')
        self.mjd = t.mjd

        del t  # to save memory

        # -- IV STOKES PARAMS --
        self.i_tab, self.v_tab = self.__make_iv_tabs(self.lhc_tab, self.rhc_tab)
        # -- RMS --
        self.rmsLhc = self.__calculate_rms(self.lhc_tab, [25, 300], [-300, -25])
        self.rmsRhc = self.__calculate_rms(self.rhc_tab, [25, 300], [-300, -25])
        self.rmsIhc = self.__calculate_rms(self.i_tab, [25, 300], [-300, -25])
        self.rmsVhc = self.__calculate_rms(self.v_tab, [25, 300], [-300, -25])
        # -- doppler tracking --
        self.velocity_table = self.__generate_velocity_tab(self.v_lsr, self.__dopp_vto, self.__restFreq, self.__freqRang)

        # -- coordinates --
        self.__epoch = header["EQUINOX"]
        self.__ra = header['SRC_RA']
        self.__dec = header['SRC_DEC']
        self.__azimuth_angle = header["AZ"]
        self.__zenital_distance = header["Z"]

        # -- molecule --
        self.__molecule = header["MOLECULE"]


    def get_header_data(self) -> dict:
        """
        Get a dictionary with basic header data
        :return: a dictionary with fits file header data
        """
        return {
            "Source name": str(self.sourcename),
            "V_lsr (km/s)": str(self.v_lsr),
            "Molecule": str(self.__molecule),
            "Frequency (MHz)": str(self.__restFreq),
            "Bandwidth (MHz)": str(self.__freqRang),
            "Obs. time (iso)": str(self.isotime),
            "Obs. time (mjd)": str(self.mjd),
            "Epoch": str(self.__epoch),
            "RA": str(self.__ra),
            "DEC": str(self.__dec),
            "AZ": str(self.__azimuth_angle),
            "Z": str(self.__zenital_distance),
            "RMS I": str(round(self.rmsIhc, 3)),
            "RMS V": str(round(self.rmsVhc, 3)),
            "RMS LHC": str(round(self.rmsLhc, 3)),
            "RMS RHC": str(round(self.rmsRhc, 3))
        }


    def __generate_velocity_tab(
            self,
            v_lsr: float,
            dopp_vto: float,
            rest_freq: float,
            freq_rang: float) -> np.ndarray:
        """
        generates doppler-tracked velocity table
        """
        full_velocity = v_lsr + dopp_vto
        c = 299792.458  # km/s
        beta = full_velocity / c
        gamma = 1.0 / np.sqrt(1.0 - beta * beta)
        f_centr = rest_freq * (gamma * (1.0 - beta))
        f_beg = f_centr - (freq_rang / 2.0)
        # --
        freqs: np.ndarray = np.linspace(f_beg, f_beg + freq_rang, len(self.lhc_tab))
        vels = -c * ((freqs / rest_freq) - 1.0)
        # --
        vels: np.ndarray = vels[::-1]
        return vels

    def __calculate_rms(
            self,
            tab: np.ndarray,
            left_rms_chan_idxs: list[int],
            right_rms_chan_idxs: list[int]) -> float:
        """
        Calculates rms (root-mean-square) errors for the spectrum
        This is done by calculating the noise levels on the left and right part of the spectrum
        :param tab: table with a spectrum samples
        :param left_rms_chan_idxs: left-window channel indexes
        :param right_rms_chan_idxs: right-window channel indexes
        :return: rms noise value
        """
        suma: float = 0.0
        suma += np.sum(tab[left_rms_chan_idxs[0]:left_rms_chan_idxs[1]] * tab[left_rms_chan_idxs[0]:left_rms_chan_idxs[1]])
        suma += np.sum(tab[right_rms_chan_idxs[0]:right_rms_chan_idxs[1]] * tab[right_rms_chan_idxs[0]:right_rms_chan_idxs[1]])
        suma /= (abs(left_rms_chan_idxs[1] - left_rms_chan_idxs[0]) + abs(right_rms_chan_idxs[1] - right_rms_chan_idxs[0])) - 1.0
        rms = np.sqrt(suma)
        return rms

    def __read_pol_data(self, header):
        """
        Reads polarization data from a .fits file header
        Typically, Pol 1 decodes LHC and Pol 2 - RHC polarization
        Edge 25 channels are zeroed to avoid spectral edge data overflow
        :param header: a header with spectral data
        :return: lhc and rhc tables
        """
        lhc_tab = np.asarray(header['Pol 1'])[::-1]
        rhc_tab = np.asarray(header['Pol 2'])[::-1]
        lhc_tab[0:25] = 0.0
        lhc_tab[-25:] = 0.0
        rhc_tab[0:25] = 0.0
        rhc_tab[-25:] = 0.0
        return lhc_tab, rhc_tab

    def __make_iv_tabs(self, lhc_tab: np.ndarray, rhc_tab: np.ndarray):
        """
        Makes the I and V stokes parameters tables, based on input of the LHC and RHC tables data
        """
        i_tab = (lhc_tab + rhc_tab) / 2.0
        v_tab = (rhc_tab - lhc_tab) / 2.0
        return i_tab, v_tab

    def get_dataframe(self) -> pd.DataFrame:
        """
        returns a pandas dataframe with this spectrum
        """
        return pd.DataFrame(np.column_stack((self.velocity_table, self.i_tab, self.lhc_tab, self.rhc_tab, self.v_tab)),
                            columns=["Velocity", "I", "LHC", "RHC", "V"])

    def get_integrated_flux_density(self, min_chan: int, max_chan: int) -> np.ndarray:
        """
        Returns the integrated flux density of the obs, based on min and max channels
        """
        channels = np.asarray(range(1, len(self.i_tab) + 1))
        indices = np.logical_and(channels > min_chan, channels < max_chan)
        velocity = self.velocity_table[indices]
        i_integrated = np.trapezoid(self.i_tab[indices], velocity)
        v_integrated = np.trapezoid(self.v_tab[indices], velocity)
        lhc_integrated = np.trapezoid(self.lhc_tab[indices], velocity)
        rhc_integrated = np.trapezoid(self.rhc_tab[indices], velocity)
        return np.asarray([self.mjd, i_integrated, v_integrated, lhc_integrated, rhc_integrated])

    def make_slice(self, indices: np.ndarray):
        """
        Returns the slice of a spectrum
        """
        new_slice = copy.deepcopy(self)
        # slice the arrays
        new_slice.i_tab = new_slice.i_tab[indices]
        new_slice.v_tab = new_slice.v_tab[indices]
        new_slice.lhc_tab = new_slice.lhc_tab[indices]
        new_slice.rhc_tab = new_slice.rhc_tab[indices]
        new_slice.velocity_table = new_slice.velocity_table[indices]
        return new_slice

    def __str__(self):
        return repr(self.mjd)

    def __del__(self):
        if getattr(self, "_tmp_file", None) is not None:
            self._tmp_file.close()

