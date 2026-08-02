__doc__ = "This file contains simple set of classes that allow simple fits_files reading. There are two classes:\
    - Spectrum, which allows for reading single fits file\
    - setOfSpec, which reads a bunch of the .fits files\
For performance reasons, classes below utilize fitsio package for reading fits files."

import fitsio as fits
import numpy as np
from astropy.time import Time
import pandas as pd
import copy


class FitsSpectrum:
    def __init__(
            self,
            filename: str):
        """
        initializes the class instance
        """
        self.filename = filename
        self.read_data_from_header(self.__read_data_header(filename))

    def __read_data_header(self, filename):
        """
        reads and returns data header
        """
        return fits.FITS(filename)[1]

    def read_data_from_header(self, data_section_of_fits_file):
        """
        reads data from FITS file header
        assumes, that data header was read before
        """
        header = data_section_of_fits_file.read_header()
        self.lhcTab, self.rhcTab = self.__read_pol_data(data_section_of_fits_file.read())
        # --- reading key values ---
        self.sourcename = header['OBJECT']
        # -- DOPPLER TRACKING --
        self.Vlsr = header['VSYS']  # systemic velocity
        self.__freqRang = header['FRQ_RANG']  # frequency range
        self.__restFreq = header['FREQ'] / 1000000.0  # rest frequency (in FITS file it is in Hz, we convert it to MHz)

        try:  # TSYS
            self.__tsys1 = header['TSYS1']
            self.__tsys2 = header['TSYS2']
        except:
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
        self.iTab, self.vTab = self.__make_iv_tabs(self.lhcTab, self.rhcTab)
        # -- RMS --
        self.rmsLhc = self.__calculate_rms(self.lhcTab, [25, 300], [-300, -25])
        self.rmsRhc = self.__calculate_rms(self.rhcTab, [25, 300], [-300, -25])
        self.rmsIhc = self.__calculate_rms(self.iTab, [25, 300], [-300, -25])
        self.rmsVhc = self.__calculate_rms(self.vTab, [25, 300], [-300, -25])
        # -- doppler tracking --
        self.velocityTable = self.__generate_velocity_tab(self.Vlsr, self.__dopp_vto, self.__restFreq, self.__freqRang)

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
            "V_lsr": str(self.Vlsr),
            "Molecule": str(self.__molecule),
            "Frequency": str(self.__restFreq),
            "Band width": str(self.__freqRang),
            "Obs. time": str(self.isotime),
            "Obs. time (mjd)": str(self.mjd),
            "Epoch": str(self.__epoch),
            "RA": str(self.__ra),
            "DEC": str(self.__dec),
            "AZ": str(self.__azimuth_angle),
            "Z": str(self.__zenital_distance)
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
        freqs: np.ndarray = np.linspace(f_beg, f_beg + freq_rang, len(self.lhcTab))
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
        self.lhcTab = np.asarray(header['Pol 1'])[::-1]
        self.rhcTab = np.asarray(header['Pol 2'])[::-1]
        self.lhcTab[0:25] = 0.0
        self.lhcTab[-25:] = 0.0
        self.rhcTab[0:25] = 0.0
        self.rhcTab[-25:] = 0.0

        self.rhcTab[-1] = 0.0
        self.lhcTab[-1] = 0.0
        return self.lhcTab, self.rhcTab

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
        return pd.DataFrame(np.column_stack((self.velocityTable, self.iTab, self.lhcTab, self.rhcTab, self.vTab)),
                            columns=["Velocity", "I", "LHC", "RHC", "V"])

    def get_integrated_flux_density(self, min_chan: int, max_chan: int) -> np.ndarray:
        """
        Returns the integrated flux density of the obs, based on min and max channels
        """
        channels = np.asarray(range(1, len(self.iTab) + 1))
        indices = np.logical_and(channels > min_chan, channels < max_chan)
        velocity = self.velocityTable[indices]
        i_integrated = np.trapezoid(self.iTab[indices], velocity)
        v_integrated = np.trapezoid(self.vTab[indices], velocity)
        lhc_integrated = np.trapezoid(self.lhcTab[indices], velocity)
        rhc_integrated = np.trapezoid(self.rhcTab[indices], velocity)
        return np.asarray([self.mjd, i_integrated, v_integrated, lhc_integrated, rhc_integrated])

    def make_slice(self, indices: np.ndarray):
        """
        Returns the slice of a spectrum
        """
        new_slice = copy.deepcopy(self)
        # slice the arrays
        new_slice.iTab = new_slice.iTab[indices]
        new_slice.vTab = new_slice.vTab[indices]
        new_slice.lhcTab = new_slice.lhcTab[indices]
        new_slice.rhcTab = new_slice.rhcTab[indices]
        new_slice.velocityTable = new_slice.velocityTable[indices]
        return new_slice

    def __str__(self):
        return repr(self.mjd)



